import os
import sys
import math
import numpy as np
import pandas as pd
from PIL import Image
from math import floor
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
# print(BASE_DIR)
sys.path.append(BASE_DIR)
from utils.utils import *
from utils.file_utils import save_hdf5
from datasets.wsi_dataset import Wsi_Region
from wsi_core.WholeSlideImage import WholeSlideImage


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile

def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level = -1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)
    
    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)
    
    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap

def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object

# TODO: Debug here!
def compute_from_patches(wsi_object, clam_pred=None, model=None, feature_extractor=None, batch_size=512,  
    attn_save_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):    
    top_left = wsi_kwargs['top_left']
    bot_right = wsi_kwargs['bot_right']
    patch_size = wsi_kwargs['patch_size']
    
    roi_dataset = Wsi_Region(wsi_object, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size)
    print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    print('number of batches: ', len(roi_loader))
    mode = "w"
    print("Model type: ", model)

    from datasets.dataset_survival import Generic_MIL_Survival_Dataset
    dataset = Generic_MIL_Survival_Dataset(
        csv_path="/data/lmx/CMT/dataset_csv/tcga_blca_survival_prediction_all_clean.csv",
        data_dir="/data/lmx/Dataset/TCGA-DATASET/FEATURES_DIRECTORY/tcga_blca/pt_files",
        shuffle=False,
        seed=1,
        print_info=True,
        patient_strat=False,
        n_bins=4,
        label_col='survival_months',
        ignore=[],
    )

    train_dataset, val_dataset = dataset.return_splits(from_id=False,
                                                       csv_path='/data/lmx/CMT/splits/5foldcv/tcga_blca/splits_0.csv')
    for i in range(5):
        train_dataset.set_split_id(split_id=i)
        val_dataset.set_split_id(split_id=i)
    print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
    datasets = (train_dataset, val_dataset)
    train_splits, val_splits = datasets
    omic_loader = get_split_loader(val_splits, batch_size=1)

    for idx, (roi, coords) in enumerate(roi_loader):
        roi = roi.to(device)
        coords = coords.numpy()
        
        with torch.no_grad():
            features = feature_extractor(roi)

            if attn_save_path is not None:
                for batch_idx, (data_WSI, data_omic, label, event_time, censor) in enumerate(omic_loader):
                    data_omic = data_omic.type(torch.FloatTensor).to(device)
                    logits, Y_prob, _, _, Y_hat, A = model(features, data_omic)

                # A = model(features, attention_only=True)  # (2,512)

                # if A.size(0) > 1: #CLAM multi-branch attention # (2)
                #     A = A[clam_pred] # (512,)

                A = A.view(-1, 1).cpu().numpy() # (512,1)

                if ref_scores is not None:
                    for score_idx in range(len(A)):
                        A[score_idx] = score2percentile(A[score_idx], ref_scores)

                asset_dict = {'attention_scores': A, 'coords': coords}
                save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
    
        if idx % math.ceil(num_batches * 0.05) == 0:
            print('procssed {} / {}'.format(idx, num_batches))

        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            save_hdf5(feat_save_path, asset_dict, mode=mode)

        mode = "a"
    return attn_save_path, feat_save_path, wsi_object 