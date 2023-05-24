from collections import OrderedDict
from os.path import join
import pdb
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(BASE_DIR)
sys.path.append(BASE_DIR)
from models.model_utils import *


##########################
#### MaskedMLP ####
##########################
class MaskedOmics(nn.Module):
    def __init__(
        self, 
        device="cpu",
        df_comp=None,
        input_dim=1577,
        dim_per_path_1=8,
        dim_per_path_2=16,
        dropout=0.1,
        num_classes=4,
        ):
        super(MaskedOmics, self).__init__()

        self.df_comp = df_comp
        self.input_dim = input_dim
        self.dim_per_path_1 = dim_per_path_1
        self.dim_per_path_2 = dim_per_path_2
        self.dropout = dropout
        self.num_classes = num_classes

        #---> mask_1
        # df = [genes, pathways]
        self.num_genomics = self.df_comp
        # self.num_genomics = self.df_comp.shape[1]
        M_raw = torch.Tensor(self.df_comp.values)
        self.mask_1 = torch.repeat_interleave(M_raw, self.dim_per_path_1, dim=1)

        self.fc_1_weight = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(self.input_dim, self.dim_per_path_1*self.num_genomics)))
        self.fc_1_bias = nn.Parameter(torch.rand(self.dim_per_path_1*self.num_genomics))

        self.fc_2_weight = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(self.dim_per_path_1*self.num_genomics, self.dim_per_path_2*self.num_genomics)))
        self.fc_2_bias = nn.Parameter(torch.rand(self.dim_per_path_2*self.num_genomics))

        self.mask_2 = np.zeros([self.dim_per_path_1*self.num_genomics, self.dim_per_path_2*self.num_genomics])
        for (row, col) in zip(range(0, self.dim_per_path_1*self.num_genomics, self.dim_per_path_1), range(0, self.dim_per_path_2*self.num_genomics, self.dim_per_path_2)):
            self.mask_2[row:row+self.dim_per_path_1, col:col+self.dim_per_path_2] = 1
        self.mask_2 = torch.Tensor(self.mask_2)

        #---> to_logits 
        self.to_logits = nn.Sequential(
            nn.Linear(self.num_genomics*self.dim_per_path_2, self.num_genomics*self.dim_per_path_2//4), nn.ReLU(), nn.Dropout(self.dropout),    
            nn.Linear(self.num_genomics*self.dim_per_path_2//4, self.num_classes)
        )
        
        #---> manually put on device
        self.fc_1_weight.to(device)
        self.fc_1_bias.to(device)
        self.mask_1 = self.mask_1.to(device)

        self.fc_2_weight.to(device)
        self.fc_2_bias.to(device)
        self.mask_2 = self.mask_2.to(device)

        # self.enc.to(device)
        self.to_logits.to(device)

    def forward(self, **kwargs):

        x = kwargs['x_omic']

        #---> apply mask to fc_1 and apply fc_1
        out = torch.matmul(x, self.fc_1_weight * self.mask_1) + self.fc_1_bias

        #---> apply mask to fc_2 and apply fc_2
        out = torch.matmul(out, self.fc_2_weight * self.mask_2) + self.fc_2_bias

        #---> get logits
        logits = self.to_logits(out).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat, None, None
    

##########################
#### MLP for Genomics ####
##########################
class MLPOmics(nn.Module):
    def __init__(
        self, 
        input_dim,
        n_classes=4, 
        projection_dim = 512, 
        dropout = 0.1, 
        ):
        super(MLPOmics, self).__init__()
        
        # self
        self.projection_dim = projection_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(projection_dim//2, projection_dim//2), nn.ReLU(), nn.Dropout(dropout)
        ) 

        self.to_logits = nn.Sequential(
                nn.Linear(projection_dim//2, n_classes)
            )

    def forward(self, **kwargs):
        self.cuda()

        #---> unpack
        data_omics = kwargs["x_omic"].float().cuda().squeeze()
        
        #---> project omics data to projection_dim/2
        data = self.net(data_omics) #[B, n]

        #---->predict
        # logits = self.to_logits(data) #[B, n_classes]
        logits = self.to_logits(data).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat, None, None
    
    def captum(self, omics):

        self.cuda()

        #---> unpack
        data_omics = omics.float().cuda().squeeze()
        
        #---> project omics data to projection_dim/2
        data = self.net(data_omics) #[B, n]

        #---->predict
        logits = self.to_logits(data) #[B, n_classes]

        #---> get risk 
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)

        #---> return risk 
        return risk
    

##########################
#### Genomic FC Model ####
##########################
class SNN(nn.Module):
    def __init__(self, input_dim: int, model_size_omic: str='small', n_classes: int=4):
        super(SNN, self).__init__()
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256, 256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        self.classifier = nn.Linear(hidden[-1], n_classes)
        init_max_weights(self)


    def forward(self, **kwargs):
        x = kwargs['x_omic']
        features = self.fc_omic(x)

        logits = self.classifier(features).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat, None, None

    def relocate(self):
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if torch.cuda.device_count() > 1:
                device_ids = list(range(torch.cuda.device_count()))
                self.fc_omic = nn.DataParallel(self.fc_omic, device_ids=device_ids).to('cuda:0')
            else:
                self.fc_omic = self.fc_omic.to(device)


            self.classifier = self.classifier.to(device)