import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(BASE_DIR)
sys.path.append(BASE_DIR)
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from nystrom_attention import NystromAttention, Nystromformer
from models.model_utils import SNN_Block, BilinearFusion, LRBilinearFusion, Attn_Net_Gated, MLP_Block


class MGCT(nn.Module):
    """
    Mutual-Guided Cross-Modality Transformer

    """

    def __init__(self, stage1_num_layers=1, stage2_num_layers=2, n_classes=4, dropout=0.25,
                 omic_sizes=[100, 200, 300, 400, 500, 600], fusion='concat', 
                 model_size_wsi: str = 'small', model_size_omic: str = 'small') -> None:
        super(MGCT, self).__init__()

        self.n_classes = n_classes
        self.fusion = fusion
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        # [1024, 512, 384, 256] -> [1024, 768, 640, 512, 384, 256]
        self.size_dict_omic = {'small': [256, 256], 'middle': [256, 256, 256, 256, 256], 'big': [1024, 1024, 1024, 256]}

        # WSI patches feature extractor
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(in_features=size[0], out_features=size[1]), nn.ReLU()]
        fc.append(nn.Dropout(p=dropout))
        self.wsi_net = nn.Sequential(*fc)

        # Genomic data feature extractor
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            omic_fc = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                omic_fc.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*omic_fc))
        self.omic_net = nn.ModuleList(sig_networks)

        # Stage 1: WSI features and Genomic features fusion
        self.stage1_layers = nn.ModuleList([
            CrossModalityFusionLayer(dim=256, ffn_dim=512, num_heads=8, drop=dropout, fusion=fusion)
            for _ in range(stage1_num_layers)
        ])

        # Stage 2: stage1 fusion embedding and WSI feature fusion again
        self.stage2_layers = nn.ModuleList([
            CrossModalityFusionLayer(dim=256, ffn_dim=512, num_heads=8, drop=dropout, fusion=fusion)
            for _ in range(stage2_num_layers)
        ])

        # Classifier
        self.cls_drop = nn.Dropout(p=dropout)
        self.act = nn.ReLU()
        self.classifier = nn.Linear(size[2], n_classes)

    def forward(self, **kwargs):
        ## WSI and Genomic feature embeddings after feature extractor
        wsi_feats = kwargs['x_path']
        omic_feats = [kwargs['x_omic%d' % i] for i in range(1,7)]

        wsi_feats_embed = self.wsi_net(wsi_feats).unsqueeze(1)

        omic_feats = [self.omic_net[idx].forward(sig_feat) for idx, sig_feat in enumerate(omic_feats)] ### each omic signature goes through it's own FC layer
        omic_feats_embed = torch.stack(omic_feats).unsqueeze(1) ### omic embeddings are stacked (to be used in co-attention)

        # omic_feats_embed = self.omic_net(omic_feats)  # omic_feats_embed: (256, )
        # omic_feats_embed = omic_feats_embed.unsqueeze(0).unsqueeze(0)  # (1, 1, 256)

        ## Stage 1: WSI features and Genomic features fusion
        for stage1_layer in self.stage1_layers:
            stage1_output, omic_wsi_mgca = stage1_layer(wsi_feats_embed, omic_feats_embed, attn_mask=None)

        ## Stage 2: stage1 fusion embedding and WSI feature fusion again
        for stage2_layer in self.stage2_layers:
            stage2_output, wsi_omic_mgca = stage2_layer(stage1_output.unsqueeze(0).unsqueeze(0), wsi_feats_embed,
                                                        attn_mask=None)

        ## Final output feature embedding
        output = self.act(self.cls_drop(stage2_output))

        ## Classifier
        logits = self.classifier(output).unsqueeze(0)

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S, dim=1)
        attention_scores = {'omic_wsi_mgca': omic_wsi_mgca, 'wsi_omic_mgca': wsi_omic_mgca}
        return hazards, S, Y_hat, attention_scores
    
        # return logits, Y_prob, hazards, S, Y_hat

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wsi_net = self.wsi_net.to(device)
        self.omic_net = self.omic_net.to(device)
        self.stage1_layers = self.stage1_layers.to(device)
        self.stage2_layers = self.stage2_layers.to(device)
        self.classifier = self.classifier.to(device)


class CrossModalityFusionLayer(nn.Module):
    """
        Fusion layer for WSI and Genomic feature embeddings
        Modified by Multimodal Fusion with Co-Attention Networks for Fake News Detection from ACL 2021
    Info:
        Paper: https://aclanthology.org/2021.findings-acl.226.pdf
        Code: https://github.com/wuyang45/MCAN_code
    Args:
        dim: input dimension (default: 256)
        ffn_dim: the input dimension of feed-forward network (default: 512)
        num_heads: the number of cross-modality attention heads (default: 8)
        drop: dropout mechanism (default: 0.5)
        fusion_type: the method of fusion two-modality data (concat, bilinear, lrbilinear)
    """

    def __init__(self, dim=256, ffn_dim=512, num_heads=8, drop=0.5, fusion='concat') -> None:
        super(CrossModalityFusionLayer, self).__init__()
        self.fusion = fusion
        from torch.nn.modules import MultiheadAttention
        self.wsi_omic_attn = MultiheadAttention(dim, num_heads, drop)
        self.omic_wsi_attn = MultiheadAttention(dim, num_heads, drop)

        self.wsi_omic_pool = Attn_Net_Gated(L=256, D=256, dropout=drop, n_classes=1)
        self.omic_wsi_pool = Attn_Net_Gated(L=256, D=256, dropout=drop, n_classes=1)

        self.wsi_omic_ffn = FeedForward(dim, ffn_dim, drop)
        self.omic_wsi_ffn = FeedForward(dim, ffn_dim, drop)

        self.wsi_rho = nn.Sequential(*[nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(p=drop)])
        self.omic_rho = nn.Sequential(*[nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(p=drop)])

        if self.fusion == 'concat':
            self.fusion_layer = nn.Sequential(*[nn.Linear(dim * 2, dim), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.fusion_layer = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        elif self.fusion == 'lrbilinear':
            self.fusion_layer = LRBilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8)
        else:
            raise NotImplementedError

    def forward(self, wsi_feats, omic_feats, attn_mask=None):
        # wsi_feats: (15231, 1, 256), omic_feats: (1, 1, 256)
        ## genomic-guided WSI feature embedding
        wsi_omic_output, wsi_omic_MGCA = self.wsi_omic_attn(omic_feats, wsi_feats, wsi_feats,
                                                            attn_mask)  # wsi_omic_output: (1, 1, 256)
        # print("wsi_omic_MGCA", wsi_omic_MGCA.squeeze(0).shape)
        wsi_omic_A, wsi_omic_output = self.wsi_omic_pool(
            wsi_omic_output.squeeze(1))  # wsi_omic_A: (1, 1), wsi_omic_output: (1, 256)
        wsi_omic_A = torch.transpose(wsi_omic_A, 1, 0)  # wsi_omic_A: (1, 1) heatmap attention
        wsi_omic_output = torch.mm(F.softmax(wsi_omic_A, dim=1), wsi_omic_output)  # wsi_omic_output: (1, 256)

        wsi_omic_output = self.wsi_omic_ffn(wsi_omic_output)
        wsi_omic_output = self.wsi_rho(wsi_omic_output).squeeze(0)  # wsi_omic_output: (256, )

        ## WSI-guided genomic feature embedding
        omic_wsi_output, omic_wsi_MGCA = self.omic_wsi_attn(wsi_feats, omic_feats, omic_feats, attn_mask)
        omic_wsi_A, omic_wsi_output = self.omic_wsi_pool(omic_wsi_output.squeeze(1))
        omic_wsi_A = torch.transpose(omic_wsi_A, 1, 0)
        omic_wsi_output = torch.mm(F.softmax(omic_wsi_A, dim=1), omic_wsi_output)

        omic_wsi_output = self.omic_wsi_ffn(omic_wsi_output)
        omic_wsi_output = self.omic_rho(omic_wsi_output).squeeze(0)

        if self.fusion == 'concat':
            output = self.fusion_layer(torch.cat([wsi_omic_output, omic_wsi_output], axis=0))
        # todo: forward for bilinear and lrbilinear
        elif self.fusion == 'bilinear':
            output = self.fusion_layer(wsi_omic_output.unsqueeze(dim=0), omic_wsi_output.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'lrbilinear':
            output = self.fusion_layer(wsi_omic_output.unsqueeze(dim=0), omic_wsi_output.unsqueeze(dim=0)).squeeze()
        else:
            raise NotImplementedError
        # print("Final shared representation (h_final):\n", output.shape)

        return output, wsi_omic_MGCA.squeeze(2)


class FeedForward(nn.Module):
    def __init__(self, dim=256, ffn_dim=512, drop=0.5) -> None:
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        # self.dwconv = DWConv(ffn_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.ffn_drop = nn.Dropout(p=drop)

    def forward(self, x):
        # (1, 256)
        x = self.fc1(x)  # (1, 512)
        x = self.act(x)
        x = self.ffn_drop(x)
        x = self.fc2(x)  # (1, 256)
        return x  # (1, 256)


if __name__ == '__main__':
    wsi_feats = torch.randn((1500, 1024))  # 15231 patches with 1024-dim embedding size
    omic_feats = torch.randn((2181))
    model = MGCT(stage1_num_layers=1, stage2_num_layers=2, fusion='concat')
    from utils.utils import print_network
    print_network(model)
