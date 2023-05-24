import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import MultiheadAttention, TransformerEncoderLayer, TransformerEncoder
from nystrom_attention import NystromAttention, Nystromformer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(BASE_DIR)
sys.path.append(BASE_DIR)
from models.model_utils import SNN_Block, Attn_Net_Gated, BilinearFusion, MLP_Block, Reg_Block


class MGCT_Layer(nn.Module):
    """ alpha: Genomic -> WSI Attention
        beta: WSI -> Genomic Attention
        gamma: Integration Attention """
    def __init__(self, dim=256, fusion='concat', num_attn_heads=1, num_trans_heads=8, num_trans_layer=1, drop_rate=0.25):
        super(MGCT_Layer, self).__init__()
        self.fusion = fusion

        trans_layer = TransformerEncoderLayer(d_model=dim, nhead=num_trans_heads, dim_feedforward=512, dropout=drop_rate)

        self.alpha = MultiheadAttention(embed_dim=dim, num_heads=num_attn_heads)
        self.alpha_trans = TransformerEncoder(encoder_layer=trans_layer, num_layers=num_trans_layer)
        # self.alpha_ffn = FeedForward(dim=dim, ffn_dim=dim*2)
        self.alpha_gap = Attn_Net_Gated(L=dim, D=dim, dropout=drop_rate, n_classes=1)
        self.alpha_rho = nn.Sequential(*[nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(p=drop_rate)])

        self.beta  = MultiheadAttention(embed_dim=dim, num_heads=num_attn_heads)
        self.beta_trans = TransformerEncoder(encoder_layer=trans_layer, num_layers=num_trans_layer)
        # self.beta_ffn = FeedForward(dim=dim, ffn_dim=dim * 2)
        self.beta_gap = Attn_Net_Gated(L=dim, D=dim, dropout=drop_rate, n_classes=1)
        self.beta_rho = nn.Sequential(*[nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(p=drop_rate)])

        if fusion == 'concat':
            self.gamma = nn.Sequential(*[nn.Linear(dim * 2, dim), nn.ReLU()])
        elif fusion == 'bilinear':
            self.gamma = BilinearFusion(dim1=dim, dim2=dim, scale_dim1=8, scale_dim2=8, mmhid=dim)
        elif fusion == 'attn':
            self.gamma = MultiheadAttention(embed_dim=dim, num_heads=num_attn_heads)

    def forward(self, wsi_feat_embed, omic_feat_embed, return_attn=False):
        # alpha: Genomic-Guided WSI Feature Embedding
        ## Genomic -> WSI
        ### omic_feat_embed: (1,1,256), wsi_feats_embed: (15231,1,256)
        omic_wsi_embed, alpha_attn = self.alpha(omic_feat_embed, wsi_feat_embed, wsi_feat_embed)
        A_alpha, omic_wsi_embed = self.alpha_gap(omic_wsi_embed.squeeze(1))
        A_alpha = torch.transpose(A_alpha, 1, 0)
        omic_wsi_embed = torch.mm(F.softmax(A_alpha, dim=1), omic_wsi_embed)
        omic_wsi_embed = self.alpha_trans(omic_wsi_embed)
        # omic_wsi_embed = self.alpha_ffn(omic_wsi_embed)
        omic_wsi_embed = self.alpha_rho(omic_wsi_embed)

        # beta: WSI-Guided Genomic Feature Embedding
        ## WSI -> Genomic
        wsi_omic_embed, beta_attn = self.beta(wsi_feat_embed, omic_feat_embed, omic_feat_embed)
        A_beta, wsi_omic_embed = self.beta_gap(wsi_omic_embed.squeeze(1))
        A_beta = torch.transpose(A_beta, 1, 0)
        wsi_omic_embed = torch.mm(F.softmax(A_beta, dim=1), wsi_omic_embed)
        wsi_omic_embed = self.beta_trans(wsi_omic_embed)
        # wsi_omic_embed = self.beta_ffn(wsi_omic_embed)
        wsi_omic_embed = self.beta_rho(wsi_omic_embed)

        ## Integration between Genomic-Guided WSI Feature Embedding and WSI-Guided Genomic Feature Embedding

        if self.fusion == 'concat':
            ### omic_wsi_embed: (1,256), wsi_omic_embed: (1,256)
            fused_embed = self.gamma(torch.cat([omic_wsi_embed, wsi_omic_embed], axis=1))
        elif self.fusion == 'bilinear':
            fused_embed = self.gamma(omic_wsi_embed, wsi_omic_embed)
        elif self.fusion == 'attn':
            fused_embed, gamma_attn = self.gamma(omic_wsi_embed, wsi_omic_embed, wsi_omic_embed)
        else:
            raise NotImplementedError

        if return_attn:
            return fused_embed, alpha_attn, beta_attn, A_alpha, A_beta
        else:
            return fused_embed


class MGCT_Surv(nn.Module):
    def __init__(self, fusion='concat', stage1_num_layers=1, stage2_num_layers=2,
                 n_classes=4, num_attn_heads=1, num_trans_heads=8, num_trans_layer=1, 
                 drop_rate=0.25, omic_sizes=[100, 200, 300, 400, 500, 600],
                 return_attn=False, model_size_wsi: str='small', model_size_omic: str='small'):
        super(MGCT_Surv, self).__init__()
        self.fusion = fusion
        self.return_attn = return_attn
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], "big": [1024, 1024, 1024, 384]}
        self.n_classes = n_classes

        # WSI patches feature extractor
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
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
            MGCT_Layer(size[2], fusion, num_attn_heads, num_trans_heads=8, num_trans_layer=1, drop_rate=drop_rate)
            for _ in range(stage1_num_layers)
        ])

        # Stage 2: stage1 fusion embedding and WSI feature fusion again
        self.stage2_layers = nn.ModuleList([
            MGCT_Layer(size[2], fusion, num_attn_heads, num_trans_heads=8, num_trans_layer=1, drop_rate=drop_rate)
            for _ in range(stage2_num_layers)
        ])

        # Classifier
        # self.cls_drop = nn.Dropout(p=drop_rate)
        # self.act = nn.ReLU()
        self.classifier = nn.Linear(size[2], n_classes)

    def forward(self, **kwargs):
        ## WSI and Genomic feature embeddings after feature extractor
        wsi_feats = kwargs['x_path']
        omic_feats = [kwargs['x_omic%d' % i] for i in range(1,7)]

        wsi_feats_embed = self.wsi_net(wsi_feats).unsqueeze(1)

        omic_feats = [self.omic_net[idx].forward(sig_feat) for idx, sig_feat in enumerate(omic_feats)] ### each omic signature goes through it's own FC layer
        omic_feats_embed = torch.stack(omic_feats).unsqueeze(1) ### omic embeddings are stacked (to be used in co-attention)

        ## Stage 1: WSI features and Genomic features fusion
        for stage1_layer in self.stage1_layers:
            ### wsi_feats_embed: (15231,1,256), omic_feats_embed: (1,1,256)
            if self.return_attn:
                stage1_output, s1_alpha_attn, s1_beta_attn, s1_A_alpha, s1_A_beta = stage1_layer(wsi_feats_embed, omic_feats_embed, return_attn=True)
            else:
                stage1_output = stage1_layer(wsi_feats_embed, omic_feats_embed, return_attn=False)

        ## Stage 2: stage1 fusion embedding and WSI feature fusion again
        for stage2_layer in self.stage2_layers:
            ### stage1_output: (1,256), wsi_feats_embed: (15231,1,256)
            if self.return_attn:
                stage2_output, s2_alpha_attn, s2_beta_attn, s2_A_alpha, s2_A_beta = stage2_layer(stage1_output, wsi_feats_embed.squeeze(1), return_attn=True)
            else:
                stage2_output = stage2_layer(stage1_output, wsi_feats_embed.squeeze(1), return_attn=False)

        logits = self.classifier(stage2_output)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S, dim=1)
   
        if self.return_attn is False:
            attention_scores = None
        else:
            attention_scores = {'s1_alpha_attn': s1_alpha_attn, 's1_beta_attn': s1_beta_attn, 's2_alpha_attn': s2_alpha_attn, 's2_beta_attn': s2_beta_attn,
                                's1_A_alpha': s1_A_alpha, 's1_A_beta': s1_A_beta, 's2_A_alpha': s2_A_alpha, 's2_A_beta': s2_A_beta}
            
        return hazards, S, Y_hat, attention_scores
        
    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wsi_net = self.wsi_net.to(device)
        self.omic_net = self.omic_net.to(device)
        self.stage1_layers = self.stage1_layers.to(device)
        self.stage2_layers = self.stage2_layers.to(device)
        self.classifier = self.classifier.to(device)


class FeedForward(nn.Module):
    def __init__(self, dim=256, ffn_dim=512, drop=0.5) -> None:
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.act = nn.ReLU()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.drop1 = nn.Dropout(p=drop)
        self.drop2 = nn.Dropout(p=drop)

    def forward(self, x):
        x_ = x
        x = self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))
        x = self.norm2(x_ + x)
        return x  # (1, 256)

if __name__ == '__main__':
    wsi_feats = torch.randn((1500, 1024))  # 15231 patches with 1024-dim embedding size
    omic_feats = torch.randn((2181))
    model = MGCT_Surv(stage1_num_layers=1, stage2_num_layers=2, num_attn_heads=1, fusion='concat')
    from utils.utils import print_network
    print_network(model)