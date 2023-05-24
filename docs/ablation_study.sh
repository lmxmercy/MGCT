DATA_ROOT_DIR='/data/lmx/Dataset/TCGA-DATASET/20X_FEATURES_DIRECTORY' # work on 20X magnification of whole slide imaging
LOG_DIR='/data/lmx/MCAT/logs' # the folder for saving nohup logs
DATASET='tcga_blca'
MODEL='new_mgct'

# TODO: Ablation study
# omic_net
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type $MODEL --max_epochs 20 \
                       --split_dir $DATASET --fusion='concat' \
                       --num_attn_heads=1 --mode='coattn' --omic_net='reg' \
                       1>$LOG_DIR/$MODEL-train-result.out 2>$LOG_DIR/$MODEL-train-error.out 
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type $MODEL --max_epochs 20 \
                       --split_dir $DATASET --fusion='concat' \
                       --num_attn_heads=1 --mode='coattn' --omic_net='mlp' \
                       1>$LOG_DIR/$MODEL-train-result.out 2>$LOG_DIR/$MODEL-train-error.out 
# w/o linear
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type $MODEL --max_epochs 20 \
                       --split_dir $DATASET --fusion='concat' \
                       --num_attn_heads=1 --mode='coattn' --omic_net='snn' \
                       1>$LOG_DIR/$MODEL-train-result.out 2>$LOG_DIR/$MODEL-train-error.out 
# fusion method
## Bilinear
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type $MODEL --max_epochs 20 \
                       --split_dir $DATASET --fusion='bilinear' \
                       --num_attn_heads=1 --mode='coattn' \
                       1>$LOG_DIR/$MODEL-train-result.out 2>$LOG_DIR/$MODEL-train-error.out 
## LRBilinear
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type $MODEL --max_epochs 20 \
                       --split_dir $DATASET --fusion='lrbilinear' \
                       --num_attn_heads=1 --mode='coattn' \
                       1>$LOG_DIR/$MODEL-train-result.out 2>$LOG_DIR/$MODEL-train-error.out 
# ## Addition
# nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type $MODEL --max_epochs 20 \
#                        --split_dir $DATASET --fusion='add' \
#                        --num_attn_heads=1 --mode='coattn' \
#                        1>$LOG_DIR/$MODEL-train-result.out 2>$LOG_DIR/$MODEL-train-error.out 
# ## Hadamard Product
# nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type $MODEL --max_epochs 20 \
#                        --split_dir $DATASET --fusion='hadamard' \
#                        --num_attn_heads=1 --mode='coattn' \
#                        1>$LOG_DIR/$MODEL-train-result.out 2>$LOG_DIR/$MODEL-train-error.out 
# ## AFF
# nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type $MODEL --max_epochs 20 \
#                        --split_dir $DATASET --fusion='aff' \
#                        --num_attn_heads=1 --mode='coattn' \
#                        1>$LOG_DIR/$MODEL-train-result.out 2>$LOG_DIR/$MODEL-train-error.out 
# ## iAFF
# nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type $MODEL --max_epochs 20 \
#                        --split_dir $DATASET --fusion='iaff' \
#                        --num_attn_heads=1 --mode='coattn' \
#                        1>$LOG_DIR/$MODEL-train-result.out 2>$LOG_DIR/$MODEL-train-error.out 

# FeedForward Network
# nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type $MODEL --max_epochs 20 \
#                        --split_dir $DATASET --fusion='concat' \
#                        --num_attn_heads=1 --mode='coattn' \
#                        1>$LOG_DIR/$MODEL-train-result.out 2>$LOG_DIR/$MODEL-train-error.out 
# GAP: gated-attention pooling
# nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type $MODEL --split_dir $DATASET \ 
#                         --fusion='concat' --num_attn_heads=1 --mode='coattn' \
#                         1>$LOG_DIR/Ablation-train-result.out 2>$LOG_DIR/Ablation-train-error.out 