DATA_ROOT_DIR='/data/lmx/Dataset/TCGA-DATASET/20X_FEATURES_DIRECTORY' # work on 20X magnification of whole slide imaging
MODEL='new_mgct' # model for training
LOG_DIR='/data/lmx/MCAT/logs' # the folder for saving nohup logs
DATASET='tcga_brca'

nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type $MODEL --max_epochs 20 \
                       --split_dir='tcga_blca' --fusion='concat' --stage1_num_layers=1 --stage2_num_layers=2 \
                       --num_attn_heads=1 --num_trans_head=0 --num_trans_layer=0 \
                       1>$LOG_DIR/$MODEL-train-result.out 2>$LOG_DIR/$MODEL-train-error.out 
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type $MODEL --max_epochs 20 \
                       --split_dir='tcga_brca' --fusion='concat' --stage1_num_layers=1 --stage2_num_layers=2 \
                       --num_attn_heads=1 --num_trans_head=0 --num_trans_layer=0 \
                       1>$LOG_DIR/$MODEL-train-result.out 2>$LOG_DIR/$MODEL-train-error.out 
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type $MODEL --max_epochs 20 \
                       --split_dir='tcga_luad' --fusion='concat' --stage1_num_layers=1 --stage2_num_layers=2 \
                       --num_attn_heads=1 --num_trans_head=0 --num_trans_layer=0 \
                       1>$LOG_DIR/$MODEL-train-result.out 2>$LOG_DIR/$MODEL-train-error.out 
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type $MODEL --max_epochs 20 \
                       --split_dir='tcga_ucec' --fusion='concat' --stage1_num_layers=1 --stage2_num_layers=2 \
                       --num_attn_heads=1 --num_trans_head=0 --num_trans_layer=0 \
                       1>$LOG_DIR/$MODEL-train-result.out 2>$LOG_DIR/$MODEL-train-error.out 