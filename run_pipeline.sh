ROOT_DIR='/data/lmx/Dataset/TCGA-DATASET/20X_FEATURES_DIRECTORY'
DATASET='tcga_blca'
DATA_ROOT_DIR=$ROOT_DIR/$DATASET
MODEL='mgct'
LOG_DIR='/data/lmx/MGCT/logs'

nohup python -u main.py --data_root_dir $DATA_ROOT_DIR \
                            --model_type $MODEL --max_epochs 20 \
                            1>$LOG_DIR/$MODEL-train-result.out 2>$LOG_DIR/$MODEL-train-error.out       