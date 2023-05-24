DATA_ROOT_DIR='/data/lmx/Dataset/TCGA-DATASET/20X_FEATURES_DIRECTORY' # work on 20X magnification of whole slide imaging
LOG_DIR='/data/lmx/MCAT/logs' # the folder for saving nohup logs
DATASET='tcga_blca' 

# TODO: Baselines, apply_signatures, TCGA-BLCA
# TODO: Baselines, w/o apply_signatures, TCGA-BLCA
## Genomic Only
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type='mlp' --max_epochs 20 \
                       --split_dir $DATASET --mode='omic' --fusion='None' \
                       1>$LOG_DIR/Baselines-train-result.out 2>$LOG_DIR/Baselines-train-error.out 
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type='snn' --max_epochs 20 \
                       --split_dir $DATASET --mode='omic' --fusion='None' \
                       1>$LOG_DIR/Baselines-train-result.out 2>$LOG_DIR/Baselines-train-error.out 
## WSI Only
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type='deepset' --max_epochs 20 \
                       --split_dir $DATASET --mode='path' --fusion='None' \
                       1>$LOG_DIR/Baselines-train-result.out 2>$LOG_DIR/Baselines-train-error.out 
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type='amil' --max_epochs 20 \
                       --split_dir $DATASET --mode='path' --fusion='None' \
                       1>$LOG_DIR/Baselines-train-result.out 2>$LOG_DIR/Baselines-train-error.out 
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type='mi_fcn' --max_epochs 20 \
                       --split_dir $DATASET --mode='cluster' --fusion='None' \
                       1>$LOG_DIR/Baselines-train-result.out 2>$LOG_DIR/Baselines-train-error.out 
## Multimodal
### Deep Sets (Concat)
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type='deepset' --max_epochs 20 \
                       --split_dir $DATASET --mode='pathomic' --fusion='concat' \
                       1>$LOG_DIR/Baselines-train-result.out 2>$LOG_DIR/Baselines-train-error.out 
### Deep Sets (Bilinear Pooling)
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type='deepset' --max_epochs 20 \
                       --split_dir $DATASET --mode='pathomic' --fusion='bilinear' \
                       1>$LOG_DIR/Baselines-train-result.out 2>$LOG_DIR/Baselines-train-error.out 
### Attention MIL (Concat)
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type='amil' --max_epochs 20 \
                       --split_dir $DATASET --mode='pathomic' --fusion='concat' \
                       1>$LOG_DIR/Baselines-train-result.out 2>$LOG_DIR/Baselines-train-error.out 
### Attention MIL (Bilinear Pooling)
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type='amil' --max_epochs 20 \
                       --split_dir $DATASET --mode='pathomic' --fusion='bilinear' \
                       1>$LOG_DIR/Baselines-train-result.out 2>$LOG_DIR/Baselines-train-error.out 
### DeepAttnMISL (Concat)
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type='mi_fcn' --max_epochs 20 \
                       --split_dir $DATASET --mode='clusteromic' --fusion='concat' \
                       1>$LOG_DIR/Baselines-train-result.out 2>$LOG_DIR/Baselines-train-error.out 
### DeepAttnMISL (Bilinear Pooling)
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type='mi_fcn' --max_epochs 20 \
                       --split_dir $DATASET --mode='clusteromic' --fusion='bilinear' \
                       1>$LOG_DIR/Baselines-train-result.out 2>$LOG_DIR/Baselines-train-error.out 
### PORPOISE (Concat)
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type='porpoise' --max_epochs 20 \
                       --split_dir $DATASET --mode='pathomic' --fusion='concat' \
                       1>$LOG_DIR/Baselines-train-result.out 2>$LOG_DIR/Baselines-train-error.out 
### PORPOISE (Bilinear Pooling)
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type='porpoise' --max_epochs 20 \
                       --split_dir $DATASET --mode='pathomic' --fusion='bilinear' \
                       1>$LOG_DIR/Baselines-train-result.out 2>$LOG_DIR/Baselines-train-error.out 
### MCAT (Concat)
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type='mcat' --max_epochs 20 \
                       --split_dir $DATASET --mode='coattn' --fusion='concat' \
                       1>$LOG_DIR/Baselines-train-result.out 2>$LOG_DIR/Baselines-train-error.out 
### MCAT (Bilinear Pooling)
nohup python -u main.py --data_root_dir $DATA_ROOT_DIR --model_type='mcat' --max_epochs 20 \
                       --split_dir $DATASET --mode='coattn' --fusion='bilinear' \
                       1>$LOG_DIR/Baselines-train-result.out 2>$LOG_DIR/Baselines-train-error.out 