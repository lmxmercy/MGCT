# MGCT
The PyTorch implementation for the paper:

MGCT: Mutual-Guided Cross-Modality Transformer for Survival Outcome Prediction using Integrative Histipathology-Genomic Features. (Accepted by IEEE BIBM 2023) [arXiv](https://arxiv.org/abs/2311.11659)

## Pre-requisites
* Linux (Tested on Ubuntu 18.04) 
* NVIDIA GPU (Tested on a workstation with one NVIDIA Quadro GV100 GPU)
* Python (3.7.7), opencv-python (4.6.0.66), openslide-python (1.2.0), torch (1.13.1), torchvision (0.14.1), etc.
* Also, you can create the anaconda environment directly by the command ``` conca env create -f docs/environment.yml```

## Data Preparation
### Dataset Downloading
* For downloading the diagnostic WSIs, please refer to the [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov/) 
* For Genomic data, we use the collected molecular feature and signatures from the [PORPOISE](https://github.com/mahmoodlab/PORPOISE)

### Whole Slide Images Processing
To process the WSI data, we used the [CLAM WSI-analysis toolbox](https://github.com/mahmoodlab/CLAM/tree/master) open-source repository. First, the tissue regions in each biopsy slide are segmented. The 256 x 256 patches without spatial overlapping are extracted from the segmented tissue regions at the desired magnification. Consequently, a pretrained truncated ResNet50 is used to encode raw image patches into 1024-dim feature vector. Using the CLAM toolbox, the features are saved as matrices of torch tensors of size N x 1024, where N is the number of patches from each WSI (varies from slide to slide). Please refer to [CLAM](https://github.com/mahmoodlab/CLAM/tree/master) for the details on tissue segmentation and feature extraction. In this paper, we worked on the 20X magnification of Whole Slide Images and the parameter setting can be found [here](https://github.com/mahmoodlab/CLAM/issues/119). The extracted features then serve as input (in a .pt file) to the network. The following folder structure is assumed for the extracted features vectors:    
```bash
DATA_ROOT_DIR/
    └──TCGA_BLCA/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
    └──TCGA_BRCA/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
    └──TCGA_GBMLGG/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
    └──TCGA_LUAD/
        ├── slide_1.ptd
        ├── slide_2.pt
        └── ...
    └──TCGA_UCEC/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
    ...
```
DATA_ROOT_DIR is the base directory of all datasets / cancer type (e.g. the directory to your SSD). Within DATA_ROOT_DIR, each folder contains a list of .pt files for that dataset / cancer type.

### Training-Validation Splits
For evaluating the algorithm's performance, we randomly partitioned each dataset using 5-fold cross-validation which same as the [PORPOISE](https://github.com/mahmoodlab/PORPOISE). Splits for each cancer type are found in the [splits/5foldcv](https://github.com) folder, which each contain **splits_{k}.csv** for k = 1 to 5. In each **splits_{k}.csv**, the first column corresponds to the TCGA Case IDs used for training, and the second column corresponds to the TCGA Case IDs used for validation. Alternatively, one could define their own splits, however, the files would need to be defined in this format. 

## Running Experiments
* To run experiments using our compared cutting-edge baselines, experiments can be run using the [shell script](docs/baselines.sh)
* To run experiment using our proposed MGCT, you can run the [main.py](main.py) as following:

```python
python main.py --data_root_dir <PATH TO WSIs> --split_dir <CANCER TYPE SPLITS> --model_type mgct --fusion concat --mode coattn --stage1_num_layers 1 --stage2_num_layers 2
```

