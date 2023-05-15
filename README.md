
# Few-Shot Classification of Screen Defects with Class-Agnostic Mask and Context-Based Classifier

The official PyTorch implementation of Few-Shot Classification of Screen Defects with Class-Agnostic Mask and Context-Based Classifier.

## Prerequisites

We have tested in a [**Python=3.7**](https://www.python.org/) environment with [**PyTorch=1.7.0**](https://pytorch.org/get-started/previous-versions/). Other environments may work as well. 

```python
conda create --name FSL python=3.7
conda activate FSL
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.1
pip install terminaltables==3.1.0
pip install scikit-learn==1.0.2
pip install matplotlib==3.5.3
pip install PyYAML==6.0
pip install tqdm==4.64.1
pip install pandas==1.3.4
pip install albumentations==1.0.3
pip install opencv-python==4.5.3.56 #albumentations should be installed before opencv-python
#opencv-python-headless==4.6.0.66
pip install seaborn==0.12.2
```

## Dataset

Data splits are placed under  `data/` folder.

For the NEU-CLS dataset, you can download from [here](http://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/list/index.htm).  Then, you can place images `IMAGES/` under `data/NEU/` folder.

For the magnetic tile defect (MTD) dataset, you can download from [here](https://github.com/Charmve/Surface-Defect-Detection/tree/master/Magnetic-Tile-Defect). Then, you can place images (such as `MT_Blowhole/Imgs/`,`MT_Break/Imgs/`, etc) under `data/Magnetic/` folder.

## Code Structures
There are four parts in the code.

 - `configs`: It contains configuration files for experiments.
 - `data`: Data splits for the data sets.
- `dataloader`: Dataloader of different datasets.
 - `models`: It contains the backbone network and training protocols for the experiment.

## Training scripts

- Train NEU-CLS

  ```shell
  cd code_FSL_Cls_Screen
  python train.py -default_arg_path ./configs/NEU/ours5shot.yaml
  # dataset. (1) dataroot/(all data splits in ./data/NEU)；(2) dataroot/IMAGES/*.jpg
  ```
  
- Train MTD
    ```shell
    cd code_FSL_Cls_Screen
    python train.py -default_arg_path ./configs/Magnetic/ours5shot.yaml
    # dataset. (1) dataroot/(all data splits in ./data/Magnetic)；(2) dataroot/(all images, such as MT_Blowhole/Imgs/*.jpg)
    ```


Remember to change `dataroot` into your own data root, or you will encounter errors.

## Acknowledgment

We thank the following repos providing helpful components/functions/dataset in our work.

- [CEC](https://github.com/icoz69/CEC-CVPR2021)
- [FACT](https://github.com/zhoudw-zdw/CVPR22-Fact)
- [simclr](https://github.com/google-research/simclr)
- [Charmve/Surface-Defect-Detection](https://github.com/Charmve/Surface-Defect-Detection)

This repository also contains the implementation of compared methods in our paper. If you have  any questions, please contact us: 3493132244@qq.com. Thank you. :smile:

