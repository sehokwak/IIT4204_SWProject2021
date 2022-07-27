# Pascal VOC Weakly Supervised Semantic Segmentation
This is a repository for the final project of IIT4204, SW Project, Yonsei University

## Prerequisites
- Python3( >= 3.6)
- PyTorch( >= 1.1)
- OpenCV

## Data preparation
- PascalVOC [download link](https://drive.google.com/file/d/19FGnKyWBq4gU9Qm0x0w83lc6k-o5Eb3V/view?usp=sharing)

## Execution

### Run on Google Colab GPU
Set number of classes in `settings.py` and `script/train.sh`
```
NUM_CLASSES=10   # 5 or 10
```


Set `HOMEROOT` path in `settings.py`,  `script/train.sh`, and `FinalColab.ipynb`
```
HOMEROOT="path/to/IIT4204_SWProject2021"
```

Run `FinalColab.ipynb` on Google Colab

## Results
Read the final paper [here](https://github.com/sehokwak/IIT4204_SWProject2021/blob/main/files/finalreport.pdf)