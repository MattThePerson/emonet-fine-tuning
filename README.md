# EmoNet Fine Tuning for Affecting Computing (2025)

Fork of [EmoNet](https://github.com/face-analysis/emonet) indended for fine tuning and other testing.

By Matt Stirling as part of Affective Computing (2025), University of Oulu. 


## Setup

1. Install PyTorch: \
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
2. Install requirements.txt \
`pip install -r requirements.txt`


## Running

In order to run scripts in `scripts/` folder, recommended to run as modules:

```python -m scripts.train_emonet [ARGS]```


## 

| SCRIPT | DESCRIPTION |
| --- | --- |
| `scripts/train_emonet.py` | Script for fine tuning EmoNet on dataset with compatible labels.  |
| `scripts/evaluate_emonet.py` | Script for evaluating EmoNet given dataset and pretrained params.  |

