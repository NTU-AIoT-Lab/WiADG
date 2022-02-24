# Robust Wifi-enabled Device-free Gesture recognition via Unsupervised Adversarial Domain Adaptation

### Introduction
In this paper, we propose a WiFi-enabled device-free adaptive gesture recognition scheme, WiADG, that is able to identify human gestures accurately and consistently under environmental dynamics via adversarial domain adaptation. ([link](https://ieeexplore.ieee.org/abstract/document/8487345))

### Requirements
- cuda
- python3
- pytorch

### Guidance
- Train the source encoder and classifier
```
python train_src.py
```
- Test the source on the target dataset
```
python test_src.py
```
- Train the target encoder
```
python train_adapt.py
```
- Test the target encoder with source classifier on the target dataset
```
python test_tgt.py
```

### Performance
| Method | Conf room --> Office | Office --> Conf room |
| :----: | :----: | :----: |
| Source-only | 49.7% | 32.7% |
| Target-only | 96.7% | 93.0% |
| WiADG (Ours) | 83.3% | 66.6% |

### Reference
```
@inproceedings{zou2018robust,
  title={Robust wifi-enabled device-free gesture recognition via unsupervised adversarial domain adaptation},
  author={Zou, Han and Yang, Jianfei and Zhou, Yuxun and Xie, Lihua and Spanos, Costas J},
  booktitle={2018 27th International Conference on Computer Communication and Networks (ICCCN)},
  pages={1--8},
  year={2018},
  organization={IEEE},
  doi={10.1109/ICCCN.2018.8487345}
}
```
