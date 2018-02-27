# ADDA-wifi-gesture
WiFi-based gesture recognition using Adversarial Discriminative Domain Adaptation.

### Requirements
- cuda
- python3
- pytorch

### Guidance
- train_src.py: train the source encoder and classifier
- test_src.py: test the source on the target dataset
- train_adapt.py: train the target encoder
- test_tgt: test the target encoder with source classifier on the target dataset

### Performance
| Source A: 96.7% | Source B: 93.0% |
| A ---> B: 49.7% | B ---> A: 32.7% |

After domain adaptation
A dataset: A target encoder + B source classifier --> 70.1%
B dataset: B target encoder + A source classifier --> 52.3%
