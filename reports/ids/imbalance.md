# Imbalance study (dev sample)

Each strategy is calibrated and thresholded on validation (benign FPR <= 1%), then scored on test.

| Strategy | Val macro-F1 | Test macro-F1 | Test benign FPR | Recall Infiltration | Recall WebAttack | Recall Bot | Train rows | Seconds |
|---|---|---|---|---|---|---|---|---|
| LightGBM, unweighted | 0.7153 | 0.7081 | 0.044% | 0.000 | 0.773 | 0.944 | 111,515 | 8 |
| LightGBM, sqrt class weights | 0.9998 | 0.9910 | 0.014% | 1.000 | 1.000 | 1.000 | 111,515 | 19 |
| LightGBM, inverse-frequency weights | 0.9997 | 0.9966 | 0.003% | 1.000 | 1.000 | 1.000 | 111,515 | 10 |
| LightGBM, random oversampling | 0.9967 | 0.9934 | 0.003% | 1.000 | 0.955 | 1.000 | 117,746 | 13 |
| LightGBM, SMOTE | 0.9994 | 0.9941 | 0.007% | 1.000 | 0.955 | 1.000 | 117,746 | 11 |
| MLP, cross-entropy + sqrt weights | 0.9799 | 0.9541 | 0.308% | 0.875 | 0.864 | 1.000 | 111,515 | 13 |
| MLP, focal loss (gamma=2) + sqrt weights | 0.9493 | 0.8888 | 0.243% | 0.250 | 0.864 | 0.972 | 111,515 | 12 |
