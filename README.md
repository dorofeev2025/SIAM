# SIAM
Statistical Identification of Anomalies in Mahalanobis Space

## Description
This repository contains software that supplements the following paper:
Oleg Melnikov, Yurii Dorofieiev, Yurii Shakhnovskiy, Huy Truong, Victoria Degeler,
A multivariate statistical framework for detection, classification and pre-localization of anomalies in water distribution networks,
Expert Systems with Applications, Volume 313, 2026, 131450, ISSN 0957-4174,
https://doi.org/10.1016/j.eswa.2026.131450.
(https://www.sciencedirect.com/science/article/pii/S0957417426003635)

## Installation
1. Clone or download this repository.
2. Install the required libraries via this command:

    ``pip install -r requirement.txt``

## Training model

User can train SIAM model using this default command:
```
python train.py 
```

## Detection of Anomalies

To detect anomalies, run the following command:
```
python detect.py   
```
This should produce Fig.12(a) from the paper cited above.

## License

MIT license. See the LICENSE file for more details.
