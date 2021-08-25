# kaggle-SETI-2ndPlaceSolution

This is the 2nd place solution for the SETI Breakthrough Listen - E.T. Signal Search competition.

## Archive contents
kaggle_model.tgz          : original kaggle model upload - contains original code, additional training examples, corrected labels, etc
comp_etc                     : contains ancillary information for prediction - clustering of training/test examples
comp_mdl                     : model binaries used in generating solution
comp_preds                   : model predictions
train_code                  : code to rebuild models from scratch
predict_code                : code to generate predictions from model binaries

## Hardware
* Ubuntu 18.04.5 LTS
* 1~4 x NVIDIA Tesla-V100-SXM2-32GB 

## Software
* This docker image were used for all executions.
https://hub.docker.com/r/hirune924/pikachu
* If you need, you can see requirements.txt.

## Data setup 
```
mkdir data/
cd data/
kaggle competitions download -c seti-breakthrough-listen
unzip seti-breakthrough-listen.zip
```
## Training
## Inference
