# kaggle-SETI-2ndPlaceSolution

This is the 2nd place solution for the SETI Breakthrough Listen - E.T. Signal Search competition.

## Hardware
* Ubuntu 18.04.5 LTS
* 1~4 x NVIDIA Tesla-V100-SXM2-32GB 

## Software
* This docker image were used for all executions.
    - https://hub.docker.com/r/hirune924/pikachu
* If you need, you can see requirements.txt.

## Data setup 
```
mkdir data/
cd data/
kaggle competitions download -c seti-breakthrough-listen
unzip seti-breakthrough-listen.zip
```
## Training
[Training Process](train/README.md)

## Prediction
[Prediction Process](train/README.md)
