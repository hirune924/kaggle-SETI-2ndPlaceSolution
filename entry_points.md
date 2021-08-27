# train models
## train with warmup
* options
    - batch_size: batch size
    - epoch: train epoch
    - height: image height
    - width: image width
    - model_name: model architecture
    - drop_rate: Dropout rate
    - drop_path_rate: Stochastic Depth rate
    - lr: learning rate
    - fold: validation fold
    - data_dir: data root directory
    - model_path: resume pretrained model
    - output_dir: output directory 
```
python step1/train.py
```

## train with pseudo label
* options
    - batch_size: batch size
    - epoch: train epoch
    - height: image height
    - width: image width
    - model_name: model architecture
    - drop_rate: Dropout rate
    - drop_path_rate: Stochastic Depth rate
    - lr: learning rate
    - fold: validation fold
    - data_dir: data root directory
    - model_path: resume pretrained model
    - output_dir: output directory 
    - pseudo: pseudo label file path

```
python final/train.py
```

## predict
* options
    - batch_size: batch size
    - height: image height
    - width: image width
    - model_name: model architecture
    - data_dir: data root directory
    - model_dir:  ckpt directory
    - output_dir: output directory 
    - submission_fname: result file name

```
python predict/predict.py
```