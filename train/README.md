# Train process
Steps 1 through 4 are for creating pseudo labels. And the final step is to train the model for the final predict.
## step1
* step1 train
```
pip install -U timm

python step1/train.py fold=0 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
data_dir=${DATA_DIR} output_dir=step1/output/fold0_0

python step1/train.py fold=1 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
data_dir=${DATA_DIR} output_dir=step1/output/fold1_0

python step1/train.py fold=2 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
data_dir=${DATA_DIR} output_dir=step1/output/fold2_0

python step1/train.py fold=3 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
data_dir=${DATA_DIR} output_dir=step1/output/fold3_0

python step1/train.py fold=4 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
data_dir=${DATA_DIR} output_dir=step1/output/fold4_0



python step1/train.py fold=0 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=step1/output/fold0_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step1/output/fold0_1

python step1/train.py fold=1 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=step1/output/fold1_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step1/output/fold1_1

python step1/train.py fold=2 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=step1/output/fold2_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step1/output/fold2_1

python step1/train.py fold=3 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=step1/output/fold3_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step1/output/fold3_1

python step1/train.py fold=4 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.4 drop_path_rate=0.2 \
model_path=step1/output/fold4_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step1/output/fold4_1
```

* step1 predict(make pseudo label)
```
pip install -U timm
python step1/predict.py batch_size=100 model_name=tf_efficientnet_b5_ns data_dir=${DATA_DIR} output_dir=step1/output/ model_dir=step1/output submission_fname=submission.csv
```
## step2
* step2 train
```
python step2/train.py fold=0 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step1/output/fold0_1/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step2/output/fold0_0 pseudo=step1/output/submission.csv

python step2/train.py fold=1 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step1/output/fold1_1/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step2/output/fold1_0 pseudo=step1/output/submission.csv

python step2/train.py fold=2 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step1/output/fold2_1/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step2/output/fold2_0 pseudo=step1/output/submission.csv

python step2/train.py fold=3 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step1/output/fold3_1/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step2/output/fold3_0 pseudo=step1/output/submission.csv

python step2/train.py fold=4 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step1/output/fold4_1/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step2/output/fold4_0 pseudo=step1/output/submission.csv
```

* step2 predict(make pseudo label)
```
pip install -U timm
python step2/predict.py batch_size=100 model_name=tf_efficientnet_b5_ns data_dir=${DATA_DIR} output_dir=step2/output/ model_dir=step2/output submission_fname=submission.csv snap=1
```
## step3
* step3 train
```
python step3/train.py fold=0 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step2/output/fold0_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step3/output/fold0_0 pseudo=step2/output/submission.csv

python step3/train.py fold=1 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step2/output/fold1_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step3/output/fold1_0 pseudo=step2/output/submission.csv

python step3/train.py fold=2 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step2/output/fold2_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step3/output/fold2_0 pseudo=step2/output/submission.csv

python step3/train.py fold=3 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step2/output/fold3_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step3/output/fold3_0 pseudo=step2/output/submission.csv

python step3/train.py fold=4 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step2/output/fold4_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step3/output/fold4_0 pseudo=step2/output/submission.csv
```

* step3 predict(make pseudo label)
```
pip install -U timm
python step3/predict.py batch_size=100 model_name=tf_efficientnet_b5_ns data_dir=${DATA_DIR} output_dir=step3/output/ model_dir=step3/output submission_fname=submission.csv snap=1
```
## step4
* step4 train
```
python step4/train.py fold=0 batch_size=36 epoch=10 height=512 width=768 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold0_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step4/output/fold0_0 pseudo=step3/output/submission.csv

python step4/train.py fold=1 batch_size=36 epoch=10 height=512 width=768 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold1_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step4/output/fold1_0 pseudo=step3/output/submission.csv

python step4/train.py fold=2 batch_size=36 epoch=10 height=512 width=768 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold2_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step4/output/fold2_0 pseudo=step3/output/submission.csv

python step4/train.py fold=3 batch_size=36 epoch=10 height=512 width=768 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold3_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step4/output/fold3_0 pseudo=step3/output/submission.csv

python step4/train.py fold=4 batch_size=36 epoch=10 height=512 width=768 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold4_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=step4/output/fold4_0 pseudo=step3/output/submission.csv
```

* step4 predict(make pseudo label)
```
pip install -U timm
python step4/predict.py batch_size=100 model_name=tf_efficientnet_b5_ns data_dir=${DATA_DIR} output_dir=step4/output/ model_dir=step4/output submission_fname=submission.csv snap=1  height=512 width=768
```

## final step

* 22021941
```
python final/train.py fold=0 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold0_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021941/fold0_0 pseudo=step3/output/submission.csv

python final/train.py fold=1 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold1_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021941/fold1_0 pseudo=step3/output/submission.csv

python final/train.py fold=2 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold2_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021941/fold2_0 pseudo=step3/output/submission.csv

python final/train.py fold=3 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold3_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021941/fold3_0 pseudo=step3/output/submission.csv

python final/train.py fold=4 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold4_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021941/fold4_0 pseudo=step3/output/submission.csv
```
* 22021976
```
python final/train.py fold=0 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold0_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021976/fold0_0 pseudo=step3/output/submission.csv seed=5656

python final/train.py fold=1 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold1_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021976/fold1_0 pseudo=step3/output/submission.csv seed=5656

python final/train.py fold=2 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold2_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021976/fold2_0 pseudo=step3/output/submission.csv seed=5656

python final/train.py fold=3 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold3_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021976/fold3_0 pseudo=step3/output/submission.csv seed=5656

python final/train.py fold=4 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold4_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021976/fold4_0 pseudo=step3/output/submission.csv seed=5656
```

* 22021977
```
python step4/train.py fold=0 batch_size=36 epoch=10 height=512 width=768 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold0_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021977/fold0_0 pseudo=step3/output/submission.csv

python step4/train.py fold=1 batch_size=36 epoch=10 height=512 width=768 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold1_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021977/fold1_0 pseudo=step3/output/submission.csv

python step4/train.py fold=2 batch_size=36 epoch=10 height=512 width=768 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold2_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021977/fold2_0 pseudo=step3/output/submission.csv

python step4/train.py fold=3 batch_size=36 epoch=10 height=512 width=768 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold3_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021977/fold3_0 pseudo=step3/output/submission.csv

python step4/train.py fold=4 batch_size=36 epoch=10 height=512 width=768 model_name=tf_efficientnet_b5_ns drop_rate=0.5 drop_path_rate=0.3 \
model_path=step3/output/fold4_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021977/fold4_0 pseudo=step3/output/submission.csv
```

* 22021766
```
pip install -U timm

python step1/train.py fold=0 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.4 drop_path_rate=0.2 \
data_dir=${DATA_DIR} output_dir=final/22021766/fold0_0

python step1/train.py fold=1 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.4 drop_path_rate=0.2 \
data_dir=${DATA_DIR} output_dir=final/22021766/fold1_0

python step1/train.py fold=2 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.4 drop_path_rate=0.2 \
data_dir=${DATA_DIR} output_dir=final/22021766/fold2_0

python step1/train.py fold=3 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.4 drop_path_rate=0.2 \
data_dir=${DATA_DIR} output_dir=final/22021766/fold3_0

python step1/train.py fold=4 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.4 drop_path_rate=0.2 \
data_dir=${DATA_DIR} output_dir=final/22021766/fold4_0


python step1/train.py fold=0 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.4 drop_path_rate=0.2 \
model_path=final/22021766/fold0_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021766/fold0_1

python step1/train.py fold=1 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.4 drop_path_rate=0.2 \
model_path=final/22021766/fold1_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021766/fold1_1

python step1/train.py fold=2 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.4 drop_path_rate=0.2 \
model_path=final/22021766/fold2_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021766/fold2_1

python step1/train.py fold=3 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.4 drop_path_rate=0.2 \
model_path=final/22021766/fold3_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021766/fold3_1

python step1/train.py fold=4 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.4 drop_path_rate=0.2 \
model_path=final/22021766/fold4_0/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22021766/fold4_1
```
* 22022048
```
python final/train.py fold=0 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.5 drop_path_rate=0.3 \
model_path=final/22021766/fold0_1/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22022048/fold0_0 pseudo=step4/output/submission.csv

python final/train.py fold=1 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.5 drop_path_rate=0.3 \
model_path=final/22021766/fold1_1/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22022048/fold1_0 pseudo=step4/output/submission.csv

python final/train.py fold=2 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.5 drop_path_rate=0.3 \
model_path=final/22021766/fold2_1/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22022048/fold2_0 pseudo=step4/output/submission.csv

python final/train.py fold=3 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.5 drop_path_rate=0.3 \
model_path=final/22021766/fold3_1/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22022048/fold3_0 pseudo=step4/output/submission.csv

python final/train.py fold=4 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.5 drop_path_rate=0.3 \
model_path=final/22021766/fold4_1/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22022048/fold4_0 pseudo=step4/output/submission.csv
```
* 22022067
```
python final/train.py fold=0 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.5 drop_path_rate=0.3 \
model_path=final/22021766/fold0_1/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22022067/fold0_0 pseudo=step4/output/submission.csv seed=5656

python final/train.py fold=1 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.5 drop_path_rate=0.3 \
model_path=final/22021766/fold1_1/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22022067/fold1_0 pseudo=step4/output/submission.csv seed=5656

python final/train.py fold=2 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.5 drop_path_rate=0.3 \
model_path=final/22021766/fold2_1/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22022067/fold2_0 pseudo=step4/output/submission.csv seed=5656

python final/train.py fold=3 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.5 drop_path_rate=0.3 \
model_path=final/22021766/fold3_1/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22022067/fold3_0 pseudo=step4/output/submission.csv seed=5656

python final/train.py fold=4 batch_size=36 epoch=20 height=512 width=512 model_name=tf_efficientnetv2_m_in21ft1k drop_rate=0.5 drop_path_rate=0.3 \
model_path=final/22021766/fold4_1/ckpt/last.ckpt data_dir=${DATA_DIR} output_dir=final/22022067/fold4_0 pseudo=step4/output/submission.csv seed=5656
```
