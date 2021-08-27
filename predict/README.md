# Prediction process
Here is the inference process for reproducing the final submission.
The following script is intended to be run from the same directory hierarchy as this README.
* predict1
```
python inference.py batch_size=100 model_name=tf_efficientnet_b5_ns data_dir=${DATA_DIR} output_dir=${PRED_RESULT_DIR} model_dir=ckpts/22021941 submission_fname=22021941-submission.csv
```
* predict2
```
python inference.py batch_size=100 model_name=tf_efficientnet_b5_ns data_dir=${DATA_DIR} output_dir=${PRED_RESULT_DIR} model_dir=ckpts/22021976 submission_fname=22021976-submission.csv
```
* predict3
```
python inference.py batch_size=100 model_name=tf_efficientnet_b5_ns data_dir=${DATA_DIR} output_dir=${PRED_RESULT_DIR} model_dir=ckpts/22021977 submission_fname=22021977-submission.csv
```
* predict4
```
python inference.py batch_size=100 model_name=tf_efficientnetv2_m_in21ft1k data_dir=${DATA_DIR} output_dir=${PRED_RESULT_DIR} model_dir=ckpts/22022048 submission_fname=22022048-submission.csv
```
* predict5
```
python inference.py batch_size=100 model_name=tf_efficientnetv2_m_in21ft1k data_dir=${DATA_DIR} output_dir=${PRED_RESULT_DIR} model_dir=ckpts/22022067 submission_fname=22022067-submission.csv
```
* blend submissions
```
python blend.py csv_dir=${PRED_RESULT_DIR} sub_name=blend_result\blend-22021976-22021977-22021941-22022048-22022067.csv
```