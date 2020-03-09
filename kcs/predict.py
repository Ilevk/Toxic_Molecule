import tensorflow as tf      # tensorflow version _2.0 이상
import pandas     as pd
import numpy      as np
import os


model_save_path = './Train_Result'           # 학습완료 후 체크포인트 저장 폴더 위치, Default : 소스파일디렉토리/Train_Result
test_data_path = './predict_input.csv'          # 학습용 데이터 파일 경로, Default : 소스파일디렉토리와 같음


batch_size              = 10
input_data_size         = 1024



###.............. 테스트 모듈 Body   .......................

# read csv file
if not os.path.isfile(test_data_path):
    print('학습데이터 파일이 경로에 존재하지 않습니다.')

dataset         = pd.read_csv(test_data_path)
print(dataset.head())

feature         = dataset.iloc[:,1:1025].values.astype(float)     ### ecfp Fingerprint 사용 예시 (ecfp0 ~ ecfp1023) column index 1~1024
nrow, ncol      = feature.shape

# reshape input feature
feature         = tf.reshape( feature , (-1 , input_data_size , 1 , 1) )   ### (-1 , 1024 , 1 , 1) )

test_dataset    = tf.data.Dataset.from_tensor_slices( feature )
test_dataset    = test_dataset.batch( batch_size )


# load model and weight
model = tf.keras.models.load_model( model_save_path ) 

y_ = model.predict( test_dataset )
model.summary()

print(y_)
y_ = np.argmax(y_,1)


# save result file
result = dataset[['SMILES']].copy()
result['label'] = y_
result.to_csv('./predict_output.csv', index=False)
