%tensorflow_version 2.x             # Colab에서 tensorflow version 2.0 사용

import tensorflow as tf             # tensorflow version 2.0 이상
import pandas as pd
import numpy as np
import os

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras import Input
from sklearn.metrics import classification_report,confusion_matrix


'''
입력 파라미터들
'''
###   학습 세팅
epoch         = 20                  ### 설정 학습 EPOCH... type:int
batch_size    = 10                  ### 데이터 배치 크기... type:int
learning_rate = 1e-5                ### 학습률... type:float


###   모델 구조 기본파라미터
conv_block_info = [32,3,64,3,64,3]  ### Conv Block Filter, Kernal_Size 파라미터... type:list
fc_block_info   = [128]             ### Fully Connected Layer Unis 파라미터... type:list
drop_out_rate   = 0.2               ### Dropout Rate... type:float  

  
model_save_path = './Train_Result'  # 학습완료 후 체크포인트 저장 폴더 위치, Default : 소스파일디렉토리/Train_Result
try:
  if not(os.path.isdir(model_save_path)):
    os.makedirs(os.path.join(model_save_path))

except OSError as e:
  if e.errno != errno.EEXIST:
    print('학습 결과를 저장할 폴더 생성에 실패하였습니다.')
    raise


train_data_path = './train.csv'      # 학습용 데이터 파일 경로, Default : 소스파일디렉토리와 같음
if not os.path.isfile(train_data_path):
    print('학습데이터 파일이 경로에 존재하지 않습니다.')


###.............. 학습 모듈 Body   .......................

### read csv file
dataset         = pd.read_csv(train_data_path)
print(dataset.head())

feature         = dataset.iloc[:,1:1025].values.astype(float)       ### ecfp Fingerprint 사용 예시 (ecfp0 ~ ecfp1023) column index 1~1024
label           = dataset['label'].values
nrow, ncol      = feature.shape


### reshape input feature
input_data_size = 1024
feature         = tf.reshape(feature, (-1, input_data_size, 1, 1))  ### (-1, 1024, 1, 1) )
input_shape     = (input_data_size, 1, 1)

### Using tf.data.Dataset
full_dataset    = tf.data.Dataset.from_tensor_slices((feature , label))

train_size      = int(0.9 * nrow)         ### train, validation data split (9:1)
train_dataset   = full_dataset.take(train_size)
val_dataset     = full_dataset.skip(train_size)

train_dataset   = train_dataset.shuffle(train_size).batch(batch_size).repeat()
val_dataset     = val_dataset.batch(batch_size)


### Model
model = tf.keras.Sequential()
model.add(Input(shape=input_shape))


for i in range(0, len(conv_block_info), 2):
  filters_ = int(conv_block_info[i])
  kernel_size_ = int(conv_block_info[i+1])

  model.add(Conv2D(filters=filters_, kernel_size=(kernel_size_, 1), strides=(1, 1), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D((2, 1), padding='same'))


model.add(Flatten())


for fc_unit in fc_block_info:
  fc_unit_ = int(fc_unit)

  model.add(Dense(fc_unit_, activation='relu'))
  model.add(Dropout(drop_out_rate))


model.add(Dense(2, activation='softmax'))
model.summary()


adam = tf.keras.optimizers.Adam(learning_rate = learning_rate )
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()


model.compile(optimizer = adam, 
             loss = loss_fn,
             metrics = ['accuracy'])

model.fit( train_dataset, epochs = epoch, steps_per_epoch = train_size//batch_size)


### test data evaluate
val_pred = model.predict( val_dataset )
val_pred = np.argmax(val_pred,1)

val_label = []
for data, label in val_dataset:
  val_label.extend(label.numpy().tolist())

print('\n-----[Evaluation]-----')
print(classification_report(val_label, val_pred, digits=4))
print('\n-----[Confusion Matrix]-----')
print(confusion_matrix(val_label, val_pred))


### Export the model to a SavedModel
model.save( model_save_path, save_format='tf' )