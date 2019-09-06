import os.path
import gc
import psutil

import numpy as np
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

#Kerasの設定
import keras
from keras import layers
from keras import backend as K
from keras.models import Model
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils.vis_utils import plot_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint


# 元画像の解像度の設定
pix = 256

# train-test split
data = np.load('road_network_'+str(pix) +'.npy')
print(data.shape)

print('Memory Usage: {}'.format(psutil.virtual_memory()))

data = data.astype('float32') / 255
data = 1-data
x_train, x_test = train_test_split(data, test_size=0.1, random_state=0)

x_train = np.reshape(x_train, (len(x_train), pix, pix, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), pix, pix, 1))  # adapt this if using `channels_first` image data format
print(x_train.shape, x_test.shape)

del data
gc.collect()
print('Memory Usage: {}'.format(psutil.virtual_memory()))

# Convolutional Auto Encoder
input_img = Input(shape=(pix, pix, 1))

### encoder
#1
x = Conv2D(64, (5, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
print(x.shape)

#2
x = Conv2D(32, (5, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
print(x.shape)

#3
x = Conv2D(16, (5, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
print(x.shape)

#4
x = Conv2D(10, (5, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
print(x.shape)


#5
x = Conv2D(10, (5, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
print('encoded:',encoded.shape)

### decoder
#1
x = Conv2D(10, (5, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)

#2
x = Conv2D(10, (5, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)

#3
x = Conv2D(16, (5, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)

#4
x = Conv2D(32, (5, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)

#5
x = Conv2D(64, (5, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)

#5
x = Conv2D(1, (5, 3), padding='same')(x)
x = BatchNormalization()(x)
decoded = Activation('sigmoid')(x)

model = Model(input_img, decoded)
model.compile(optimizer='adam', loss='binary_crossentropy')

es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
chkpt = 'chkpt_' + str(pix) + '/CAE_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model.summary()
model.load_weights('chkpt_256/CAE_weights.02-0.36-0.29.hdf5')

"""
es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
chkpt = 'chkpt/CAE_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model = model_from_json(open('model_cae/model_128.json').read())
model.compile(optimizer='adadelta', loss='binary_crossentropy')
model.load_weights('model_cae/model_weights_128.hdf5')

model.summary()
"""

history = model.fit(x_train, x_train,
                epochs=30,
                batch_size=128,
                shuffle=True,
                callbacks=[es_cb, cp_cb],
               validation_data=(x_test,x_test))

# save result
print('save the architecture of a model')
open('model_cae/model_' + str(pix) +'.json', 'w').write(model.to_json())
print('save weights')
model.save_weights('model_cae/model_weights_' + str(pix) +'.hdf5')

