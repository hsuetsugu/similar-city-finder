import os.path
import gc
import psutil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

pix =256

# MODEL
model_fl = 'model_cae/model_256.json'
model = model_from_json(open(model_fl).read())
model.compile(optimizer='adadelta', loss='binary_crossentropy')

prm_fl ="chkpt_256/CAE_weights.11-0.13-0.13.hdf5"
model.load_weights(prm_fl)
# model.summary()

# DATA
master = pd.read_pickle('result/master_city_only.pkl')
city_images = np.load('result/city_images_256.npy')

generated_imgs = model.predict(city_images)
# get encoded results
# !!! need to adjust get_layer(index)  !!!
encoder = Model(inputs=model.input,
                        outputs=model.get_layer(index=20).output)
encoded_imgs = encoder.predict(city_images)

np.save('result/encoded_images_' + str(pix) + '.npy',encoded_imgs)
np.save('result/generated_images_' + str(pix) + '.npy',generated_imgs)
