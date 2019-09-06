import os.path
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
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import model_from_json

# 元画像の解像度の設定
pix = 128

x = np.load('road_network.npy')
print(x.shape)

x = x.astype('float32') / 255
x = 1- x
x = x.reshape((len(x), np.prod(x.shape[1:])))
print(x.shape)

x_train, x_test = train_test_split(x, test_size=0.2, random_state=0)
print(x_train.shape, x_test.shape)

# Auto Encoder
encoding_dim = 3920

input_img = layers.Input(shape=(pix*pix,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
decoded = layers.Dense(pix*pix, activation='sigmoid')(encoded)
autoencoder = Model(input=input_img, output=decoded)

encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = layers.Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# plot_model(autoencoder, to_file="architecture.png", show_shapes=True)

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
               validation_data=(x_test, x_test))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


# save result
print('save the architecture of a model')
open('model/ae_model.json', 'w').write(autoencoder.to_json())
open('model/e_model.json', 'w').write(encoder.to_json())
open('model/d_model.json', 'w').write(decoder.to_json())

print('save weights')
autoencoder.save_weights('model/ae_model_weights.hdf5')
encoder.save_weights('model/e_model_weights.hdf5')
decoder.save_weights('model/d_model_weights.hdf5')



