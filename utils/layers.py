from keras.layers import Conv2D, UpSampling2D, Activation
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

def conv2d(layer_input, filters, f_size=4, normalize=True):
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    if normalize:
        d = InstanceNormalization()(d)
    d = Activation('relu')(d)
    return d

def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
    if dropout_rate:
        u = Dropout(dropout_rate)(u)
    u = InstanceNormalization()(u)
    u = Concatenate()([u, skip_input])
    return u
