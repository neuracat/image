from keras.models import Model
from keras.layers import Input, Concatenate, Dropout
from utils.layers import conv2d, deconv2d

def build_generator(img_shape, channels):
    d0 = Input(shape=img_shape)
    d1 = conv2d(d0, 64, normalize=False)
    d2 = conv2d(d1, 128)
    d3 = conv2d(d2, 256)
    d4 = conv2d(d3, 512)
    
    u1 = deconv2d(d4, d3, 256)
    u2 = deconv2d(u1, d2, 128)
    u3 = deconv2d(u2, d1, 64)

    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

    return Model(d0, output_img)
