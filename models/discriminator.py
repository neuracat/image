from keras.models import Model
from keras.layers import Input
from utils.layers import conv2d

def build_discriminator(img_shape):
    img = Input(shape=img_shape)
    d1 = conv2d(img, 64, normalize=False)
    d2 = conv2d(d1, 128)
    d3 = conv2d(d2, 256)
    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d3)
    return Model(img, validity)
