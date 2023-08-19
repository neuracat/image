

import numpy as np
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from models.generator import build_generator
from models.discriminator import build_discriminator

IMG_SHAPE = (128, 128, 3)
BATCH_SIZE = 1
EPOCHS = 100
LR = 0.0002
BETA_1 = 0.5


def load_data(dataset_path, batch_size=BATCH_SIZE, img_res=IMG_SHAPE):
    datagen = ImageDataGenerator(rescale=1./255.)
    data_loader = datagen.flow_from_directory(dataset_path, target_size=img_res[:2], batch_size=batch_size, class_mode=None)
    return data_loader


g_AB = build_generator(IMG_SHAPE, IMG_SHAPE[2])
g_BA = build_generator(IMG_SHAPE, IMG_SHAPE[2])
d_A = build_discriminator(IMG_SHAPE)
d_B = build_discriminator(IMG_SHAPE)


optimizer = Adam(LR, BETA_1)

d_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
d_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])



if __name__ == "__main__":

    data_loader_A = load_data('path_to_dataset_A')
    data_loader_B = load_data('path_to_dataset_B')

 
