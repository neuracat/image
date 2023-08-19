import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.losses import mean_squared_error
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

input_A = Input(shape=IMG_SHAPE)
input_B = Input(shape=IMG_SHAPE)

fake_B = g_AB(input_A)
fake_A = g_BA(input_B)

reconstr_A = g_BA(fake_B)
reconstr_B = g_AB(fake_A)


valid_A = d_A(fake_A)
valid_B = d_B(fake_B)

combined_model = Model(inputs=[input_A, input_B], outputs=[valid_A, valid_B, reconstr_A, reconstr_B])
combined_model.compile(loss=['mse', 'mse', 'mae', 'mae'], optimizer=optimizer)

def train(epochs, batch_size):
    for epoch in range(epochs):
        for batch_i, (imgs_A, imgs_B) in enumerate(zip(load_data('path_to_dataset_A'), load_data('path_to_dataset_B'))):
            
            
            valid = np.ones((batch_size,) + d_A.output_shape[1:])
            fake = np.zeros((batch_size,) + d_A.output_shape[1:])
            
          
            fake_B = g_AB.predict(imgs_A)
            fake_A = g_BA.predict(imgs_B)
            
           
            dA_loss_real = d_A.train_on_batch(imgs_A, valid)
            dA_loss_fake = d_A.train_on_batch(fake_A, fake)
            
            dB_loss_real = d_B.train_on_batch(imgs_B, valid)
            dB_loss_fake = d_B.train_on_batch(fake_B, fake)
            
            
            d_loss = 0.5 * np.add(0.5 * np.add(dA_loss_real, dA_loss_fake), 0.5 * np.add(dB_loss_real, dB_loss_fake))
            
            
            g_loss = combined_model.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B])
            
            print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss[0]} | D Accuracy: {100 * d_loss[1]}] [G loss: {g_loss[0]}]")

if __name__ == "__main__":
    train(EPOCHS, BATCH_SIZE)

