import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING logs

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, ReLU, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load and preprocess the CelebA dataset
def load_data(data_dir, img_size=(64, 64)):
    imgs = []
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        img = load_img(img_path, target_size=img_size)
        img = img_to_array(img)
        imgs.append(img)
    imgs = np.array(imgs)
    imgs = (imgs.astype(np.float32) - 127.5) / 127.5  # Normalize the images to [-1, 1]
    return imgs

# Generator Model
def build_generator():
    model = Sequential()
    model.add(Input(shape=(100,)))
    model.add(Dense(8*8*256, activation="relu"))
    model.add(Reshape((8, 8, 256)))
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(ReLU())
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(ReLU())
    model.add(Conv2DTranspose(32, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(ReLU())
    model.add(Conv2D(3, kernel_size=3, strides=1, padding="same", activation='tanh'))
    return model

# Discriminator Model
def build_discriminator():
    model = Sequential()
    model.add(Input(shape=(64, 64, 3)))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# GAN Model combining the Generator and Discriminator
def build_gan(generator, discriminator):
    discriminator.trainable = False
    z = Input(shape=(100,))
    img = generator(z)
    validity = discriminator(img)
    gan = Model(z, validity)
    return gan

# Training the GAN
def train(epochs, batch_size=128, save_interval=50, data_dir=r'C:/Users/DELL/Desktop/Interview for Generative AI Engineer Position/Deeplearning/celeba/img_align_celeba/img_align_celeba'):
    X_train = load_data(data_dir)
    half_batch = int(batch_size / 2)
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    
    # Compile the models
    optimizer = Adam(0.0002, 0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    # Training loop
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]
        noise = np.random.normal(0, 1, (half_batch, 100))
        gen_imgs = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)
        
        # Print the progress
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}] [G loss: {g_loss}]")
        
        # Save generated images at save_interval
        if epoch % save_interval == 0:
            save_imgs(generator, epoch)

# Save generated images for visualization
def save_imgs(generator, epoch, image_grid_rows=4, image_grid_columns=4):
    noise = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale images to [0, 1]
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(4, 4), sharey=True, sharex=True)
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[cnt, :, :, :])
            axs[i, j].axis('off')
            cnt += 1
    plt.show()
    fig.savefig(f"gan_generated_image_epoch_{epoch}.png")

# Main function to run the training
if __name__ == '__main__':
    train(epochs=10000, batch_size=64, save_interval=200, data_dir=r'C:/Users/DELL/Desktop/Interview for Generative AI Engineer Position/Deeplearning/celeba/img_align_celeba/img_align_celeba')
