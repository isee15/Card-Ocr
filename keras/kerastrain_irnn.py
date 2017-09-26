# -*- coding: UTF-8 -*-

from __future__ import print_function

import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.layers import Input, Dense, Activation
from keras.layers import K, Lambda, LSTM
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.utils.vis_utils import plot_model
from util import split_image

data_dir = './data'
train_data_dir = os.path.join(data_dir, 'train')
test_data_dir = os.path.join(data_dir, 'test')

# dimensions of our images.
img_width, img_height = 28, 28
charset_size = 11
nb_validation_samples = 11
nb_samples_per_epoch = 11
nb_nb_epoch = 100
txt = "0123456789X"

hidden_units = 128
learning_rate = 0.001


def train(model):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=1024,
        color_mode="grayscale",
        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=1024,
        color_mode="grayscale",
        class_mode='categorical')

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(lr=learning_rate),
                  metrics=['accuracy'])
    model.fit_generator(train_generator,
                        steps_per_epoch=nb_samples_per_epoch,
                        epochs=nb_nb_epoch,
                        validation_data=validation_generator,
                        validation_steps=nb_validation_samples)


def rgb_to_grayscale(input):
    """Average out each pixel across its 3 RGB layers resulting in a grayscale image"""
    return K.mean(input, axis=3)


def rgb_to_grayscale_output_shape(input_shape):
    return input_shape[:-1]


def build_model(input_shape=(28, 28, 1), classes=charset_size):
    img_input = Input(shape=input_shape)

    # SimpleRNN
    # x = Lambda(rgb_to_grayscale, rgb_to_grayscale_output_shape)(img_input)
    # x = SimpleRNN(hidden_units,
    #               kernel_initializer=initializers.RandomNormal(stddev=0.001),
    #               recurrent_initializer=initializers.Identity(gain=1.0),
    #               activation='relu',
    #               input_shape=(28, 28))(x)
    # x = Dense(charset_size)(x)
    # x = Activation('softmax')(x)
    # model = Model(img_input, x, name='model')

    # Hierarchical recurrent neural network
    # x = TimeDistributed(LSTM(hidden_units))(img_input)
    # x = LSTM(hidden_units)(x)
    # x = Dense(charset_size)(x)
    # x = Activation('softmax')(x)
    # model = Model(img_input, x, name='model')

    # LSTM
    x = Lambda(rgb_to_grayscale, rgb_to_grayscale_output_shape)(img_input)
    x = LSTM(hidden_units)(x)
    x = Dense(charset_size)(x)
    x = Activation('softmax')(x)
    model = Model(img_input, x, name='model')

    plot_model(model, show_shapes=True)
    # 训练的正确率和误差，acc和loss 验证集正确率和误差val_acc和val_loss
    return model


def decode(y):
    y = np.argmax(y, axis=1)
    print(y)
    return ''.join([txt[x] for x in y])


model = build_model()
train(model)
model.save("./model.h5")


# model = load_model("./model.h5")


def squareImage(image, size=(28, 28)):
    wh1 = image.width / image.height
    wh2 = size[0] / size[1]
    newsize = ((int)(size[1] * wh1), (int)(size[1]))
    if wh1 > wh2:
        newsize = ((int)(size[0]), (int)(size[0] / wh1))

    image = image.resize(newsize, Image.ANTIALIAS)
    img_padded = Image.new("L", size, 255)
    img_padded.paste(image, (int((size[0] - image.size[0]) / 2), int((size[1] - image.size[1]) / 2)))
    return img_padded


fig, axes = plt.subplots(3, 6, figsize=(7, 6))
print(axes.shape)
filename = "445281198606095334.png"
image = Image.open(filename).convert("LA")
cimgs = split_image(image)
i = 0
for img in cimgs:
    ii = i + 1
    img.save("data/train/" + filename[i] + "/" + str(i) + "-" + filename)
    img = squareImage(img)
    x = img_to_array(img)
    # reshape to array rank 4
    x = x.reshape((-1,) + x.shape)
    print(x.shape)
    y_pred = model.predict(x)
    print(model.predict(x))
    print(i)
    axes[(int)(i / 6), (int)(i % 6)].set_title('predict:%s' % (decode(y_pred)))
    axes[(int)(i / 6), (int)(i % 6)].imshow(img, cmap='gray')
    i += 1

plt.tight_layout()
plt.show()

