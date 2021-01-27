#from keras.applications.resnet50 import ResNet50
from resnet50_64 import ResNet50
from keras.layers import Flatten, Dense, Dropout,Input,GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD, Adam
import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from keras.utils import to_categorical
import math
from keras.preprocessing.image import ImageDataGenerator

import keras.backend as K
K.set_image_data_format('channels_last')


BATCH_SIZE = 64
img_width, img_height = 64, 64
nb_epoch = 100
num_classes = 7

TRAIN_DIR = './data/train'
VALID_DIR = './data/val'
num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

gen = ImageDataGenerator(
     rotation_range=20,
     width_shift_range=0.2,
     height_shift_range=0.2,
     horizontal_flip=True)

val_gen = ImageDataGenerator(
     rotation_range=20,
     width_shift_range=0.2,
     height_shift_range=0.2,
     horizontal_flip=True)

train_batches = gen.flow_from_directory(TRAIN_DIR,
                                  target_size=(img_width, img_height),
                                  class_mode='categorical',
                                  shuffle=True,
                                  batch_size=BATCH_SIZE)


val_batches = val_gen.flow_from_directory(VALID_DIR,
                                          target_size=(img_width, img_height),
                                          class_mode='categorical',
                                          shuffle=True,
                                          batch_size=BATCH_SIZE)


image_input = Input(shape=(img_width, img_height, 3))

model = ResNet50(input_tensor=image_input, include_top= True, weights='imagenet')
model.summary()

x = model.get_layer('average_pooling2d_1').input
x = GlobalAveragePooling2D(name='avg_pool')(x)
#    x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
out = Dense(num_classes, activation='softmax', name='output_layer')(x)
resnet_model = Model(inputs=image_input, outputs=out)

# for layer in model.layers[:80]:
#     layer.trainable = True

# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
#model.compile(loss='binary_crossentropy',
#                            optimizer=SGD(lr=1e-3, momentum=0.8),
#                            metrics=['accuracy'])

resnet_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
resnet_model.summary()

from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(patience=10)
checkpointer = ModelCheckpoint('resnet50_best.h5', verbose=1, save_best_only=True)

hist = resnet_model.fit_generator(train_batches,
                                        steps_per_epoch=num_train_steps,
                                        epochs=nb_epoch,
                                        callbacks=[early_stopping, checkpointer],
                                        validation_data=val_batches,
                                        validation_steps=num_valid_steps)



resnet_model.save('./resnet50model.h5')

###############################################################################
import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']

xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#plt.style.use(['classic'])

