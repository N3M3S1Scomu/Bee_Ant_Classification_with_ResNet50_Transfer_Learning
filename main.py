
import PIL
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.optimizers import Adam

####################### RESNET TRANSFER LEARNING ##############################
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import cv2

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from keras.optimizers import RMSprop
from keras import Model, layers

base_dir = Path("hymenoptera_data")
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')

################### DATASET #############################

train_datagen = ImageDataGenerator(
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=32,
    class_mode='binary',
    target_size=(224, 224))

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    shuffle=False,
    class_mode='binary',
    target_size=(224, 224))

##################### MODEL ################################

pretrained_model=ResNet50(include_top=False,weights="imagenet",input_shape=(224,224,3))

for layer in pretrained_model.layers:
    layer.trainable=False

model=Sequential()

model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dense(2,activation="softmax"))

model.summary()

model.compile(optimizer=Adam(learning_rate=0.001),loss="sparse_categorical_crossentropy",metrics=["accuracy"])

##################### TRAINING ############################

history = model.fit_generator(
    generator=train_generator,
    epochs=10,
    validation_data=validation_generator,
)
"""
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.grid()
plt.title('Model doğruluğu')
plt.ylabel('Doğruluk')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
#plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('Model kaybı')
plt.ylabel('Kayıp')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()
"""
###################### PREDICTION ################################
img=cv2.imread("bee.jpg")
img=cv2.resize(img,(224,224))
img=np.expand_dims(img,axis=0)
print(img.shape)

preds=model.predict(img)

pred=np.argmax(preds,axis=-1)
prob_val=np.argmax(preds)

print(preds,pred,prob_val)

################### OUTPUT ############################
"""
Epoch 1/10
8/8 [==============================] - 36s 4s/step - loss: 8.9163 - accuracy: 0.8115 - val_loss: 10.9076 - val_accuracy: 0.9281
Epoch 2/10
8/8 [==============================] - 34s 4s/step - loss: 3.3986 - accuracy: 0.9549 - val_loss: 8.1188 - val_accuracy: 0.9346
Epoch 3/10
8/8 [==============================] - 33s 4s/step - loss: 1.3309 - accuracy: 0.9754 - val_loss: 8.2169 - val_accuracy: 0.9346
Epoch 4/10
8/8 [==============================] - 33s 4s/step - loss: 1.0094 - accuracy: 0.9795 - val_loss: 6.6458 - val_accuracy: 0.9542
Epoch 5/10
8/8 [==============================] - 33s 4s/step - loss: 0.9001 - accuracy: 0.9836 - val_loss: 9.8241 - val_accuracy: 0.9085
Epoch 6/10
8/8 [==============================] - 33s 4s/step - loss: 1.5812 - accuracy: 0.9795 - val_loss: 7.1233 - val_accuracy: 0.9346
Epoch 7/10
8/8 [==============================] - 33s 4s/step - loss: 0.0963 - accuracy: 0.9877 - val_loss: 7.7816 - val_accuracy: 0.9216
Epoch 8/10
8/8 [==============================] - 33s 4s/step - loss: 0.2288 - accuracy: 0.9959 - val_loss: 8.0540 - val_accuracy: 0.9346
Epoch 9/10
8/8 [==============================] - 33s 4s/step - loss: 0.0407 - accuracy: 0.9959 - val_loss: 8.7465 - val_accuracy: 0.9281
Epoch 10/10
8/8 [==============================] - 33s 4s/step - loss: 0.4408 - accuracy: 0.9918 - val_loss: 7.0359 - val_accuracy: 0.9346
(1, 224, 224, 3)
1/1 [==============================] - 1s 1s/step

[[0. 1.]] [1] 1
"""


