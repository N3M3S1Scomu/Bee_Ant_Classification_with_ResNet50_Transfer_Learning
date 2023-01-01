
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




