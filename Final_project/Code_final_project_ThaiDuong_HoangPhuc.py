import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.optimizers import Adam 
from keras.optimizers import SGD 
from keras.optimizers import RMSprop
import pathlib


training_dir = pathlib.Path('../input/radardataset/training_set')
training_count = len(list(training_dir.glob('*/*.png')))
print(training_count)

test_dir = pathlib.Path('../input/radardataset/test_set')
test_count = len(list(test_dir.glob('*/*.png')))
print(test_count)

batch_size = 32
img_height = 128
img_width = 128
train_ds = tf.keras.utils.image_dataset_from_directory(
  training_dir,
  validation_split=0,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  validation_split=0,
  seed=113,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

def create_model():
    input = keras.Input(shape=(128,128,3))
    conv = layers.Conv2D(64, (7,7), strides=(2,2), padding="same", name="conv_")(input)
    relu = layers.ReLU()(conv)
    batchnorm = layers.BatchNormalization(epsilon=0.00010, name="batchnorm_")(relu)
    
    conv_5 = layers.Conv2D(64, (1,1), strides=(2,2), padding="same", name="conv_5_")(batchnorm)
    relu_5 = layers.ReLU()(conv_5)
    maxpool = layers.MaxPool2D(pool_size=(5,5), strides=(1,1), padding="same")(relu_5)
    conv_1 = layers.Conv2D(16, (1,1), padding="same", name="conv_1_")(maxpool)
    relu_1 = layers.ReLU()(conv_1)
    conv_2 = layers.Conv2D(16, (1,1), padding="same", name="conv_2_")(maxpool)
    relu_2 = layers.ReLU()(conv_2)
    conv_3 = layers.Conv2D(64, (3,3), padding="same", name="conv_3_")(relu_1)
    relu_3 = layers.ReLU()(conv_3)
    batchnorm_3 = layers.BatchNormalization(epsilon=0.00010, name="batchnorm_3_")(relu_3)
    conv_4 = layers.Conv2D(64, (5,5), padding="same", name="conv_4_")(relu_2)
    relu_4 = layers.ReLU()(conv_4)
    batchnorm_4 = layers.BatchNormalization(epsilon=0.00010, name="batchnorm_4_")(relu_4)
    depthcat = layers.Concatenate(axis=-1)([batchnorm_3, batchnorm_4])
    conv_6 = layers.Conv2D(64, (1,1), padding="same", name="conv_6_")(depthcat)
    relu_6 = layers.ReLU()(conv_6)
    batchnorm_6 = layers.BatchNormalization(epsilon=0.00010, name="batchnorm_6_")(relu_6)
    addition = layers.Add()([maxpool, batchnorm_6])
    
    
    conv_11 = layers.Conv2D(64, (1,1), padding="same", name="conv_11_")(batchnorm)
    relu_11 = layers.ReLU()(conv_11)
    maxpool_1 = layers.MaxPool2D(pool_size=(5,5), strides=(1,1), padding="same")(relu_11)
    conv_7 = layers.Conv2D(16, (1,1), padding="same", name="conv_7_")(maxpool_1)
    relu_7 = layers.ReLU()(conv_7)
    conv_8 = layers.Conv2D(16, (1,1), padding="same", name="conv_8_")(maxpool_1)
    relu_8 = layers.ReLU()(conv_8)
    conv_9 = layers.Conv2D(64, (3,3), padding="same", name="conv_9_")(relu_7)
    relu_9 = layers.ReLU()(conv_9)
    conv_10 = layers.Conv2D(64, (5,5), padding="same", name="conv_10_")(relu_8)
    relu_10 = layers.ReLU()(conv_10)
    depthcat_1 = layers.Concatenate(axis=-1)([relu_9, relu_10])
    conv_12 = layers.Conv2D(64, (1,1), padding="same", name="conv_12_")(depthcat_1)
    relu_12 = layers.ReLU()(conv_12)
    batchnorm_12 = layers.BatchNormalization(epsilon=0.00010, name="batchnorm_12_")(relu_12)
    addition_1 = layers.Add()([maxpool_1, batchnorm_12])
    
    
    conv_17 = layers.Conv2D(64, (1,1), strides=(2,2), padding="same", name="conv_17_")(addition_1)
    relu_17 = layers.ReLU()(conv_17)
    maxpool_2 = layers.MaxPool2D(pool_size=(5,5), strides=(1,1), padding="same")(relu_17)
    conv_13 = layers.Conv2D(16, (1,1), padding="same", name="conv_13_")(maxpool_2)
    relu_13 = layers.ReLU()(conv_13)
    conv_14 = layers.Conv2D(16, (1,1), padding="same", name="conv_14_")(maxpool_2)
    relu_14 = layers.ReLU()(conv_14)
    conv_15 = layers.Conv2D(64, (3,3), padding="same", name="conv_15_")(relu_13)
    relu_15 = layers.ReLU()(conv_15)
    batchnorm_15 = layers.BatchNormalization(epsilon=0.00010, name="batchnorm_15_")(relu_15)
    conv_16 = layers.Conv2D(64, (5,5), padding="same", name="conv_16_")(batchnorm_15)
    relu_16 = layers.ReLU()(conv_16)
    batchnorm_16 = layers.BatchNormalization(epsilon=0.00010, name="batchnorm_16_")(relu_16)
    depthcat_2 = layers.Concatenate(axis=-1)([batchnorm_15, batchnorm_16])
    conv_18 = layers.Conv2D(64, (1,1), padding="same", name="conv_18_")(depthcat_2)
    relu_18 = layers.ReLU()(conv_18)
    batchnorm_18 = layers.BatchNormalization(epsilon=0.00010, name="batchnorm_18_")(relu_18)
    addition_2 = layers.Add()([maxpool_2, batchnorm_18])
    addition_3 = layers.Add()([addition_2, addition])

    
    conv_19 = layers.Conv2D(128, (1,1), padding="same", name="conv_19_")(addition_3)
    relu_19 = layers.ReLU()(conv_19)
    batchnorm_19 = layers.BatchNormalization(epsilon=0.00010, name="batchnorm_19_")(relu_19)
    gapool = layers.GlobalAveragePooling2D(keepdims=True)(batchnorm_19)
    fc = layers.Reshape((-1,), name="fc_preFlatten1")(gapool)
    fc = layers.Dense(84, name="fc_")(fc)
    relu_21 = layers.ReLU()(fc)
    batchnorm_21 = layers.BatchNormalization(epsilon=0.00010, name="batchnorm_21_")(relu_21)
    fc_1 = layers.Dense(8, name="fc_1_")(batchnorm_21)
    softmax = layers.Softmax()(fc_1)

    model = keras.Model(inputs=[input], outputs=[softmax])
    return model

model = create_model()
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001,momentum=0.9),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min')

# Train the model
epochs=40   #tang len 40 epoch
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs, shuffle=True, callbacks=learning_rate_reduction)

accuracy = model.evaluate(val_ds)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()