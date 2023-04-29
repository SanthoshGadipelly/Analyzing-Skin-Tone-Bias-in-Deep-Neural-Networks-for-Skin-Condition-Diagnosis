#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[49]:


df = pd.read_csv('final_dataset.csv')


# In[50]:


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 114
EPOCHS = 20


# In[51]:


# Set a random seed for reproducibility
RANDOM_SEED = 42

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory=None,
    x_col="local_filename",
    y_col="label",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=False,  # Disable shuffling to keep the order of data consistent
    seed=RANDOM_SEED
)

test_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory=None,
    x_col="local_filename",
    y_col="label",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=RANDOM_SEED
)


# In[52]:


model = tf.keras.models.Sequential([
    tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])


# In[53]:


# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    metrics=['accuracy']
)


# In[54]:


model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS
)


# In[55]:


test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




