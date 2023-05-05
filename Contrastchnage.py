#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageEnhance


def reduce_contrast(image, contrast_factor):
    pil_image = Image.open(image)
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_image = enhancer.enhance(contrast_factor)
    enhanced_image.save("reduced_contrast_" + image)

    return "reduced_contrast_" + image


def train_and_evaluate_model(train_df, test_df):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=None,
        x_col="local_filename",
        y_col="label",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=RANDOM_SEED
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=None,
        x_col="local_filename",
        y_col="label",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=RANDOM_SEED
    )

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

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        metrics=['accuracy']
    )

    model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=EPOCHS
    )

    test_loss, test_acc = model.evaluate(test_generator)
    return test_acc


df = pd.read_csv('final_dataset.csv')

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 110
EPOCHS = 20
RANDOM_SEED = 42

train_df = df[df['fitzpatrick_scale'].isin([1, 2])]
test_df = df[df['fitzpatrick_scale'].isin([5, 6])]

# Train and evaluate model with original images
accuracy1 = train_and_evaluate_model(train_df, test_df)
print(f'Accuracy of the model trained on original images: {accuracy1}')

# Reduce the contrast of Fitzpatrick scale 1, 2 images
reduced_contrast_train_df = train_df.copy()
reduced_contrast_train_df["local_filename"] = reduced_contrast_train_df["local_filename"].apply(
    lambda x: reduce_contrast(x, 0.5))

# Train and evaluate model with reduced contrast images
accuracy2 = train_and_evaluate_model(reduced_contrast_train_df, test_df)
print(f'Accuracy of the model trained on reduced contrast images: {accuracy2}')

plt.bar(['Original Images', 'Reduced Contrast Images'], [accuracy1, accuracy2])
plt.ylabel('Accuracy')
plt.title('Comparison of model accuracy with original and reduced contrast images')
plt.show()

