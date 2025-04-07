# %%
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import shutil
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

df = pd.read_csv('./postprocessed_data/cats_filtered.csv', index_col=0).reset_index(drop=True)
df.columns
print(df.info)

df.head()
# %%
raw_path = './raw_data/cat-breeds-dataset/images'
labeled_path = './postprocessed_data/labeled_imgs'

os.makedirs(labeled_path, exist_ok=True)

for breed in os.listdir(raw_path):
    breed_path = os.path.join(raw_path, breed)
    
    if os.path.isdir(breed_path):
        safe_breed = breed.replace(' ', '_')
        # Create breed subdirectory in labeled_path
        breed_dest_dir = os.path.join(labeled_path, safe_breed)
        os.makedirs(breed_dest_dir, exist_ok=True)
        
        for i, image_name in enumerate(sorted(os.listdir(breed_path))):
            image_path = os.path.join(breed_path, image_name)
            
            if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                ext = os.path.splitext(image_name)[1].lower()
                new_filename = f"{safe_breed}_{i}{ext}"
                # Copy to breed-specific subdirectory
                new_filepath = os.path.join(breed_dest_dir, new_filename)
                shutil.copy(image_path, new_filepath)
                print(f"Copied and renamed: {image_path} -> {new_filepath}")
# %% Data Pipeline 
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    labeled_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    labeled_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical', 
    subset='validation',
    shuffle=False
)

# %% Class Balance Verification
if len(train_generator.class_indices) == 0:
    raise ValueError("No classes found. Check directory structure and image paths.")

if len(train_generator.classes) == 0:
    raise ValueError("Generator found zero samples. Verify your image file formats.")

# Modified class weight calculation
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Visualize distribution
plt.figure(figsize=(12,6))
pd.Series(train_generator.classes).value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xticks([])
plt.show()


# %% CNN Architecture (From Scratch)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

# Freeze base layers
base_model.trainable = False  

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_generator.class_indices), 
                        activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
