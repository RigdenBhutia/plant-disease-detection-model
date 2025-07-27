from google.colab import drive
drive.mount('/content/drive')


import zipfile
import os

zip_path = '/content/drive/MyDrive/archive.zip'
extract_to = '/content/drive/MyDrive/unzipped_plant_data'

# Unzip if not already unzipped
if not os.path.exists(extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✅ Dataset unzipped.")
else:
    print("✅ Dataset already exists.")


import os
import shutil
import random
from tqdm import tqdm

# Paths
extracted_dataset_path = '/content/drive/MyDrive/unzipped_plant_data/PlantVillage' # Use the extracted directory path
output_base_dir = '/content/drive/MyDrive/Plant_Village_Dataset_Split'

# Step 1: Clear output directory if it exists
if os.path.exists(output_base_dir):
    shutil.rmtree(output_base_dir)

# Step 2: Re-create train and val directories
os.makedirs(os.path.join(output_base_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, 'val'), exist_ok=True)

# Step 3: Split and copy
classes = [d for d in os.listdir(extracted_dataset_path) if os.path.isdir(os.path.join(extracted_dataset_path, d))]

for cls in tqdm(classes, desc="Splitting"):
    class_dir = os.path.join(extracted_dataset_path, cls)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(images) == 0:
        print(f"⚠️ No images found in class: {cls}")
        continue

    random.shuffle(images)
    split_index = int(0.8 * len(images))
    train_imgs = images[:split_index]
    val_imgs = images[split_index:]

    # Create target class folders
    train_cls_path = os.path.join(output_base_dir, 'train', cls)
    val_cls_path = os.path.join(output_base_dir, 'val', cls)
    os.makedirs(train_cls_path, exist_ok=True)
    os.makedirs(val_cls_path, exist_ok=True)

    # Copy files
    for img in train_imgs:
        shutil.copy(os.path.join(class_dir, img), os.path.join(train_cls_path, img))

    for img in val_imgs:
        shutil.copy(os.path.join(class_dir, img), os.path.join(val_cls_path, img))

print("✅ Dataset split complete.")


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Set dataset path
train_path = '/content/drive/MyDrive/Plant_Village_Dataset_Split/train'
val_path = '/content/drive/MyDrive/Plant_Village_Dataset_Split/val'
image_size = (224, 224)
batch_size = 32

# ✅ Step 1 & 2: Data augmentation + proper validation split
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    val_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)



# ✅ Model definition
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)


# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ✅ Step 3: Add EarlyStopping to prevent overtraining
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ✅ Model training with callbacks
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[early_stop]
)


# ✅ Accuracy Plot
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


model.save('/content/drive/MyDrive/plant_disease_model.keras')

import tensorflow as tf
tf.saved_model.save(model, '/content/plant_disease_model_tf')


import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('/content/drive/MyDrive/plant_disease_model.keras')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Ensure it's in inference mode
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_enable_resource_variables = False  # <== CRUCIAL LINE

# Convert
tflite_model = converter.convert()

# Save
with open('/content/drive/MyDrive/plant_disease_model_fixed.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ TFLite model saved successfully.")


converter = tf.lite.TFLiteConverter.from_saved_model('/content/plant_disease_model_tf')
tflite_model = converter.convert()

# Save the .tflite model
with open('/content/drive/MyDrive/plant_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Successfully converted to TFLite and saved!")

