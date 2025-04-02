import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(gpus))
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Dataset paths
train_data_path = r"C:\Users\ABIJITH RAJA B\Desktop\next gen connect\data"
validation_data_path = r"C:\Users\ABIJITH RAJA B\Desktop\next gen connect\validation"

# Ensure dataset folders exist
for path in [train_data_path, validation_data_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset folder not found: {path}")

# Image Preprocessing (Normalization & Augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load Training Data
train_generator = train_datagen.flow_from_directory(
    train_data_path, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

# Load Validation Data
validation_generator = validation_datagen.flow_from_directory(
    validation_data_path, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

# Number of classes
num_classes = len(train_generator.class_indices)
print(f"Detected {num_classes} classes.")
if num_classes != 36:
    raise ValueError(f"Expected 36 classes, but found {num_classes}. Check dataset structure.")

# Load MobileNetV2 and Unfreeze Last Few Layers
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
for layer in base_model.layers[-10:]:  # Unfreezing last 10 layers for better fine-tuning
    layer.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Early stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train Model
epochs = 7
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=[early_stopping]
)

# Save Model in New Format
model_save_path = r"C:\Users\ABIJITH RAJA B\Desktop\next gen connect\isl_gesture_model.keras"
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# Evaluate Model
loss, accuracy = model.evaluate(validation_generator)
print(f"Final Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Final Validation Loss: {loss:.4f}")
