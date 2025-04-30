print("--- Script Starting ---")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns # For plotting confusion matrix
from sklearn.metrics import confusion_matrix, classification_report # For evaluation metrics

print("--- Imports Successful ---")

# --- Step 1: Define Constants and Paths ---
img_height = 128
img_width = 128
batch_size = 32
validation_split = 0.2
random_seed = 123

data_dir = os.getcwd() # Gets the current working directory
print(f"--- Using image data from: {data_dir} ---")

# --- Step 2: Load Image Data ---
try:
    print("--- Loading training data... ---")
    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=validation_split,
      subset="training",
      seed=random_seed,
      image_size=(img_height, img_width),
      batch_size=batch_size)
    print("--- Training data loaded. ---")

    print("--- Loading validation data... ---")
    val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=validation_split,
      subset="validation",
      seed=random_seed,
      image_size=(img_height, img_width),
      batch_size=batch_size)
    print("--- Validation data loaded. ---")

except Exception as e:
    print(f"\n!!! ERROR loading data: {e} !!!")
    print("--- Please ensure image files are valid and folders are correct. ---")
    exit()

# --- Step 3: Get Class Names ---
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\n--- Classes found: {class_names} ---")
print(f"--- Number of classes: {num_classes} ---")

# --- Step 4: Configure Datasets for Performance ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
print("--- Datasets configured for performance. ---")

# --- Step 5: Define Data Augmentation Layers ---
data_augmentation = Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ],
  name="data_augmentation"
)
print("--- Data augmentation layers defined. ---")

# --- Step 6: Define CNN Model Architecture ---
print("--- Defining the CNN model architecture... ---")
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),

  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.25),

  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.25),

  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dropout(0.5),

  layers.Dense(num_classes, activation='softmax', name="outputs")
])
print("--- Model defined. ---")

# --- Step 7: Compile Model ---
print("--- Compiling the model... ---")
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
print("--- Model compiled. ---")

# --- Step 8: Print Model Summary ---
print("\n--- Model Summary: ---")
# Build the model with input shape to enable summary printing
model.build(input_shape=(None, img_height, img_width, 3))
model.summary()


# --- Step 9: Define Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True)

print("--- Callbacks defined (EarlyStopping). ---")


# --- Step 10: Train Model ---
print("\n--- Starting model training (with early stopping)... ---")
epochs = 50 # Max epochs; early stopping might halt training sooner

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[early_stopping]
)
print("--- Model training finished. ---")

# --- Step 11: Evaluate Model ---
print("\n--- Evaluating model performance on validation set... ---")
loss, accuracy = model.evaluate(val_ds, verbose=0)
print(f"\n--- Final Validation Set Performance ---")
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy*100:.2f}%")

# --- Step 12: Visualize Training History ---
print("\n--- Plotting training history... ---")
actual_epochs = len(history.history['loss'])

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(range(actual_epochs), history.history['accuracy'], label='Training Accuracy')
plt.plot(range(actual_epochs), history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])

plt.subplot(1, 2, 2)
plt.plot(range(actual_epochs), history.history['loss'], label='Training Loss')
plt.plot(range(actual_epochs), history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, max(plt.ylim())])

plt.suptitle('Model Training History (with Augmentation, Dropout, Early Stopping)')
plt.show()


# --- Step 13: Generate Confusion Matrix ---
print("\n--- Generating predictions for confusion matrix... ---")
val_predictions_proba = model.predict(val_ds)
val_predictions = np.argmax(val_predictions_proba, axis=1)

val_labels = []
for images, labels in val_ds.unbatch():
    val_labels.append(labels.numpy())
val_labels = np.array(val_labels)

if len(val_predictions) != len(val_labels):
     print(f"!!! WARNING: Prediction ({len(val_predictions)}) and label ({len(val_labels)}) counts mismatch for validation set! Skipping Confusion Matrix. !!!")
else:
    print("--- Calculating and plotting confusion matrix... ---")
    cm = confusion_matrix(val_labels, val_predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Validation Set')
    plt.show()

    print("\n--- Classification Report for Validation Set: ---")
    print(classification_report(val_labels, val_predictions, target_names=class_names, zero_division=0))


# --- Step 14: Save Trained Model ---
print("\n--- Saving the trained model... ---")
try:
    model.save('aircraft_classifier_model.keras')
    print("--- Model saved as aircraft_classifier_model.keras ---")
except Exception as e:
    print(f"!!! ERROR saving model: {e} !!!")


print("\n--- Script Finished ---")