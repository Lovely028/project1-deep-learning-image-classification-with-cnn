# Link to three notebooks, presentation, and report here: https://drive.google.com/drive/folders/1T3CsPVH2A6vZF5F7wdscqvbCjphdfK_A?usp=sharing

# Animal Image Classification Project

## Overview

This project demonstrates how to build, train, evaluate, and test a Convolutional Neural Network (CNN) model for classifying animal images into multiple classes. The workflow covers:

- Mounting Google Drive to access dataset files
- Extracting and organizing the dataset
- Loading and preprocessing images
- Building a CNN model
- Training with validation
- Plotting training metrics and generating classification reports
- Creating an interactive manual image classification demo using Gradio

---

## Setup and Data Preparation

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Mount your Google Drive to access the ZIP archive containing the dataset.

---

### 2. Extract Dataset

```python
import zipfile
import os

# Define extraction paths
animals_path = '/content/drive/MyDrive/archive.zip'  # Path to your ZIP file in Drive
extract_path = '/content/animals_dataset'

# Extract the ZIP file
with zipfile.ZipFile(animals_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction complete!")
```

---

### 3. Explore Dataset Structure

```python
raw_img_path = os.path.join(extract_path, 'archive', 'raw-img')
print(os.listdir(raw_img_path))  # List class folders
```

Check the folder names to confirm class names (Italian names may require renaming or mapping to English).

---

## Load and Preprocess Dataset

```python
import tensorflow as tf

img_height = 160
img_width = 160
batch_size = 16

# Load train and validation datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    raw_img_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    raw_img_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Optimize dataset loading
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Get class names
class_names = train_ds.class_names
print("Classes:", class_names)
```

---

## Visualize Sample Images

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()
```

---

## Define and Compile CNN Model

```python
from tensorflow.keras import layers, models

cnn_model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Train the Model

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = cnn_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    #callbacks=[early_stop]
)
```

---

## Plot Training Accuracy and Loss

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('CNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

---

## Generate Classification Report

```python
# Extract true labels and images from validation dataset
y_true = []
images = []

for img_batch, label_batch in val_ds:
    images.append(img_batch)
    y_true.append(label_batch)

images = np.concatenate(images)
y_true = np.concatenate(y_true)

# Predict classes
cnn_preds = cnn_model.predict(images)
y_pred_cnn = np.argmax(cnn_preds, axis=1)

# Print classification report
print("
Classification Report for CNN Model:")
print(classification_report(y_true, y_pred_cnn, target_names=class_names))
```

---

## Manual Testing with Gradio

```python
!pip install gradio --quiet

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

model = cnn_model  # Use your trained model

def classify_image(img):
    img = img.resize((img_height, img_width))  # Resize to model input size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0) / 255.0  # Normalize and batch

    predictions = model.predict(img_array)
    confidence_scores = predictions[0]
    predicted_index = np.argmax(confidence_scores)
    predicted_label = class_names[predicted_index]

    return {class_names[i]: float(confidence_scores[i]) for i in range(len(class_names))}

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Animal Image Classifier",
    description="Upload an image of an animal to get predictions from the trained CNN model."
)

interface.launch(share=True)
```

---

## Summary

- Mounted Google Drive and extracted dataset
- Loaded and preprocessed images for training/validation
- Built and trained a CNN model
- Evaluated model performance with plots and classification report
- Created an interactive Gradio app for manual image classification testing


##########################################################################################################




#  Animal Image Classification with MobileNetV2 (Transfer Learning)

This project uses **MobileNetV2** with **transfer learning and fine-tuning** to classify animal images into 10 categories using the **Animal-10 dataset**.

##  Dataset

- Source: [Animal-10 Dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10)  
- Format: 10 class folders in `raw-img/` directory
- Classes (in Italian): Cane (Dog), Gatto (Cat), Cavallo (Horse), Elefante (Elephant), Farfalla (Butterfly), Gallina (Chicken), Mucca (Cow), Pecora (Sheep), Ragno (Spider), Scoiattolo (Squirrel)
- Dataset split:
  - 80% Training
  - 20% Validation

##  Preprocessing

- Images resized to **224x224**
- Normalized to range `[-1, 1]` using MobileNetV2 preprocessing
- Augmentation: Random flip, rotation, and zoom
- Data loaded using `image_dataset_from_directory` in TensorFlow

##  Model: MobileNetV2 + Custom Classifier

- **Base Model**: MobileNetV2 (pretrained on ImageNet, `include_top=False`)
- **Custom Head**:
  - Global Average Pooling
  - Dense(256, ReLU) + Dropout(0.5)
  - Dense(128, ReLU) + Dropout(0.3)
  - Output Layer: Dense(10, softmax)
- Phase 1: Feature extraction (base model frozen)
- Phase 2: Fine-tuning (last 50 layers of base model unfrozen)

##  Training

- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Early stopping with patience of 3 epochs
- Batch size: 32
- Epochs:
  - Phase 1: 10
  - Phase 2 (Fine-tuning): 10

## Results

### Evaluation Metrics (on Validation Set):

| Metric     | Score     |
|------------|-----------|
| Accuracy   | 0.92 (example) |
| Precision  | 0.91 (weighted) |
| Recall     | 0.92 (weighted) |
| F1-score   | 0.91 (weighted) |

> *Note: Actual scores may vary depending on training run and dataset splits.*

### Confusion Matrix & Classification Report

- Included to analyze class-wise performance and confusions
- Useful to detect overfitting, class imbalance, or misclassification

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy, Matplotlib, Scikit-learn
- Google Colab (used for this project)
- Kaggle API (to download dataset, if applicable)

## Future Improvements

- Try other models: ResNet, EfficientNet
- Hyperparameter tuning
- Model deployment (Flask, Streamlit, or TFLite)
- Label cleanup (translate Italian to English)

###############################################################################################################

NoteBOOK 3 


# Animal Image Classification with TensorFlow

This project demonstrates how to train image classification models on the Animals-10 dataset using TensorFlow and Keras. It includes dataset preparation, model training with MobileNetV2 and ResNet50, fine-tuning, evaluation, and deployment with a Gradio web interface.

---

## Table of Contents

- [Dataset Preparation](#dataset-preparation)  
- [Exploring the Dataset](#exploring-the-dataset)  
- [Creating TensorFlow Datasets](#creating-tensorflow-datasets)  
- [Model Building and Training](#model-building-and-training)  
- [Fine-Tuning ResNet50](#fine-tuning-resnet50)  
- [Model Evaluation](#model-evaluation)  
- [Saving and Loading Models](#saving-and-loading-models)  
- [Deployment with Gradio](#deployment-with-gradio)  

---

## Dataset Preparation

1. **Mount Google Drive in Colab** to access dataset files:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Set paths** and extract the dataset ZIP file:

    ```python
    import zipfile
    import os

    animals_path = '/content/drive/MyDrive/animals.zip'  # Path to dataset ZIP in Drive
    extract_path = '/content/animals_dataset'

    with zipfile.ZipFile(animals_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    print("Extraction complete!")
    ```

---

## Exploring the Dataset

- Check extracted folders and class subfolders:

    ```python
    raw_img_path = os.path.join(extract_path, 'raw-img')
    print(os.listdir(raw_img_path))  # List class folders
    ```

- Preview some sample images from each class:

    ```python
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    for folder in os.listdir(raw_img_path):
        folder_path = os.path.join(raw_img_path, folder)
        image_files = os.listdir(folder_path)[:5]

        print(f"\nClass: {folder}")
        plt.figure(figsize=(15, 3))
        for idx, image_file in enumerate(image_files):
            img_path = os.path.join(folder_path, image_file)
            img = mpimg.imread(img_path)
            plt.subplot(1, 5, idx + 1)
            plt.imshow(img)
            plt.title(f"{folder}\n{image_file[:10]}...", fontsize=8)
            plt.axis('off')
        plt.show()
    ```

---

## Creating TensorFlow Datasets

- Load images into training and validation datasets with 20% validation split:

    ```python
    import tensorflow as tf

    batch_size = 16
    img_height = 160
    img_width = 160

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        raw_img_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        raw_img_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    ```

---

## Model Building and Training

- Build a MobileNetV2 model with data augmentation:

    ```python
    from tensorflow.keras import layers, models

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(train_ds, validation_data=val_ds, epochs=15)
    ```

---

## Fine-Tuning ResNet50

- Load ResNet50 base model, freeze layers, train, then fine-tune last 50 layers:

    ```python
    base_model_resnet = tf.keras.applications.ResNet50(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model_resnet.trainable = False

    model_resnet = models.Sequential([
        base_model_resnet,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model_resnet.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

    history_resnet = model_resnet.fit(train_ds, validation_data=val_ds, epochs=10)

    # Fine-tuning
    base_model_resnet.trainable = True
    for layer in base_model_resnet.layers[:-50]:
        layer.trainable = False

    model_resnet.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

    history_fine = model_resnet.fit(train_ds, validation_data=val_ds, epochs=20, initial_epoch=10)
    ```

---

## Model Evaluation

- Evaluate model performance and plot confusion matrix:

    ```python
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    y_pred_probs = model.predict(val_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_true = np.concatenate([y for x, y in val_ds], axis=0)

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    ```

---

## Saving and Loading Models

- Save your trained model for future use:

    ```python
    model_resnet.save("resnet_finetuned_my_model.keras")
    ```

- Load the model later:

    ```python
    from tensorflow.keras.models import load_model

    model = load_model("resnet_finetuned_my_model.keras")
    ```

---

## Deployment with Gradio

- Simple Gradio interface for image classification:

    ```python
    import gradio as gr
    from PIL import Image
    import numpy as np
    import tensorflow as tf

    class_names = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina',
                   'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']

    def preprocess_image(image):
        image = image.resize((160, 160))
        image = np.array(image)
        image = tf.cast(image, tf.float32)
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image

    def classify_image(image):
        image = preprocess_image(image)
        preds = model(image)
        preds = tf.nn.softmax(preds[0]).numpy()
        return {class_names[i]: float(preds[i]) for i in range(len(class_names))}

    interface = gr.Interface(fn=classify_image,
                             inputs=gr.Image(type="pil"),
                             outputs=gr.Label(num_top_classes=3))
    interface.launch()
    ```

---

## References

- TensorFlow Image Classification Tutorial: https://www.tensorflow.org/tutorials/images/classification  
- MobileNetV2 Model: https://keras.io/api/applications/mobilenet/  
- ResNet50 Model: https://keras.io/api/applications/resnet/  
- Gradio Documentation: https://gradio.app/


# project1-deep-learning-image-classification-with-cnn
