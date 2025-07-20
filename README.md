# Aircraft Type Classification using a Convolutional Neural Network

This project demonstrates the development and training of a Convolutional Neural Network (CNN) to classify images of military aircraft into three distinct categories: **Fighter**, **Bomber**, and **Helicopter**. The model is built using Python with the TensorFlow and Keras libraries.

## Project Goal
The primary objective of this project is to build an accurate image classification model that can distinguish between different types of military aircraft based on their visual silhouettes. This serves as a practical introduction to computer vision and deep learning concepts.

## Key Features
- **3-Class Image Classification**: Classifies aircraft into Fighter, Bomber, and Helicopter categories.
- **Convolutional Neural Network**: Implements a custom CNN architecture to automatically learn distinguishing features from images.
- **Data Augmentation**: Utilizes random transformations (flips, rotations, zooms) to increase the diversity of the training data and improve model generalization.
- **Regularization**: Employs Dropout layers to prevent overfitting.
- **Efficient Training**: Uses Early Stopping to monitor validation loss and halt training when the model's performance on unseen data no longer improves, restoring the best-performing weights.
- **Detailed Evaluation**: Provides a comprehensive analysis of the model's performance, including accuracy, training history plots, a confusion matrix, and a classification report.

## Dataset
- **Source**: The dataset consists of aircraft silhouettes and simple line drawings sourced from various online image searches.
- **Content**: The dataset contains a total of **142 images**, organized into three class folders:
  - `Fighter/`
  - `Bomber/`
  - `Helicopter/`
- **Data Split**: The dataset is automatically split into:
  - **Training Set**: 114 images (~80%)
  - **Validation Set**: 28 images (~20%)

## Technologies Used
- **Python 3.10**
- **TensorFlow & Keras**: For building and training the neural network.
- **Scikit-learn**: For generating the confusion matrix and classification report.
- **Matplotlib & Seaborn**: For data visualization (plots and heatmaps).
- **NumPy**: For numerical operations.

## Setup and Installation
To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [your-repository-url]
    cd [repository-folder-name]
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required libraries:** A `requirements.txt` file is provided for easy installation.
    ```bash
    pip install -r requirements.txt
    ```
    *If you don't have a `requirements.txt` file, create one and add the following lines:*
    ```
    tensorflow
    scikit-learn
    matplotlib
    seaborn
    numpy
    ```

## How to Run
1.  **Prepare the Dataset**: Ensure your dataset is located in the root of the project directory, with images organized into the `Fighter/`, `Bomber/`, and `Helicopter/` subfolders.
2.  **Execute the main script**: Run the `load_data.py` script from your terminal.
    ```bash
    python load_data.py
    ```
3.  **View the Results**: The script will output the training progress to the console. After training, it will display the training history plot and the confusion matrix plot. The final trained model will be saved as `aircraft_classifier_model.keras`.

## Model Architecture
The model is a Sequential CNN with the following structure. The `model.summary()` output provides a detailed overview:

```text
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ data_augmentation (Sequential)       │ (None, 128, 128, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ rescaling (Rescaling)                │ (None, 128, 128, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d (Conv2D)                      │ (None, 128, 128, 32)        │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 64, 64, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 64, 64, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 32, 32, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 32, 32, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 32, 32, 128)         │          73,856 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_2 (MaxPooling2D)       │ (None, 16, 16, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 16, 16, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 32768)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 256)                 │       8,388,864 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ outputs (Dense)                      │ (None, 3)                   │             771 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 8,482,883 (32.36 MB)
 Trainable params: 8,482,883 (32.36 MB)
 Non-trainable params: 0 (0.00 B)
```

## Results and Evaluation
The model achieved a final **validation accuracy of 82.14%** after being trained with early stopping, which restored the best weights from epoch 17.

**Key Insights:**
- **Helicopter**: The model performs exceptionally well, correctly identifying all 12 test samples.
- **Fighter**: Good performance, with 9/13 samples correctly identified. The main confusion is with the Bomber class.
- **Bomber**: This is the most challenging class for the model, likely due to fewer training examples and visual similarity to fighters in silhouette form.

## Future Improvements
- **Expand Dataset**: Collect more images, especially for the Bomber class, to improve performance and reduce class imbalance.
- **Use Real Photographs**: Transition from silhouettes to real-world photographs to create a more challenging and practical model.
- **Apply Transfer Learning**: Implement a pre-trained model (e.g., ResNet50, MobileNetV2) to leverage knowledge from large-scale datasets and potentially achieve higher accuracy on real photos.
- **Hyperparameter Tuning**: Systematically experiment with different learning rates, batch sizes, and network architectures to further optimize performance.
