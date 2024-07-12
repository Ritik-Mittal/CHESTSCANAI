# Chest X-Ray Image Classification

This repository contains a deep learning project for classifying chest X-ray images using the NIH Chest X-ray dataset. The model achieves an accuracy of approximately 93%.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)


## Introduction

This project aims to classify various chest conditions from X-ray images using a deep learning model. The NIH Chest X-ray dataset, consisting of over 100,000 images of 14 different conditions, is used for training and validation.
- Atelectasis
- Cardiomegaly
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Consolidation
- Edema
- Emphysema
- Fibrosis
- Pleural Thickening
- Hernia

## Dataset

The dataset used in this project is the NIH Chest X-ray dataset. It consists of around 45 GB of images with labels for 14 different conditions.

## Installation

To run this project, you need to have Python and the following packages installed:

```bash
pip install tensorflow numpy pandas matplotlib pillow scikit-learn
```

## Data Preparation

1. **Download the dataset** from the NIH website or Kaggle.
2. **Extract the images** to a directory and place the metadata files in the same directory.
3. **Run the data preparation script** to organize the images and create the necessary dataframes.

```python
# Example code snippet to organize images
from chestx_ray8 import get_folder_paths, extract_img_paths_dict

data_dir_path = "/path/to/dataset"
subdirectory_paths = get_folder_paths(data_dir_path)

with ThreadPoolExecutor(max_workers=4) as pool:
    img_paths_dict_list = list(pool.map(extract_img_paths_dict, subdirectory_paths))
```

## Model Architecture

The model is built on top of a pre-trained VGG16 convolutional neural network with custom dense layers added for classification.

```python
from keras.applications import VGG16
from keras.layers import Flatten, Dense
from keras.models import Model

vgg16_cnn = VGG16(include_top=False, input_shape=(224, 224, 3))

def create_custom_head():
    flattened = Flatten()(vgg16_cnn.output)
    dense1 = Dense(units=512, activation='relu')(flattened)
    output = Dense(units=15, activation='sigmoid')(dense1)
    model = Model(inputs=vgg16_cnn.input, outputs=output)
    return model

vgg_model = create_custom_head()
vgg_model.summary()
```

## Training

The model is trained using a custom training loop with metrics for accuracy, precision, and recall.

```python
# Example training loop
epochs = 10
for epoch in range(epochs):
    for X_train_mb, Y_train_mb in training_data_generator(df_train):
        y_pred, loss_func_value = training_step(X_train_mb, Y_train_mb)
        # Update metrics and print results
```

## Results

The model achieves approximately 93% accuracy on the validation set. Below are the performance metrics for the first epoch:

```plaintext
Epoch # 1, BCEL value = 0.2126
Epoch # 1, Accuracy = 0.9318, Precision = 0.6915, Recall = 0.3318
```

## Usage

To use the trained model for inference, run the following:

```python
# Example inference code
image_path = "path/to/xray/image.png"
image = Image.open(image_path).convert('RGB')
image = np.array(image)
image = tf.image.resize(image, size=(224, 224))
image = tf.cast(image, dtype='float32')
image = (1/255.0) * image

prediction = vgg_model.predict(np.expand_dims(image, axis=0))
print(prediction)
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

