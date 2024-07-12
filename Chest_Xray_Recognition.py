
!pip install --upgrade tensorflow-io

import os
import itertools
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as mplt
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

img_array = mplt.imread("/kaggle/input/data/images_001/images/00000001_000.png")

img_array

mplt.imshow(img_array)

data_dir_path = "/kaggle/input/data"

def get_folder_paths(data_dir_path):
    subdirectory_paths = []

    for root, dirs, files in os.walk(data_dir_path):
        for directory in dirs:
            img_dir_apth = os.path.join(root, directory)
            if os.path.basename(img_dir_apth) == 'images':
                subdirectory_paths.append(img_dir_apth)

    subdirectory_paths.sort()

    return subdirectory_paths

subdirectory_paths = get_folder_paths(data_dir_path)

subdirectory_paths

len(subdirectory_paths)

def extract_img_paths_dict(dir_path):
    img_paths_dict = {}

    directory_name = os.path.dirname(dir_path)
    dir_name = os.path.basename(directory_name)

    img_paths_list = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            img_path = os.path.join(root, file)
            img_paths_list.append(img_path)

    img_paths_list.sort()
    img_paths_dict[dir_name] = img_paths_list

    return img_paths_dict

with ThreadPoolExecutor(max_workers=4) as pool:
    img_paths_dict_list = list(pool.map(extract_img_paths_dict, subdirectory_paths))

len(img_paths_dict_list)

imgs_paths_dict = {}
for img_paths_dict_item in img_paths_dict_list:
    imgs_paths_dict.update(img_paths_dict_item)

len(imgs_paths_dict)

len(imgs_paths_dict['images_001'])

sum(len(lst) for lst in imgs_paths_dict.values())

imgs_paths_dict

img_paths_df = pd.DataFrame(columns=['Image_Paths'])

for key, paths in imgs_paths_dict.items():

    path_data = pd.DataFrame({'Image_Paths': paths})

    img_paths_df = pd.concat([img_paths_df, path_data])

img_paths_df.reset_index(drop=True, inplace=True)

img_paths_df

"""# Custom Data Generator"""

bbox = pd.read_csv("/kaggle/input/data/BBox_List_2017.csv")
bbox = bbox.rename(columns = {'Bbox [x' : 'X',
                  'y' : 'Y',
                  'w' : 'Width',
                  'h]' : 'Height'})
bbox = bbox.iloc[:,:6]

bbox

data_entry_df = pd.read_csv("/kaggle/input/data/Data_Entry_2017.csv")

data_entry_df = data_entry_df.rename(columns={'OriginalImage[Width' : 'OriginalImageWidth',
                                   'Height]' : 'OriginalImageHeight',
                                   'OriginalImagePixelSpacing[x' : 'OriginalImagePixelSpacingX',
                                   'y]' : 'OriginalImagePixelSpacingY'})
data_entry_df = data_entry_df.drop('Unnamed: 11', axis=1)

data_entry_df

unique_labels = list(set(itertools.chain.from_iterable(data_entry_df['Finding Labels'].apply(lambda x : x.split('|')))))

unique_labels

one_hot_labels = pd.DataFrame(0.0, index=np.arange(len(img_paths_df)), columns=unique_labels)

for index, rows in data_entry_df.iterrows():
    labels = rows['Finding Labels'].split('|')
    for label in labels:
        one_hot_labels.iloc[index][label] = 1.0

data = pd.concat([img_paths_df['Image_Paths'], data_entry_df.iloc[:, :2], one_hot_labels], axis=1)

data

df_test = data[data['Image Index'].isin(bbox['Image Index'])]

df_test

for idx in df_test.index:
    data = data.drop(idx, axis=0)

data.reset_index(drop=True, inplace=True)
data



df_train, df_val = train_test_split(data, test_size=0.2, random_state=42)

def training_data_generator(df_train, mb_size=288):
    batches = [df_train[i:i+mb_size] for i in range(0, len(df_train), mb_size)]

    for batch in batches:
        images = []
        ohe_values_arrays = []

        for img_path in batch['Image_Paths']:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)

            image = tf.image.resize(image, size=(224, 224))
            image = tf.cast(image, dtype='float32')

            images.append(image)

        images = np.array(images)

        one_hot_columns = batch.columns[3:]

        for path in batch['Image_Paths']:
            ohe_values = batch[batch['Image_Paths'] == path][one_hot_columns].iloc[0]
            ohe_values_arrays.append(ohe_values.values)

        one_hot_arrays = np.array(ohe_values_arrays)

        yield (1/255.0) * images, one_hot_arrays

X_train_mb, Y_train_mb = training_data_generator(df_train, 8).__next__()

X_train_mb.shape

Y_train_mb

"""['No Finding', 'Consolidation', 'Mass', 'Nodule', 'Atelectasis', 'Effusion', 'Fibrosis', 'Pneumonia', 'Edema', 'Cardiomegaly', 'Hernia', 'Infiltration', 'Pneumothorax', 'Pleural_Thickening', 'Emphysema']"""

def val_data_generator(df_val, mb_size=108):
    batches = [df_val[i:i+mb_size] for i in range(0, len(df_val), mb_size)]

    for batch in batches:
        images = []
        ohe_values_arrays = []

        for img_path in batch['Image_Paths']:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)

            image = tf.image.resize(image, size=(224, 224))
            image = tf.cast(image, dtype='float32')

            images.append(image)

        images = np.array(images)

        one_hot_columns = batch.columns[3:]

        for path in batch['Image_Paths']:
            ohe_values = batch[batch['Image_Paths'] == path][one_hot_columns].iloc[0]
            ohe_values_arrays.append(ohe_values.values)

        one_hot_arrays = np.array(ohe_values_arrays)

        yield (1/255.0) * images, one_hot_arrays

import tensorflow as tf
from keras.applications import VGG16
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.losses import BinaryCrossentropy
from keras.metrics import Accuracy, Precision, Recall
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

vgg16_cnn = VGG16(include_top=False, input_shape=(224,224,3,))

vgg16_cnn.summary()

for layer in vgg16_cnn.layers:
    layer.trainable = False

for layer in vgg16_cnn.layers[-2:]:
    layer.trainable = True

def create_custom_head():

    flattened = Flatten()(vgg16_cnn.output)
    dense1 = Dense(units=512, activation='relu')(flattened)
    output = Dense(units=15, activation='sigmoid')(dense1)

    model = Model(inputs=vgg16_cnn.input, outputs=output)

    return model

vgg_model = create_custom_head()

vgg_model.summary()

epochs = 10
optimizer = Adam()
binary_crossentropy = BinaryCrossentropy()
train_accuracy_metric = Accuracy()
train_precision_metric = Precision()
train_recall_metric = Recall()

val_accuracy_metric = Accuracy()
val_precision_metric = Precision()
val_recall_metric = Recall()
threshold = 0.5

@tf.function
def training_step(X_train_mb, Y_train_mb):
    with tf.GradientTape() as Tape:
        y_pred = vgg_model(X_train_mb, training=True)
        loss_func_value = binary_crossentropy(Y_train_mb, y_pred)

    grads = Tape.gradient(loss_func_value, vgg_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, vgg_model.trainable_weights))

    return y_pred, loss_func_value



for epoch in range(epochs):
    for X_train_mb, Y_train_mb in training_data_generator(df_train):

        y_pred, loss_func_value = training_step(X_train_mb, Y_train_mb)
        y_pred_binary = tf.where(y_pred >= threshold, 1.0, 0.0)

        train_accuracy_metric.update_state(y_true=Y_train_mb, y_pred=y_pred_binary)
        train_precision_metric.update_state(y_true=Y_train_mb, y_pred=y_pred_binary)
        train_recall_metric.update_state(y_true=Y_train_mb, y_pred=y_pred_binary)

    training_acc = train_accuracy_metric.result()
    training_pre = train_precision_metric.result()
    training_rec = train_recall_metric.result()

    print("Epoch # {}, BCEL value = {}".format((epoch+1), loss_func_value))
    print("Epoch # {}, Accuracy = {}, Precision = {}, Recall = {}".format((epoch+1), training_acc, training_pre, training_rec))

    train_accuracy_metric.reset_state()
    train_precision_metric.reset_state()
    train_recall_metric.reset_state()

    for X_val_mb, Y_val_mb in val_data_generator(df_val):
        val_y_pred = vgg_model(X_val_mb)
        val_y_pred_binary = tf.where(val_y_pred >= threshold, 1.0, 0.0)
        val_loss_func_value = binary_crossentropy(Y_val_mb, val_y_pred)

        val_accuracy_metric.update_state(y_true=Y_val_mb, y_pred=val_y_pred_binary)
        val_precision_metric.update_state(y_true=Y_val_mb, y_pred=val_y_pred_binary)
        val_recall_metric.update_state(y_true=Y_val_mb, y_pred=val_y_pred_binary)

    val_acc = val_accuracy_metric.result()
    val_pre = val_precision_metric.result()
    val_rec = val_recall_metric.result()

    print("Epoch # {}, val_BCEL value = {}".format((epoch+1), val_loss_func_value))
    print("Epoch # {}, val_Accuracy = {}, val_Precision = {}, val_Recall = {}\n\n".format((epoch+1), val_acc, val_pre, val_rec))

    val_accuracy_metric.reset_state()
    val_precision_metric.reset_state()
    val_recall_metric.reset_state()

"""    Epoch # 1, BCEL value = 0.21260681748390198
    Epoch # 1, Accuracy = 0.9317789077758789, Precision = 0.6915146112442017, Recall = 0.3318246304988861
    Epoch # 1, val_BCEL value = 0.23309804499149323
    Epoch # 1, val_Accuracy = 0.9319309592247009, val_Precision = 0.6889708042144775, val_Recall = 0.34654876589775085
    
    Epoch # 1, BCEL value = 0.21001923084259033
    Epoch # 1, Accuracy = 0.9287501573562622, Precision = 0.6526876091957092, Recall = 0.315387099981308
    Epoch # 1, val_BCEL value = 0.22609342634677887
    Epoch # 1, val_Accuracy = 0.9318230748176575, val_Precision = 0.6914955377578735, val_Recall = 0.3409155607223511
"""