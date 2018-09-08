
# coding: utf-8

# # Dog Breed Classifier
# 
# This notebook leverages a pretrained InceptionV3 model (on ImageNet) to prepare a _Dog Breed Classifier_
# It showcases how __Transfer Learning__ can be utilized to prepare high performing models

# In[1]:

# Pandas and Numpy for data structures and util fucntions
import re
#import tqdm
import itertools
import numpy as np
import pandas as pd
from numpy.random import rand
from datetime import datetime
pd.options.display.max_colwidth = 600

# Scikit Imports
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split

# Matplot Imports
import matplotlib.pyplot as plt
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

plt.rcParams.update(params)
get_ipython().run_line_magic('matplotlib', 'inline')

# pandas display data frames as tables
from IPython.display import display, HTML

import warnings
warnings.filterwarnings('ignore')


# In[2]:

import os
import math
import pathlib
import shutil

from keras import regularizers
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.layers import Conv2D,MaxPooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation,Dense,Flatten
from keras.models import Sequential,load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical


# ## Load Dataset

# In[3]:

train_folder = 'train/'
test_folder = 'test/'


# In[4]:

data_labels = pd.read_csv('labels/labels.csv')
data_labels.head()


# ## Check Number of Classes in the Dataset

# In[5]:

target_labels = data_labels['breed']
len(set(target_labels))


# ## Prepare Labels
# Deep Learning models work with one hot encoded outputs or target variables

# In[6]:

labels_ohe_names = pd.get_dummies(target_labels, sparse=True)
labels_ohe = np.asarray(labels_ohe_names)
print(labels_ohe.shape)
print(labels_ohe[:2])


# We add another column to the labels dataset to identify image path

# In[7]:

data_labels['image_path'] = data_labels.apply( lambda row: (train_folder + row["id"] + ".jpg" ), axis=1)
data_labels.head()


# ## Prepare Train-Test Datasets
# We use a 70-30 split to prepare the two dataset

# In[8]:

train_data = np.array([img_to_array(
                            load_img(img, 
                                     target_size=(299, 299))
                       ) for img 
                           in data_labels['image_path'].values.tolist()
                      ]).astype('float32')


# In[9]:

train_data.shape


# In[10]:

x_train, x_test, y_train, y_test = train_test_split(train_data, 
                                                    target_labels, 
                                                    test_size=0.3, 
                                                    stratify=np.array(target_labels), 
                                                    random_state=42)


# In[11]:

x_train.shape, x_test.shape


# Prepare Validation Dataset

# In[12]:

x_train, x_val, y_train, y_val = train_test_split(x_train, 
                                                    y_train, 
                                                    test_size=0.15, 
                                                    stratify=np.array(y_train), 
                                                    random_state=42)


# In[13]:

x_train.shape, x_val.shape


# Prepare target variables for train, test and validation datasets

# In[14]:

y_train_ohe = pd.get_dummies(y_train.reset_index(drop=True)).as_matrix()
y_val_ohe = pd.get_dummies(y_val.reset_index(drop=True)).as_matrix()
y_test_ohe = pd.get_dummies(y_test.reset_index(drop=True)).as_matrix()

y_train_ohe.shape, y_test_ohe.shape, y_val_ohe.shape


# ## Data Augmentation
# 
# Since number of samples per class are not very high, we utilize data augmentation to prepare different variations of different samples available. We do this using the ```ImageDataGenerator utility``` from ```keras```

# In[15]:

BATCH_SIZE = 32


# In[16]:

# Create train generator.
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=30, 
                                   width_shift_range=0.2,
                                   height_shift_range=0.2, 
                                   horizontal_flip = 'true')
train_generator = train_datagen.flow(x_train, y_train_ohe, shuffle=False, batch_size=BATCH_SIZE, seed=1)


# In[17]:

# Create validation generator
val_datagen = ImageDataGenerator(rescale = 1./255)
val_generator = train_datagen.flow(x_val, y_val_ohe, shuffle=False, batch_size=BATCH_SIZE, seed=1)


# ## Prepare Deep Learning Classifier
# 
# * Load InceptionV3 pretrained on ImageNet without its top/classification layer
# * Add additional custom layers on top of InceptionV3 to prepare custom classifier

# In[18]:

# Get the InceptionV3 model so we can do transfer learning
base_inception = InceptionV3(weights='imagenet', include_top = False, input_shape=(299, 299, 3))


# In[19]:

# Add a global spatial average pooling layer
out = base_inception.output
out = GlobalAveragePooling2D()(out)
out = Dense(512, activation='relu')(out)
out = Dense(512, activation='relu')(out)
total_classes = y_train_ohe.shape[1]
predictions = Dense(total_classes, activation='softmax')(out)


# * Stack the two models (InceptionV3 and custom layers) on top of each other 
# * Compile the model and view its summary

# In[20]:

model = Model(inputs=base_inception.input, outputs=predictions)

# only if we want to freeze layers
for layer in base_inception.layers:
    layer.trainable = False
    
# Compile 
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy']) 

model.summary()


# ## Model Training
# We train the model with a Batch Size of 32 for just 15 Epochs.
# 
# The model utilizes the power of transfer learning to achieve a validation accuracy of about __81%__ !

# In[21]:

# Train the model
batch_size = BATCH_SIZE
train_steps_per_epoch = x_train.shape[0] // batch_size
val_steps_per_epoch = x_val.shape[0] // batch_size

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_steps_per_epoch,
                              validation_data=val_generator,
                              validation_steps=val_steps_per_epoch,
                              epochs=15,
                              verbose=1)


# Save the Model

# In[22]:

model.save('dog_breed.hdf5')


# ## Visualize Model Performance

# In[35]:

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
t = f.suptitle('Deep Neural Net Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epochs = list(range(1,16))
ax1.plot(epochs, history.history['acc'], label='Train Accuracy')
ax1.plot(epochs, history.history['val_acc'], label='Validation Accuracy')
ax1.set_xticks(epochs)
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epochs, history.history['loss'], label='Train Loss')
ax2.plot(epochs, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(epochs)
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")


# ## Test Model Performance
# 
# Step 1 is to prepare the training dataset. Since we scaled training data, test data should also be scaled in a similar manner. 
# 
# _Note: Deep Learning models are very sensitive to scaling._

# In[67]:

# scaling test features
x_test /= 255.


# In[69]:

test_predictions = model.predict(x_test)
test_predictions


# In[70]:

predictions = pd.DataFrame(test_predictions, columns=labels_ohe_names.columns)
predictions.head()


# In[71]:

test_labels = list(y_test)
predictions = list(predictions.idxmax(axis=1))
predictions[:10]


# ## Analyze Test Performance

# In[72]:

import model_evaluation_utils as meu


# In[73]:

meu.get_metrics(true_labels=test_labels, 
                predicted_labels=predictions)


# In[74]:

meu.display_classification_report(true_labels=test_labels, 
                                  predicted_labels=predictions, 
                                  classes=list(labels_ohe_names.columns))


# In[75]:

meu.display_confusion_matrix_pretty(true_labels=test_labels, 
                                    predicted_labels=predictions, 
                                    classes=list(labels_ohe_names.columns))


# The model achieves a test accuracy of approximately __86%__

# ## Visualize Model Performance
# Visualize model performance with actual images, labels and prediction confidence

# In[112]:

grid_width = 5
grid_height = 5
f, ax = plt.subplots(grid_width, grid_height)
f.set_size_inches(15, 15)
batch_size = 25
dataset = x_test

label_dict = dict(enumerate(labels_ohe_names.columns.values))
model_input_shape = (1,)+model.get_input_shape_at(0)[1:]
random_batch_indx = np.random.permutation(np.arange(0,len(dataset)))[:batch_size]

img_idx = 0
for i in range(0, grid_width):
    for j in range(0, grid_height):
        actual_label = np.array(y_test)[random_batch_indx[img_idx]]
        prediction = model.predict(dataset[random_batch_indx[img_idx]].reshape(model_input_shape))[0]
        label_idx = np.argmax(prediction)
        predicted_label = label_dict.get(label_idx)
        conf = round(prediction[label_idx], 2)
        ax[i][j].axis('off')
        ax[i][j].set_title('Actual: '+actual_label+'\nPred: '+predicted_label + '\nConf: ' +str(conf))
        ax[i][j].imshow(dataset[random_batch_indx[img_idx]])
        img_idx += 1

plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.5, hspace=0.55)    

