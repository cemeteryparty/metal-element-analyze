from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, Lambda
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import os

base_path = "/content/drive/MyDrive/midterm/"
# gradeset: https://drive.google.com/drive/folders/1Qo7aQWCeqEmgRQ3pFRDIZWHbiQdIsxhz

def CNN_Model(Shape = (64, 64, 3)):
	inputs = Input(shape = Shape)
	x = Conv2D(16, (3, 3), activation = "relu", padding = "same")(inputs)
	x = MaxPooling2D((2, 2), padding = "same")(x)
	x = Conv2D(8, (3, 3), activation = "relu", padding = "same")(x)
	x = MaxPooling2D((2, 2), padding = "same")(x)
	x = Conv2D(8, (3, 3), activation = "relu", padding = "same")(x)
	x = Flatten()(x)
	x = Dense(units = 128, activation = "relu")(x)
	x = Dense(units = 128, activation = "relu")(x)
	outputs = Dense(units = 3, activation = "sigmoid")(x)
	
	return Model(inputs = inputs, outputs = outputs)

imgcsv = pd.read_csv(base_path + "image.csv", header = None)
x = np.reshape(imgcsv.values, (177, 128, 128, 3))
y = pd.read_csv(base_path + "label.csv", encoding = 'big5')
y = y.drop(columns = "file name").values

x_train, y_train, x_test, y_test = train_test_split(x, y, test_size = 0.2)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)