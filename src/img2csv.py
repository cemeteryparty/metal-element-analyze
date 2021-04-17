from PIL import Image
import numpy as np
import pandas as pd
import os

base_path = "/content/drive/MyDrive/midterm/"

file_list = []
for root, dirs, files in os.walk(base_path + "photos"):
	for file in files:
		filename = os.path.join(root, file)
		file_list.append(filename)
file_list.sort()
#print(file_list)

img_list = []
size = (128, 128)
for i in range(len(file_list)):
	im = Image.open(file_list[i])
	im = im.resize(size, Image.BILINEAR)
	imarray = np.array(im)
	img_list.append(imarray)
x = np.asarray(img_list)


# Label
# y = pd.read_csv(base_path + "label.csv", encoding = 'big5')
# y = y.drop(columns = 'file name')

xx = np.reshape(x, (177, 128 * 128 *3))
np.savetxt(base_path + "image.csv", xx, delimiter = ",")

#########################################################
from sklearn.model_selection import train_test_split
import numpy as np

imgcsv = pd.read_csv(base_path + "image.csv", header = None)
x = np.reshape(imgcsv.values, (177, 128, 128, 3))
y = pd.read_csv(base_path + "label.csv", encoding = 'big5')
y = y.drop(columns = "file name").values

x_train, x_part, y_train, y_part = train_test_split(x, y, test_size = 0.2)
for sample in x_part:
	plt.figure(figsize = (20, 4))
	plt.subplot(141)
	plt.imshow(sample.astype(np.uint8))
	plt.subplot(142)
	plt.imshow(np.rot90(sample, k = 1).astype(np.uint8))
	plt.subplot(143)
	plt.imshow(np.flip(sample, axis = 1).astype(np.uint8))
	plt.subplot(144)
	plt.imshow(np.flip(sample, axis = 0).astype(np.uint8))
	plt.show()