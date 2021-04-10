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
file_list

img_list = []
size = (128, 128)

for i in range(len(file_list)):
	im = Image.open(file_list[i])
	im = im.resize(size, Image.BILINEAR)
	imarray = np.array(im)
	img_list.append(imarray)

x = np.asarray(img_list)


# Label
y = pd.read_csv(base_path + "label.csv", encoding = 'big5')
# y = y.drop(columns = 'file name')

xx = np.reshape(x, (177, 128 * 128 *3))
np.savetxt(base_path + "image.csv", xx, delimiter = ",")

# imgcsv = pd.read_csv(base_path + "image.csv", header = None)
# x = np.reshape(imgcsv.values, (177, 128, 128, 3))