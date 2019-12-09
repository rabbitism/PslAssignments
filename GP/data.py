import os

import cv2
import numpy

from sklearn.model_selection import train_test_split


def load_data_from_files(path: str):
	
	filenames = os.listdir(path)
	if (filenames is None or len(filenames) == 0):
		raise FileNotFoundError()
	image = cv2.imread(os.path.sep.join([path, filenames[0]]))
	data = numpy.empty((len(filenames), image.size), dtype=int)
	for i, filename in enumerate(filenames):
		image = cv2.imread(os.path.sep.join([path, filename]))
		image.resize(image.size)
		data[i,...]=image
	return data

def load_data():
	
	benign_data = load_data_from_files('./benign_resized')
	malignant_data = load_data_from_files('./malignant_resized')

	benign_label = numpy.zeros((benign_data.shape[0]), dtype=int)
	malignant_label = numpy.ones((malignant_data.shape[0]), dtype=int)

	x = numpy.concatenate([benign_data, malignant_data], axis=0)
	y = numpy.concatenate([benign_label, malignant_label])
	return x, y
	
	
if __name__ == "__main__":
	#data = load_data_from_files("./benign_resized")
	#print(data.shape)
	x, y = load_data()
	print(x)
	print(y)