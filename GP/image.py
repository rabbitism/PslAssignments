import numpy as np
from cv2 import cv2 as cv2
import os

def resize_images(source_dir: str, target_dir: str, width: int, height: int):
	print("Converting image from " + source_dir + " to " + target_dir)
	filenames = os.listdir(source_dir)
	for filename in filenames:
		image = cv2.imread(os.path.sep.join([source_dir, filename]))
		print(image.shape)
		image_resized = cv2.resize(image, (height, width), interpolation=cv2.INTER_NEAREST)
		cv2.imwrite(os.path.sep.join([target_dir, filename]), image_resized)
	
if __name__ == "__main__":
	resize_images("./benign", "./benign_resized", 450, 600)
	resize_images("./malignant", "./malignant_resized", 450, 600)