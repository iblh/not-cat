# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-f", "--folder", required=True,
	help="path to input image")
args = vars(ap.parse_args())

hits = 0
count = 0
label = args["folder"].split('/')[1]


# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["folder"])))


# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	print("{}: {}".format('loading image', count))
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# classify the input image
	(notCat, cat) = model.predict(image)[0]

	# build the hits
	count +=1
	if label == 'cat':
		hits += 1 if cat > notCat else 0
	else:
		hits += 1 if notCat > cat else 0

# print hits accuracy
hits = "{}: {:.2f}%".format('accuracy', hits / count * 100)
print(hits)