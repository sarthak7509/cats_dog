import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import pickle

class_names = ["cats","dogs"]
lb = pickle.loads(open("output/lb.pickle", "rb").read())
model = load_model("output/classifier.model")
for i in range(0,7):
	img = cv2.imread("test_img/cat/cat.{}.jpg".format(i))
	image = load_img("test_img/cat/cat.{}.jpg".format(i),target_size=(224,224))
	image = img_to_array(image)/ 255.0
	image = np.expand_dims(image,axis=0)

	label_preds = model.predict(image)

	print(class_names[np.argmax(label_preds[0])])


	img = imutils.resize(img, width=600)
	cv2.imshow("img",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()