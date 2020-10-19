from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
from tensorflow.keras.applications import MobileNetV2,VGG16
from tensorflow.keras.layers import AveragePooling2D,Dropout,Flatten,Dense,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os

LB_PATH = os.path.sep.join(["output", "lb.pickle"])
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",default="dataset",help="path to input dataset")
ap.add_argument("-p","--plot",type=str,default="output/plot.jpg",help="path to loss/accuracy plot")
ap.add_argument("-m","--model",default="output/classifier.model",help="path to input dataset")
args = vars(ap.parse_args())

INIT_LR = 1e-4
EPOCHS = 20
BS = 32
print("[+] loading images")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []
for imagePath in imagePaths:
	#taking labels from the dir name
	label = imagePath.split(os.path.sep)[-2]
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)
	#append data and label
	data.append(image)
	labels.append(label)

#converting into data and labels np array
data = np.array(data,dtype="float32") 
labels = np.array(labels)

#binarizing the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#labels 
print("[+] saving label binarizer...")
f = open(LB_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()

#split into test and train data
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

#image generator
aug = ImageDataGenerator(
	rotation_range=20,
	rescale=1/255,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
#initialized pretrained model
ConvBase = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

headModel = ConvBase.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)


model = Model(inputs=ConvBase.input, outputs=headModel)
for layer in baseModel.layers:
	layer.trainable = False
"""if len(lb.classes_) == 2:
	loss = "binary_crossentropy"
elif len(lb.classes_) > 2:
	loss = "categorical_crossentropy"""


print("[+]Compiling Model")
opt = Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])
print("[+]training dataset to the model")

H = model.fit(
	aug.flow(trainX,trainY,batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
	)
print("[+] evaluating model")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

print("[+]Saving the model")
model.save(args["model"],save_format="h5")
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
