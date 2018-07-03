import numpy as numpy
from skimage import color, exposure, transform
from skimage import io 
import os
import glob
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
K.set_image_data_format('channels_first')

NUM_CLASSES = 120
IMG_SIZE = 256

def preprocess_img(img):
	hsv = color.rgb2hsv(img)
	hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
	img = color.hsv2rgb(hsv)

	min_edge = min(img.shape[:-1])
	center = img.shape[0] // 2, img.shape[1] // 2
	img = img[center[0] - min_edge//2 : center[0] + min_edge//2, center[1] - min_edge//2 : center[1] + min_edge//2]

	img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

	img = np.rollaxis(img, -1)

	return img

def load_labels(path):
	label_df = pd.read_csv(path, usecols=['id', 'label'])
	return Series(label_df.label.values,index=df.id).to_dict()

def get_class(img_path, label_dictionary):
	file_id = img_path.split('.')[0]
	return label_dictionary[file_id]

root_dir = ''
label_dictionary = load_labels(root_dir + "labels")

train_labels = []
test_labels = []

train_images = []
test_images = []
eval_images = []

paths = glob.glob(os.path.join(root_dir + "train", '*.jpg'))
eval_paths = glob.glob(os.path.join(root_dir + "test", '*.jpg'))
np.shuffle(paths)
np.shuffle(eval_paths)
split = int(.8 * len(paths))
train_paths = paths[:split]
test_paths = paths[split:]

for img_path in train_paths:
	img = preprocess_img(io.imread(img_path))
	label = get_class(img_path, label_dictionary)
	train_images.append(img)
	train_labels.append(label)

for img_path in test_paths:
	img = preprocess_img(io.imread(img_path))
	label = get_class(img_path, label_dictionary)
	test_images.append(img)
	test_labels.append(label)

for img_path in eval_paths:
	img = preprocess_img(io.imread(img_path))
	eval_images.append(img)

X_train = np.array(train_imgs, dtype='float32')
Y_train = np.eye(NUM_CLASSES, dtype='uint8')[train_labels]


def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(3, IMG_SIZE, IMG_SIZE),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

model = cnn_model()
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

batch_size = 32
epochs = 30

model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[LearningRateScheduler(lr_schedule),
                     ModelCheckpoint('image_classification_model.h5', save_best_only=True)])

X_test = test_images
Y_test = test_labels

y_pred = model.predict_classes(X_test)
acc = np.sum(y_pred == y_test) / np.size(y_pred)
print("Test accuracy = {}".format(acc))
