# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from glob import glob
from random import shuffle
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution1D, concatenate, SpatialDropout1D, GlobalMaxPool1D, GlobalAvgPool1D, Embedding, \
    Conv2D, SeparableConv1D, Add, BatchNormalization, Activation, GlobalAveragePooling2D, LeakyReLU, Flatten
from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, \
    Lambda, Multiply, LSTM, Bidirectional, PReLU, MaxPooling1D
from keras.layers.pooling import _GlobalPooling1D
from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy
from keras.models import Model
from keras.applications.nasnet import NASNetMobile, NASNetLarge, preprocess_input
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from imgaug import augmenters as iaa
import imgaug as ia

train_path = '/mnt/d/project/AI.Health.kaggle/train/'
test_path = '/mnt/d/project/AI.Health.kaggle/test/'
train_label_path = '/mnt/d/project/AI.Health.kaggle/train_labels.csv'
df_train = pd.read_csv(train_label_path)
id_label_map = {k: v for k, v in zip(df_train.id.values, df_train.label.values)}
print(df_train.head())


def get_id_from_file_path(file_path):
    return file_path.split(os.path.sep)[-1].replace('.tif', '')


# labeled_files = glob(train_path+'*.tif')[:10]
# test_files = glob(test_path+'*.tif')[:10]


labeled_files = ['/mnt/d/project/AI.Health.kaggle/train/00001b2b5609af42ab0ab276dd4cd41c3e7745b5.tif',
                 '/mnt/d/project/AI.Health.kaggle/train/000020de2aa6193f4c160e398a8edea95b1da598.tif',
                 '/mnt/d/project/AI.Health.kaggle/train/00004aab08381d25d315384d646f5ce413ea24b1.tif',
                 '/mnt/d/project/AI.Health.kaggle/train/0000d563d5cfafc4e68acb7c9829258a298d9b6a.tif',
                 '/mnt/d/project/AI.Health.kaggle/train/0000da768d06b879e5754c43e2298ce48726f722.tif',
                 '/mnt/d/project/AI.Health.kaggle/train/0000f8a4da4c286eee5cf1b0d2ab82f979989f7b.tif',
                 '/mnt/d/project/AI.Health.kaggle/train/00010f78ea8f878117500c445a658e5857f4e304.tif',
                 '/mnt/d/project/AI.Health.kaggle/train/00011545a495817817c6943583b294c900a137b8.tif',
                 '/mnt/d/project/AI.Health.kaggle/train/000126ec42770c7568204e2f6e07eb9a07d5e121.tif',
                 '/mnt/d/project/AI.Health.kaggle/train/00014e39b5df5f80df56f18a0a049d1cc6de430a.tif']
test_files = ['/mnt/d/project/AI.Health.kaggle/test/00006537328c33e284c973d7b39d340809f7271b.tif',
              '/mnt/d/project/AI.Health.kaggle/test/0000ec92553fda4ce39889f9226ace43cae3364e.tif',
              '/mnt/d/project/AI.Health.kaggle/test/00024a6dee61f12f7856b0fc6be20bc7a48ba3d2.tif',
              '/mnt/d/project/AI.Health.kaggle/test/000253dfaa0be9d0d100283b22284ab2f6b643f6.tif',
              '/mnt/d/project/AI.Health.kaggle/test/000270442cc15af719583a8172c87cd2bd9c7746.tif',
              '/mnt/d/project/AI.Health.kaggle/test/000309e669fa3b18fb0ed6a253a2850cce751a95.tif',
              '/mnt/d/project/AI.Health.kaggle/test/000360e0d8358db520b5c7564ac70c5706a0beb0.tif',
              '/mnt/d/project/AI.Health.kaggle/test/00040095a4a671280aeb66cb0c9231e6216633b5.tif',
              '/mnt/d/project/AI.Health.kaggle/test/000698b7df308d75ec9559ef473a588c513a68aa.tif',
              '/mnt/d/project/AI.Health.kaggle/test/0006e1af5670323331d09880924381d67d79eda0.tif']
print(labeled_files)
print(test_files)
print("labeled_files size :", len(labeled_files))
print("test_files size :", len(test_files))

train, val = train_test_split(labeled_files, test_size=0.1, random_state=101010)


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def get_seq():
    def sometimes(aug): return iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},  # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # translate by -20 to +20 percent (per axis)
                rotate=(-10, 10),  # rotate by -45 to +45 degrees
                shear=(-5, 5),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                # convert images into their superpixel representation
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.0)),  # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(3, 5)),  # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 5)),  # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),  # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5),  # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.05), per_channel=0.5),  # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),
                ]),
                iaa.Invert(0.01, per_channel=True),  # invert color channels
                iaa.Add((-2, 2), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-1, 1)),  # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.9, 1.1), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-1, 0),
                        first=iaa.Multiply((0.9, 1.1), per_channel=True),
                        second=iaa.ContrastNormalization((0.9, 1.1))
                    )
                ]),
                # move pixels locally around (with random strengths)
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),  # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
                random_order=True
            )
        ],
        random_order=True
    )
    return seq


def data_gen(list_files, id_label_map, batch_size, augment=False):
    seq = get_seq()
    while True:
        shuffle(list_files)
        for batch in chunker(list_files, batch_size):
            X = [cv2.imread(x) for x in batch]
            Y = [id_label_map[get_id_from_file_path(x)] for x in batch]
            if augment:
                X = seq.augment_images(X)
            X = [preprocess_input(x) for x in X]

            yield np.array(X), np.array(Y)


def get_model_classif_nasnet():
    inputs = Input((96, 96, 3))
    base_model = NASNetMobile(include_top=False, input_shape=(96, 96, 3))  # , weights=None
    x = base_model(inputs)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(1, activation="sigmoid", name="3_")(out)
    model = Model(inputs, out)
    model.compile(optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=['acc'])
    model.summary()

    return model


model = get_model_classif_nasnet()

batch_size = 1
h5_path = "model.h5"
checkpoint32 = ModelCheckpoint(h5_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint64 = ModelCheckpoint(h5_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history = model.fit_generator(
    data_gen(train, id_label_map, batch_size, augment=True),
    validation_data=data_gen(val, id_label_map, batch_size),
    epochs=2, verbose=1,
    callbacks=[checkpoint32],
    steps_per_epoch=9,#len(train) // batch_size,
    validation_steps=1)#len(val) // batch_size)
# batch_size = 5
# history = model.fit_generator(
#     data_gen(train, id_label_map, batch_size, augment=True),
#     validation_data=data_gen(val, id_label_map, batch_size),
#     epochs=6, verbose=1,
#     callbacks=[checkpoint64],
#     steps_per_epoch=len(train) // batch_size,
#     validation_steps=len(val) // batch_size)

model.load_weights(h5_path)

preds = []
ids = []

for batch in chunker(test_files, batch_size):
    X = [preprocess_input(cv2.imread(x)) for x in batch]
    ids_batch = [get_id_from_file_path(x) for x in batch]
    X = np.array(X)
    preds_batch = ((model.predict(X).ravel()
                    * model.predict(X[:, ::-1, :, :]).ravel()
                    * model.predict(X[:, ::-1, ::-1, :]).ravel()
                    * model.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()
    preds += preds_batch
    ids += ids_batch


df = pd.DataFrame({'id': ids, 'label': preds})
df.to_csv("baseline_nasnet.csv", index=False)
print(df.head())
