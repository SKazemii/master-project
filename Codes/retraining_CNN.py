

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
import os
import sys
import timeit
from pathlib import Path as Pathlb

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# keras imports
import tensorflow as tf

from PIL import Image


from sklearn import preprocessing as pre
from sklearn.model_selection import train_test_split

sns.set()

    
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    
from MLPackage import Features as feat
from MLPackage import Utilities as util
from MLPackage import config as cfg

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"






project_dir = cfg.configs["paths"]["project_dir"]


fig_dir = cfg.configs["paths"]["fig_dir"]
tbl_dir = cfg.configs["paths"]["tbl_dir"]
results_dir = cfg.configs["paths"]["results_dir"]
temp_dir = cfg.configs["paths"]["temp_dir"]
log_path = cfg.configs["paths"]["log_path"]



def create_logger(level):
    loggerName = "retraining-pretrained-CNN"
    Pathlb(log_path).mkdir(parents=True, exist_ok=True)
    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    logger = logging.getLogger(loggerName)
    logger.setLevel(level)
    formatter_colored = logging.Formatter(blue + '[%(asctime)s]-' + yellow + '[%(name)s @%(lineno)d]' + reset + blue + '-[%(levelname)s]' + reset + bold_red + '\t\t%(message)s' + reset, datefmt='%m/%d/%Y %I:%M:%S %p ')
    formatter = logging.Formatter('[%(asctime)s]-[%(name)s @%(lineno)d]-[%(levelname)s]\t\t%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p ')
    file_handler = logging.FileHandler( os.path.join(log_path, loggerName + f'_loger{os.getpid()}.log'), mode = 'w')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    stream_handler.setFormatter(formatter_colored)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
logger = create_logger(logging.DEBUG)


logger.info("Importing libraries....")

configs = cfg.configs

if configs['CNN']["image_feature"]=="tile":
    configs['CNN']["image_size"] = (120, 80, 3)


# configs['CNN']["image_size"] = (32, 32, 3)
# configs['CNN']["class_numbers"] = 10




# # ##################################################################
# #                phase 1: Reading image
# # ##################################################################
# logger.info("Reading stepscan dataset....")
# with h5py.File(cfg.configs["paths"]["stepscan_dataset.h5"], "r") as hdf:
#     barefoots = hdf.get("/barefoot/data")[:]
#     metadata = hdf.get("/barefoot/metadata")[:]

# data = barefoots.transpose(0,2,3,1)

# np.save(cfg.configs["paths"]["stepscan_data.npy"], data)
# np.save(cfg.configs["paths"]["stepscan_meta.npy"], metadata)





# # ##################################################################
# #                phase 2: extracting image features
# # ##################################################################
# metadata = np.load(cfg.configs["paths"]["stepscan_meta.npy"])
# data = np.load(cfg.configs["paths"]["stepscan_data.npy"])

# logger.info(f"barefoots.shape: {data.shape}")
# logger.info(f"metadata.shape: {metadata.shape}")


# plt.imshow(data[1,:,:,:].sum(axis=2))
# plt.show()




# ## Extracting Image Features
# features = list()
# labels = list()

# for label, sample in zip(metadata, data):
#     try:
#         B = sample.sum(axis=1).sum(axis=0)
#         A = np.trim_zeros(B)

#         aa = np.where(B == A[0])
#         bb = np.where(B == A[-1])

#         if aa[0][0]<bb[0][0]:
#             features.append(feat.prefeatures(sample[10:70, 10:50, aa[0][0]:bb[0][0]]))
#             labels.append(label)
#         else:
#             print(aa[0][0],bb[0][0])
#             k=sample
#             l=label
    
#     except Exception as e:
#         print(e)
#         continue
    

# logger.info(f"len prefeatures: {len(features)}")
# logger.info(f"prefeatures.shape: {features[0].shape}")
# logger.info(f"labels.shape: {labels[0].shape}")

# np.save(cfg.configs["paths"]["stepscan_image_feature.npy"], features)
# np.save(cfg.configs["paths"]["stepscan_image_label.npy"], labels)


# # ##################################################################
# #                phase 3: Making Base Model
# # ##################################################################
try:
    logger.info(f"Loading { cfg.configs['CNN']['base_model'] } model...")
    base_model = eval("tf.keras.applications." + cfg.configs["CNN"]["base_model"] + "(weights=cfg.configs['CNN']['weights'], include_top=cfg.configs['CNN']['include_top'])")
    logger.info("Successfully loaded base model and model...")

except Exception as e: 
    base_model = None
    logger.error("The base model could NOT be loaded correctly!!!")
    print(e)


base_model.trainable = False

CNN_name = cfg.configs["CNN"]["base_model"].split(".")[0]

input = tf.keras.layers.Input(shape=cfg.configs["CNN"]["image_size"], dtype = tf.float64, name="original_img")
x = tf.cast(input, tf.float32)
x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
x = tf.keras.layers.RandomRotation(0.2)(x)
x = tf.keras.layers.RandomZoom(0.1)(x)
x = eval("tf.keras.applications." + CNN_name + ".preprocess_input(x)")
x = base_model(x)
x = tf.keras.layers.GlobalMaxPool2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(128,  activation='relu', name="last_dense")(x) # kernel_regularizer=tf.keras.regularizers.l2(0.0001),
x = tf.keras.layers.Dropout(0.2)(x)
output = tf.keras.layers.Dense(cfg.configs['CNN']['class_numbers'], name="prediction")(x) # activation='softmax',

## The CNN Model
model = tf.keras.models.Model(inputs=input, outputs=output, name=cfg.configs['CNN']['base_model'])

# Freeze the layers 
for layer in model.layers[-2:]:
    layer.trainable = True


# for i,layer in enumerate(model.layers):
#     print(i,layer.name,layer.trainable)

# model.summary() 
# tf.keras.utils.plot_model(model, to_file=cfg.configs['CNN']['base_model'] + ".png", show_shapes=True)
# plt.show()





# # ##################################################################
# #                phase 4: processing labels
# # ##################################################################
metadata = np.load(configs["paths"]["stepscan_image_label.npy"])
logger.info("metadata shape: {}".format(metadata.shape))

features = np.load(configs["paths"]["stepscan_image_feature.npy"])
logger.info("features shape: {}".format(features.shape))



indices = metadata[:,0]
le = pre.LabelEncoder()
le.fit(indices)

logger.info(f"Number of subjects: {len(np.unique(indices))}")

labels = le.transform(indices)

# labels = tf.keras.utils.to_categorical(labels, num_classes=len(np.unique(indices)))





# # ##################################################################
# #                phase 5: Loading Image features
# # ##################################################################



# #CD, PTI, Tmax, Tmin, P50, P60, P70, P80, P90, P100
logger.info("batch_size: {}".format(configs["CNN"]["batch_size"]))

maxvalues = [np.max(features[...,ind]) for ind in range(len(cfg.image_feature_name))]

for i in range(len(cfg.image_feature_name)):
    features[..., i] = features[..., i]/maxvalues[i]


if configs['CNN']["image_feature"]=="tile":
    images = util.tile(features)

else:
    image_feature_name = dict(zip(cfg.image_feature_name, range(len(cfg.image_feature_name))))
    ind = image_feature_name[configs['CNN']["image_feature"]]
    
    images = features[...,ind]
    images = images[...,tf.newaxis]
    images = np.concatenate((images, images, images), axis=-1)


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# # Normalize pixel values to be between 0 and 1
# train_images, test_images = train_images / 255.0, test_images / 255.0

# images = train_images 
# labels = train_labels
# # labels = tf.keras.utils.to_categorical(train_labels,10)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
logger.info(f"images: {images.shape}")
logger.info(f"labels: {labels.shape}")

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.15, random_state=42, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)




AUTOTUNE = tf.data.AUTOTUNE


train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000)
train_ds = train_ds.batch(configs['CNN']["batch_size"])
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(1000)
val_ds = val_ds.batch(configs['CNN']["batch_size"])
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(configs['CNN']["batch_size"])
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)






# # ##################################################################
# #                phase 6: Loading Image features
# # ##################################################################

model.compile(
    optimizer=tf.keras.optimizers.Adam(), #learning_rate=0.001
    loss=tf.keras.losses.SparseCategoricalCrossentropy (from_logits=True), 
    metrics=["Accuracy"]
    )


TensorBoard_logs = configs["paths"]["TensorBoard_logs"] + "/"
time = int(timeit.timeit()*1_000_000)

checkpoint = [
        tf.keras.callbacks.ModelCheckpoint(    configs["CNN"]["saving_path"], save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(  monitor="val_loss", factor=0.5, patience=30, min_lr=0.0001),
        tf.keras.callbacks.EarlyStopping(      monitor="val_loss", patience=90, verbose=1),
        tf.keras.callbacks.TensorBoard(        log_dir=TensorBoard_logs+str(time))   
    ]    


history = model.fit(
    train_ds,    
    batch_size=configs["CNN"]["batch_size"],
    callbacks=[checkpoint],
    epochs=configs["CNN"]["epochs"],
    validation_data=val_ds,
    verbose=1#configs["CNN"]["verbose"],
)


model.save(configs["CNN"]["saving_path"])
# plt.plot(history.history['accuracy'], label='accuracy')
# # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.show()

test_loss, test_acc = model.evaluate(test_ds, verbose=2)

print(" test_loss ", test_loss, " test_acc ", test_acc)
logger.info("Done!!")
sys.exit()






loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')


# train_writer = tf.summary.create_file_writer("logs/train/")
# test_writer = tf.summary.create_file_writer("logs/test/")
train_step = test_step = 0

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy.update_state(labels, predictions)

@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)




for epoch in range(configs['CNN']["epochs"]):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)


    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )

