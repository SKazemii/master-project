# TF 1.14 gives lots of warnings for deprecations ready for the switch to TF 2.0
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import glob

from datetime import datetime
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Input, concatenate
from tensorflow.keras.layers import Layer, BatchNormalization, MaxPooling2D, Concatenate, Lambda, Flatten, Dense
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, concatenate, Lambda, BatchNormalization
from keras.models import Sequential, Model, Input

from tensorflow.keras.initializers import glorot_uniform, he_uniform
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score
import math
# from pylab import dist
import json

from tensorflow.python.client import device_lib
import matplotlib.gridspec as gridspec

import Project as prg

from sklearn import model_selection



setting = {

        "dataset_name": 'casia',

        "_classifier_name": 'TM',
        "_combination": True,

        "_CNN_weights": 'imagenet',
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": '',
        "_CNN_epochs": 500,
        "_CNN_optimizer": 'adam',
        "_val_size": 0.2,

        "_min_number_of_sample": 30,
        "_known_imposter": 3,
        "_unknown_imposter": 5,
        "_number_of_unknown_imposter_samples": 1.0,  # Must be less than 1


        "_waveletname": 'coif1',
        "_pywt_mode": 'constant',
        "_wavelet_level": 4,


        "_p_training_samples": 11,
        "_train_ratio": 34,
        "_ratio": False,


        "_KNN_n_neighbors": 5,
        "_KNN_metric": 'euclidean',
        "_KNN_weights": 'uniform',
        "_SVM_kernel": 'linear',


        "_KFold": 10,
        "_random_runs": 20,
        "_persentage": 0.95,
        "_normilizing": 'z-score',

    }
    
A = prg.Pipeline(setting)


A._CNN_batch_size = 64
A._CNN_epochs = 800
A._CNN_optimizer = tf.keras.optimizers.Adadelta()
A._val_size = 0.2
    
image_feature_name = ['P100']#["CD", "PTI", "Tmin", "Tmax", "P50", "P60", "P70", "P80", "P90", "P100"]
dataset_name = "casia"

CNN_name = "lightweight_CNN"

pre_images, labels = A.loading_pre_features_image(dataset_name)
pre_image = A.loading_image_features_from_list(pre_images, image_feature_name)

encoded_labels = A.label_encoding(labels)

outputs = len(labels['ID'].unique())

images_feat_norm = A.normalizing_image_features(pre_image)

X_train, X_test, y_train, y_test = model_selection.train_test_split(images_feat_norm, encoded_labels, test_size=0.15, random_state=A._random_state, stratify=encoded_labels)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.1, random_state=A._random_state, stratify=y_train)

x_train_w_h = 11


def create_batch(batch_size, data, data_y):
    x_anchors = list()
    x_positives = list()
    x_negatives = list()

    for i in range(0, batch_size):
        random_index = random.randint(0, data.shape[0] - 1)
        x_anchor = data[random_index]
        y = data_y[random_index]

        indices_for_pos = np.squeeze(np.where(data_y == y))
        indices_for_neg = np.squeeze(np.where(data_y != y))

        
        x_positive = data[indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]]
        x_negative = data[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]]
        
        x_anchors.append( x_anchor)
        x_positives.append( x_positive)
        x_negatives.append( x_negative)
    
    ai = np.array(x_anchors)
    pi = np.array(x_positives)
    ni = np.array(x_negatives)
    return [ai, pi, ni]


anchor_image, positive_image, negative_image = create_batch(256, X_train, y_train)


fig, ax = plt.subplots(nrows=3, ncols=3)
for i in range(3):
    ax[i, 0].imshow(anchor_image[i,...])
    ax[i, 1].imshow(positive_image[i,...])
    ax[i, 2].imshow(negative_image[i,...])
plt.show()


emb_size = 128
# emb_size = 2048
embedding_model = Sequential()
embedding_model.add(Conv2D(64, (5,5), activation='relu', input_shape=(60,40,1)))
embedding_model.add(BatchNormalization())
embedding_model.add(MaxPooling2D())
embedding_model.add(Dropout(0.5))
embedding_model.add(Conv2D(128, (5,5), activation='relu'))
embedding_model.add(BatchNormalization())
embedding_model.add(MaxPooling2D())
embedding_model.add(Dropout(0.5))
embedding_model.add(Conv2D(256, (3,3), activation='relu'))
embedding_model.add(BatchNormalization())
embedding_model.add(Flatten())
embedding_model.add(Dense(emb_size, activation='sigmoid'))
embedding_model.add(Lambda(lambda x:tf.keras.backend.l2_normalize(x, axis=1)))
embedding_model.summary()

in_anc = Input(shape=(60,40,1))
in_pos = Input(shape=(60,40,1))
in_neg = Input(shape=(60,40,1))

em_anc = embedding_model(in_anc)
em_pos = embedding_model(in_pos)
em_neg = embedding_model(in_neg)

out = concatenate([em_anc, em_pos, em_neg], axis=1)

siamese_net = Model(
    [in_anc, in_pos, in_neg],
    out
)

siamese_net.summary()

# L2 Distance
def triplet_loss(alpha, emb_dim):
    def loss(y_true, y_pred):
        anc, pos, neg = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
        distance1 = tf.sqrt(tf.reduce_sum(tf.pow(anc - pos, 2), 1, keepdims=True))
        distance2 = tf.sqrt(tf.reduce_sum(tf.pow(anc - neg, 2), 1, keepdims=True))
        return tf.reduce_mean(tf.maximum(distance1 - distance2 + alpha, 0.))
    return loss

batch_size = 64
epochs = 50
opt = tf.keras.optimizers.Adam(lr = 0.001)
steps_per_epoch = 100
siamese_net.compile(loss=triplet_loss(alpha=0.2, emb_dim=emb_size), optimizer=opt, metrics=['accuracy'])
# siamese_net.summary()

def data_generator(batch_size, emb_size):
    while True:
        x = create_batch(batch_size, X_train, y_train)
        # x = preprocess_data.get_triplet_batch(batch_size)
        y = np.zeros((batch_size, 3*emb_size))
        yield x,y

save_path = os.path.join(os.getcwd(), "model_weights_triplet_loss_2048.h5")


history = siamese_net.fit(
    data_generator(batch_size, emb_size),
    epochs=epochs, 
    steps_per_epoch=steps_per_epoch,
    verbose=True
)        



breakpoint()
import pandas as pd
hist_df = pd.DataFrame(history.history) 

fig, ax = plt.subplots(1,2,figsize=(10,6))
ax[0].plot(hist_df['Accuracy'], label='Train Accuracy')
ax[0].plot(hist_df['val_Accuracy'], label = 'Val Accuracy')

ax[0].set_title('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].legend()


# summarize history for loss
ax[1].plot(hist_df['loss'], label='Train Loss')
ax[1].plot(hist_df['val_loss'], label='Val Loss')
ax[1].set_title('Loss')
ax[1].set_ylabel('loss')
ax[1].set_xlabel('epoch')
ax[1].legend()

plt.show()


test_embedding_output1 = embedding_model.predict(np.expand_dims(anchor_image[3], axis=0))
test_embedding_output2 = embedding_model.predict(np.expand_dims(positive_image[3], axis=0))
test_embedding_output3 = embedding_model.predict(np.expand_dims(negative_image[3], axis=0))
print(test_embedding_output1.shape)
distance1 = tf.sqrt(tf.reduce_sum(tf.pow(test_embedding_output1 - test_embedding_output2, 2), 1, keepdims=True))
distance2 = tf.sqrt(tf.reduce_sum(tf.pow(test_embedding_output1 - test_embedding_output3, 2), 1, keepdims=True))
loss = tf.reduce_mean(tf.maximum(distance1 - distance2 + 0.2, 0.))
distance1, distance2, loss
breakpoint()





def compute_metrics(probs,yprobs):
    '''
    Returns
        fpr : Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i]
        tpr : Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
        thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
        auc : Area Under the ROC Curve metric
    '''
    # calculate AUC
    auc = roc_auc_score(yprobs, probs)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(yprobs, probs)
    
    return fpr, tpr, thresholds,auc

def draw_roc(fpr, tpr,thresholds, auc):
    #find threshold
    targetfpr=1e-3
    _, idx = find_nearest(fpr,targetfpr)
    threshold = thresholds[idx]
    recall = tpr[idx]
    
    
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.title('AUC: {0:.3f}\nSensitivity : {2:.1%} @FPR={1:.0e}\nThreshold={3})'.format(auc,targetfpr,recall,abs(threshold) ))
    # show the plot
    plt.show()
    
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1],idx-1
    else:
        return array[idx],idx
    
def draw_interdist(network, epochs):
    interdist = compute_interdist(network)
    
    data = []
    for i in range(num_classes):
        data.append(np.delete(interdist[i,:],[i]))

    fig, ax = plt.subplots()
    ax.set_title('Evaluating embeddings distance from each other after {0} epochs'.format(epochs))
    ax.set_ylim([0,3])
    plt.xlabel('Classes')
    plt.ylabel('Distance')
    ax.boxplot(data,showfliers=False,showbox=True)
    locs, labels = plt.xticks()
    plt.xticks(locs,np.arange(num_classes))

    plt.show()
       
def compute_interdist(network):
    '''
    Computes sum of distances between all classes embeddings on our reference test image: 
        d(0,1) + d(0,2) + ... + d(0,9) + d(1,2) + d(1,3) + ... d(8,9)
        A good model should have a large distance between all theses embeddings
        
    Returns:
        array of shape (num_classes,num_classes) 
    '''
    res = np.zeros((num_classes,num_classes))
    
    ref_images = np.zeros((num_classes, x_test_w_h))
    
    #generates embeddings for reference images
    for i in range(num_classes):
        ref_images[i,:] = x_test[i]
    
    ref_embeddings = network.predict(ref_images)
    
    for i in range(num_classes):
        for j in range(num_classes):
            res[i,j] = math.dist(ref_embeddings[i],ref_embeddings[j])
    return res

def DrawTestImage(network, images, refidx=0):
    '''
    Evaluate some pictures vs some samples in the test set
        image must be of shape(1,w,h,c)
    
    Returns
        scores : result of the similarity scores with the basic images => (N)
    
    '''
    nbimages = images.shape[0]
    
    
    #generates embedings for given images
    image_embedings = network.predict(images)
    
    #generates embedings for reference images
    ref_images = np.zeros((num_classes,x_test_w_h))
    for i in range(num_classes):
        images_at_this_index_are_of_class_i = np.squeeze(np.where(y_test == i))
        ref_images[i,:] = x_test[images_at_this_index_are_of_class_i[refidx]]
        
        
    ref_embedings = network.predict(ref_images)
            
    for i in range(nbimages):
        # Prepare the figure
        fig=plt.figure(figsize=(16,2))
        subplot = fig.add_subplot(1,num_classes+1,1)
        plt.axis("off")
        plotidx = 2
            
        # Draw this image    
        plt.imshow(np.reshape(images[i], (x_train_w, x_train_h)),vmin=0, vmax=1,cmap='Greys')
        subplot.title.set_text("Test image")
            
        for ref in range(num_classes):
            #Compute distance between this images and references
            dist = compute_dist(image_embedings[i,:],ref_embedings[ref,:])
            #Draw
            subplot = fig.add_subplot(1,num_classes+1,plotidx)
            plt.axis("off")
            plt.imshow(np.reshape(ref_images[ref, :], (x_train_w, x_train_h)),vmin=0, vmax=1,cmap='Greys')
            subplot.title.set_text(("Class {0}\n{1:.3e}".format(ref,dist)))
            plotidx += 1

def generate_prototypes(x_data, y_data, embedding_model):
    classes = np.unique(y_data)
    prototypes = {}

    for c in classes:
        #c = classes[0]
        # Find all images of the chosen test class
        locations_of_c = np.where(y_data == c)[0]

        imgs_of_c = x_data[locations_of_c]

        imgs_of_c_embeddings = embedding_model.predict(imgs_of_c)

        # Get the median of the embeddings to generate a prototype for the class (reshaping for PCA)
        prototype_for_c = np.median(imgs_of_c_embeddings, axis = 0).reshape(1, -1)
        # Add it to the prototype dict
        prototypes[c] = prototype_for_c
        
    return prototypes
         
def test_one_shot_prototypes(network, sample_embeddings):
    distances_from_img_to_test_against = []
    # As the img to test against is in index 0, we compare distances between img@0 and all others
    for i in range(1, len(sample_embeddings)):
        distances_from_img_to_test_against.append(compute_dist(sample_embeddings[0], sample_embeddings[i]))
    # As the correct img will be at distances_from_img_to_test_against index 0 (sample_imgs index 1),
    # If the smallest distance in distances_from_img_to_test_against is at index 0, 
    # we know the one shot test got the right answer
    is_min = distances_from_img_to_test_against[0] == min(distances_from_img_to_test_against)
    is_max = distances_from_img_to_test_against[0] == max(distances_from_img_to_test_against)
    return int(is_min and not is_max)
    
def n_way_accuracy_prototypes(n_val, n_way, network):
    num_correct = 0
    
    for val_step in range(n_val):
        num_correct += load_one_shot_test_batch_prototypes(n_way, network)
        
    accuracy = num_correct / n_val * 100
        
    return accuracy

def load_one_shot_test_batch_prototypes(n_way, network):
    
    labels = np.unique(y_test)
    # Reduce the label set down from size n_classes to n_samples 
    labels = np.random.choice(labels, size = n_way, replace = False)

    # Choose a class as the test image
    label = random.choice(labels)
    # Find all images of the chosen test class
    imgs_of_label = np.where(y_test == label)[0]

    # Randomly select a test image of the selected class, return it's index
    img_of_label_idx = random.choice(imgs_of_label)

    # Expand the array at the selected indexes into useable images
    img_of_label = np.expand_dims(x_test[img_of_label_idx],axis=0)
    
    sample_embeddings = []
    # Get the anchor image embedding
    anchor_prototype = network.predict(img_of_label)
    sample_embeddings.append(anchor_prototype)
    
    # Get the prototype embedding for the positive class
    positive_prototype = prototypes[label]
 
    sample_embeddings.append(positive_prototype)
    
    # Get the negative prototype embeddings
    # Remove the selected test class from the list of labels based on it's index 
    label_idx_in_labels = np.where(labels == label)[0]
    other_labels = np.delete(labels, label_idx_in_labels)
    
    # Get the embedding for each of the remaining negatives
    for other_label in other_labels:
        negative_prototype = prototypes[other_label]
        sample_embeddings.append(negative_prototype)
                
    correct = test_one_shot_prototypes(network, sample_embeddings)

    return correct

def visualise_n_way_prototypes(n_samples, network):
    labels = np.unique(y_test)
    # Reduce the label set down from size n_classes to n_samples 
    labels = np.random.choice(labels, size = n_samples, replace = False)

    # Choose a class as the test image
    label = random.choice(labels)
    # Find all images of the chosen test class
    imgs_of_label = np.where(y_test == label)[0]

    # Randomly select a test image of the selected class, return it's index
    img_of_label_idx = random.choice(imgs_of_label)

    # Get another image idx that we know is of the test class for the sample set
    label_sample_img_idx = random.choice(imgs_of_label)

    # Expand the array at the selected indexes into useable images
    img_of_label = np.expand_dims(x_test[img_of_label_idx],axis=0)
    label_sample_img = np.expand_dims(x_test[label_sample_img_idx],axis=0)
    
    # Make the first img in the sample set the chosen test image, the second the other image
    sample_imgs = np.empty((0, x_test_w_h))
    sample_imgs = np.append(sample_imgs, img_of_label, axis=0)
    sample_imgs = np.append(sample_imgs, label_sample_img, axis=0)
    
    sample_embeddings = []
    
    # Get the anchor embedding image
    anchor_prototype = network.predict(img_of_label)
    sample_embeddings.append(anchor_prototype)
    
    # Get the prototype embedding for the positive class
    positive_prototype = prototypes[label]
    sample_embeddings.append(positive_prototype)

    # Get the negative prototype embeddings
    # Remove the selected test class from the list of labels based on it's index 
    label_idx_in_labels = np.where(labels == label)[0]
    other_labels = np.delete(labels, label_idx_in_labels)
    # Get the embedding for each of the remaining negatives
    for other_label in other_labels:
        negative_prototype = prototypes[other_label]
        sample_embeddings.append(negative_prototype)
        
        # Find all images of the other class
        imgs_of_other_label = np.where(y_test == other_label)[0]
        # Randomly select an image of the selected class, return it's index
        another_sample_img_idx = random.choice(imgs_of_other_label)
        # Expand the array at the selected index into useable images
        another_sample_img = np.expand_dims(x_test[another_sample_img_idx],axis=0)
        # Add the image to the support set
        sample_imgs = np.append(sample_imgs, another_sample_img, axis=0)
    
    distances_from_img_to_test_against = []
    
    # As the img to test against is in index 0, we compare distances between img@0 and all others
    for i in range(1, len(sample_embeddings)):
        distances_from_img_to_test_against.append(compute_dist(sample_embeddings[0], sample_embeddings[i]))
        
    # + 1 as distances_from_img_to_test_against doesn't include the test image
    min_index = distances_from_img_to_test_against.index(min(distances_from_img_to_test_against)) + 1
    
    return sample_imgs, min_index

def evaluate(embedding_model, epochs = 0):
    probs,yprob = compute_probs(embedding_model, x_test[:500, :], y_test[:500])
    fpr, tpr, thresholds, auc = compute_metrics(probs,yprob)
    draw_roc(fpr, tpr, thresholds, auc)
    draw_interdist(embedding_model, epochs)

    for i in range(3):
        DrawTestImage(embedding_model, np.expand_dims(x_train[i],axis=0))


# Hyperparams
batch_size = 256
epochs = 100
steps_per_epoch = int(x_train.shape[0]/batch_size)
val_steps = int(x_test.shape[0]/batch_size)
alpha = 0.2
num_hard = int(batch_size * 0.5) # Number of semi-hard triplet examples in the batch
lr = 0.00006
optimiser = 'Adam'
emb_size = 10

with tf.device("/cpu:0"):
    # Create the embedding model
    print("Generating embedding model... \n")
    embedding_model = create_embedding_model(emb_size)
    
    print("\nGenerating SNN... \n")
    # Create the SNN
    siamese_net = create_SNN(embedding_model)
    # Compile the SNN
    optimiser_obj = Adam(lr = lr)
    siamese_net.compile(loss=triplet_loss, optimizer= optimiser_obj)
    
    # Store visualisations of the embeddings using PCA for display next to "after training" for comparisons
    num_vis = 500 # Take only the first num_vis elements of the test set to visualise
    embeddings_before_train = embedding_model.predict(x_test[:num_vis, :])
    pca = PCA(n_components=2)
    decomposed_embeddings_before = pca.fit_transform(embeddings_before_train)


# Display evaluation the untrained model
print("\nEvaluating the model without training for a baseline...\n")
evaluate(embedding_model)

# Set up logging directory
## Use date-time as logdir name:
#dt = datetime.now().strftime("%Y%m%dT%H%M")
#logdir = os.path.join("PATH/TO/LOGDIR",dt)

## Use a custom non-dt name:
name = "snn-example-run"
logdir = os.path.join(os.getcwd(), name)

if not os.path.exists(logdir):
    os.mkdir(logdir)

## Callbacks:
# Create the TensorBoard callback
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir = logdir,
    histogram_freq=0,
    batch_size=batch_size,
    write_graph=True,
    write_grads=True, 
    write_images = True, 
    update_freq = 'epoch', 
    profile_batch=0
)

# Training logger
csv_log = os.path.join(logdir, 'training.csv')
csv_logger = CSVLogger(csv_log, separator=',', append=True)

# Only save the best model weights based on the val_loss
checkpoint = ModelCheckpoint(os.path.join(logdir, 'snn_model-{epoch:02d}-{val_loss:.2f}.h5'),
                             monitor='val_loss', verbose=1,
                             save_best_only=True, save_weights_only=True, 
                             mode='auto')

# Save the embedding mode weights based on the main model's val loss
# This is needed to reecreate the emebedding model should we wish to visualise
# the latent space at the saved epoch
class SaveEmbeddingModelWeights(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.best = np.Inf
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("SaveEmbeddingModelWeights requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.best:
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            #if self.verbose == 1:
                #print("Saving embedding model weights at %s" % filepath)
            embedding_model.save_weights(filepath, overwrite = True)
            self.best = current

# Save the embedding model weights if you save a new snn best model based on the model checkpoint above
emb_weight_saver = SaveEmbeddingModelWeights(os.path.join(logdir, 'emb_model-{epoch:02d}.h5'))


callbacks = [tensorboard, csv_logger, checkpoint, emb_weight_saver]


# Save model configs to JSON
model_json = siamese_net.to_json()
with open(os.path.join(logdir, "siamese_config.json"), "w") as json_file:
    json_file.write(model_json)
    json_file.close()
    
model_json = embedding_model.to_json()
with open(os.path.join(logdir, "embedding_config.json"), "w") as json_file:
    json_file.write(model_json)
    json_file.close()
    

hyperparams = {'batch_size' : batch_size,
              'epochs' : epochs, 
               'steps_per_epoch' : steps_per_epoch, 
               'val_steps' : val_steps, 
               'alpha' : alpha, 
               'num_hard' : num_hard, 
               'optimiser' : optimiser,
               'lr' : lr,
               'emb_size' : emb_size
              }


with open(os.path.join(logdir, "hyperparams.json"), "w") as json_file:
    json.dump(hyperparams, json_file)
    
# Set the model to TB
tensorboard.set_model(siamese_net)


def delete_older_model_files(filepath):
    
    model_dir = filepath.split("emb_model")[0]
    
    # Get model files
    model_files = os.listdir(model_dir)

    # Get only the emb_model files
    emb_model_files = [file for file in model_files if "emb_model" in file]
    # Get the epoch nums of the emb_model_files
    emb_model_files_epoch_nums = [int(file.split("-")[1].split(".h5")[0]) for file in emb_model_files]

    # Find all the snn model files
    snn_model_files = [file for file in model_files if "snn_model" in file]

    # Sort, get highest epoch num
    emb_model_files_epoch_nums.sort()
    highest_epoch_num = str(emb_model_files_epoch_nums[-1]).zfill(2)

    # Filter the emb_model and snn_model file lists to remove the highest epoch number ones
    emb_model_files_without_highest = [file for file in emb_model_files if highest_epoch_num not in file]
    snn_model_files_without_highest = [file for file in snn_model_files if ("-" + highest_epoch_num + "-") not in file]

    # Delete the non-highest model files from the subdir
    if len(emb_model_files_without_highest) != 0:
        print("Deleting previous best model file")
    for model_file_list in [emb_model_files_without_highest, snn_model_files_without_highest]:
        for file in model_file_list:
            os.remove(os.path.join(model_dir, file))





# Display sample batches. This has to be performed after the embedding model is created
# as create_batch_hard utilises the model to see which batches are actually hard.

examples = create_batch(1)
print("Example triplet batch:")
plot_triplets(examples)

print("Example semi-hard triplet batch:")
ex_hard = create_hard_batch(1, 1, split="train")
plot_triplets(ex_hard)

def get_num_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

## Training:
#print("Logging out to Tensorboard at:", logdir)
print("Starting training process!")
print("-------------------------------------")

# # Make the model work over the two GPUs we have
# num_gpus = get_num_gpus()
# parallel_snn = multi_gpu_model(siamese_net, gpus = num_gpus)
# batch_per_gpu = int(batch_size / num_gpus)

siamese_net.compile(loss=triplet_loss, optimizer=optimiser_obj)

siamese_history = siamese_net.fit(
    data_generator(batch_size, num_hard),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks, 
    workers = 0, 
    validation_data = data_generator(batch_size, num_hard, split="test"), 
    validation_steps = val_steps)

print("-------------------------------------")
print("Training complete.")
