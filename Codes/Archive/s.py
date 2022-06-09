import warnings

from numpy.core.einsumfunc import _update_other_results
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, os, logging, timeit, pprint, copy
from pathlib import Path as Pathlb


from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import multiprocessing

from itertools import product


import sklearn as sk
import tensorflow as tf
from tensorflow import keras

# from sklearn.decomposition import PCA
# from sklearn.decomposition import IncrementalPCA
# from sklearn.decomposition import KernelPCA
# from sklearn.decomposition import SparsePCA
# from sklearn.decomposition import TruncatedSVD
# from sklearn.decomposition import FastICA
# from sklearn.decomposition import MiniBatchDictionaryLearning


# from sklearn.manifold import Isomap
# from sklearn.manifold import TSNE
# from sklearn.manifold import LocallyLinearEmbedding

# from sklearn.random_projection import GaussianRandomProjection
# from sklearn.random_projection import SparseRandomProjection






SLURM_JOBID = str(os.environ.get('SLURM_JOBID',default=os.getpid()))

import extracting_deep_featrues as extract
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    
from MLPackage import config as cfg
from MLPackage import Utilities as util


e = pd.read_excel(r"C:\Project\master-project\results\Result.xlsx")
e.drop("Unnamed: 0",  axis=1, inplace=True)

df_all = e[e["subject ID"]==4]
df_all = df_all[df_all["# positive samples training"]==5]
df_all = df_all[df_all["clasifier"]=="KNN"]
plt.plot(df_all.iloc[0,24:124], np.linspace(0, 1, 100))
plt.plot(df_all.iloc[0,124:225], np.linspace(0, 1, 100))
plt.figure()
plt.plot(df_all.iloc[3,24:124], np.linspace(0, 1, 100))
plt.plot(df_all.iloc[3,124:225], np.linspace(0, 1, 100))
ff = np.linspace(0, 1, 100)
print(ff[60])
print(ff[1])
df_all.iloc[:,10:25]

breakpoint()
EER, t_idx = util.compute_eer(df_all.iloc[0,24:124], df_all.iloc[0,124:225])
print(EER, t_idx)