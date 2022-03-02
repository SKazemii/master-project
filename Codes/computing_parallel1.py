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






SLURM_JOBID = str(os.environ.get('SLURM_JOBID', default=os.getpid()))

# import extracting_deep_featrues as extract
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    
from MLPackage import config as cfg
from MLPackage import Utilities as util


project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "Manuscripts", "src", "figures")
tbl_dir = os.path.join(project_dir, "Manuscripts", "src", "tables")
results_dir = os.path.join(project_dir, "results")
dataset_dir = os.path.join(project_dir, "Datasets")
temp_dir = os.path.join(project_dir, "temp")
log_path = os.path.join(project_dir, 'logs')

Pathlb(log_path).mkdir(parents=True, exist_ok=True)
Pathlb(dataset_dir).mkdir(parents=True, exist_ok=True)
Pathlb(temp_dir).mkdir(parents=True, exist_ok=True)
Pathlb(results_dir ).mkdir(parents=True, exist_ok=True)
Pathlb(fig_dir).mkdir(parents=True, exist_ok=True)
Pathlb(tbl_dir).mkdir(parents=True, exist_ok=True)




def create_logger(level):
    loggerName = Pathlb(__file__).stem
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
    file_handler = logging.FileHandler( os.path.join(log_path, f"{SLURM_JOBID}_" + loggerName + '_loger.log'), mode = 'w')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    stream_handler.setFormatter(formatter_colored)


    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
logger = create_logger(logging.DEBUG)




time = int(timeit.default_timer() * 1_000_000)

def collect_results(result):
    global time
    test = os.environ.get('SLURM_JOB_NAME', default="Mode_1")
    excel_path = os.path.join(cfg.configs["paths"]["results_dir"], f"Result_all_{test}.xlsx")
    # excel_path = os.path.join(cfg.configs["paths"]["results_dir"], "Result.xlsx")

    if os.path.isfile(excel_path):
        Results_DF = pd.read_excel(excel_path, index_col = 0)
    else:
        Results_DF = pd.DataFrame(columns=util.columnsname_result_DF)

    Results_DF = Results_DF.append(result)
    try:
        Results_DF.to_excel(excel_path, columns=util.columnsname_result_DF)
    except:
        Results_DF.to_excel(excel_path+str(time)+'.xlsx', columns=util.columnsname_result_DF)



def main():

    # ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=4))
    # pool = multiprocessing.Pool(processes=ncpus)
    # logger.info(f"CPU count: {ncpus}")


    test = os.environ.get('SLURM_JOB_NAME', default="Mode_1")
    for _ in ["Mode_2"]:#["Mode_1", "Mode_2", "Mode_3"]:
        logger.info(f"test name: {test}")

        if test=="Mode_1": # handcrafted features first pipeline
            p0 = [i for i in range(6,23,3)]
            p1 = [5, 7, 9]
            p2 = [1, 2, 5, 10, 20, 50, 100]
            space = list(product(p0, p1, p2))
            space = space[104:]

        elif test=="Mode_2A": 
            p0 = [3, 6, 9,]
            p1 = ["mahalanobis"]
            p2 = [2, 3, 6, 9, 15, 21, 27, 30, 45, 60, 90, 120, 150, 180, 210, 360, 500, 1000]
            space = list(product(p0, p1, p2))

        elif test=="Mode_2B": 
            p0 = [10, 11, 12]
            p1 = ["mahalanobis"]
            p2 = [2, 3, 6, 9, 15, 21, 27, 30, 45, 60, 90, 120, 150, 180, 210, 360, 500, 1000]
            space = list(product(p0, p1, p2))

        elif test=="Mode_2C": 
            p0 = [13, 14, 15]
            p1 = ["mahalanobis"]
            p2 = [2, 3, 6, 9, 15, 21, 27, 30, 45, 60, 90, 120, 150, 180, 210, 360, 500, 1000]
            space = list(product(p0, p1, p2))

        elif test=="Mode_2D": 
            p0 = [18, 21, 24, 27]
            p1 = ["mahalanobis"]
            p2 = [2, 3, 6, 9, 15, 21, 27, 30, 45, 60, 90, 120, 150, 180, 210, 360, 500, 1000]
            space = list(product(p0, p1, p2))


        elif test=="Mode_3": 
            p0 = [24, 27]#(i,j) for i in range(3,30,3) for j in range(3,30,3) if i+j<=30]
            p1 = [5, 7, 9]
            p2 = [1, 2, 5, 10, 20, 50, 100]
            space = list(product(p0, p1, p2))
            space = space[-1:]


        for idx, parameters in enumerate(space):
            logger.info(f"[step {idx+1} out of {len(space)}], parameters: {parameters}")
            configs = copy.deepcopy(cfg.configs)

            if test=="Mode_3": # handcrafted features first pipeline
                configs["Pipeline"]["classifier"] = "KNN"
                configs["Pipeline"]["training_ratio"] = parameters[2]
                configs["features"]["category"] = "hand_crafted"
                configs["features"]["combination"] = True
                configs['dataset']["dataset_name"] = "casia"
                configs["Pipeline"]["train_ratio"] = parameters[0]
                configs["classifier"]["KNN"]["n_neighbors"] = parameters[1]
                collect_results(util.pipeline(configs))
                
            elif test== "Mode_2A" | "Mode_2B" | "Mode_2C" | "Mode_2D": # handcrafted features first pipeline
                configs["Pipeline"]["classifier"] = "TM"
                configs["Pipeline"]["train_ratio"] = parameters[0]
                configs["features"]["category"] = "hand_crafted"
                configs["features"]["combination"] = True
                configs['dataset']["dataset_name"] = "casia"
                configs["Pipeline"]["test_ratio"] = parameters[2]
                collect_results(util.pipeline(configs))
                      
            elif test=="Mode_1": # handcrafted features first pipeline
                configs["Pipeline"]["classifier"] = "KNN"
                configs["Pipeline"]["training_ratio"] = parameters[2]
                configs["features"]["category"] = "hand_crafted"
                configs["features"]["combination"] = True
                configs['dataset']["dataset_name"] = "casia"
                configs["Pipeline"]["train_ratio"] = parameters[0]
                configs["classifier"]["KNN"]["n_neighbors"] = parameters[1]
                collect_results(util.pipeline(configs))

            
    logger.info("Done!!!")



if __name__ == "__main__":
    logger.info("Starting !!!")
    tic = timeit.default_timer()
    main()
    toc = timeit.default_timer()
    logger.info("Done ({:2.2f} process time)!!!\n\n\n".format(toc-tic))




