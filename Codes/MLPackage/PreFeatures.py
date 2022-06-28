import re
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import distance
import scipy.io
import collections

from pathlib import Path as Pathlb
import pywt, sys, random
import os, logging, timeit, pprint, copy, multiprocessing, glob
from itertools import product

# keras imports
import tensorflow as tf
from tensorflow import keras 

from keras import preprocessing, callbacks
from keras.models import Model, load_model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten
import keras.backend as K

from sklearn.cluster import KMeans
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn import preprocessing as sk_preprocessing
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import svm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import OneClassSVM
# from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

import optunity
import optunity.metrics

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.getcwd(), "MLPackage"))
import Butterworth
import convertGS2BW
 


project_dir = os.getcwd()
log_path = os.path.join(project_dir, "logs")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def create_logger(level):
    loggerName = Pathlb(__file__).stem
    Pathlb(log_path).mkdir(parents=True, exist_ok=True)
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    logger = logging.getLogger(loggerName)
    logger.setLevel(level)
    t1 = ( blue + "[%(asctime)s]-" + yellow + "[%(name)s @%(lineno)d]" + reset + blue + "-[%(levelname)s]" + reset + bold_red )
    t2 = "%(message)s" + reset
    # breakpoint()
    formatter_colored = logging.Formatter(t1 + t2, datefmt="%m/%d/%Y %I:%M:%S %p ")
    formatter = logging.Formatter(
        "[%(asctime)s]-[%(name)s @%(lineno)d]-[%(levelname)s]      %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p ",
    )
    file_handler = logging.FileHandler(
        os.path.join(log_path, loggerName + "_loger.log"), mode="w"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    stream_handler.setFormatter(formatter_colored)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


logger = create_logger(logging.DEBUG)

class Database(object):
    def __init__(self,):
        super().__init__()

    def load_H5():
        pass

    def print_paths(self):
        logger.info(self._h5_path)
        logger.info(self._data_path)
        logger.info(self._meta_path)

    def set_dataset_path(self, dataset_name: str) -> None:
        """setting path for dataset"""
        if dataset_name == "casia":
            self._h5_path = os.path.join(
                os.getcwd(), "Datasets", "Casia-D", "reserve.h5"
            )
            self._data_path = os.path.join(
                os.getcwd(), "Datasets", "Casia-D", "Data-barefoot.npy"
            )
            self._meta_path = os.path.join(
                os.getcwd(), "Datasets", "Casia-D", "Metadata-barefoot.npy"
            )
            self._pre_features_path = os.path.join(
                os.getcwd(), "Datasets", "Casia-D", "pre_features"
            )
            self._features_path = os.path.join(
                os.getcwd(), "Datasets", "Casia-D", "features"
            )

        elif dataset_name == "casia-shod":
            self._h5_path = os.path.join(
                os.getcwd(), "Datasets", "Casia-D", "reserve.h5"
            )
            self._data_path = os.path.join(
                os.getcwd(), "Datasets", "Casia-D", "Data-shod.npy"
            )
            self._meta_path = os.path.join(
                os.getcwd(), "Datasets", "Casia-D", "Metadata-shod.npy"
            )
            self._pre_features_path = os.path.join(
                os.getcwd(), "Datasets", "Casia-D", "pre_features-shod"
            )
            self._features_path = os.path.join(
                os.getcwd(), "Datasets", "Casia-D", "features-shod"
            )

        elif dataset_name == "stepscan":
            self._h5_path = os.path.join(
                os.getcwd(), "Datasets", "stepscan", "footpressures_align.h5"
            )
            self._data_path = os.path.join(
                os.getcwd(), "Datasets", "stepscan", "Data-barefoot.npy"
            )
            self._meta_path = os.path.join(
                os.getcwd(), "Datasets", "stepscan", "Metadata-barefoot.npy"
            )
            self._pre_features_path = os.path.join(
                os.getcwd(), "Datasets", "stepscan", "pre_features"
            )
            self._features_path = os.path.join(
                os.getcwd(), "Datasets", "stepscan", "features"
            )

        elif dataset_name == "sfootbd":
            self._h5_path = os.path.join(os.getcwd(), "Datasets", "sfootbd", ".h5")
            self._mat_path = os.path.join(os.getcwd(), "Datasets", "sfootbd", "SFootBD")
            self._txt_path = os.path.join(
                os.getcwd(), "Datasets", "sfootbd", "IndexFiles"
            )
            self._data_path = os.path.join(
                os.getcwd(), "Datasets", "sfootbd", "SFootBD", "Data.npy"
            )
            self._meta_path = os.path.join(
                os.getcwd(), "Datasets", "sfootbd", "SFootBD", "Meta.npy"
            )
            self._pre_features_path = os.path.join(
                os.getcwd(), "Datasets", "sfootbd", "pre_features"
            )
            self._features_path = os.path.join(
                os.getcwd(), "Datasets", "sfootbd", "features"
            )
            if not (
                os.path.isfile(self._data_path) and os.path.isfile(self._meta_path)
            ):
                self.mat_to_numpy()

        else:
            logger.error("The name is not valid!!")
            sys.exit()

        Pathlb(self._pre_features_path).mkdir(parents=True, exist_ok=True)
        Pathlb(self._features_path).mkdir(parents=True, exist_ok=True)

        return None

    def extracting_labels(self, dataset_name: str) -> np.ndarray:
        if dataset_name == "casia":
            return self._meta[:, 0:2]

        elif dataset_name == "casia-shod":
            return self._meta[:, 0:2]

        elif dataset_name == "stepscan":
            return self._meta[:-1, 0:2]

        elif dataset_name == "sfootbd":
            g = glob.glob(self._txt_path + "\*.txt", recursive=True)
            label = list()
            label1 = list()
            for i in g:
                with open(i) as f:
                    for line in f:
                        label.append(line.split(" ")[0])
                        label1.append(line.split(" ")[1][:-1])

            file_name = np.load(self._meta_path)
            lst = list()
            for i in file_name:
                ind = label1.index(i[0][:-4])
                lst.append(int(label[ind]))

            return np.array(lst)

        else:
            logger.error("The name is not valid!!")

    def print_dataset(self):
        logger.info(f"sample size: {self._sample_size}")
        logger.info(f"samples: {self._samples}")

    def mat_to_numpy(self):
        g = glob.glob(self._mat_path + "\*.mat", recursive=True)
        data = list()
        label = list()
        for i in g:
            if i.endswith(".mat"):
                logger.info(i.split("\\")[-1] + " is loading...")
                temp = scipy.io.loadmat(i)

                X = 2200 - temp["dataL"].shape[0]
                temp["dataL"] = np.pad(temp["dataL"], ((0, X), (0, 0)), "constant")
                X = 2200 - temp["dataR"].shape[0]
                temp["dataR"] = np.pad(temp["dataR"], ((0, X), (0, 0)), "constant")

                temp = np.concatenate((temp["dataL"], temp["dataR"]), axis=1)
                data.append(temp)
                label.append(i.split("\\")[-1])

        data = np.array(data)
        label = np.array(label)
        label = label[..., np.newaxis]
        np.save(self._data_path, data)
        np.save(self._meta_path, label)

    def loaddataset(self, dataset_name: str) -> np.ndarray:
        self.set_dataset_path(dataset_name)

        data = np.load(self._data_path, allow_pickle=True)
        if dataset_name == "stepscan":
            data = data[:-1]
        self._meta = np.load(self._meta_path, allow_pickle=True)
        self._sample_size = data.shape[1:]
        self._samples = data.shape[0]
        labels = self.extracting_labels(dataset_name)
        self.print_dataset()
        if data.shape[0] != labels.shape[0]:
            logger.error(
                f"The data and labels are not matched!! data shape ({data.shape[0]}) != labels shape {labels.shape[0]}"
            )
            sys.exit()
        return data, labels

class PreFeatures(Database):
    def __init__(self, dataset_name: str):
        super().__init__()
        self._test_id = int(timeit.default_timer() * 1_000_000)
        self._features_set = dict()
        self.set_dataset_path(dataset_name)

    @staticmethod
    def plot_COP(Footprint3D_1, nname):
        import matplotlib.animation as animation

        figure = plt.figure()

        gs = figure.add_gridspec(1, 1)
        ax1 = figure.add_subplot(gs[0, 0])

        (dot,) = plt.plot([0], [0], "ro")

        def func(i):
            ML = list()
            AP = list()
            for ii in range(Footprint3D_1.shape[2]):
                temp = Footprint3D_1[:, :, ii]
                temp2 = ndimage.measurements.center_of_mass(temp)
                ML.append(temp2[1])
                AP.append(temp2[0])

            lowpass = Butterworth.Butterworthfilter(
                mode="lowpass", fs=100, cutoff=5, order=4
            )
            ML = lowpass.filter(ML)
            AP = lowpass.filter(AP)

            # ml, ap = A.plot_COP(data[0])
            ax1.imshow(Footprint3D_1[..., i])
            ax1.axis("off")
            ax1.plot(ML, AP, "w")

            dot.set_data(ML[i], AP[i])
            return dot

        myani = animation.FuncAnimation(
            figure, func, frames=np.arange(0, 100, 1), interval=10
        )
        # plt.show()

        FFwriter = animation.PillowWriter(fps=30)
        myani.save(nname, writer=FFwriter)

    @staticmethod
    def plot_GRF(Footprint3D_1, Footprint3D_2):
        import matplotlib.animation as animation

        # breakpoint()
        figure = plt.figure()

        gs = figure.add_gridspec(2, 2)
        ax1 = figure.add_subplot(gs[0, 0])
        ax2 = figure.add_subplot(gs[0, 1])
        ax3 = figure.add_subplot(gs[1, :])

        (dot,) = plt.plot([0], [0], "ro")
        (dot2,) = plt.plot([0], [0], "r*")

        def func(i):
            GRF_1 = list()
            for ii in range(Footprint3D_1.shape[2]):
                temp = Footprint3D_1[:, :, ii].sum()
                GRF_1.append(temp)

            GRF_2 = list()
            for ii in range(Footprint3D_2.shape[2]):
                temp = Footprint3D_2[:, :, ii].sum()
                GRF_2.append(temp)

            dd_1 = Footprint3D_1[..., i]
            dd_2 = Footprint3D_2[..., i]

            ax1.imshow(np.rot90(dd_1, 3))
            ax1.axis("off")
            ax1.set_title("footprint subject 1")

            ax2.imshow(np.rot90(dd_2, 3))
            ax2.axis("off")
            ax2.set_title("footprint subject 2")

            ax3.plot(np.arange(0, 100, 1), GRF_1, "b", label="GRF of subject 1")
            ax3.plot(np.arange(0, 100, 1), GRF_2, "g", label="GRF of subject 2")
            ax3.set_xlabel("time")
            ax3.set_ylabel("Pressure")
            ax3.set_title("GRF")

            dot.set_data(i, GRF_1[i])
            dot2.set_data(i, GRF_2[i])
            return dot, dot2

        ax3.legend()
        myani = animation.FuncAnimation(
            figure, func, frames=np.arange(0, 100, 1), interval=10
        )
        plt.show()

        # FFwriter = animation.PillowWriter(fps=30)
        # myani.save('GRF_two_sub.gif', writer=FFwriter)

    def plot_preimages(self, Footprint3D_1, Footprint3D_2):
        figure, axs = plt.subplots(1, 2)
        breakpoint()
        imgs = self.pre_image(Footprint3D_1)
        imgs1 = self.pre_image(Footprint3D_2)
        x = ["CD", "PTI", "Tmin", "Tmax", "P50", "P60", "P70", "P80", "P90", "P100"]
        axs[0].imshow(imgs[..., 9])
        axs[0].axis("off")
        axs[0].set_title(f"{x[9]} for subject 1")

        axs[1].imshow(imgs1[..., 9])
        axs[1].axis("off")
        axs[1].set_title(f"{x[9]} for subject 2")
        # for i in range(5):
        #     axs[0,i].imshow(imgs[...,i])
        #     axs[0,i].axis('off')
        #     axs[0,i].set_title(f"{x[i]}")

        #     axs[1,i].imshow(imgs[...,i+5])
        #     axs[1,i].axis('off')
        #     axs[1,i].set_title(f"{x[i+5]}")

        plt.show()

    @staticmethod
    def plot_footprint3D(Footprint3D):
        plt.imshow(Footprint3D)
        plt.axis("off")
        plt.show()

    @staticmethod
    def computeCOPTimeSeries(Footprint3D):
        """
        computeCOPTimeSeries(Footprint3D)
        Footprint3D : [x,y,t] image
        return COPTS : RD, AP, ML COP time series
        """

        ML = list()
        AP = list()

        for i in range(Footprint3D.shape[2]):
            temp = Footprint3D[:, :, i]
            temp2 = ndimage.measurements.center_of_mass(temp)
            ML.append(temp2[1])
            AP.append(temp2[0])

        lowpass = Butterworth.Butterworthfilter(
            mode="lowpass", fs=100, cutoff=5, order=4
        )
        ML = lowpass.filter(ML)
        AP = lowpass.filter(AP)

        ML_f = ML - np.mean(ML)
        AP_f = AP - np.mean(AP)

        a = ML_f**2
        b = AP_f**2
        RD_f = np.sqrt(a + b)

        COPTS = np.stack((RD_f, AP_f, ML_f), axis=0)
        return COPTS

    @staticmethod
    def computeCOATimeSeries(Footprint3D, Binarize="otsu", Threshold=1):
        """
        computeCOATimeSeries(Footprint3D)
        Footprint3D : [x,y,t] image
        Binarize = 'simple', 'otsu', 'adaptive'
        Threshold = 1
        return COATS : RD, AP, ML COA time series
        """
        GS2BW_object = convertGS2BW.convertGS2BW(mode=Binarize, TH=Threshold)
        aML = list()
        aAP = list()
        for i in range(Footprint3D.shape[2]):
            temp = Footprint3D[:, :, i]

            BW, threshold = GS2BW_object.GS2BW(temp)

            temp3 = ndimage.measurements.center_of_mass(BW)

            aML.append(temp3[1])
            aAP.append(temp3[0])

        lowpass = Butterworth.Butterworthfilter(
            mode="lowpass", fs=100, cutoff=5, order=4
        )
        aML = lowpass.filter(aML)
        aAP = lowpass.filter(aAP)
        aML_f = aML - np.mean(aML)
        aAP_f = aAP - np.mean(aAP)

        a = aML_f**2
        b = aAP_f**2
        aRD_f = np.sqrt(a + b)

        COATS = np.stack((aRD_f, aAP_f, aML_f), axis=0)

        return COATS

    @staticmethod
    def computeGRF(Footprint3D):
        """
        computeGFR(Footprint3D)
        Footprint3D: [x,y,t] image
        return GFR: [t] time series signal
        """
        GRF = list()

        for frame in range(Footprint3D.shape[2]):
            temp = Footprint3D[:, :, frame].sum()
            GRF.append(temp)

        return np.array(GRF)

    @staticmethod
    def pre_image(Footprint3D, eps=5):
        """
        prefeatures(Footprint3D)
        Footprint3D: [x,y,t] image
        return pre_images: [x, y, 10] (CD, PTI, Tmin, Tmax, P50, P60, P70, P80, P90, P100)

        If The 30th percentile of a is 24.0: This means that 30% of values fall below 24.

        """

        temp = np.zeros(Footprint3D.shape)
        temp[Footprint3D > eps] = 1
        CD = np.sum(temp, axis=2)

        PTI = np.sum(Footprint3D, axis=2)

        Tmax = np.argmax(Footprint3D, axis=2)

        temp = Footprint3D.copy()
        temp[Footprint3D < eps] = 0
        x = np.ma.masked_array(temp, mask=temp == 0)
        Tmin = np.argmin(
            x,
            axis=2,
        )

        P50 = np.percentile(Footprint3D, 50, axis=2)
        P60 = np.percentile(Footprint3D, 60, axis=2)
        P70 = np.percentile(Footprint3D, 70, axis=2)
        P80 = np.percentile(Footprint3D, 80, axis=2)
        P90 = np.percentile(Footprint3D, 90, axis=2)
        P100 = np.percentile(Footprint3D, 100, axis=2)

        pre_images = np.stack(
            (CD, PTI, Tmin, Tmax, P50, P60, P70, P80, P90, P100), axis=-1
        )

        return pre_images

    def extracting_pre_features(
        self, dataset_name: str, combination: bool = True
    ) -> pd.DataFrame:
        self.set_dataset_path(dataset_name)
        data, labels = self.loaddataset(dataset_name)
        GRFs = list()
        COPs = list()
        COAs = list()
        pre_images = list()
        # i=0
        for sample, label in zip(data, labels):
            # logger.info(  i )
            # i = i+1

            if (
                combination == True
                and label[1] == 0
                and (dataset_name == "casia" or dataset_name == "casia-shod")
            ):
                sample = np.fliplr(sample)

            COA = self.computeCOATimeSeries(sample, Binarize="simple", Threshold=0)
            COA = COA.flatten()

            GRF = self.computeGRF(sample)

            COP = self.computeCOPTimeSeries(sample)
            COP = COP.flatten()

            pre_image = self.pre_image(sample)

            COPs.append(COP)
            COAs.append(COA)
            GRFs.append(GRF)
            pre_images.append(pre_image)

        GRFs = pd.DataFrame(
            np.array(GRFs),
            columns=["GRF_" + str(i) for i in range(np.array(GRFs).shape[1])],
        )
        COPs = pd.DataFrame(
            np.array(COPs),
            columns=["COP_" + str(i) for i in range(np.array(COPs).shape[1])],
        )
        COAs = pd.DataFrame(
            np.array(COAs),
            columns=["COA_" + str(i) for i in range(np.array(COAs).shape[1])],
        )
        pre_images = np.array(pre_images)

        self.saving_pre_features(GRFs, COPs, COAs, pre_images, labels, combination)
        return GRFs, COPs, COAs, pre_images, labels

    def saving_pre_features(
        self, GRFs, COPs, COAs, pre_images, labels, combination: bool = True
    ):
        pd.DataFrame(
            labels,
            columns=[
                "ID",
                "side",
            ],
        ).to_parquet(os.path.join(self._pre_features_path, f"label.parquet"))

        GRFs.to_parquet(
            os.path.join(self._pre_features_path, f"GRF_{combination}.parquet")
        )
        COAs.to_parquet(
            os.path.join(self._pre_features_path, f"COA_{combination}.parquet")
        )
        COPs.to_parquet(
            os.path.join(self._pre_features_path, f"COP_{combination}.parquet")
        )
        np.save(
            os.path.join(self._pre_features_path, f"pre_images_{combination}.npy"),
            pre_images,
        )

    def loading_pre_features_COP(
        self, dataset_name: str, combination: bool = True
    ) -> pd.DataFrame:
        self.set_dataset_path(dataset_name)
        try:
            labels = pd.read_parquet(
                os.path.join(self._pre_features_path, f"label.parquet")
            )
            COPs = pd.read_parquet(
                os.path.join(self._pre_features_path, f"COP_{combination}.parquet")
            )
            logger.info("COP curve were loaded!!!")

        except Exception as e:
            logger.error(e)
            logger.info("extraxting COP curve!!!")
            _, COPs, _, _, labels = self.extracting_pre_features(
                dataset_name, combination
            )

        self._features_set["COPs"] = {
            "columns": COPs.columns,
            "number_of_features": COPs.shape[1],
            "number_of_samples": COPs.shape[0],
        }

        return COPs, labels

    def loading_pre_features_COA(
        self, dataset_name: str, combination: bool = True
    ) -> pd.DataFrame:
        self.set_dataset_path(dataset_name)
        try:
            labels = pd.read_parquet(
                os.path.join(self._pre_features_path, f"label.parquet")
            )
            COAs = pd.read_parquet(
                os.path.join(self._pre_features_path, f"COA_{combination}.parquet")
            )
            logger.info("COA curve were loaded!!!")

        except Exception as e:
            logger.error(e)
            logger.info("extraxting COA curve!!!")
            _, _, COAs, _, labels = self.extracting_pre_features(
                dataset_name, combination
            )

        self._features_set["COAs"] = {
            "columns": COAs.columns,
            "number_of_features": COAs.shape[1],
            "number_of_samples": COAs.shape[0],
        }

        return COAs, labels

    def loading_pre_features_GRF(
        self, dataset_name: str, combination: bool = True
    ) -> pd.DataFrame:
        self.set_dataset_path(dataset_name)
        try:
            labels = pd.read_parquet(
                os.path.join(self._pre_features_path, f"label.parquet")
            )
            GRFs = pd.read_parquet(
                os.path.join(self._pre_features_path, f"GRF_{combination}.parquet")
            )
            logger.info("GRF curve were loaded!!!")

        except Exception as e:
            logger.error(e)
            logger.info("extraxting GRF curve!!!")
            GRFs, _, _, _, labels = self.extracting_pre_features(
                dataset_name, combination
            )

        self._features_set["GRFs"] = {
            "columns": GRFs.columns,
            "number_of_features": GRFs.shape[1],
            "number_of_samples": GRFs.shape[0],
        }

        return GRFs, labels

    def loading_pre_features_image(
        self, dataset_name: str, combination: bool = True
    ) -> pd.DataFrame:
        self.set_dataset_path(dataset_name)
        try:
            labels = pd.read_parquet(
                os.path.join(self._pre_features_path, f"label.parquet")
            )
            pre_images = np.load(
                os.path.join(self._pre_features_path, f"pre_images_{combination}.npy")
            )
            logger.info("image features were loaded!!!")
        except Exception as e:
            logger.error(e)
            logger.info("extraxting image features!!!")
            _, _, _, pre_images, labels = self.extracting_pre_features(
                dataset_name, combination
            )
        return pre_images, labels

    def loading_pre_features(
        self, dataset_name: str, combination: bool = True
    ) -> pd.DataFrame:
        self.set_dataset_path(dataset_name)
        try:
            labels = pd.read_parquet(
                os.path.join(self._pre_features_path, f"label.parquet")
            )
            GRFs = pd.read_parquet(
                os.path.join(self._pre_features_path, f"GRF_{combination}.parquet")
            )
            COAs = pd.read_parquet(
                os.path.join(self._pre_features_path, f"COA_{combination}.parquet")
            )
            COPs = pd.read_parquet(
                os.path.join(self._pre_features_path, f"COP_{combination}.parquet")
            )
            pre_images = np.load(
                os.path.join(self._pre_features_path, f"pre_images_{combination}.npy")
            )
            logger.info(" all pre features were loaded!!!")

        except Exception as e:
            logger.error(e)
            logger.info("extraxting pre features!!!")
            GRFs, COPs, COAs, pre_images, labels = self.extracting_pre_features(
                dataset_name, combination
            )

        self._features_set["GRFs"] = {
            "columns": GRFs.columns,
            "number_of_features": GRFs.shape[1],
            "number_of_samples": GRFs.shape[0],
        }

        self._features_set["COAs"] = {
            "columns": COAs.columns,
            "number_of_features": COAs.shape[1],
            "number_of_samples": COAs.shape[0],
        }

        self._features_set["COPs"] = {
            "columns": COPs.columns,
            "number_of_features": COPs.shape[1],
            "number_of_samples": COPs.shape[0],
        }
        # self._CNN_image_size = pre_images.shape
        return GRFs, COPs, COAs, pre_images, labels


