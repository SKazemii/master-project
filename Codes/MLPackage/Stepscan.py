import re
from unicodedata import name
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


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.getcwd(), "Codes", "MLPackage"))
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
    t1 = (
        blue
        + "[%(asctime)s]-"
        + yellow
        + "[%(name)s @%(lineno)d]"
        + reset
        + blue
        + "-[%(levelname)s]"
        + reset
        + bold_red
    )
    t2 = " %(message)s" + reset
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


class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="bac", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = tf.keras.metrics.TruePositives()
        self.tn = tf.keras.metrics.TrueNegatives()
        self.fp = tf.keras.metrics.FalsePositives()
        self.fn = tf.keras.metrics.FalseNegatives()
        self.bac = self.add_weight(name="bac", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        tp = self.tp(y_true, y_pred)
        tn = self.tn(y_true, y_pred)
        fp = self.fp(y_true, y_pred)
        fn = self.fn(y_true, y_pred)
        spec = (tn) / (fp + tn + 1e-6)
        sen = (tp) / (tp + fn + 1e-6)
        self.bac.assign((sen + spec) / 2)

    def result(self):
        return self.bac

    def reset_state(self):
        self.tp.reset_states()
        self.tn.reset_states()
        self.fp.reset_states()
        self.fn.reset_states()
        self.bac.assign(0)


class Database(object):
    def __init__(
        self,
    ):
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


class Features(PreFeatures):
    COX_feature_name = [
        "MDIST_RD",
        "MDIST_AP",
        "MDIST_ML",
        "RDIST_RD",
        "RDIST_AP",
        "RDIST_ML",
        "TOTEX_RD",
        "TOTEX_AP",
        "TOTEX_ML",
        "MVELO_RD",
        "MVELO_AP",
        "MVELO_ML",
        "RANGE_RD",
        "RANGE_AP",
        "RANGE_ML",
        "AREA_CC",
        "AREA_CE",
        "AREA_SW",
        "MFREQ_RD",
        "MFREQ_AP",
        "MFREQ_ML",
        "FDPD_RD",
        "FDPD_AP",
        "FDPD_ML",
        "FDCC",
        "FDCE",
    ]

    GRF_feature_name = [
        "max_value_1",
        "max_value_1_ind",
        "max_value_2",
        "max_value_2_ind",
        "min_value",
        "min_value_ind",
        "mean_value",
        "std_value",
        "sum_value",
    ]

    _pre_image_names = [
        "CD",
        "PTI",
        "Tmin",
        "Tmax",
        "P50",
        "P60",
        "P70",
        "P80",
        "P90",
        "P100",
    ]

    def __init__(
        self,
        dataset_name: str,
        combination: bool = True,
        waveletname: str = "coif1",
        pywt_mode: str = "constant",
        wavelet_level: int = 4,
    ):
        super().__init__(dataset_name)
        self._waveletname = waveletname
        self._pywt_mode = pywt_mode
        self._wavelet_level = wavelet_level
        self.dataset_name = dataset_name
        self._combination = combination

    @staticmethod
    def computeMDIST(COPTS):
        """
        computeMDIST(COPTS)
        MDIST : Mean Distance
        COPTS [3,t] : RD, AP, ML COP time series
        return MDIST [3] : [MDIST_RD, MDIST_AP, MDIST_ML]
        """

        MDIST = np.mean(np.abs(COPTS), axis=1)

        return MDIST

    @staticmethod
    def computeRDIST(COPTS):
        """
        computeRDIST(COPTS)
        RDIST : RMS Distance
        COPTS [3,t] : RD, AP, ML COP time series
        return RDIST [3] : [RDIST_RD, RDIST_AP, RDIST_ML]
        """
        RDIST = np.sqrt(np.mean(COPTS**2, axis=1))

        return RDIST

    @staticmethod
    def computeTOTEX(COPTS):
        """
        computeTOTEX(COPTS)
        TOTEX : Total Excursions
        COPTS [3,t] : RD, AP, ML COP time series
        return TOTEX [3] : TOTEX_RD, TOTEX_AP, TOTEX_ML
        """

        TOTEX = list()
        TOTEX.append(
            np.sum(np.sqrt((np.diff(COPTS[2, :]) ** 2) + (np.diff(COPTS[1, :]) ** 2)))
        )
        TOTEX.append(np.sum(np.abs(np.diff(COPTS[1, :]))))
        TOTEX.append(np.sum(np.abs(np.diff(COPTS[2, :]))))

        return TOTEX

    @staticmethod
    def computeRANGE(COPTS):
        """
        computeRANGE(COPTS)
        RANGE : Range
        COPTS [3,t] : RD, AP, ML COP time series
        return RANGE [3] : RANGE_RD, RANGE_AP, RANGE_ML
        """
        RANGE = list()
        # print(cdist(COPTS[1:2,:].T, COPTS[1:2,:].T).shape)
        # print(cdist(COPTS[1:2,:].T, COPTS[1:2,:].T))
        # sys.exit()
        RANGE.append(np.max(distance.cdist(COPTS[1:2, :].T, COPTS[1:2, :].T)))
        RANGE.append(np.max(COPTS[1, :]) - np.min(COPTS[1, :]))
        RANGE.append(np.max(COPTS[2, :]) - np.min(COPTS[2, :]))

        return RANGE

    @staticmethod
    def computeMVELO(COPTS, T=1):
        """
        computeMVELO(COPTS,varargin)
        MVELO : Mean Velocity
        COPTS [3,t] : RD, AP, ML COP time series
        T : the period of time selected for analysis (CASIA-D = 1s)
        return MVELO [3] : MVELO_RD, MVELO_AP, MVELO_ML
        """

        MVELO = list()
        MVELO.append(
            (np.sum(np.sqrt((np.diff(COPTS[2, :]) ** 2) + (np.diff(COPTS[1, :]) ** 2))))
            / T
        )
        MVELO.append((np.sum(np.abs(np.diff(COPTS[1, :])))) / T)
        MVELO.append((np.sum(np.abs(np.diff(COPTS[2, :])))) / T)

        return MVELO

    def computeAREACC(self, COPTS):
        """
        computeAREACC(COPTS)
        AREA-CC : 95% Confidence Circle Area
        COPTS [3,t] : RD (AP, ML) COP time series
        return AREACC [1] : AREA-CC
        """

        MDIST = self.computeMDIST(COPTS)
        RDIST = self.computeRDIST(COPTS)
        z05 = 1.645  # z.05 = the z statistic at the 95% confidence level
        SRD = np.sqrt(
            (RDIST[0] ** 2) - (MDIST[0] ** 2)
        )  # the standard deviation of the RD time series

        AREACC = np.pi * ((MDIST[0] + (z05 * SRD)) ** 2)
        return AREACC

    def computeAREACE(self, COPTS):
        """
        computeAREACE(COPTS)
        AREA-CE : 95% Confidence Ellipse Area
        COPTS [3,t] : (RD,) AP, ML COP time series
        return AREACE [1] : AREA-CE
        """

        F05 = 3
        RDIST = self.computeRDIST(COPTS)
        SAP = RDIST[1]
        SML = RDIST[2]
        SAPML = np.mean(COPTS[2, :] * COPTS[1, :])
        AREACE = 2 * np.pi * F05 * np.sqrt((SAP**2) * (SML**2) - (SAPML**2))

        return AREACE

    @staticmethod
    def computeAREASW(COPTS, T=1):
        """
        computeAREASW(COPTS, T)
        AREA-SW : Sway area
        COPTS [3,t] : RD, AP, ML COP time series
        T : the period of time selected for analysis (CASIA-D = 1s)
        return AREASW [1] : AREA-SW
        """

        AP = COPTS[1, :]
        ML = COPTS[2, :]

        AREASW = np.sum(np.abs((AP[1:] * ML[:-1]) - (AP[:-1] * ML[1:]))) / (2 * T)

        return AREASW

    def computeMFREQ(self, COPTS, T=1):
        """
        computeMFREQ(COPTS, T)
        MFREQ : Mean Frequency
        COPTS [3,t] : RD, AP, ML COP time series
        T : the period of time selected for analysis (CASIA-D = 1s)
        return MFREQ [3] : MFREQ_RD, MFREQ_AP, MFREQ_ML
        """

        TOTEX = self.computeTOTEX(COPTS)
        MDIST = self.computeMDIST(COPTS)

        MFREQ = list()
        MFREQ.append(TOTEX[0] / (2 * np.pi * T * MDIST[0]))
        MFREQ.append(TOTEX[1] / (4 * np.sqrt(2) * T * MDIST[1]))
        MFREQ.append(TOTEX[2] / (4 * np.sqrt(2) * T * MDIST[2]))

        return MFREQ

    def computeFDPD(self, COPTS):
        """
        computeFDPD(COPTS)
        FD-PD : Fractal Dimension based on the Plantar Diameter of the Curve
        COPTS [3,t] : RD, AP, ML COP time series
        return FDPD [3] : FD-PD_RD, FD-PD_AP, FD-PD_ML
        """

        N = COPTS.shape[1]
        TOTEX = self.computeTOTEX(COPTS)
        d = self.computeRANGE(COPTS)
        Nd = [elemt * N for elemt in d]
        dev = [i / j for i, j in zip(Nd, TOTEX)]

        FDPD = np.log(N) / np.log(dev)
        # sys.exit()
        return FDPD

    def computeFDCC(self, COPTS):
        """
        computeFDCC(COPTS)
        FD-CC : Fractal Dimension based on the 95% Confidence Circle
        COPTS [3,t] : RD, (AP, ML) COP time series
        return FDCC [1] : FD-CC_RD
        """

        N = COPTS.shape[1]
        MDIST = self.computeMDIST(COPTS)
        RDIST = self.computeRDIST(COPTS)
        z05 = 1.645
        # z.05 = the z statistic at the 95% confidence level
        SRD = np.sqrt(
            (RDIST[0] ** 2) - (MDIST[0] ** 2)
        )  # the standard deviation of the RD time series

        d = 2 * (MDIST[0] + z05 * SRD)
        TOTEX = self.computeTOTEX(COPTS)

        FDCC = np.log(N) / np.log((N * d) / TOTEX[0])
        return FDCC

    def computeFDCE(self, COPTS):
        """
        computeFDCE(COPTS)
        FD-CE : Fractal Dimension based on the 95% Confidence Ellipse
        COPTS [3,t] : (RD,) AP, ML COP time series
        return FDCE [2] : FD-CE_AP, FD-CE_ML
        """

        N = COPTS.shape[1]
        F05 = 3
        RDIST = self.computeRDIST(COPTS)
        SAPML = np.mean(COPTS[2, :] * COPTS[1, :])

        d = np.sqrt(
            8 * F05 * np.sqrt(((RDIST[1] ** 2) * (RDIST[2] ** 2)) - (SAPML**2))
        )
        TOTEX = self.computeTOTEX(COPTS)

        FDCE = np.log(N) / np.log((N * d) / TOTEX[0])

        return FDCE

    @staticmethod
    def computeGRFfeatures(GRF):
        """
        computeGRFfeatures(GRF)
        GRF: [t] time series signal
        return GFR features: [9] (max_value_1, max_value_1_ind, max_value_2, max_value_2_ind, min_value, min_value_ind, mean_value, std_value, sum_value)
        """
        # handcraft_features = list()
        L = int(len(GRF) / 2)

        max_value_1 = np.max(GRF[:L])
        max_value_1_ind = np.argmax(GRF[:L])
        max_value_2 = np.max(GRF[L:])
        max_value_2_ind = L + np.argmax(GRF[L:])

        min_value = np.min(GRF[max_value_1_ind:max_value_2_ind])
        min_value_ind = max_value_1_ind + np.argmin(
            GRF[max_value_1_ind:max_value_2_ind]
        )

        mean_value = np.mean(GRF)
        std_value = np.std(GRF)
        sum_value = np.sum(GRF)

        return [
            max_value_1,
            max_value_1_ind,
            max_value_2,
            max_value_2_ind,
            min_value,
            min_value_ind,
            mean_value,
            std_value,
            sum_value,
        ]

    def wt_feature(self, signal):
        """
        wt_feature(signal, waveletname, pywt_mode, wavelet_level)
        signal: [t] time series signal
        wavelet_level = 4 or pywt.dwt_max_level(100, waveletname)
        pywt_mode = "constant"
        waveletname = "coif1"
            haar family: haar
            db family: db1, db2, db3, db4, db5, db6, db7, db8, db9, db10, db11, db12, db13, db14, db15, db16, db17, db18, db19, db20, db21, db22, db23, db24, db25, db26, db27, db28, db29, db30, db31, db32, db33, db34, db35, db36, db37, db38
            sym family: sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9, sym10, sym11, sym12, sym13, sym14, sym15, sym16, sym17, sym18, sym19, sym20
            coif family: coif1, coif2, coif3, coif4, coif5, coif6, coif7, coif8, coif9, coif10, coif11, coif12, coif13, coif14, coif15, coif16, coif17
            bior family: bior1.1, bior1.3, bior1.5, bior2.2, bior2.4, bior2.6, bior2.8, bior3.1, bior3.3, bior3.5, bior3.7, bior3.9, bior4.4, bior5.5, bior6.8
            rbio family: rbio1.1, rbio1.3, rbio1.5, rbio2.2, rbio2.4, rbio2.6, rbio2.8, rbio3.1, rbio3.3, rbio3.5, rbio3.7, rbio3.9, rbio4.4, rbio5.5, rbio6.8
            dmey family: dmey
            gaus family: gaus1, gaus2, gaus3, gaus4, gaus5, gaus6, gaus7, gaus8
            mexh family: mexh
            morl family: morl
            cgau family: cgau1, cgau2, cgau3, cgau4, cgau5, cgau6, cgau7, cgau8
            shan family: shan
            fbsp family: fbsp
            cmor family: cmor


        return dwt_coeff: Discrete Wavelet Transform coeff
        """

        dwt_coeff = pywt.wavedec(
            signal, self._waveletname, mode=self._pywt_mode, level=self._wavelet_level
        )
        dwt_coeff = np.concatenate(dwt_coeff).ravel()

        return dwt_coeff

    ## COA
    def extraxting_COA_handcrafted(self, COAs: np.ndarray) -> pd.DataFrame:
        COA_handcrafted = list()
        for _, sample in COAs.iterrows():
            sample = sample.values.reshape(3, 100)

            MDIST = self.computeMDIST(sample)
            RDIST = self.computeRDIST(sample)
            TOTEX = self.computeTOTEX(sample)
            MVELO = self.computeMVELO(sample)
            RANGE = self.computeRANGE(sample)
            AREACC = self.computeAREACC(sample)
            AREACE = self.computeAREACE(sample)
            AREASW = self.computeAREASW(sample)
            MFREQ = self.computeMFREQ(sample)
            FDPD = self.computeFDPD(sample)
            FDCC = self.computeFDCC(sample)
            FDCE = self.computeFDCE(sample)

            COA_handcrafted.append(
                np.concatenate(
                    (
                        MDIST,
                        RDIST,
                        TOTEX,
                        MVELO,
                        RANGE,
                        [AREACC],
                        [AREACE],
                        [AREASW],
                        MFREQ,
                        FDPD,
                        [FDCC],
                        [FDCE],
                    ),
                    axis=0,
                )
            )

        COA_handcrafted = pd.DataFrame(
            np.array(COA_handcrafted), columns=self.COX_feature_name
        )

        self.saving_dataframe(COA_handcrafted, "COA_handcrafted")

        return COA_handcrafted

    def saving_dataframe(self, data: pd.DataFrame, name: str) -> None:
        Pathlb(self._features_path).mkdir(parents=True, exist_ok=True)
        data.to_parquet(
            os.path.join(self._features_path, f"{name}_{self._combination}.parquet")
        )

    def loading_COA_handcrafted(self, COAs: np.ndarray) -> pd.DataFrame:
        try:
            COA_handcrafted = pd.read_parquet(
                os.path.join(
                    self._features_path, f"COA_handcrafted_{self._combination}.parquet"
                )
            )
            logger.info("loading COA handcrafted features!!!")

        except Exception as e:
            logger.error(e)
            logger.info("extraxting COA handcrafted features!!!")
            COA_handcrafted = self.extraxting_COA_handcrafted(COAs)

        self._features_set["COA_handcrafted"] = {
            "columns": COA_handcrafted.columns,
            "number_of_features": COA_handcrafted.shape[1],
            "number_of_samples": COA_handcrafted.shape[0],
        }
        return COA_handcrafted

    ## COP
    def extraxting_COP_handcrafted(self, COPs: np.ndarray) -> pd.DataFrame:
        COP_handcrafted = list()
        for _, sample in COPs.iterrows():
            sample = sample.values.reshape(3, 100)

            MDIST = self.computeMDIST(sample)
            RDIST = self.computeRDIST(sample)
            TOTEX = self.computeTOTEX(sample)
            MVELO = self.computeMVELO(sample)
            RANGE = self.computeRANGE(sample)
            AREACC = self.computeAREACC(sample)
            AREACE = self.computeAREACE(sample)
            AREASW = self.computeAREASW(sample)
            MFREQ = self.computeMFREQ(sample)
            FDPD = self.computeFDPD(sample)
            FDCC = self.computeFDCC(sample)
            FDCE = self.computeFDCE(sample)

            COP_handcrafted.append(
                np.concatenate(
                    (
                        MDIST,
                        RDIST,
                        TOTEX,
                        MVELO,
                        RANGE,
                        [AREACC],
                        [AREACE],
                        [AREASW],
                        MFREQ,
                        FDPD,
                        [FDCC],
                        [FDCE],
                    ),
                    axis=0,
                )
            )

        COP_handcrafted = pd.DataFrame(
            np.array(COP_handcrafted), columns=self.COX_feature_name
        )
        self.saving_dataframe(COP_handcrafted, "COP_handcrafted")
        return COP_handcrafted

    def loading_COP_handcrafted(self, COPs: np.ndarray) -> pd.DataFrame:
        try:
            COP_handcrafted = pd.read_parquet(
                os.path.join(
                    self._features_path, f"COP_handcrafted_{self._combination}.parquet"
                )
            )
            logger.info("loading COP handcrafted features!!!")

        except Exception as e:
            logger.error(e)
            logger.info("extraxting COP handcrafted features!!!")
            COP_handcrafted = self.extraxting_COP_handcrafted(COPs)

        self._features_set["COP_handcrafted"] = {
            "columns": COP_handcrafted.columns,
            "number_of_features": COP_handcrafted.shape[1],
            "number_of_samples": COP_handcrafted.shape[0],
        }
        return COP_handcrafted

    ## GRF
    def extraxting_GRF_handcrafted(self, GRFs: np.ndarray) -> pd.DataFrame:
        GRF_handcrafted = list()
        for _, sample in GRFs.iterrows():
            GRF_handcrafted.append(self.computeGRFfeatures(sample))

        GRF_handcrafted = pd.DataFrame(
            np.array(GRF_handcrafted), columns=self.GRF_feature_name
        )
        self.saving_dataframe(GRF_handcrafted, "GRF_handcrafted")
        return GRF_handcrafted

    def loading_GRF_handcrafted(self, GRFs: np.ndarray) -> pd.DataFrame:
        try:
            GRF_handcrafted = pd.read_parquet(
                os.path.join(
                    self._features_path, f"GRF_handcrafted_{self._combination}.parquet"
                )
            )
            logger.info("loading GRF handcrafted features!!!")

        except Exception as e:
            logger.error(e)
            logger.info("extraxting GRF handcrafted features!!!")
            GRF_handcrafted = self.extraxting_GRF_handcrafted(GRFs)

        self._features_set["GRF_handcrafted"] = {
            "columns": GRF_handcrafted.columns,
            "number_of_features": GRF_handcrafted.shape[1],
            "number_of_samples": GRF_handcrafted.shape[0],
        }
        return GRF_handcrafted

    ## GRF WPT
    def extraxting_GRF_WPT(self, GRFs: np.ndarray) -> pd.DataFrame:
        GRF_WPT = list()
        for _, sample in GRFs.iterrows():
            GRF_WPT.append(self.wt_feature(sample))

        GRF_WPT = pd.DataFrame(
            np.array(GRF_WPT),
            columns=["GRF_WPT_" + str(i) for i in range(np.array(GRF_WPT).shape[1])],
        )
        self.saving_dataframe(GRF_WPT, "GRF_WPT")

        return GRF_WPT

    def loading_GRF_WPT(self, GRFs: np.ndarray) -> pd.DataFrame:
        try:
            GRF_WPT = pd.read_parquet(
                os.path.join(
                    self._features_path, f"GRF_WPT_{self._combination}.parquet"
                )
            )
            logger.info("loading GRF WPT features!!!")

        except Exception as e:
            logger.error(e)
            logger.info("extraxting GRF WPT features!!!")
            GRF_WPT = self.extraxting_GRF_WPT(GRFs)

        self._features_set["GRF_WPT"] = {
            "columns": GRF_WPT.columns,
            "number_of_features": GRF_WPT.shape[1],
            "number_of_samples": GRF_WPT.shape[0],
        }
        return GRF_WPT

    ## COP WPT
    def extraxting_COP_WPT(self, COPs: np.ndarray) -> pd.DataFrame:
        COP_WPT = list()
        for _, sample in COPs.iterrows():
            sample = sample.values.reshape(3, 100)
            wt_COA_RD = self.wt_feature(sample[0, :])
            wt_COA_AP = self.wt_feature(sample[1, :])
            wt_COA_ML = self.wt_feature(sample[2, :])
            COP_WPT.append(np.concatenate((wt_COA_RD, wt_COA_AP, wt_COA_ML), axis=0))

        COP_WPT = pd.DataFrame(
            np.array(COP_WPT),
            columns=["COP_WPT_" + str(i) for i in range(np.array(COP_WPT).shape[1])],
        )
        self.saving_dataframe(COP_WPT, "COP_WPT")
        return COP_WPT

    def loading_COP_WPT(self, COPs: np.ndarray) -> pd.DataFrame:
        try:
            COP_WPT = pd.read_parquet(
                os.path.join(
                    self._features_path, f"COP_WPT_{self._combination}.parquet"
                )
            )
            logger.info("loading COP WPT features!!!")

        except Exception as e:
            logger.error(e)
            logger.info("extraxting COP WPT features!!!")
            COP_WPT = self.extraxting_COP_WPT(COPs)

        self._features_set["COP_WPT"] = {
            "columns": COP_WPT.columns,
            "number_of_features": COP_WPT.shape[1],
            "number_of_samples": COP_WPT.shape[0],
        }
        return COP_WPT

    ## COA WPT
    def extraxting_COA_WPT(self, COAs: np.ndarray) -> pd.DataFrame:
        COA_WPT = list()
        for _, sample in COAs.iterrows():
            sample = sample.values.reshape(3, 100)
            wt_COA_RD = self.wt_feature(sample[0, :])
            wt_COA_AP = self.wt_feature(sample[1, :])
            wt_COA_ML = self.wt_feature(sample[2, :])
            COA_WPT.append(np.concatenate((wt_COA_RD, wt_COA_AP, wt_COA_ML), axis=0))

        COA_WPT = pd.DataFrame(
            np.array(COA_WPT),
            columns=["COA_WPT_" + str(i) for i in range(np.array(COA_WPT).shape[1])],
        )
        self.saving_dataframe(COA_WPT, "COA_WPT")
        return COA_WPT

    def loading_COA_WPT(self, COAs: np.ndarray) -> pd.DataFrame:
        try:
            COA_WPT = pd.read_parquet(
                os.path.join(
                    self._features_path, f"COA_WPT_{self._combination}.parquet"
                )
            )
            logger.info("loading COA WPT features!!!")

        except Exception as e:
            logger.error(e)
            logger.info("extraxting COA WPT features!!!")
            COA_WPT = self.extraxting_COA_WPT(COAs)

        self._features_set["COA_WPT"] = {
            "columns": COA_WPT.columns,
            "number_of_features": COA_WPT.shape[1],
            "number_of_samples": COA_WPT.shape[0],
        }
        return COA_WPT

    ## deep
    @staticmethod
    def resize_images(images, labels):
        # breakpoint()
        if len(images.shape) < 4:
            images = tf.expand_dims(images, -1)

        images = tf.image.grayscale_to_rgb(images)
        images = tf.image.resize(images, (224, 224))
        return images, labels

    def extraxting_deep_features(
        self, data: tuple, pre_image_name: str, CNN_base_model: str
    ) -> pd.DataFrame:
        # self._CNN_base_model = CNN_base_model

        try:
            logger.info(f"Loading { CNN_base_model } model...")
            base_model = eval(
                f"tf.keras.applications.{CNN_base_model}(weights='{self._CNN_weights}', include_top={self._CNN_include_top})"
            )
            logger.info("Successfully loaded base model and model...")
            base_model.trainable = False
            CNN_name = CNN_base_model.split(".")[0]
            logger.info(f"MaduleName: {CNN_name}")

        except Exception as e:
            base_model = None
            logger.error("The base model could NOT be loaded correctly!!!")
            logger.error(e)

        pre_image_norm = self.normalizing_pre_image(data[0], pre_image_name)

        train_ds = tf.data.Dataset.from_tensor_slices((pre_image_norm, data[1]))
        train_ds = train_ds.batch(self._CNN_batch_size)
        logger.info(f"batch_size: {self._CNN_batch_size}")
        train_ds = train_ds.map(self.resize_images)

        input = tf.keras.layers.Input(
            shape=(224, 224, 3), dtype=tf.float64, name="original_img"
        )
        x = tf.cast(input, tf.float32)
        x = eval("tf.keras.applications." + CNN_name + ".preprocess_input(x)")
        x = base_model(x)
        output = tf.keras.layers.GlobalMaxPool2D()(x)

        model = tf.keras.Model(input, output, name=CNN_name)

        # if self._verbose==True:
        #     model.summary()
        #     tf.keras.utils.plot_model(model, to_file=CNN_name + ".png", show_shapes=True)

        Deep_features = np.zeros((1, model.layers[-1].output_shape[1]))
        for image_batch, _ in train_ds:

            feature = model(image_batch)
            Deep_features = np.append(Deep_features, feature, axis=0)

            if (Deep_features.shape[0] - 1) % 256 == 0:
                logger.info(
                    f" ->>> ({os.getpid()}) completed images: "
                    + str(Deep_features.shape[0])
                )

        Deep_features = Deep_features[1:, :]
        logger.info(f"Deep features shape: {Deep_features.shape}")

        # time = int(timeit.default_timer() * 1_000_000)
        exec(
            f"deep_{pre_image_name}_{CNN_name} = pd.DataFrame(Deep_features, columns=['deep_{pre_image_name}_{CNN_name}_'+str(i) for i in range(Deep_features.shape[1])])"
        )

        self.saving_deep_features(
            eval(f"deep_{pre_image_name}_{CNN_name}"),
            f"deep_{pre_image_name}_{CNN_name}_{self._combination}",
        )

        return eval(f"deep_{pre_image_name}_{CNN_name}")

    def normalizing_pre_image(
        self, pre_images: np.ndarray, pre_image_name: str
    ) -> np.ndarray:
        if not pre_image_name in self._pre_image_names:
            raise Exception("Invalid pre image name!!!")

        i = self._pre_image_names.index(pre_image_name)
        maxvalues = np.max(pre_images[..., i])
        return pre_images[..., i] / maxvalues

    def saving_deep_features(self, data: pd.DataFrame, name: str) -> None:
        Pathlb(self._features_path).mkdir(parents=True, exist_ok=True)
        # exec(f"data.to_parquet(os.path.join(self._features_path, f'deep_{pre_image_name}_{CNN_name}_{self._combination}.parquet'))")
        exec(f"data.to_parquet(os.path.join(self._features_path, f'{name}.parquet'))")

    def loading_deep_features(
        self, data: tuple, pre_image_name: str, CNN_base_model: str
    ) -> pd.DataFrame:
        CNN_name = CNN_base_model.split(".")[0]
        if not pre_image_name in self._pre_image_names:
            raise Exception("Invalid pre image name!!!")
        try:
            logger.info(f"loading deep features from {pre_image_name}!!!")
            exec(
                f"df = pd.read_parquet(os.path.join(self._features_path, f'deep_{pre_image_name}_{CNN_name}_{self._combination}.parquet'))"
            )

        except Exception as e:
            logger.error(e)
            logger.info(f"extraxting deep features from {pre_image_name}!!!")
            exec(
                f"df = self.extraxting_deep_features(data, pre_image_name, CNN_base_model)"
            )

        self._features_set[f"deep_{pre_image_name}_{CNN_name}"] = {
            "columns": eval("df.columns"),
            "number_of_features": eval("df.shape[1]"),
            "number_of_samples": eval("df.shape[0]"),
        }
        return eval("df")

    def loading_deep_features_from_list(
        self, data: tuple, pre_image_names: list, CNN_base_model: str
    ) -> pd.DataFrame:
        """loading deep features from a list of image features"""
        CNN_name = CNN_base_model.split(".")[0]
        sss = []
        for pre_image_name in pre_image_names:
            if not pre_image_name in self._pre_image_names:
                raise Exception("Invalid pre image name!!!")
            try:
                exec(
                    f"{pre_image_name} = pd.read_parquet(os.path.join(self._features_path, f'deep_{pre_image_name}_{CNN_name}_{self._combination}.parquet'))"
                )
                logger.info(f"loading deep features from {pre_image_name}!!!")

            except Exception as e:
                logger.error(e)
                logger.info(f"extraxting deep features  from {pre_image_name}!!!")
                exec(
                    f"{pre_image_name} = self.extraxting_deep_features(data, pre_image_name, CNN_base_model)"
                )

            self._features_set[f"deep_{pre_image_name}_{CNN_name}"] = {
                "columns": eval(f"{pre_image_name}.columns"),
                "number_of_features": eval(f"{pre_image_name}.shape[1]"),
                "number_of_samples": eval(f"{pre_image_name}.shape[0]"),
            }

            sss.append(eval(f"{pre_image_name}"))
        return sss

    ## images
    def extraxting_pre_image(
        self, pre_images: np.ndarray, pre_image_name: str
    ) -> pd.DataFrame:
        if not pre_image_name in self._pre_image_names:
            raise Exception("Invalid pre image name!!!")

        pre_image = list()
        for idx in range(pre_images.shape[0]):

            sample = pre_images[idx, ..., self._pre_image_names.index(pre_image_name)]
            sample = sample.reshape(-1)
            pre_image.append(sample)

        exec(
            f"I = pd.DataFrame(np.array(pre_image), columns=['{pre_image_name}_pixel_'+str(i) for i in range(np.array(pre_image).shape[1])]) "
        )
        exec(f"self.saving_pre_image(I, '{pre_image_name}')")

        return eval("I")

    def saving_pre_image(self, data, pre_image_name: str) -> None:
        Pathlb(self._features_path).mkdir(parents=True, exist_ok=True)
        exec(
            f"data.to_parquet(os.path.join(self._features_path, '{pre_image_name}_{self._combination}.parquet'))"
        )

    def loading_pre_image(
        self, pre_images: np.ndarray, pre_image_name: str
    ) -> pd.DataFrame:
        """loading a pre image from a excel file."""
        if not pre_image_name in self._pre_image_names:
            raise Exception("Invalid pre image name!!!")
        try:
            exec(
                f"I = pd.read_parquet(os.path.join(self._features_path, '{pre_image_name}_{self._combination}.parquet'))"
            )
            logger.info(f"loading {pre_image_name} features!!!")

        except Exception as e:
            logger.error(e)
            logger.info(f"extraxting {pre_image_name} features!!!")
            exec(f"I = self.extraxting_pre_image(pre_images, pre_image_name)")

        self._features_set[f"{pre_image_name}"] = {
            "columns": eval(f"I.columns"),
            "number_of_features": eval(f"I.shape[1]"),
            "number_of_samples": eval(f"I.shape[0]"),
        }
        return eval("I")

    def loading_pre_image_from_list(
        self, pre_images: np.ndarray, list_pre_image: list
    ) -> list:
        """loading multiple pre image features from list"""
        sss = []
        for pre_image_name in list_pre_image:
            if not pre_image_name in self._pre_image_names:
                raise Exception("Invalid pre image name!!!")

            try:
                exec(
                    f"{pre_image_name} = pd.read_parquet(os.path.join(self._features_path, '{pre_image_name}_{self._combination}.parquet'))"
                )
                logger.info(f"loading {pre_image_name} features!!!")

            except Exception as e:
                logger.error(e)
                logger.info(f"extraxting {pre_image_name} features!!!")
                exec(
                    f"{pre_image_name} = self.extraxting_pre_image(pre_images, pre_image_name)"
                )

            self._features_set[f"{pre_image_name}"] = {
                "columns": eval(f"{pre_image_name}.columns"),
                "number_of_features": eval(f"{pre_image_name}.shape[1]"),
                "number_of_samples": eval(f"{pre_image_name}.shape[0]"),
            }

            sss.append(eval(f"{pre_image_name}"))
        return sss

    ## trained models
    def normalizing_pre_image_1(self, pre_images: np.ndarray) -> np.ndarray:
        assert len(pre_images.shape) == 4, "the shape of image feature is not correct"
        ww = []
        for i in range(pre_images.shape[3]):
            maxvalues = np.max(pre_images[..., i])
            ww.append(pre_images[..., i] / maxvalues)
        return np.transpose(np.array(ww), (1, 2, 3, 0))

    def extracting_deep_feature_from_model(
        self,
        model: tf.keras.models.Model,
        feature_layer_name: str,
        data: tuple,
        pre_image_name: str,
    ) -> pd.DataFrame:
        try:
            logger.info(f"Loading { model.name } model...")
            logger.info("Successfully loaded base model and model...")
            model.trainable = False

        except Exception as e:
            base_model = None
            logger.error("The base model could NOT be loaded correctly!!!")
            logger.error(e)

        # x = model.layers[-2].output
        x = model.get_layer(name=feature_layer_name).output
        model = Model(inputs=model.input, outputs=x, name=model.name)
        model.summary()

        pre_image_norm = self.normalizing_pre_image_1(data[0])

        train_ds = tf.data.Dataset.from_tensor_slices((pre_image_norm, data[1]))
        train_ds = train_ds.batch(self._CNN_batch_size)
        logger.info(f"batch_size: {self._CNN_batch_size}")
        # train_ds = train_ds.map(self.resize_images)

        if model.name == "ResNet50":
            train_ds = train_ds.map(self.resize_images)
        else:
            assert (
                model.input.shape[1:] == data[0].shape[1:]
            ), f"image and input are not equal"

        Deep_features = np.zeros((1, model.layers[-1].output_shape[1]))
        for image_batch, _ in train_ds:
            # breakpoint()
            feature = model(image_batch)
            Deep_features = np.append(Deep_features, feature, axis=0)

            if (Deep_features.shape[0] - 1) % 256 == 0:
                logger.info(
                    f" ->>> ({os.getpid()}) completed images: "
                    + str(Deep_features.shape[0])
                )

        Deep_features = Deep_features[1:, :]
        logger.info(f"Deep features shape: {Deep_features.shape}")

        # time = int(timeit.default_timer() * 1_000_000)
        pre_image_name = "_".join(pre_image_name)
        exec(
            f"deep_{pre_image_name}_{model.name} = pd.DataFrame(Deep_features, columns=['deep_{pre_image_name}_{model.name}_trained_'+str(i) for i in range(Deep_features.shape[1])])"
        )
        # exec(f"deep_{pre_image_name}_{model.name} = pd.concat([deep_{pre_image_name}_{model.name},data[1].reset_index(drop=True)], axis=1)")
        self.saving_deep_features(
            eval(f"deep_{pre_image_name}_{model.name}"),
            f"deep_{pre_image_name}_{model.name}_{self._combination}_trained",
        )

        return eval(f"deep_{pre_image_name}_{model.name}")

    def loading_deep_feature_from_model(
        self,
        model: tf.keras.models.Model,
        feature_layer_name: str,
        data: tuple,
        pre_image_name: str,
    ) -> pd.DataFrame:
        """loading a pre image from a excel file."""
        CNN_name = model.name
        pre_image_name_ = "_".join(pre_image_name)
        try:
            logger.info(f"loading deep features from {pre_image_name_}!!!")
            exec(
                f"df = pd.read_parquet(os.path.join(self._features_path, f'deep_{pre_image_name_}_{CNN_name}_{self._combination}_trained.parquet'))"
            )

        except Exception as e:
            logger.error(e)
            logger.info(f"extraxting deep features from {pre_image_name_}!!!")
            exec(
                f"df = self.extracting_deep_feature_from_model(model, feature_layer_name, data, pre_image_name)"
            )

        self._features_set[f"deep_{pre_image_name_}_{CNN_name}_trained"] = {
            "columns": eval("df.columns"),
            "number_of_features": eval("df.shape[1]"),
            "number_of_samples": eval("df.shape[0]"),
        }
        return eval("df")

    ## rest of the code
    def pack(self, list_features: list, labels: pd.DataFrame) -> pd.DataFrame:
        """
        list of features=[
            [GRFs, COAs, COPs,
            COA_handcrafted, COP_handcrafted, GRF_handcrafted,
            deep_P100_resnet50, deep_P80_resnet50, deep_P90_resnet50,
            P50, P60, P70,
            COA_WPT, COP_WPT, GRF_WPT]
        """
        return pd.concat(list_features + [labels], axis=1)

    def filtering_subjects_and_samples(
        self, DF_features_all: pd.DataFrame
    ) -> pd.DataFrame:
        subjects, samples = np.unique(DF_features_all["ID"].values, return_counts=True)

        ss = [
            a[0]
            for a in list(zip(subjects, samples))
            if a[1] >= self._min_number_of_sample
        ]
        if self._known_imposter + self._unknown_imposter > len(ss):
            raise Exception(
                f"Invalid _known_imposter and _unknown_imposter!!! self._known_imposter:{self._known_imposter}, self._unknown_imposter:{self._unknown_imposter}, len(ss):{len(ss)}"
            )

        self._known_imposter_list = ss[: self._known_imposter]
        self._unknown_imposter_list = ss[-self._unknown_imposter :]

        if self._unknown_imposter == 0:
            self._unknown_imposter_list = []

        DF_unknown_imposter = DF_features_all[
            DF_features_all["ID"].isin(self._unknown_imposter_list)
        ]
        DF_known_imposter = DF_features_all[
            DF_features_all["ID"].isin(self._known_imposter_list)
        ]

        DF_unknown_imposter = DF_unknown_imposter.groupby("ID", group_keys=False).apply(
            lambda x: x.sample(
                frac=self._number_of_unknown_imposter_samples,
                replace=False,
                random_state=self._random_state,
            )
        )

        #  pre_image, labels = data[0], data[1]

        #     subjects, samples = np.unique(labels["ID"].values, return_counts=True)

        #     unknown_imposter =  pre_image[labels["ID"].isin(self._unknown_imposter_list)], labels[labels["ID"].isin(self._unknown_imposter_list)]
        #     known_imposter =    pre_image[labels["ID"].isin(self._known_imposter_list)], labels[labels["ID"].isin(self._known_imposter_list)]
        return DF_known_imposter, DF_unknown_imposter

    def extract_deep_features(self, train_ds, binary_model1):
        Deep_features = np.zeros((1, binary_model1.layers[-1].output_shape[1]))
        temp1 = np.zeros((1, 1))
        for image_batch, label_batch in train_ds:
            feature = binary_model1(image_batch)
            Deep_features = np.append(Deep_features, feature, axis=0)
            temp1 = np.append(temp1, label_batch, axis=0)

            if (Deep_features.shape[0] - 1) % 256 == 0:
                logger.info(
                    f" ->>> ({os.getpid()}) completed images: "
                    + str(Deep_features.shape[0])
                )

        Deep_features = Deep_features[1:, :]
        temp1 = temp1[1:, :]
        df = pd.DataFrame(
            np.concatenate((Deep_features, temp1), axis=1),
            columns=["Deep_" + str(i) for i in range(Deep_features.shape[1])] + ["ID"],
        )
        self._features_set[f"deep_second_trained"] = {
            "columns": df.columns[:-1],
            "number_of_features": df.shape[1],
            "number_of_samples": df.shape[0],
        }
        return df


class Classifier(Features):
    def __init__(self, dataset_name, classifier_name):
        super().__init__(dataset_name)
        self._classifier_name = classifier_name

    def binarize_labels(self, DF_known_imposter, DF_unknown_imposter, subject):
        DF_known_imposter_binariezed = DF_known_imposter.copy().drop(["side"], axis=1)
        DF_known_imposter_binariezed["ID"] = DF_known_imposter_binariezed["ID"].map(
            lambda x: 1.0 if x == subject else 0.0
        )

        DF_unknown_imposter_binariezed = DF_unknown_imposter.copy().drop(
            ["side"], axis=1
        )
        DF_unknown_imposter_binariezed["ID"] = DF_unknown_imposter_binariezed["ID"].map(
            lambda x: 1.0 if x == "a" else 0.0
        )

        return DF_known_imposter_binariezed, DF_unknown_imposter_binariezed

    def down_sampling(self, DF):

        number_of_negatives = DF[DF["ID"] == 0].shape[0]
        if self._ratio == True:
            self._n_training_samples = int(self._p_training_samples * self._train_ratio)
            if number_of_negatives < (self._p_training_samples * self._train_ratio):
                self._n_training_samples = int(number_of_negatives)

        else:
            self._n_training_samples = self._train_ratio
            if number_of_negatives < self._train_ratio:
                self._n_training_samples = int(number_of_negatives)

        DF_positive_samples_train = DF[DF["ID"] == 1].sample(n=self._p_training_samples, replace=False, random_state=self._random_state)
        DF_negative_samples_train = DF[DF["ID"] == 0].sample(n=self._n_training_samples, replace=False, random_state=self._random_state)

        return pd.concat([DF_positive_samples_train, DF_negative_samples_train], axis=0)

    def down_sampling_new(self, DF, sample):

        number_of_negatives = DF[DF["ID"] == 0].shape[0]
        number_of_positives = DF[DF["ID"] == 1].shape[0]

        if (sample < min(number_of_negatives, number_of_positives)):  
            sample = min(number_of_negatives, number_of_positives)     

        DF_positive_samples_train = DF[DF["ID"] == 1].sample(n=sample, replace=False, random_state=self._random_state)
        DF_negative_samples_train = DF[DF["ID"] == 0].sample(n=sample, replace=False, random_state=self._random_state)

        return pd.concat([DF_positive_samples_train, DF_negative_samples_train], axis=0)

    def scaler(self, df_train, *args):

        if self._normilizing == "minmax":
            scaling = sk_preprocessing.MinMaxScaler()

        elif self._normilizing == "z-score":
            scaling = sk_preprocessing.StandardScaler()

        elif self._normilizing == "z-mean":
            scaling = sk_preprocessing.StandardScaler(with_std=False)

        else:
            raise KeyError(self._normilizing)

        Scaled_train = scaling.fit_transform(df_train.iloc[:, :-1])
        Scaled_train = pd.DataFrame(
            np.concatenate((Scaled_train, df_train.iloc[:, -1:].values), axis=1),
            columns=df_train.columns,
        )

        Scaled_df = list()
        Scaled_df.append(Scaled_train)
        for df in args:
            Scaled_test = scaling.transform(df.iloc[:, :-1])
            Scaled_df.append(
                pd.DataFrame(
                    np.concatenate((Scaled_test, df.iloc[:, -1:].values), axis=1),
                    columns=df.columns,
                )
            )

        # Scaled_val = scaling.transform(df_val.iloc[:, :-1])
        # Scaled_val  = pd.DataFrame(np.concatenate((Scaled_val,  df_val.iloc[:, -1:].values),  axis = 1), columns=df_val.columns)

        # Scaled_test_U = pd.DataFrame(columns=df_U.columns)

        # if df_U.shape[0] != 0:
        #     Scaled_test_U = scaling.transform(df_U.iloc[:, :-1])
        #     Scaled_test_U  = pd.DataFrame(np.concatenate((Scaled_test_U,  df_U.iloc[:, -1:].values),  axis = 1), columns=df_U.columns)

        # return Scaled_train, Scaled_test, Scaled_test_U, Scaled_val
        return tuple(Scaled_df)

    def projector_archive(self, df_train, df_test, df_test_U, listn):
        if self._persentage == 1.0:
            num_pc = df_train.shape[1] - 1
            columnsName = ["PC" + str(i) for i in list(range(1, num_pc + 1))] + ["ID"]

            df_train.columns = columnsName
            df_test.columns = columnsName
            df_test_U.columns = columnsName

            return df_train, df_test, df_test_U, num_pc

        elif self._persentage != 1.0:

            principal = PCA(svd_solver="full")
            N = list()
            for ind, feat in enumerate(listn):
                # breakpoint()
                col = self._features_set[feat]["columns"]

                PCA_out_train = principal.fit_transform(df_train.loc[:, col])
                PCA_out_test = principal.transform(df_test.loc[:, col])

                variance_ratio = np.cumsum(principal.explained_variance_ratio_)
                high_var_PC = np.zeros(variance_ratio.shape)
                high_var_PC[variance_ratio <= self._persentage] = 1

                N.append(int(np.sum(high_var_PC)))
                columnsName = [
                    listn[ind] + "_PC" + str(i) for i in list(range(1, N[ind] + 1))
                ]

                exec(
                    f"df_train_pc_{ind} = pd.DataFrame(PCA_out_train[:,:N[ind]], columns = columnsName)"
                )
                exec(
                    f"df_test_pc_{ind} = pd.DataFrame(PCA_out_test[:,:N[ind]], columns = columnsName)"
                )

                if df_test_U.shape[0] != 0:
                    PCA_out_test_U = principal.transform(df_test_U.loc[:, col])
                    exec(
                        f"df_test_U_pc_{ind} = pd.DataFrame(PCA_out_test_U[:,:N[ind]], columns = columnsName)"
                    )

            tem = [("df_train_pc_" + str(i)) for i in range(len(listn))] + [
                'df_train["ID"]'
            ]
            exec(f"df_train_pc = pd.concat({tem}, axis=1)".replace("'", ""))
            tem = [("df_test_pc_" + str(i)) for i in range(len(listn))] + [
                'df_test["ID"]'
            ]
            exec(f"df_test_pc = pd.concat({tem}, axis=1)".replace("'", ""))

            exec(f"df_test_U_pc = pd.DataFrame(columns = columnsName)".replace("'", ""))
            if df_test_U.shape[0] != 0:
                tem = [("df_test_U_pc_" + str(i)) for i in range(len(listn))] + [
                    'df_test_U["ID"]'
                ]
                exec(f"df_test_U_pc = pd.concat({tem}, axis=1)".replace("'", ""))

            num_pc = np.sum(N)

            return eval("df_train_pc"), eval("df_test_pc"), eval("df_test_U_pc"), num_pc

    def projector(self, listn, df_train, *args):
        if self._persentage == 1.0:
            num_pc = df_train.shape[1] - 1
            columnsName = ["PC" + str(i) for i in list(range(1, num_pc + 1))] + ["ID"]

            df_train.columns = columnsName
            projected_df = list()
            projected_df.append(df_train)
            for df in args:
                df.columns = columnsName
                projected_df.append(df)

            projected_df.append(num_pc)
            return tuple(projected_df)

        elif self._persentage != 1.0:

            principal = PCA(svd_solver="full")
            N = list()
            for ind, feat in enumerate(listn):
                # breakpoint()
                col = self._features_set[feat]["columns"]

                PCA_out_train = principal.fit_transform(df_train.loc[:, col])

                variance_ratio = np.cumsum(principal.explained_variance_ratio_)
                high_var_PC = np.zeros(variance_ratio.shape)
                high_var_PC[variance_ratio <= self._persentage] = 1

                N.append(int(np.sum(high_var_PC)))
                columnsName = [
                    listn[ind] + "_PC" + str(i) for i in list(range(1, N[ind] + 1))
                ]

                exec(
                    f"df_train_pc_{ind} = pd.DataFrame(PCA_out_train[:,:N[ind]], columns = columnsName)"
                )

                for i, df in enumerate(args):
                    PCA_out_df = principal.transform(df.loc[:, col])
                    exec(
                        f"df{i}_pc_{ind} = pd.DataFrame(PCA_out_df[:,:N[ind]], columns=columnsName)"
                    )

            tem = [("df_train_pc_" + str(i)) for i in range(len(listn))] + [
                'df_train["ID"]'
            ]
            exec(f"df_train_pc = pd.concat({tem}, axis=1)".replace("'", ""))
            h = list()
            h.append(eval("df_train_pc"))
            for ii, df in enumerate(args):
                tem = [(f"df{ii}_pc_" + str(i)) for i in range(len(listn))] + [
                    'df["ID"]'
                ]
                exec(f"df{ii}_pc = pd.concat({tem}, axis=1)".replace("'", ""))
                h.append(eval(f"df{ii}_pc "))

            # exec( f"df_test_U_pc = pd.DataFrame(columns = columnsName)".replace("'",""))
            # if df_test_U.shape[0] != 0:
            #     tem = [("df_test_U_pc_"+str(i)) for i in range(len(listn))] + ['df_test_U["ID"]']
            #     exec( f"df_test_U_pc = pd.concat({tem}, axis=1)".replace("'",""))

            num_pc = np.sum(N)
            h.append(num_pc)
            return tuple(h)

    def FXR_calculater(self, x_train, y_pred):
        FRR = list()
        FAR = list()

        for tx in self._THRESHOLDs:
            E1 = np.zeros((y_pred.shape))
            E1[y_pred >= tx] = 1

            e = pd.DataFrame([x_train.values, E1]).T
            e.columns = ["y", "pred"]
            e["FAR"] = e.apply(lambda x: 1 if x["y"] < x["pred"] else 0, axis=1)
            e["FRR"] = e.apply(lambda x: 1 if x["y"] > x["pred"] else 0, axis=1)

            a1 = e.sum()
            N = e.shape[0] - a1["y"]
            P = a1["y"]
            FRR.append(a1["FRR"] / P)
            FAR.append(a1["FAR"] / N)

        return FRR, FAR

    def balancer(self, DF, method="random", ratio=1):  # None, DEND, MDIST, Random
        pos_samples = DF[DF["ID"] == 1]
        n = pos_samples.shape[0]
        neg_samples = DF[
            DF["ID"] == 0
        ]  # .sample()#, random_state=cfg.config["Pipeline"]["random_state"])
        neg_samples = self.template_selection(
            neg_samples, method=method, k_cluster=n * ratio, verbose=False
        )
        DF_balanced = pd.concat([pos_samples, neg_samples])
        return DF_balanced, pos_samples.shape[0]

    @staticmethod
    def compute_eer(FAR, FRR):
        """Returns equal error rate (EER) and the corresponding threshold."""
        abs_diffs = np.abs(np.subtract(FRR, FAR))
        min_index = np.argmin(abs_diffs)
        min_index = 99 - np.argmin(abs_diffs[::-1])
        eer = np.mean((FAR[min_index], FRR[min_index]))

        return eer, min_index

    def template_selection(self, DF, method, k_cluster, verbose=True):
        if DF.shape[0] < k_cluster:
            k_cluster = DF.shape[0]

        if method == "DEND":
            kmeans = KMeans(n_clusters=k_cluster, random_state=self._random_state)
            kmeans.fit(DF.iloc[:, :-2].values)
            clusters = np.unique(kmeans.labels_)
            col = DF.columns

            DF1 = DF.copy().reset_index(drop=True)
            for i, r in DF.reset_index(drop=True).iterrows():

                DF1.loc[i, "dist"] = distance.euclidean(
                    kmeans.cluster_centers_[kmeans.labels_[i]], r[:-2].values
                )
                DF1.loc[i, "label"] = kmeans.labels_[i]
            DF_clustered = list()

            for cluster in clusters:
                mean_cluster = DF1[DF1["label"] == cluster].sort_values(
                    by=["dist"],
                )
                DF_clustered.append(mean_cluster.iloc[0, :-2])

            DF_clustered = pd.DataFrame(DF_clustered, columns=col)

        elif method == "MDIST":
            A = distance.squareform(distance.pdist(DF.iloc[:, :-2].values)).mean(axis=1)
            i = np.argsort(A)[:k_cluster]
            DF_clustered = DF.iloc[i, :]
            DF_clustered = pd.DataFrame(
                np.concatenate((DF_clustered, DF.iloc[:, -2:].values), axis=1),
                columns=DF.columns,
            )

        elif method == "None":
            DF_clustered = pd.DataFrame(DF, columns=DF.columns)

        elif method == "Random":
            DF_clustered = pd.DataFrame(DF, columns=DF.columns).sample(n=k_cluster)

        if verbose:
            logger.info(
                f"\tApplying template selection with method '{method}' [orginal shape: {DF.shape}, output shape{DF_clustered.shape}]"
            )
        return DF_clustered

    def ML_classifier_archive(
        self,
        a,
        x_train,
        *args,
    ):
        PP = f"./temp/shod1-dend/{self._classifier_name}/{str(a)}/{str(self._unknown_imposter)}/"

        if self._classifier_name == "knn":
            classifier = knn(
                n_neighbors=self._KNN_n_neighbors,
                metric=self._KNN_metric,
                weights=self._KNN_weights,
            )

            best_model = classifier.fit(
                x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values
            )
            y_pred_tr = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]
            FRR_t, FAR_t = self.FXR_calculater(x_train["ID"], y_pred_tr)
            EER, t_idx = self.compute_eer(FRR_t, FAR_t)
            TH = self._THRESHOLDs[t_idx]

        elif self._classifier_name == "TM":
            positives = x_train[x_train["ID"] == 1.0]
            negatives = x_train[x_train["ID"] == 0.0]
            (
                similarity_matrix_positives,
                similarity_matrix_negatives,
            ) = self.compute_score_matrix(positives, negatives)
            client_scores, imposter_scores = self.compute_scores(
                similarity_matrix_positives, similarity_matrix_negatives, criteria="min"
            )
            y_pred_tr = np.append(client_scores.data, imposter_scores.data)

            FRR_t, FAR_t = self.FXR_calculater(x_train["ID"], y_pred_tr)
            EER, t_idx = self.compute_eer(FRR_t, FAR_t)
            TH = self._THRESHOLDs[t_idx]

            # self.plot_eer(FRR_t, FAR_t)
            # Pathlb(PP).mkdir(parents=True, exist_ok=True)
            # plt.savefig(PP+f"EER_{str(self._known_imposter)}.png")

            # EER1 = list()
            # TH1 = list()
            # for _ in range(self._random_runs):
            #     DF, _ = self.balancer(x_train, method="Random")

            #     positives = DF[DF["ID"]== 1.0]
            #     negatives = DF[DF["ID"]== 0.0]

            #     similarity_matrix_positives, similarity_matrix_negatives = self.compute_score_matrix(positives, negatives)
            #     client_scores, imposter_scores = self.compute_scores(similarity_matrix_positives, similarity_matrix_negatives, criteria="min")
            #     y_pred = np.append(client_scores.data, imposter_scores.data)

            #     FRR_t, FAR_t = self.FXR_calculater(x_train["ID"], y_pred)
            #     qq, t_idx = self.compute_eer(FRR_t, FAR_t)
            #     EER1.append(qq)
            #     TH1.append(self._THRESHOLDs[t_idx])
            # EER1 = np.mean(EER1)
            # TH1 = np.mean(TH1)

        elif self._classifier_name == "svm":
            classifier = svm.SVC(
                kernel=self._SVM_kernel,
                probability=True,
                random_state=self._random_state,
            )

            best_model = classifier.fit(
                x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values
            )
            y_pred_tr = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]
            FRR_t, FAR_t = self.FXR_calculater(x_train["ID"], y_pred_tr)
            EER, t_idx = self.compute_eer(FRR_t, FAR_t)
            TH = self._THRESHOLDs[t_idx]
        else:
            raise Exception(
                f"_classifier_name ({self._classifier_name}) is not valid!!"
            )

        acc = list()
        CMM = list()
        BACC = list()
        for _ in range(self._random_runs):
            DF_temp, pos_number = self.balancer(x_test, method="Random")

            if self._classifier_name == "TM":
                positives = x_train[x_train["ID"] == 1.0]
                (
                    similarity_matrix_positives,
                    similarity_matrix_negatives,
                ) = self.compute_score_matrix(positives, DF_temp)
                client_scores, imposter_scores = self.compute_scores(
                    similarity_matrix_positives,
                    similarity_matrix_negatives,
                    criteria="min",
                )
                y_pred = imposter_scores.data

            else:
                y_pred = best_model.predict_proba(DF_temp.iloc[:, :-1].values)[:, 1]

            y_pred[y_pred >= TH] = 1.0
            y_pred[y_pred < TH] = 0.0

            acc.append(accuracy_score(DF_temp.iloc[:, -1].values, y_pred) * 100)
            CM = confusion_matrix(DF_temp.iloc[:, -1].values, y_pred)
            spec = (CM[0, 0] / (CM[0, 1] + CM[0, 0] + 1e-33)) * 100
            sens = (CM[1, 1] / (CM[1, 0] + CM[1, 1] + 1e-33)) * 100
            BACC.append((spec + sens) / 2)
            CMM.append(CM)

        ACC_bd = np.mean(acc)
        CM_bd = np.array(CMM).sum(axis=0)
        BACC_bd = np.mean(BACC)
        FAR_bd = CM_bd[0, 1] / CM_bd[0, :].sum()
        FRR_bd = CM_bd[1, 0] / CM_bd[1, :].sum()

        if self._classifier_name == "TM":
            positives = x_train[x_train["ID"] == 1.0]
            (
                similarity_matrix_positives,
                similarity_matrix_negatives,
            ) = self.compute_score_matrix(positives, x_test)
            client_scores, imposter_scores = self.compute_scores(
                similarity_matrix_positives, similarity_matrix_negatives, criteria="min"
            )
            y_pred = imposter_scores.data

        else:
            y_pred = best_model.predict_proba(x_test.iloc[:, :-1].values)[:, 1]

        y_pred1 = y_pred.copy()
        y_pred[y_pred >= TH] = 1
        y_pred[y_pred < TH] = 0

        ACC_ud = accuracy_score(x_test["ID"].values, y_pred) * 100
        CM_ud = confusion_matrix(x_test.iloc[:, -1].values, y_pred)
        spec = (CM_ud[0, 0] / (CM_ud[0, 1] + CM_ud[0, 0] + 1e-33)) * 100
        sens = (CM_ud[1, 1] / (CM_ud[1, 0] + CM_ud[1, 1] + 1e-33)) * 100
        BACC_ud = (spec + sens) / 2
        FAR_ud = CM_ud[0, 1] / CM_ud[0, :].sum()
        FRR_ud = CM_ud[1, 0] / CM_ud[1, :].sum()

        AUS, FAU = 100, 0
        AUS_All, FAU_All = 100, 0

        if x_test_U.shape[0] != 0:

            AUS, FAU = [], []
            for _ in range(self._random_runs):
                numbers = x_test_U.shape[0] if x_test_U.shape[0] < 60 else 60
                temp = x_test_U.sample(n=numbers)

                if self._classifier_name == "TM":
                    positives = x_train[x_train["ID"] == 1.0]
                    (
                        similarity_matrix_positives,
                        similarity_matrix_negatives,
                    ) = self.compute_score_matrix(positives, temp)
                    client_scores, imposter_scores = self.compute_scores(
                        similarity_matrix_positives,
                        similarity_matrix_negatives,
                        criteria="min",
                    )
                    y_pred = imposter_scores.data

                else:
                    y_pred = best_model.predict_proba(temp.iloc[:, :-1].values)[:, 1]

                y_pred_U = y_pred
                y_pred_U[y_pred_U >= TH] = 1.0
                y_pred_U[y_pred_U < TH] = 0.0

                AUS.append(accuracy_score(temp["ID"].values, y_pred_U) * 100)
                FAU.append(np.where(y_pred_U == 1)[0].shape[0])
            AUS = np.mean(AUS)
            FAU = np.mean(FAU)

            if self._classifier_name == "TM":
                positives = x_train[x_train["ID"] == 1.0]
                (
                    similarity_matrix_positives,
                    similarity_matrix_negatives,
                ) = self.compute_score_matrix(positives, x_test_U)
                client_scores, imposter_scores = self.compute_scores(
                    similarity_matrix_positives,
                    similarity_matrix_negatives,
                    criteria="min",
                )
                y_pred_U = imposter_scores.data

            else:
                y_pred_U = best_model.predict_proba(x_test_U.iloc[:, :-1].values)[:, 1]

            y_pred_U1 = y_pred_U.copy()

            y_pred_U[y_pred_U >= TH] = 1.0
            y_pred_U[y_pred_U < TH] = 0.0
            AUS_All = accuracy_score(x_test_U["ID"].values, y_pred_U) * 100
            FAU_All = np.where(y_pred_U == 1)[0].shape[0]

        # #todo
        # # PP = f"./C/{self._classifier_name}/{str(a)}/{str(self._unknown_imposter)}/1/"
        # # PP1 = f"./C/{self._classifier_name}/{str(a)}/{str(self._unknown_imposter)}/2/"
        # # PP2 = f"./C/{self._classifier_name}/{str(a)}/{str(self._unknown_imposter)}/3/"
        # # PP3 = f"./C/{self._classifier_name}/{str(a)}/{str(self._unknown_imposter)}/4/"
        # # Pathlb(PP).mkdir(parents=True, exist_ok=True)
        # # Pathlb(PP1).mkdir(parents=True, exist_ok=True)
        # # Pathlb(PP2).mkdir(parents=True, exist_ok=True)
        # # Pathlb(PP3).mkdir(parents=True, exist_ok=True)
        # # breakpoint()

        # # plt.figure().suptitle(f"Number of known Imposters: {str(self._known_imposter)}", fontsize=20)
        # figure, axs = plt.subplots(1,3,figsize=(15,5))
        # figure.suptitle(f"Number of known Imposters: {str(self._known_imposter)}", fontsize=20)

        # SS = pd.DataFrame(y_pred_tr, x_train['ID'].values).reset_index()
        # SS.columns = ["Labels","train scores"]
        # sns.histplot(data=SS, x="train scores", hue="Labels", bins=100 , ax=axs[0], kde=True)
        # axs[0].plot([TH,TH],[0,13], 'r--', linewidth=2, label="unbalanced threshold")
        # # axs[0].plot([TH1,TH1],[0,10], 'g:', linewidth=2, label="balanced threshold")
        # axs[0].set_title(f"EER: {round(EER,2)}  Threshold: {round(TH,2)}  ")
        # # plt.savefig(PP+f"{str(self._known_imposter)}.png")

        # # plt.figure()
        # SS = pd.DataFrame(y_pred1,x_test['ID'].values).reset_index()
        # SS.columns = ["Labels","test scores"]
        # sns.histplot(data=SS, x="test scores", hue="Labels", bins=100, ax=axs[1], kde=True)
        # axs[1].plot([TH,TH],[0,13], 'r--', linewidth=2, label="unbalanced threshold")
        # # axs[1].plot([TH1,TH1],[0,10], 'g:', linewidth=2, label="balanced threshold")
        # axs[1].set_title(f"ACC: {round(ACC_ud,2)},    BACC: {round(BACC_ud,2)},  \n CM: {CM_ud}")
        # # plt.savefig(PP1+f"{str(self._known_imposter)}.png")

        # # plt.figure()
        # sns.histplot(y_pred_U1, bins=100, ax=axs[2], kde=True)
        # axs[2].plot([TH,TH],[0,13], 'r--', linewidth=2, label="unbalanced threshold")
        # # axs[2].plot([TH1,TH1],[0,10], 'g:', linewidth=2, label="balanced threshold")
        # axs[2].set_xlabel("unknown imposter scores")
        # axs[2].set_title(f"AUS: {round(AUS_All,2)},       FAU: {round(FAU_All,2)}")
        # # plt.savefig(PP2+f"{str(self._known_imposter)}.png")

        # # plt.figure()
        # # plt.scatter(x_train.iloc[:, 0].values, x_train.iloc[:, 1].values, c ="red", marker ="s", label="train", s = x_train.iloc[:, -1].values*22+1)
        # # plt.scatter(x_test.iloc[:, 0].values, x_test.iloc[:, 1].values,  c ="blue", marker ="*", label="test", s = x_test.iloc[:, -1].values*22+1)
        # # plt.scatter(x_test_U.iloc[:, 0].values, x_test_U.iloc[:, 1].values, c ="green", marker ="o", label="u", s = 5)
        # # plt.title(f'# training positives: {x_train[x_train["ID"]== 1.0].shape[0]},       # training negatives: {x_train[x_train["ID"]== 0.0].shape[0]} \n # test positives: {x_test[x_test["ID"]== 1.0].shape[0]},       # test negatives: {x_test[x_test["ID"]== 0.0].shape[0]}               # test_U : {x_test_U.shape[0]}')

        # # plt.xlabel("PC1")
        # # plt.ylabel("PC2")
        # # plt.legend()
        # # plt.savefig(PP3+f"{str(self._known_imposter)}.png")
        # plt.tight_layout()
        # plt.savefig(PP+f"{str(self._known_imposter)}.png")

        # plt.figure()
        # SS = pd.DataFrame(y_pred_tr, x_train['ID'].values).reset_index()
        # SS.columns = ["Labels","scores"]
        # SS1 = pd.DataFrame(y_pred_U1).reset_index()
        # SS1.columns = ["Labels","scores"]
        # SS1["Labels"] = "unknown imposters"
        # SS2 = pd.concat([SS1,SS], axis=0).reset_index()
        # SS2["Labels"] = SS2["Labels"].map(lambda x: 'user' if x==1 else 'known imposters' if x==0 else 'unknown imposters')
        # # sns.histplot(data=SS2, x="train scores", hue="Labels", bins=100 , kde=True)
        # sns.kdeplot(data=SS2, x="scores", hue="Labels")#, bins=100 , kde=True)
        # plt.plot([TH,TH],[0,13], 'r--', linewidth=2, label="unbalanced threshold")
        # # plt.plot([TH1,TH1],[0,10], 'g:', linewidth=2, label="balanced threshold")
        # plt.savefig(PP+f"kde_{str(self._known_imposter)}.png")

        # plt.show()
        # plt.close('all')

        # # breakpoint()

        results = [
            EER,
            TH,
            ACC_bd,
            BACC_bd,
            FAR_bd,
            FRR_bd,
            ACC_ud,
            BACC_ud,
            FAR_ud,
            FRR_ud,
            AUS,
            FAU,
            x_test_U.shape[0],
            AUS_All,
            FAU_All,
        ]

        return results, CM_bd, CM_ud

    def ML_classifier(self, a, **kwargs):

        if "x_train" in kwargs.keys():
            x_train = kwargs["x_train"]
        if "x_val" in kwargs.keys():
            x_val = kwargs["x_val"]
        if "x_test" in kwargs.keys():
            x_test = kwargs["x_test"]
        if "x_test_U" in kwargs.keys():
            x_test_U = kwargs["x_test_U"]

        if self._classifier_name == "knn":
            classifier = knn(n_neighbors=self._KNN_n_neighbors, metric=self._KNN_metric, weights=self._KNN_weights)

            best_model = classifier.fit( x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values )
            y_pred_tr = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]
            FRR_t, FAR_t = self.FXR_calculater(x_train["ID"], y_pred_tr)
            EER, t_idx = self.compute_eer(FRR_t, FAR_t)
            TH = self._THRESHOLDs[t_idx]

            y_pred = best_model.predict_proba(x_test.iloc[:, :-1].values)[:, 1]

        elif self._classifier_name == "tm":
            positives = x_train[x_train["ID"] == 1.0]
            negatives = x_train[x_train["ID"] == 0.0]
            similarity_matrix_positives, similarity_matrix_negatives = self.compute_score_matrix(positives, negatives)
            client_scores, imposter_scores = self.compute_scores(similarity_matrix_positives, similarity_matrix_negatives, criteria="min")
            y_pred_tr = np.append(client_scores.data, imposter_scores.data)

            FRR_t, FAR_t = self.FXR_calculater(x_train["ID"], y_pred_tr)
            EER, t_idx = self.compute_eer(FRR_t, FAR_t)
            TH = self._THRESHOLDs[t_idx]

            similarity_matrix_positives, similarity_matrix_negatives = self.compute_score_matrix(positives, x_test)
            client_scores, imposter_scores = self.compute_scores( similarity_matrix_positives, similarity_matrix_negatives, criteria="min")
            y_pred = imposter_scores.data

        elif self._classifier_name == "svm":
            classifier = svm.SVC(kernel=self._SVM_kernel, probability=True, random_state=self._random_state )
            best_model = classifier.fit(x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values)
            y_pred_tr = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]

            FRR_t, FAR_t = self.FXR_calculater(x_train["ID"], y_pred_tr)
            EER, t_idx = self.compute_eer(FRR_t, FAR_t)
            TH = self._THRESHOLDs[t_idx]

            y_pred = best_model.predict_proba(x_test.iloc[:, :-1].values)[:, 1]

        elif self._classifier_name == "lda":
            classifier = LinearDiscriminantAnalysis()
            best_model = classifier.fit(x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values)
            y_pred_tr = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]

            FRR_t, FAR_t = self.FXR_calculater(x_train["ID"], y_pred_tr)
            EER, t_idx = self.compute_eer(FRR_t, FAR_t)
            TH = self._THRESHOLDs[t_idx]

            y_pred = best_model.predict_proba(x_test.iloc[:, :-1].values)[:, 1]

        elif self._classifier_name == "rf":
            classifier = RandomForestClassifier(random_state=self._random_state)
            best_model = classifier.fit(x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values)
            y_pred_tr = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]

            FRR_t, FAR_t = self.FXR_calculater(x_train["ID"], y_pred_tr)
            EER, t_idx = self.compute_eer(FRR_t, FAR_t)
            TH = self._THRESHOLDs[t_idx]

            y_pred = best_model.predict_proba(x_test.iloc[:, :-1].values)[:, 1]

        elif self._classifier_name == "nb":
            classifier = GaussianNB()
            best_model = classifier.fit(x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values)
            y_pred_tr = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]

            FRR_t, FAR_t = self.FXR_calculater(x_train["ID"], y_pred_tr)
            EER, t_idx = self.compute_eer(FRR_t, FAR_t)
            TH = self._THRESHOLDs[t_idx]

            y_pred = best_model.predict_proba(x_test.iloc[:, :-1].values)[:, 1]

        elif self._classifier_name == "if":
            
            classifier = IsolationForest(random_state=self._random_state)
            best_model = classifier.fit(x_train.iloc[:, :-1].values)
            EER = 0
            TH = 0

            y_pred = best_model.predict(x_test.iloc[:, :-1].values)

        elif self._classifier_name == "ocsvm":
            
            classifier = OneClassSVM(kernel='rbf',nu=0.1)
            best_model = classifier.fit(x_train.iloc[:, :-1].values)
            EER = 0
            TH = 0

            y_pred = best_model.predict(x_test.iloc[:, :-1].values)

        elif self._classifier_name == "svdd":
            
            classifier = OneClassSVM(kernel='rbf',nu=0.1)
            best_model = classifier.fit(x_train.iloc[:, :-1].values)
            EER = 0
            TH = 0

            y_pred = best_model.predict(x_test.iloc[:, :-1].values)

        else:
            raise Exception(f"_classifier_name ({self._classifier_name}) is not valid!!")


               
        if self._classifier_name in ["if", "ocsvm", "svdd"]:
            y_pred = 0.5-(y_pred/2)
        else:
            y_pred[y_pred >= TH] = 1
            y_pred[y_pred < TH] = 0

        ACC_ud = accuracy_score(x_test["ID"].values, y_pred) * 100
        CM_ud = confusion_matrix(x_test.iloc[:, -1].values, y_pred)
        spec = (CM_ud[0, 0] / (CM_ud[0, 1] + CM_ud[0, 0] + 1e-33)) * 100
        sens = (CM_ud[1, 1] / (CM_ud[1, 0] + CM_ud[1, 1] + 1e-33)) * 100
        BACC_ud = (spec + sens) / 2
        FAR_ud = CM_ud[0, 1] / CM_ud[0, :].sum()
        FRR_ud = CM_ud[1, 0] / CM_ud[1, :].sum()

        AUS_All, FAU_All = "-", "-"

        if "x_test_U" in kwargs.keys():

            if self._classifier_name == "tm":
                positives = x_train[x_train["ID"] == 1.0]
                similarity_matrix_positives, similarity_matrix_negatives = self.compute_score_matrix(positives, x_test_U)
                client_scores, imposter_scores = self.compute_scores(similarity_matrix_positives, similarity_matrix_negatives, criteria="min",)
                y_pred_U = imposter_scores.data
                y_pred_U[y_pred_U >= TH] = 1.0
                y_pred_U[y_pred_U < TH] = 0.0

            elif self._classifier_name in ["if", "ocsvm", "svdd"]:
                y_pred_U = best_model.predict(x_test_U.iloc[:, :-1].values)
                y_pred_U = 0.5-(y_pred_U/2)
            else:
                y_pred_U = best_model.predict_proba(x_test_U.iloc[:, :-1].values)[:, 1]
                y_pred_U[y_pred_U >= TH] = 1.0
                y_pred_U[y_pred_U < TH] = 0.0

            
            AUS_All = accuracy_score(x_test_U["ID"].values, y_pred_U) * 100
            FAU_All = np.where(y_pred_U == 1)[0].shape[0]

        results_name = {
            "EER": EER,
            "TH": TH,
            "ACC_ud": ACC_ud,
            "BACC_ud": BACC_ud,
            "FAR_ud": FAR_ud,
            "FRR_ud": FRR_ud,
            "unknown samples": x_test_U.shape[0],
            "AUS_All": AUS_All,
            "FAU_All": FAU_All,
            "CM_ud_TN": CM_ud[0, 0],
            "CM_ud_FP": CM_ud[0, 1],
            "CM_ud_FN": CM_ud[1, 0],
            "CM_ud_TP": CM_ud[1, 1],
        }

        return results_name

    @staticmethod
    def compute_score_matrix(positive_samples, negative_samples):
        """Returns score matrix of trmplate matching"""
        positive_model = np.zeros(
            (positive_samples.shape[0], positive_samples.shape[0])
        )
        negative_model = np.zeros(
            (positive_samples.shape[0], negative_samples.shape[0])
        )

        for i in range(positive_samples.shape[0]):
            for j in range(positive_samples.shape[0]):
                positive_model[i, j] = distance.euclidean(
                    positive_samples.iloc[i, :-1], positive_samples.iloc[j, :-1]
                )
            for j in range(negative_samples.shape[0]):
                negative_model[i, j] = distance.euclidean(
                    positive_samples.iloc[i, :-1], negative_samples.iloc[j, :-1]
                )

        return (
            np.power(positive_model + 1, -1),
            np.power(negative_model + 1, -1),
        )

    @staticmethod
    def compute_scores(
        similarity_matrix_positives, similarity_matrix_negatives, criteria="min"
    ):
        if criteria == "average":
            client_scores = np.mean(
                np.ma.masked_where(
                    similarity_matrix_positives == 1, similarity_matrix_positives
                ),
                axis=0,
            )
            client_scores = np.expand_dims(client_scores, -1)

            imposter_scores = (np.sum(similarity_matrix_negatives, axis=0)) / (
                similarity_matrix_positives.shape[1]
            )
            imposter_scores = np.expand_dims(imposter_scores, -1)

        elif criteria == "min":
            client_scores = np.max(
                np.ma.masked_where(
                    similarity_matrix_positives == 1, similarity_matrix_positives
                ),
                axis=0,
            )
            client_scores = np.expand_dims(client_scores, -1)

            imposter_scores = np.max(
                np.ma.masked_where(
                    similarity_matrix_negatives == 1, similarity_matrix_negatives
                ),
                axis=0,
            )
            imposter_scores = np.expand_dims(imposter_scores, -1)

        elif criteria == "median":
            client_scores = np.median(similarity_matrix_positives, axis=0)
            client_scores = np.expand_dims(client_scores, -1)

            imposter_scores = np.median(similarity_matrix_negatives, axis=0)
            imposter_scores = np.expand_dims(imposter_scores, -1)

        return client_scores, imposter_scores

    @staticmethod
    def plot_eer(FAR, FRR):
        """Returns equal error rate (EER) and the corresponding threshold."""
        abs_diffs = np.abs(np.subtract(FRR, FAR))

        min_index = np.argmin(abs_diffs)
        # breakpoint()
        min_index = 99 - np.argmin(abs_diffs[::-1])
        plt.figure(figsize=(10, 5))
        eer = np.mean((FAR[min_index], FRR[min_index]))
        plt.plot(np.linspace(0, 1, 100), FRR, label="FRR")
        plt.plot(np.linspace(0, 1, 100), FAR, label="FAR")
        plt.plot(np.linspace(0, 1, 100)[min_index], eer, "r*", label="EER")
        # plt.savefig(path, bbox_inches='tight')

        # plt.show()
        plt.legend()


class Deep_network(PreFeatures):
    def __init__(
        self,
        dataset_name,
    ):
        super().__init__(dataset_name)
        # self._classifier_name=classifier_name

    def loading_image_features_from_list(
        self, pre_images: np.ndarray, list_pre_image: list
    ) -> np.ndarray:
        """loading multiple pre image features from list"""
        sss = []
        for pre_image_name in list_pre_image:
            if not pre_image_name in self._pre_image_names:
                raise Exception("Invalid pre image name!!!")

            sss.append(self._pre_image_names.index(pre_image_name))

        return pre_images[..., sss]

    def normalizing_image_features(self, pre_images: np.ndarray) -> np.ndarray:
        norm_pre_images = pre_images.copy()
        for i in range(norm_pre_images.shape[3]):
            maxvalues = np.max(norm_pre_images[..., i])
            norm_pre_images[..., i] = norm_pre_images[..., i] / maxvalues
        return norm_pre_images

    def label_encoding(self, labels: pd.DataFrame) -> np.ndarray:

        indices = labels["ID"]
        # logger.info("    metadata shape: {}".format(indices.shape))

        le = sk_preprocessing.LabelEncoder()
        le.fit(indices)

        # logger.info(f"Number of subjects: {len(np.unique(indices))}")

        return le.transform(indices)

    def lightweight_CNN(self, image_size, Number_of_subjects):
        """Lightweight CNN for pre-image features"""

        CNN_name = "lightweight_CNN"

        input = tf.keras.layers.Input(
            shape=image_size, dtype=tf.float64, name="original_img"
        )

        x = tf.cast(input, tf.float32)
        # x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
        # x = tf.keras.layers.RandomRotation(0.2)(x)
        # x = tf.keras.layers.RandomZoom(0.1)(x)

        x = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv2D(32, kernel_size=(7, 7), activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # x = tf.keras.layers.Conv2D(32, kernel_size=(7, 7), activation='relu')(x)
        # x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation="relu", name="last_dense")(
            x
        )  # kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        x = tf.keras.layers.Dropout(0.2)(x)
        output = tf.keras.layers.Dense(Number_of_subjects, activation='sigmoid', name="prediction")(x)  # activation='sigmoid',

        ## The CNN Model
        return tf.keras.models.Model(inputs=input, outputs=output, name=CNN_name)

    def ResNet50(self, image_size, Number_of_subjects):
        CNN_name = "ResNet50"
        try:
            logger.info(f"Loading { CNN_name } model...")
            base_model = tf.keras.applications.resnet50.ResNet50(
                weights="imagenet", include_top=False
            )
            logger.info("Successfully loaded base model and model...")
            for layer in base_model.layers:
                layer.trainable = False
            for layer in base_model.layers[-7:]:
                layer.trainable = True
            # base_model.summary()

        except Exception as e:
            base_model = None
            logger.error("The base model could NOT be loaded correctly!!!")
            logger.error(e)

        input = tf.keras.layers.Input(
            shape=(224, 224, 3), dtype=tf.float64, name="original_img"
        )
        x = tf.cast(input, tf.float32)
        x = tf.keras.applications.resnet50.preprocess_input(x)
        x = base_model(x)
        x = tf.keras.layers.GlobalMaxPool2D()(x)

        x = tf.keras.layers.Dropout(0.25)(x)
        # x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation="relu", name="last_dense")(
            x
        )  # kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        x = tf.keras.layers.Dropout(0.2)(x)
        output = tf.keras.layers.Dense(Number_of_subjects, name="prediction")(
            x
        )  # activation='softmax',
        # x = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(x)
        ## The CNN Model
        return tf.keras.models.Model(inputs=input, outputs=output, name=CNN_name)

    def CNN_model(self, update, CNN_name, pre_image_shape, outputs):
        if update == True:
            path = os.path.join(os.getcwd(), "results", CNN_name, "best.h5")
            model = load_model(path)
        else:
            # model = self.lightweight_CNN(pre_image.shape[1:], outputs)
            model = eval(f"self.{CNN_name}(pre_image_shape, outputs)")
        return model

    def train_deep_CNN( self, dataset_name: str, image_feature_name: list, CNN_name: str,  update: bool = False):
        pre_images, labels = self.loading_pre_features_image(dataset_name)
        pre_image = self.loading_image_features_from_list(
            pre_images, image_feature_name
        )
        known_imposters, _ = self.filtering_subjects_and_samples_deep(
            (pre_image, labels)
        )

        pre_image, labels = known_imposters[0], known_imposters[1]

        encoded_labels = self.label_encoding(labels)

        outputs = len(labels["ID"].unique())

        images_feat_norm = self.normalizing_image_features(pre_image)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(images_feat_norm, encoded_labels, test_size=0.2, random_state=self._random_state, stratify=encoded_labels)
        # X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2, random_state=self._random_state, stratify=y_train)#todo

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000)
        train_ds = train_ds.batch(self._CNN_batch_size)
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(1000)
        # val_ds = val_ds.batch(self._CNN_batch_size)
        # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.batch(self._CNN_batch_size)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
        # for images, labels in train_ds.take(1):
        #     print('images.shape: ', images.shape)
        #     print('labels.shape: ', labels.shape)
        #     breakpoint()

        if CNN_name == "ResNet50":
            train_ds = train_ds.map(self.resize_images)
            test_ds = test_ds.map(self.resize_images)
            # val_ds = val_ds.map(self.resize_images)

        model = self.CNN_model(update, CNN_name, pre_image.shape[1:], outputs)

        # breakpoint()
        # for layer in model.layers[:-4]: layer.trainable = False
        # print(eval('model'))
        # learning_rate=0.001
        model.compile(
            optimizer=self._CNN_optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["Accuracy"],
        )  # if softmaxt then from_logits=False otherwise True

        # TensorBoard_logs =  os.path.join( os.getcwd(), "logs", "TensorBoard_logs", "_".join(("FS", str(os.getpid()), pre_image_name, str(self._test_id)) )  )
        path = os.path.join(os.getcwd(), "results", model.name, "best.h5")

        checkpoint = [
            tf.keras.callbacks.ModelCheckpoint(
                path,
                save_best_only=True,
                monitor="val_loss",
                verbose=1,
                save_weights_only=False,
            ),
            # tf.keras.callbacks.ReduceLROnPlateau(  monitor="val_loss", factor=0.5, patience=30, min_lr=0.00001),
            # tf.keras.callbacks.EarlyStopping(      monitor="val_loss", patience=20, verbose=1),
            # tf.keras.callbacks.TensorBoard(        log_dir=TensorBoard_logs+str(self._test_id))
        ]

        history = model.fit(
            train_ds,
            batch_size=self._CNN_batch_size,
            callbacks=[checkpoint],
            epochs=self._CNN_epochs,
            validation_data=test_ds,
            verbose=self._verbose,
        )

        logger.info("best_model")
        best_model = load_model(path)
        test_loss, test_acc = best_model.evaluate(test_ds, verbose=2)
        logger.info(
            f"  test_loss: {np.round(test_loss)}, test_acc: {int(np.round(test_acc*100))}%"
        )

        # breakpoint()
        path = os.path.join(os.getcwd(), "results", model.name, "earlystop_model.h5")
        model.save(path)

        logger.info("earlystop_model")
        earlystop_model = load_model(path)
        test_loss, test_acc = earlystop_model.evaluate(test_ds, verbose=2)
        logger.info(
            f"  test_loss: {np.round(test_loss)}, test_acc: {int(np.round(test_acc*100))}%"
        )

        if update == True:
            path = os.path.join(os.getcwd(), "results", model.name, "history.csv")
            temp = pd.read_csv(path).drop("Unnamed: 0", axis=1)
            hist_df = pd.DataFrame(history.history)
            hist_df = pd.concat((temp, hist_df), axis=0).reset_index(drop=True)
            path = os.path.join(os.getcwd(), "results", model.name, "history.csv")
            hist_df.to_csv(path)

        else:
            hist_df = pd.DataFrame(history.history)
            path = os.path.join(os.getcwd(), "results", model.name, "history.csv")
            hist_df.to_csv(path)

        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        ax[0].plot(hist_df["Accuracy"], label="Train Accuracy")
        ax[0].plot(hist_df["val_Accuracy"], label="Val Accuracy")

        ax[0].set_title("Accuracy")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Accuracy")
        ax[0].legend()

        # summarize history for loss
        ax[1].plot(hist_df["loss"], label="Train Loss")
        ax[1].plot(hist_df["val_loss"], label="Val Loss")
        ax[1].set_title("Loss")
        ax[1].set_ylabel("loss")
        ax[1].set_xlabel("epoch")
        ax[1].legend()

        path = os.path.join(os.getcwd(), "results", model.name, "plot.png")
        plt.savefig(path)

        return best_model

    def filtering_subjects_and_samples_deep(self, data: tuple) -> np.ndarray:
        pre_image, labels = data[0], data[1]

        subjects, samples = np.unique(labels["ID"].values, return_counts=True)

        ss = [
            a[0]
            for a in list(zip(subjects, samples))
            if a[1] >= self._min_number_of_sample
        ]
        if self._known_imposter + self._unknown_imposter > len(ss):
            raise Exception(
                f"Invalid _known_imposter and _unknown_imposter!!! self._known_imposter:{self._known_imposter}, self._unknown_imposter:{self._unknown_imposter}, len(ss):{len(ss)}"
            )

        self._known_imposter_list = ss[: self._known_imposter]
        self._unknown_imposter_list = ss[-self._unknown_imposter :]

        if self._unknown_imposter == 0:
            self._unknown_imposter_list = []

        unknown_imposter = (
            pre_image[labels["ID"].isin(self._unknown_imposter_list)],
            labels[labels["ID"].isin(self._unknown_imposter_list)],
        )
        known_imposter = (
            pre_image[labels["ID"].isin(self._known_imposter_list)],
            labels[labels["ID"].isin(self._known_imposter_list)],
        )

        return known_imposter, unknown_imposter

    def e2e_CNN_model(self, update, CNN_name, pre_image_shape, subject: int):
        if update == True:
            path = os.path.join(os.getcwd(), "results", "e2e", CNN_name, str(subject), "best.h5")
            model = load_model(path)
        else:
            # model = self.lightweight_CNN(pre_image.shape[1:], outputs)
            model = eval(f"self.{CNN_name}(pre_image_shape, 1)")
        return model

    def train_e2e(self, train_ds:tf.data.Dataset, val_ds:tf.data.Dataset, test_ds:tf.data.Dataset, CNN_name: str, subject: int, class_weight, update: bool = False):      

        model = self.e2e_CNN_model(update, CNN_name, test_ds.element_spec[0].shape[1:] , subject)
        # model.summary()
        # breakpoint()

        METRICS = [
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="prc", curve="PR"),  # precision-recall curve
            BalancedAccuracy(name="bacc"),
        ]
        model.compile(
            optimizer=self._CNN_optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=METRICS,
        )  # if softmaxt then from_logits=False otherwise True

        # TensorBoard_logs =  os.path.join( os.getcwd(), "logs", "TensorBoard_logs", "_".join(("FS", str(os.getpid()), pre_image_name, str(self._test_id)) )  )
        path = os.path.join(os.getcwd(), "results", "e2e", model.name, str(subject), "best.h5")

        checkpoint = [
            tf.keras.callbacks.ModelCheckpoint(path, save_best_only=True, monitor="val_loss", verbose=1, save_weights_only=False),
            # tf.keras.callbacks.ReduceLROnPlateau(  monitor="val_loss", factor=0.5, patience=30, min_lr=0.00001),
            # tf.keras.callbacks.EarlyStopping(      monitor="val_loss", patience=20, verbose=1),
            # tf.keras.callbacks.TensorBoard(        log_dir=TensorBoard_logs+str(self._test_id))
        ]

        

        # 
        history = model.fit(
            train_ds,
            batch_size=self._CNN_batch_size,
            callbacks=[checkpoint],
            epochs=self._CNN_epochs,
            validation_data=val_ds,
            verbose=self._verbose,
            class_weight=class_weight,
            use_multiprocessing=True,
        )

        # breakpoint()
        logger.info("best_model")
        best_model = load_model(path, custom_objects={"BalancedAccuracy": BalancedAccuracy()})
        results = best_model.evaluate(test_ds, verbose=2)

        
        if update == True:
            path = os.path.join(os.getcwd(), "results", "e2e", model.name, str(subject), "history.csv")
            temp = pd.read_csv(path).drop("Unnamed: 0", axis=1)
            hist_df = pd.DataFrame(history.history)
            hist_df = pd.concat((temp, hist_df), axis=0).reset_index(drop=True)
            hist_df.to_csv(path)

        else:
            hist_df = pd.DataFrame(history.history)
            path = os.path.join(os.getcwd(), "results", "e2e", model.name, str(subject), "history.csv")
            hist_df.to_csv(path)

        self.plot_metrics(hist_df)
        path = os.path.join(os.getcwd(), "results", "e2e", model.name, str(subject), "metric.png")
        plt.savefig(path)

        # TH = self.treshold_CNN(best_model, train_ds)
        TH = 0.5

        predictions = np.array([])
        labels = np.array([])
        for x, y in test_ds:
            predictions = np.concatenate([predictions, best_model.predict(x).squeeze()])
            labels = np.concatenate([labels, y.numpy().squeeze()])

        self.plot_cm(labels, predictions, p=TH)
        path = os.path.join(os.getcwd(), "results", "e2e", model.name, str(subject), "cm.png")
        plt.savefig(path)
        plt.close()

        return best_model

    def test_e2e(self, train_ds:tf.data.Dataset, val_ds:tf.data.Dataset, test_ds:tf.data.Dataset, CNN_name: str, subject: int, U_data:tf.data.Dataset=None ):
        
        path = os.path.join(os.getcwd(), "results", "e2e", CNN_name, str(subject), "best.h5")
        best_model = load_model(path, custom_objects={"BalancedAccuracy": BalancedAccuracy()})
        results = best_model.evaluate(test_ds, verbose=2)
        logger.info(f"metrics pf test dataset: {dict(zip(best_model.metrics_names, np.round(results, 3)))}")
        results = dict(zip(best_model.metrics_names, np.round(results, 3)))

        results_dict = {
            "EER": "-", 
            "TH": "-",
            "ACC_ud": results['accuracy'],
            "BACC_ud": results['bacc'],
            "FAR_ud": results['fp']/(results['fp']+results['tn']),#todo
            "FRR_ud": results['fn']/(results['fn']+results['tp']),
            "unknown samples": "-",
            "AUS_All": "-",
            "FAU_All": "-",
            "CM_ud_TN": results['tn'],
            "CM_ud_FP": results['fp'],
            "CM_ud_FN": results['fn'],
            "CM_ud_TP": results['tp'],
        }

        if U_data != None:
            
            U_results = best_model.evaluate(U_data, verbose=2)
            logger.info(f"metrics pf U dataset: {dict(zip(best_model.metrics_names, np.round(U_results, 3)))}")

            U_results = dict(zip(best_model.metrics_names, np.round(U_results, 3)))
            results_dict["AUS_All"] = U_results['accuracy']
            results_dict["FAU_All"] = U_results['fp']
            

        results_dict["num_pc"] = '-'

        results_dict.update( {
            "test_id": self._test_id,
            "subject": subject,
            "combination": self._combination,
            "classifier_name": 'e2e',
            "normilizing": '-',
            "persentage": '-',
            "KFold": "-",
            "known_imposter": self._known_imposter,
            "unknown_imposter": self._unknown_imposter,
            "min_number_of_sample": self._min_number_of_sample,
        })     
        return results_dict

    def second_training(self, model, subject, train_ds, val_ds, test_ds, class_weight, update):

        if update == True:
            path = os.path.join( os.getcwd(), "results", "results", "second_train", model.name, str(subject), "best.h5", )
            binary_model = load_model(path, custom_objects={"BalancedAccuracy": BalancedAccuracy() })

        else:
            x = model.layers[-2].output
            output = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction1")(x)
            binary_model = Model(inputs=model.input, outputs=output)
            # breakpoint()
            # binary_model = self.lightweight_CNN((60,40,1), 1)

        METRICS = [
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="prc", curve="PR"),  # precision-recall curve
            BalancedAccuracy(name="bacc"),
        ]

        binary_model.compile(optimizer=self._CNN_optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=METRICS)  # if softmaxt then from_logits=False otherwise True

        path = os.path.join( os.getcwd(), "results", "results", "second_train", model.name, str(subject), "best.h5")
        checkpoint = [
            tf.keras.callbacks.ModelCheckpoint( path, save_best_only=True, monitor="val_loss", verbose=1, save_weights_only=False),
            # tf.keras.callbacks.ReduceLROnPlateau(  monitor="val_loss", factor=0.5, patience=30, min_lr=0.00001),
            # tf.keras.callbacks.EarlyStopping(      monitor="val_loss", patience=20, verbose=1),
            # tf.keras.callbacks.TensorBoard(        log_dir=TensorBoard_logs+str(self._test_id))
        ]
        
        
        history = binary_model.fit(
            train_ds,
            batch_size=self._CNN_batch_size,
            callbacks=[checkpoint],
            epochs=self._CNN_epochs,
            validation_data=val_ds,
            verbose=2,  # A._verbose,
            class_weight=class_weight,
            use_multiprocessing=True,
        )

        logger.info("best_model")
        best_model = load_model(path, custom_objects={"BalancedAccuracy": BalancedAccuracy()})
        results = best_model.evaluate(test_ds, verbose=0)

        for name, value in zip(best_model.metrics_names, results):
            print(name, ": ", value)
        print()

        # history.history["spec"] = [history.history["tn"][k] / (history.history["tn"][k] + history.history["fp"][k] + 1e-6) for k in range(len(history.history["tn"])) ]
        # history.history["sen"] = [ history.history["tp"][k] / (history.history["tp"][k] + history.history["fn"][k] + 1e-6) for k in range(len(history.history["tp"])) ]
        # history.history["bac"] = [(history.history["sen"][k] + history.history["spec"][k]) / 2 for k in range(len(history.history["sen"])) ]

        # history.history["val_spec"] = [history.history["val_tn"][k] / (history.history["val_tn"][k] + history.history["val_fp"][k] + 1e-6) for k in range(len(history.history["val_tn"]))]
        # history.history["val_sen"] = [ history.history["val_tp"][k] / (history.history["val_tp"][k] + history.history["val_fn"][k] + 1e-6) for k in range(len(history.history["val_tp"])) ]
        # history.history["val_bac"] = [ (history.history["val_sen"][k] + history.history["val_spec"][k]) / 2 for k in range(len(history.history["val_sen"])) ]

        if update == True:
            path = os.path.join( os.getcwd(), "results", "results", "second_train", model.name, str(subject), "history.csv")
            temp = pd.read_csv(path).drop("Unnamed: 0", axis=1)
            hist_df = pd.DataFrame(history.history)
            hist_df = pd.concat((temp, hist_df), axis=0).reset_index(drop=True)
            hist_df.to_csv(path)

        else:
            hist_df = pd.DataFrame(history.history)
            path = os.path.join(os.getcwd(), "results", "results", "second_train", model.name, str(subject), "history.csv")
            hist_df.to_csv(path)

        self.plot_metrics(hist_df)
        path = os.path.join(os.getcwd(), "results", "results", "second_train", model.name, str(subject), "metric.png")
        plt.savefig(path)

        # TH = self.treshold_CNN(best_model, train_ds)
        TH = 0.5

        predictions = np.array([])
        labels = np.array([])
        for x, y in test_ds:
            predictions = np.concatenate([predictions, best_model.predict(x).squeeze()])
            labels = np.concatenate([labels, y.numpy().squeeze()])


        self.plot_cm(labels, predictions, p=TH)
        best_model.evaluate(test_ds)
        path = os.path.join(os.getcwd(), "results", "results", "second_train", model.name, str(subject), "cm.png")
        plt.savefig(path)
        plt.close()

        return best_model

    def treshold_CNN(self, best_model, train_ds):

        predictions = np.array([])
        labels = np.array([])
        for x, y in train_ds:
            predictions = np.concatenate([predictions, best_model.predict(x).squeeze()])
            labels = np.concatenate([labels, y.numpy().squeeze()])
        FAR = []
        FRR = []
        for i in np.linspace(0, 1, 100): 
            cm = confusion_matrix(labels, predictions > i)
            FAR.append(cm[0][1] / (cm[0][1] + cm[0][0] + 1e-6))
            FRR.append(cm[1][0] / (cm[1][0] + cm[1][1] + 1e-6))
        eer, min_index = self.compute_eer(FAR, FRR)
        return np.linspace(0, 1, 100)[min_index]
        
    def plot_cm(self, labels, predictions, p=0.5):
        cm = confusion_matrix(labels, predictions > p)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion matrix @{:.2f}".format(p))
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")

        print("Legitimate Transactions Detected (True Negatives): ", cm[0][0])
        print("Legitimate Transactions Incorrectly Detected (False Positives): ", cm[0][1])
        print("Fraudulent Transactions Missed (False Negatives): ", cm[1][0])
        print("Fraudulent Transactions Detected (True Positives): ", cm[1][1])
        print("Total Fraudulent Transactions: ", np.sum(cm[1]))

    def plot_metrics(self, history):
        metrics = ["loss", "bacc", "accuracy"]
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        plt.figure(figsize=(10, 10))

        for n, metric in enumerate(metrics):
            name = metric.replace("_", " ").capitalize()
            plt.subplot(3, 1, n + 1)
            plt.plot(history[metric], color=colors[0], label="Train")
            plt.plot(history["val_" + metric], color=colors[0], linestyle="--", label="Val")
            plt.xlabel("Epoch")
            plt.ylabel(name)
            if metric == "loss":
                plt.ylim([0, plt.ylim()[1]])
            elif metric == "bac":
                plt.ylim([0.4, 1])
            else:
                plt.ylim([0, 1])
            plt.legend()


class Seamese(Deep_network):
    def __init__(
        self,
        dataset_name,
    ):
        super().__init__(dataset_name)

    def make_pairs(self, images, labels):
        # initialize two empty lists to hold the (image, image) pairs and
        # labels to indicate if a pair is positive or negative
        pairImages = []
        pairLabels = []
        # calculate the total number of classes present in the dataset
        # and then build a list of indexes for each class label that
        # provides the indexes for all examples with a given label
        numClasses = len(np.unique(labels))
        idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
        # loop over all images
        for idxA in range(len(images)):
            # grab the current image and label belonging to the current
            # iteration
            currentImage = images[idxA]
            label = labels[idxA]

            # randomly pick an image that belongs to the *same* class
            # label
            idxB = np.random.choice(idx[label])
            posImage = images[idxB]

            # prepare a positive pair and update the images and labels
            # lists, respectively
            pairImages.append([currentImage, posImage])
            pairLabels.append([1])

            # grab the indices for each of the class labels *not* equal to
            # the current label and randomly pick an image corresponding
            # to a label *not* equal to the current label
            negIdx = np.where(labels != label)[0]
            negImage = images[np.random.choice(negIdx)]

            # prepare a negative pair of images and update our lists
            pairImages.append([currentImage, negImage])
            pairLabels.append([0])
        # return a 2-tuple of our image pairs and labels
        return (np.array(pairImages), np.array(pairLabels))

    def euclidean_distance(self, vectors):
        # unpack the vectors into separate lists
        (featsA, featsB) = vectors
        # compute the sum of squared distances between the vectors
        sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
        # return the euclidean distance between the vectors
        return K.sqrt(K.maximum(sumSquared, K.epsilon()))

    def plot_training(self, H, plotPath):
        # construct a plot that plots and saves the training history
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H.history["loss"], label="train_loss")
        plt.plot(H.history["val_loss"], label="val_loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
        plt.savefig(plotPath)

    def build_siamese_model(self, inputShape, embeddingDim=48):
        # specify the inputs for the feature extractor network
        inputs = Input(inputShape)
        # define the first set of CONV => RELU => POOL => DROPOUT layers
        x = tf.keras.layers.Conv2D(64, (2, 2), padding="same", activation="relu")(
            inputs
        )
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        # second set of CONV => RELU => POOL => DROPOUT layers
        x = tf.keras.layers.Conv2D(64, (2, 2), padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        # prepare the final outputs
        pooledOutput = GlobalAveragePooling2D()(x)
        outputs = Dense(embeddingDim)(pooledOutput)
        # build the model
        model = Model(inputs, outputs)
        # return the model to the calling function
        return model

    def contrastive_loss(self, y, preds, margin=1):
        # explicitly cast the true class label data type to the predicted
        # class label data type (otherwise we run the risk of having two
        # separate data types, causing TensorFlow to error out)
        y = tf.cast(y, preds.dtype)
        # calculate the contrastive loss between the true labels and
        # the predicted labels
        squaredPreds = K.square(preds)
        squaredMargin = K.square(K.maximum(margin - preds, 0))
        loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
        # return the computed contrastive loss to the calling function
        return loss

    def train_Seamese_model(self, image_feature_name, dataset_name, update):
        # define the training and validation data generators
        pre_images, labels = self.loading_pre_features_image(dataset_name)
        pre_image = self.loading_image_features_from_list(
            pre_images, image_feature_name
        )

        encoded_labels = self.label_encoding(labels)

        outputs = len(labels["ID"].unique())

        images_feat_norm = self.normalizing_image_features(pre_image)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            images_feat_norm,
            encoded_labels,
            test_size=0.15,
            random_state=self._random_state,
            stratify=encoded_labels,
        )
        X_train, X_val, y_train, y_val = model_selection.train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=self._random_state,
            stratify=y_train,
        )

        # prepare the positive and negative pairs
        print("[INFO] preparing positive and negative pairs...")
        (pairTrain, labelTrain) = self.make_pairs(X_train, y_train)
        (pairTest, labelTest) = self.make_pairs(X_test, y_test)
        (pairval, labelval) = self.make_pairs(X_val, y_val)

        IMG_SHAPE = (60, 40, 1)

        # configure the siamese network
        print("[INFO] building siamese network...")
        imgA = Input(shape=IMG_SHAPE)
        imgB = Input(shape=IMG_SHAPE)
        featureExtractor = self.build_siamese_model(IMG_SHAPE)
        featsA = featureExtractor(imgA)
        featsB = featureExtractor(imgB)
        # finally, construct the siamese network
        distance = tf.keras.layers.Lambda(self.euclidean_distance)([featsA, featsB])
        model = Model(inputs=[imgA, imgB], outputs=distance, name="siamese")

        if update == True:
            path = os.path.join(os.getcwd(), "results", "siamese", "best.h5")
            model.load_weights(path)

        print("[INFO] compiling model...")
        model.compile(loss=self.contrastive_loss, optimizer="adam")

        # train the model
        print("[INFO] training model...")
        history = model.fit(
            [pairTrain[:, 0], pairTrain[:, 1]],
            labelTrain[:],
            validation_data=([pairval[:, 0], pairval[:, 1]], labelval[:]),
            callbacks=[
                calculating_threshold(
                    train=([pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:]),
                    validation=([pairval[:, 0], pairval[:, 1]], labelval[:]),
                )
            ],
            batch_size=self._CNN_batch_size,
            epochs=self._CNN_epochs,
        )

        # serialize the model to disk
        print("[INFO] saving siamese model...")
        path = os.path.join(os.getcwd(), "results", model.name, "best.h5")
        model.save(path)

        # plot the training history
        # print("[INFO] plotting training history...")
        # path = os.path.join( os.getcwd(), "results", model.name, "plot.png")
        # self.plot_training(history, path)

        if update == True:
            path = os.path.join(os.getcwd(), "results", model.name, "history.csv")
            temp = pd.read_csv(path).drop("Unnamed: 0", axis=1)
            hist_df = pd.DataFrame(history.history)
            hist_df = pd.concat((temp, hist_df), axis=0).reset_index(drop=True)
            path = os.path.join(os.getcwd(), "results", model.name, "history.csv")
            hist_df.to_csv(path)

        else:
            hist_df = pd.DataFrame(history.history)
            path = os.path.join(os.getcwd(), "results", model.name, "history.csv")
            hist_df.to_csv(path)

        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        ax[0].plot(hist_df["accuracy"], label="Train Accuracy")
        ax[0].plot(hist_df["val_accuracy"], label="Val Accuracy")

        ax[0].set_title("Accuracy")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Accuracy")
        ax[0].legend()

        # summarize history for loss
        ax[1].plot(hist_df["loss"], label="Train Loss")
        ax[1].plot(hist_df["val_loss"], label="Val Loss")
        ax[1].set_title("Loss")
        ax[1].set_ylabel("loss")
        ax[1].set_xlabel("epoch")
        ax[1].legend()

        path = os.path.join(os.getcwd(), "results", model.name, "plot.png")
        plt.savefig(path)

        plt.figure()
        preds = model.predict([pairTest[:, 0], pairTest[:, 1]])
        SS = pd.DataFrame(preds, np.squeeze(labelTest)).reset_index()
        SS.columns = ["Labels", "test scores"]
        sns.histplot(data=SS, x="test scores", hue="Labels", bins=100, kde=True)
        # plt.plot([TH,TH],[0,13], 'r--', linewidth=2, label="unbalanced threshold")
        # # axs[0].plot([TH1,TH1],[0,10], 'g:', linewidth=2, label="balanced threshold")
        # plt.set_title(f"EER: {round(EER,2)}  Threshold: {round(TH,2)}  ")
        # plt.savefig(PP+f"{str(self._known_imposter)}.png")
        plt.show()
        return model


class Pipeline(Classifier, Seamese):

    _col = [
        "test_id",
        "subject",
        "combination",
        "classifier_name",
        "normilizing",
        "persentage",
        "EER",
        "TH",
        "ACC_bd",
        "BACC_bd",
        "FAR_bd",
        "FRR_bd",
        "ACC_ud",
        "BACC_ud",
        "FAR_ud",
        "FRR_ud",
        "AUS",
        "FAU",
        "unknown_imposter_samples",
        "AUS_All",
        "FAU_All",
        "CM_bd_TN",
        "CM_bd_FP",
        "CM_bd_FN",
        "CM_bd_TP",
        "CM_ud_TN",
        "CM_ud_FP",
        "CM_ud_FN",
        "CM_ud_TP",
        "num_pc",
        "KFold",
        "p_training_samples",
        "train_ratio",
        "ratio",
        # pos_te_samples,
        # neg_te_samples,
        "known_imposter",
        "unknown_imposter",
        "min_number_of_sample",
        "number_of_unknown_imposter_samples",
        "y_train.shape[0]",
        "y_train.sum()",
        "y_val.shape[0]",
        "y_val.sum()",
        "y_test.shape[0]",
        "y_test.sum()",
    ]

    def __init__(self, kwargs):

        self.dataset_name = ""
        self._combination = 0

        self._labels = 0

        self._GRFs = pd.DataFrame()
        self._COAs = pd.DataFrame()
        self._COPs = pd.DataFrame()
        self._pre_images = pd.DataFrame()

        self._COA_handcrafted = pd.DataFrame()
        self._COP_handcrafted = pd.DataFrame()
        self._GRF_handcrafted = pd.DataFrame()

        self._GRF_WPT = pd.DataFrame()
        self._COP_WPT = pd.DataFrame()
        self._COA_WPT = pd.DataFrame()

        self._deep_features = pd.DataFrame()

        self._CNN_base_model = ""

        self._CNN_weights = "imagenet"
        self._CNN_include_top = False
        self._verbose = False
        self._CNN_batch_size = 32
        self._CNN_epochs = 10
        self._CNN_optimizer = "adam"
        self._val_size = 0.2

        #####################################################
        self._CNN_class_numbers = 97
        self._CNN_epochs = 10
        self._CNN_image_size = (60, 40, 3)

        self._min_number_of_sample = 30
        self._known_imposter = 5
        self._unknown_imposter = 30
        self._number_of_unknown_imposter_samples = 1.0  # Must be less than 1

        # self._known_imposter_list   = []
        # self._unknown_imposter_list = []

        self._waveletname = "coif1"
        self._pywt_mode = "constant"
        self._wavelet_level = 4

        self._KFold = 10
        self._random_state = 42

        self._p_training_samples = 11
        self._train_ratio = 4
        self._ratio = True

        self._classifier_name = ""

        self._KNN_n_neighbors = 5
        self._KNN_metric = "euclidean"
        self._KNN_weights = "uniform"
        self._SVM_kernel = "linear"
        self._random_runs = 10
        self._THRESHOLDs = np.linspace(0, 1, 100)
        self._persentage = 0.95
        self._normilizing = "z-score"

        self._num_pc = 0

        for (key, value) in kwargs.items():
            if key in self.__dict__:
                setattr(self, key, value)
            else:
                logger.error("key must be one of these:", self.__dict__.keys())
                raise KeyError(key)

        super().__init__(self.dataset_name, self._classifier_name)

    def run(self, DF_features_all: pd.DataFrame, feature_set_names: list):

        DF_known_imposter, DF_unknown_imposter = self.filtering_subjects_and_samples(
            DF_features_all
        )
        DF_unknown_imposter = DF_unknown_imposter.dropna()
        DF_known_imposter = DF_known_imposter.dropna()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # # extract features of shod dataset to use as unknown imposter samples
        # # it is overwrite on DF_unknown_imposter DataFrame
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # DF_features_all_shod, feature_set_names_shod = self.extracting_feature_set1('casia-shod')
        # DF_unknown_imposter = DF_features_all_shod[DF_features_all_shod['side']>=2.0].dropna()
        # subjects, samples = np.unique(DF_unknown_imposter["ID"].values, return_counts=True)

        # self._unknown_imposter_list = subjects[-self._unknown_imposter:]
        # DF_unknown_imposter =  DF_unknown_imposter[DF_unknown_imposter["ID"].isin(self._unknown_imposter_list)]

        # self.set_dataset_path('casia')
        # breakpoint()
        # ----------------------------------------------------------------

        results = list()
        for idx, subject in enumerate(self._known_imposter_list):
            # if idx not in [0, 1]: #todo: remove this block to run for all subjects.
            #     break

            if self._verbose == True:
                logger.info(
                    f"   Subject number: {idx} out of {len(self._known_imposter_list)} (subject ID is {subject})"
                )

            # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # # # droping shod samples from known imposter in training set
            # # # it is overwrite on DF_unknown_imposter DataFrame
            # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # index_of_shod_samples = DF_known_imposter[ (DF_known_imposter['side'] >= 2) & (DF_known_imposter['ID'] == subject)].index
            # DF_known_imposter1 = DF_known_imposter.drop(index_of_shod_samples)
            # #----------------------------------------------------------------

            # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # # # droping barefoot samples from unknown imposter
            # # # it is overwrite on DF_unknown_imposter DataFrame
            # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # index_of_barefoot_samples = DF_unknown_imposter[ DF_unknown_imposter['side'] <= 1 ].index
            # DF_unknown_imposter = DF_unknown_imposter.drop(index_of_barefoot_samples)
            # #----------------------------------------------------------------

            (
                DF_known_imposter_binariezed,
                DF_unknown_imposter_binariezed,
            ) = self.binarize_labels(DF_known_imposter, DF_unknown_imposter, subject)

            # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # # # applying template selection on known imposters
            # # # it is select only 200 samples from all knowwn imposters
            # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # A1 = DF_known_imposter_binariezed[DF_known_imposter_binariezed['ID'] == 1.0]
            # A2 = DF_known_imposter_binariezed[DF_known_imposter_binariezed['ID'] == 0.0]
            # A2 = self.template_selection(A2, 'DEND', 200, verbose=True)
            # DF_known_imposter_binariezed = pd.concat([A1, A2], axis=0)
            # # breakpoint()
            # #----------------------------------------------------------------

            CV = model_selection.StratifiedKFold(
                n_splits=self._KFold, shuffle=False
            )  # random_state=self._random_state,
            X = DF_known_imposter_binariezed
            U = DF_unknown_imposter_binariezed

            cv_results = list()

            ncpus = int(
                os.environ.get(
                    "SLURM_CPUS_PER_TASK", default=multiprocessing.cpu_count()
                )
            )
            pool = multiprocessing.Pool(processes=ncpus)

            for fold, (train_index, test_index) in enumerate(
                CV.split(X.iloc[:, :-1], X.iloc[:, -1])
            ):
                # breakpoint()
                # res = pool.apply_async(self.fold_calculating, args=(feature_set_names, subject, X, U, train_index, test_index, fold,))#, callback=print)#cv_results.append)
                # print(res.get())  # this will raise an exception if it happens within func

                cv_results.append(
                    self.fold_calculating(
                        feature_set_names, subject, X, U, train_index, test_index, fold
                    )
                )  # todo: comment this line to run all folds
                # break #todo: comment this line to run all folds

            pool.close()
            pool.join()
            # breakpoint()
            result = self.compacting_results(cv_results, subject)
            results.append(result)

        return pd.DataFrame(results, columns=self._col)

    def compacting_results(self, results, subject):
        # [EER, TH, ACC_bd, BACC_bd, FAR_bd, FRR_bd, ACC_ud, BACC_ud, FAR_ud, FRR_ud,]

        # return results, CM_bd, CM_ud
        # breakpoint()
        # pos_te_samples = self._p
        # neg_te_samples = self._
        # pos_tr_samples = self._
        # neg_tr_ratio = self._

        result = list()

        result.append(
            [
                self._test_id,
                subject,
                self._combination,
                self._classifier_name,
                self._normilizing,
                self._persentage,
                # configs["classifier"][CLS],
            ]
        )

        result.append(np.array(results).mean(axis=0))
        # result.append([np.array(CM_bd).mean(axis=0), np.array(CM_ud).mean(axis=0)])

        # _CNN_weights = 'imagenet'
        # _CNN_base_model = ""

        result.append(
            [
                self._KFold,
                self._p_training_samples,
                self._train_ratio,
                self._ratio,
                # pos_te_samples,
                # neg_te_samples,
                self._known_imposter,
                self._unknown_imposter,
                self._min_number_of_sample,
                self._number_of_unknown_imposter_samples,
            ]
        )

        return [val for sublist in result for val in sublist]

    def fold_calculating(self, feature_set_names: list, subject: int, X, U, train_index, test_index, fold):

        logger.info(f"\tFold number: {fold} out of {self._KFold} ({os.getpid()})")
        df_train = X.iloc[train_index, :]
        df_test = X.iloc[test_index, :]
        df_train = self.down_sampling_new(df_train, 2)
        
        df_train, df_test, df_test_U = self.scaler(df_train, df_test, U)

        df_train, df_test, df_test_U, num_pc = self.projector(feature_set_names, df_train, df_test, df_test_U, )
        results = self.ML_classifier(subject, x_train=df_train, x_test=df_test, x_test_U=df_test_U)

        results["num_pc"] = num_pc
        results.update({

            "training_samples": df_train.shape[0],
            "pos_training_samples": df_train['ID'].sum(),
            "validation_samples": 0,
            "pos_validation_samples": 0,
            "testing_samples": df_test.shape[0],
            "pos_testing_samples": df_test['ID'].sum(),
        })

        return results

    def collect_results(self, result: pd.DataFrame, pipeline_name: str) -> None:
        # result['pipeline'] = pipeline_name
        test = os.environ.get("SLURM_JOB_NAME", default=pipeline_name)
        excel_path = os.path.join(os.getcwd(), "results", f"Result__{test}.xlsx")

        if os.path.isfile(excel_path):
            Results_DF = pd.read_excel(excel_path, index_col=0)
        else:
            Results_DF = pd.DataFrame(columns=self._col)

        Results_DF = Results_DF.append(result)
        try:
            Results_DF.to_excel(excel_path)
        except Exception as e:
            logger.error(e)
            Results_DF.to_excel(excel_path[:-5] + str(self._test_id) + ".xlsx")

    def extracting_feature_set1(self, dataset_name: str) -> pd.DataFrame:
        GRFs, COPs, COAs, pre_images, labels = self.loading_pre_features(dataset_name)
        COA_handcrafted = self.loading_COA_handcrafted(COAs)
        COP_handcrafted = self.loading_COP_handcrafted(COPs)
        GRF_handcrafted = self.loading_GRF_handcrafted(GRFs)
        COA_WPT = self.loading_COA_WPT(COAs)
        COP_WPT = self.loading_COP_WPT(COPs)
        GRF_WPT = self.loading_GRF_WPT(GRFs)

        # deep_features_list = A.loading_deep_features_from_list((pre_images, labels), ['P100', 'P80'], 'resnet50.ResNet50')
        # image_from_list = A.loading_pre_image_from_list(pre_images, ['P80', 'P100'])
        # P70 = A.loading_pre_image(pre_images, 'P70')
        # P90 = A.loading_deep_features((pre_images, labels), 'P90', 'resnet50.ResNet50')

        feature_set_names = [
            "COP_handcrafted",
            "COPs",
            "COP_WPT",
            "GRF_handcrafted",
            "GRFs",
            "GRF_WPT",
        ]
        feature_set = []
        for i in feature_set_names:
            feature_set.append(eval(f"{i}"))

        return pd.concat(feature_set + [labels], axis=1), feature_set_names
