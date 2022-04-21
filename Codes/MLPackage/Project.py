import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import distance
import scipy.io

from pathlib import Path as Pathlb
import pywt, sys
import os, logging, timeit, pprint, copy, multiprocessing, glob
from itertools import product

# keras imports
import tensorflow as tf


from tensorflow.keras import preprocessing, callbacks 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten

from sklearn.cluster import KMeans
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn import preprocessing as sk_preprocessing
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import svm

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_curve)


import Butterworth
import convertGS2BW 
import matplotlib.pyplot as plt
import seaborn as sns

project_dir = os.getcwd()
log_path = os.path.join(project_dir, 'logs')



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
    t1 = blue + '[%(asctime)s]-' + yellow + '[%(name)s @%(lineno)d]' + reset + blue + '-[%(levelname)s]' + reset + bold_red
    t2 = ' %(message)s' + reset
    # breakpoint()
    formatter_colored = logging.Formatter( t1 + t2, datefmt='%m/%d/%Y %I:%M:%S %p ')
    formatter = logging.Formatter('[%(asctime)s]-[%(name)s @%(lineno)d]-[%(levelname)s]      %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p ')
    file_handler = logging.FileHandler( os.path.join(log_path, loggerName + '_loger.log'), mode = 'w')
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
    def __init__(self, ):
        super().__init__()


    def load_H5():
        pass

    def print_paths(self):
        logger.info(self._h5_path)
        logger.info(self._data_path)
        logger.info(self._meta_path)

    def set_dataset_path(self, dataset_name:str) -> None:
        """setting path for dataset"""
        if dataset_name == "casia":
            self._h5_path = os.path.join(os.getcwd(), "Datasets", "Casia-D", "footpressures_align.h5")
            self._data_path = os.path.join(os.getcwd(), "Datasets", "Casia-D", "Data-barefoot.npy")
            self._meta_path = os.path.join(os.getcwd(), "Datasets", "Casia-D", "Metadata-barefoot.npy")
            self._pre_features_path = os.path.join(os.getcwd(), "Datasets", "Casia-D", "pre_features")
            self._features_path = os.path.join(os.getcwd(), "Datasets", "Casia-D", "features")

        elif dataset_name == "stepscan":
            self._h5_path = os.path.join(os.getcwd(), "Datasets", "stepscan", "footpressures_align.h5")
            self._data_path = os.path.join(os.getcwd(), "Datasets", "stepscan", "Data-barefoot.npy")
            self._meta_path = os.path.join(os.getcwd(), "Datasets", "stepscan", "Metadata-barefoot.npy")
            self._pre_features_path = os.path.join(os.getcwd(), "Datasets", "stepscan", "pre_features")
            self._features_path = os.path.join(os.getcwd(), "Datasets", "stepscan", "features")

        elif dataset_name == "sfootbd":
            self._h5_path = os.path.join(os.getcwd(), "Datasets", "sfootbd", ".h5")
            self._mat_path = os.path.join(os.getcwd(), "Datasets", "sfootbd", "SFootBD")
            self._txt_path = os.path.join(os.getcwd(), "Datasets", "sfootbd", "IndexFiles")
            self._data_path = os.path.join(os.getcwd(), "Datasets", "sfootbd", "SFootBD", "Data.npy")
            self._meta_path = os.path.join(os.getcwd(), "Datasets", "sfootbd", "SFootBD", "Meta.npy")
            self._pre_features_path = os.path.join(os.getcwd(), "Datasets", "sfootbd", "pre_features")
            self._features_path = os.path.join(os.getcwd(), "Datasets", "sfootbd", "features")
            if not (os.path.isfile(self._data_path) and os.path.isfile(self._meta_path)):
                self.mat_to_numpy()

        else:
            logger.error("The name is not valid!!")
            sys.exit()

        Pathlb(self._pre_features_path).mkdir(parents=True, exist_ok=True)
        Pathlb(self._features_path).mkdir(parents=True, exist_ok=True)

    def extracting_labels(self, dataset_name:str) -> np.ndarray:
        if dataset_name == "casia":
            return self._meta[:,0:2]

        elif dataset_name == "stepscan":
            return self._meta[:-1,0:2]

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
                lst.append(int(label[ind]) )

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

                X = 2200 - temp['dataL'].shape[0]
                temp['dataL'] = np.pad(temp['dataL'], ((0,X),(0,0)), 'constant')
                X = 2200 - temp['dataR'].shape[0]
                temp['dataR'] = np.pad(temp['dataR'], ((0,X),(0,0)), 'constant')

                temp = np.concatenate((temp['dataL'], temp['dataR']), axis=1)
                data.append( temp )
                label.append( i.split("\\")[-1] )

        data = np.array(data)
        label = np.array(label)
        label = label[..., np.newaxis]
        np.save(self._data_path, data)
        np.save(self._meta_path, label)

    def loaddataset(self, dataset_name:str) -> np.ndarray:
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
            logger.error(f"The data and labels are not matched!! data shape ({data.shape[0]}) != labels shape {labels.shape[0]}")
            sys.exit()
        return data, labels


class PreFeatures(Database):
    def __init__(self, dataset_name:str):
        super().__init__()
        self._test_id = int(timeit.default_timer() * 1_000_000)
        self._features_set = dict()
        self.set_dataset_path(dataset_name)

        
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
        
        lowpass = Butterworth.Butterworthfilter(mode= "lowpass", fs = 100, cutoff = 5, order = 4)
        ML = lowpass.filter(ML)
        AP = lowpass.filter(AP)

        ML_f = ML - np.mean(ML)
        AP_f = AP - np.mean(AP)

        a = ML_f ** 2
        b = AP_f ** 2
        RD_f = np.sqrt(a + b)

        COPTS = np.stack((RD_f, AP_f, ML_f), axis = 0)
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
        GS2BW_object = convertGS2BW.convertGS2BW(mode = Binarize, TH = Threshold)
        aML = list()
        aAP = list()
        for i in range(Footprint3D.shape[2]):
            temp = Footprint3D[:, :, i]

            BW, threshold = GS2BW_object.GS2BW(temp)
            
            temp3 = ndimage.measurements.center_of_mass(BW)

            aML.append(temp3[1])
            aAP.append(temp3[0])


        lowpass = Butterworth.Butterworthfilter(mode= "lowpass", fs = 100, cutoff = 5, order = 4)
        aML = lowpass.filter(aML)
        aAP = lowpass.filter(aAP)
        aML_f = aML - np.mean(aML)
        aAP_f = aAP - np.mean(aAP)

        a = aML_f ** 2
        b = aAP_f ** 2
        aRD_f = np.sqrt(a + b)
        
        COATS = np.stack((aRD_f, aAP_f, aML_f), axis = 0)
        
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
    def pre_image(Footprint3D, eps = 5):
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
        x = np.ma.masked_array(temp, mask=temp==0)
        Tmin = np.argmin(x , axis=2, )

        P50  = np.percentile(Footprint3D,  50, axis=2)
        P60  = np.percentile(Footprint3D,  60, axis=2)
        P70  = np.percentile(Footprint3D,  70, axis=2)
        P80  = np.percentile(Footprint3D,  80, axis=2)
        P90  = np.percentile(Footprint3D,  90, axis=2)
        P100 = np.percentile(Footprint3D, 100, axis=2)

        pre_images = np.stack((CD, PTI, Tmin, Tmax, P50, P60, P70, P80, P90, P100), axis = -1)

        return pre_images
    
    def extracting_pre_features(self, dataset_name:str, combination:bool=True) -> pd.DataFrame:
        data, labels = self.loaddataset(dataset_name)
        GRFs = list()
        COPs = list()
        COAs = list()
        pre_images = list()
        # i=0
        for sample, label in zip(data, labels):
            # logger.info(  i )
            # i = i+1
    
            if combination==True and label[1]==0 and dataset_name=='casia':
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

        GRFs = pd.DataFrame(np.array(GRFs), columns=["GRF_"+str(i) for i in range(np.array(GRFs).shape[1])])
        COPs = pd.DataFrame(np.array(COPs), columns=["COP_"+str(i) for i in range(np.array(COPs).shape[1])]) 
        COAs = pd.DataFrame(np.array(COAs), columns=["COA_"+str(i) for i in range(np.array(COAs).shape[1])]) 
        pre_images = np.array(pre_images)

        self.saving_pre_features(GRFs, COPs, COAs, pre_images, labels, combination)
        return GRFs, COPs, COAs, pre_images, labels

    def saving_pre_features(self, GRFs, COPs, COAs, pre_images, labels, combination:bool=True):
        pd.DataFrame(labels, columns=["ID", "side",]).to_parquet(os.path.join(self._pre_features_path, f"label.parquet"))

        GRFs.to_parquet(os.path.join(self._pre_features_path, f"GRF_{combination}.parquet"))
        COAs.to_parquet(os.path.join(self._pre_features_path, f"COA_{combination}.parquet"))
        COPs.to_parquet(os.path.join(self._pre_features_path, f"COP_{combination}.parquet"))
        np.save(os.path.join(self._pre_features_path, f"pre_images_{combination}.npy"), pre_images)

    def loading_pre_features_COP(self, dataset_name:str, combination:bool=True) -> pd.DataFrame:
        try:
            labels = pd.read_parquet(os.path.join(self._pre_features_path, f"label.parquet"))
            COPs = pd.read_parquet(os.path.join(self._pre_features_path, f"COP_{combination}.parquet"))
            logger.info("COP curve were loaded!!!")

        except Exception as e: 
            logger.error(e) 
            logger.info("extraxting COP curve!!!")
            _, COPs, _, _, labels = self.extracting_pre_features(dataset_name, combination)

        self._features_set["COPs"] = {
            "columns": COPs.columns,
            "number_of_features": COPs.shape[1], 
            "number_of_samples": COPs.shape[0],           
        }  

        return COPs, labels

    def loading_pre_features_COA(self, dataset_name:str, combination:bool=True) -> pd.DataFrame:
        try:
            labels = pd.read_parquet(os.path.join(self._pre_features_path, f"label.parquet"))
            COAs = pd.read_parquet(os.path.join(self._pre_features_path, f"COA_{combination}.parquet"))
            logger.info("COA curve were loaded!!!")

                
        except Exception as e: 
            logger.error(e) 
            logger.info("extraxting COA curve!!!")
            _, _, COAs, _, labels = self.extracting_pre_features(dataset_name, combination)

        self._features_set["COAs"] = {
            "columns": COAs.columns,
            "number_of_features": COAs.shape[1], 
            "number_of_samples": COAs.shape[0],           
        }  
        
        return COAs, labels

    def loading_pre_features_GRF(self, dataset_name:str, combination:bool=True) -> pd.DataFrame:

        try:
            labels = pd.read_parquet(os.path.join(self._pre_features_path, f"label.parquet"))
            GRFs = pd.read_parquet(os.path.join(self._pre_features_path, f"GRF_{combination}.parquet"))
            logger.info("GRF curve were loaded!!!")

        except Exception as e: 
            logger.error(e) 
            logger.info("extraxting GRF curve!!!")
            GRFs, _, _, _, labels = self.extracting_pre_features(dataset_name, combination)

        self._features_set["GRFs"] = {
            "columns": GRFs.columns,
            "number_of_features": GRFs.shape[1], 
            "number_of_samples": GRFs.shape[0],           
        }  

        return GRFs, labels

    def loading_pre_features_image(self, dataset_name:str, combination:bool=True) -> pd.DataFrame:
        try:
            labels = pd.read_parquet(os.path.join(self._pre_features_path, f"label.parquet"))
            pre_images = np.load(os.path.join(self._pre_features_path, f"pre_images_{combination}.npy"))
            logger.info("image features were loaded!!!")
        except Exception as e: 
            logger.error(e) 
            logger.info("extraxting image features!!!")
            _, _, _, pre_images, labels = self.extracting_pre_features(dataset_name, combination)
        return pre_images, labels
    
    def loading_pre_features(self, dataset_name:str, combination:bool=True) -> pd.DataFrame:
        try:
            labels = pd.read_parquet(os.path.join(self._pre_features_path, f"label.parquet"))
            GRFs = pd.read_parquet(os.path.join(self._pre_features_path, f"GRF_{combination}.parquet"))
            COAs = pd.read_parquet(os.path.join(self._pre_features_path, f"COA_{combination}.parquet"))
            COPs = pd.read_parquet(os.path.join(self._pre_features_path, f"COP_{combination}.parquet"))
            pre_images = np.load(os.path.join(self._pre_features_path, f"pre_images_{combination}.npy"))
            logger.info(" all pre features were loaded!!!")
               
        except Exception as e: 
            logger.error(e) 
            logger.info("extraxting pre features!!!")
            GRFs, COPs, COAs, pre_images, labels = self.extracting_pre_features(dataset_name, combination)

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
    COX_feature_name = ['MDIST_RD', 'MDIST_AP', 'MDIST_ML', 'RDIST_RD', 'RDIST_AP', 'RDIST_ML', 
        'TOTEX_RD', 'TOTEX_AP', 'TOTEX_ML', 'MVELO_RD', 'MVELO_AP', 'MVELO_ML', 
        'RANGE_RD', 'RANGE_AP', 'RANGE_ML', 'AREA_CC',  'AREA_CE',  'AREA_SW', 
        'MFREQ_RD', 'MFREQ_AP', 'MFREQ_ML', 'FDPD_RD',  'FDPD_AP',  'FDPD_ML', 
        'FDCC',     'FDCE']

    GRF_feature_name = ["max_value_1", "max_value_1_ind", "max_value_2", "max_value_2_ind", 
        "min_value", "min_value_ind", "mean_value", "std_value", "sum_value"]

    _pre_image_names = ["CD", "PTI", "Tmin", "Tmax", "P50", "P60", "P70", "P80", "P90", "P100"]

    def __init__(self, dataset_name:str, combination:bool=True, waveletname:str="coif1", pywt_mode:str="constant", wavelet_level:int=4):
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
        RDIST = np.sqrt(np.mean(COPTS ** 2,axis=1))
        
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
        TOTEX.append(np.sum(np.sqrt((np.diff(COPTS[2,:])**2)+(np.diff(COPTS[1,:])**2))))
        TOTEX.append(np.sum(np.abs(np.diff(COPTS[1,:]))))
        TOTEX.append(np.sum(np.abs(np.diff(COPTS[2,:]))))
        
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
        RANGE.append(np.max(distance.cdist(COPTS[1:2,:].T, COPTS[1:2,:].T)))
        RANGE.append(np.max(COPTS[1,:])-np.min(COPTS[1,:]))
        RANGE.append(np.max(COPTS[2,:])-np.min(COPTS[2,:]))
        
        
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
        MVELO.append((np.sum(np.sqrt((np.diff(COPTS[2,:])**2)+(np.diff(COPTS[1,:])**2))))/T)
        MVELO.append((np.sum(np.abs(np.diff(COPTS[1,:]))))/T)
        MVELO.append((np.sum(np.abs(np.diff(COPTS[2,:]))))/T)
        
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
        z05 = 1.645 # z.05 = the z statistic at the 95% confidence level
        SRD = np.sqrt((RDIST[0]**2)-(MDIST[0]**2)) #the standard deviation of the RD time series
        
        AREACC = np.pi*((MDIST[0]+(z05*SRD))**2)
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
        SAPML = np.mean(COPTS[2,:]*COPTS[1,:])
        AREACE = 2*np.pi*F05*np.sqrt((SAP**2)*(SML**2)-(SAPML**2))

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
        
        AP = COPTS[1,:]
        ML = COPTS[2,:]

        AREASW = np.sum( np.abs((AP[1:]*ML[:-1])-(AP[:-1]*ML[1:])))/(2*T)
        
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
        MFREQ.append( TOTEX[0]/(2*np.pi*T*MDIST[0]) )
        MFREQ.append( TOTEX[1]/(4*np.sqrt(2)*T*MDIST[1]))
        MFREQ.append( TOTEX[2]/(4*np.sqrt(2)*T*MDIST[2]))

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
        Nd = [elemt*N for elemt in d]
        dev = [i / j for i, j in zip(Nd, TOTEX)]
        
        
        FDPD = np.log(N)/np.log(dev)
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
        z05 = 1.645; # z.05 = the z statistic at the 95% confidence level
        SRD = np.sqrt((RDIST[0]**2)-(MDIST[0]**2)) #the standard deviation of the RD time series

        d = 2*(MDIST[0]+z05*SRD)
        TOTEX = self.computeTOTEX(COPTS)
        
        FDCC = np.log(N)/np.log((N*d)/TOTEX[0])
        return FDCC

    def computeFDCE(self, COPTS):
        """
        computeFDCE(COPTS)
        FD-CE : Fractal Dimension based on the 95% Confidence Ellipse
        COPTS [3,t] : (RD,) AP, ML COP time series
        return FDCE [2] : FD-CE_AP, FD-CE_ML
        """
        
        
        N = COPTS.shape[1]
        F05 = 3; 
        RDIST = self.computeRDIST(COPTS)
        SAPML = np.mean(COPTS[2,:]*COPTS[1,:])
        
        d = np.sqrt(8*F05*np.sqrt(((RDIST[1]**2)*(RDIST[2]**2))-(SAPML**2)))
        TOTEX = self.computeTOTEX(COPTS)

        FDCE = np.log(N)/np.log((N*d)/TOTEX[0])
        
        return FDCE

    @staticmethod
    def computeGRFfeatures(GRF):
        """
        computeGRFfeatures(GRF)
        GRF: [t] time series signal
        return GFR features: [9] (max_value_1, max_value_1_ind, max_value_2, max_value_2_ind, min_value, min_value_ind, mean_value, std_value, sum_value)
        """
        # handcraft_features = list()
        L = int(len(GRF)/2)

        max_value_1 = np.max(GRF[:L])
        max_value_1_ind = np.argmax( GRF[:L] )
        max_value_2 = np.max(GRF[L:])
        max_value_2_ind = L + np.argmax(	GRF[L:] )

        min_value = np.min(GRF[max_value_1_ind:max_value_2_ind])
        min_value_ind = max_value_1_ind + np.argmin(	GRF[max_value_1_ind:max_value_2_ind] )

        mean_value = np.mean(GRF)
        std_value = np.std(GRF)
        sum_value = np.sum(GRF)

        return [max_value_1, max_value_1_ind,
                        max_value_2, max_value_2_ind,
                        min_value, min_value_ind,
                        mean_value, std_value, sum_value]

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
            coif family: coif1, coif2, coif3, coif4, coif5, coif6, coif7, coif8, coif9, coif10, coif11, coif12, coif13, coif14, coif15, coif16, coif17bior family: bior1.1, bior1.3, bior1.5, bior2.2, bior2.4, bior2.6, bior2.8, bior3.1, bior3.3, bior3.5, bior3.7, bior3.9, bior4.4, bior5.5, bior6.8
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

        dwt_coeff = pywt.wavedec(signal, self._waveletname, mode=self._pywt_mode, level=self._wavelet_level)
        dwt_coeff = np.concatenate(dwt_coeff).ravel()

        return dwt_coeff

    ## COA
    def extraxting_COA_handcrafted(self, COAs:np.ndarray) -> pd.DataFrame:
        COA_handcrafted = list()
        for _, sample in COAs.iterrows():
            sample = sample.values.reshape(3,100)

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

            COA_handcrafted.append(np.concatenate((MDIST, RDIST, TOTEX, MVELO, RANGE, [AREACC], [AREACE], [AREASW], MFREQ, FDPD, [FDCC], [FDCE]), axis = 0))
            
            
        COA_handcrafted = pd.DataFrame(np.array(COA_handcrafted), columns=self.COX_feature_name)

        self.saving_dataframe(COA_handcrafted, "COA_handcrafted")

        return COA_handcrafted

    def saving_dataframe(self, data:pd.DataFrame, name:str) -> None:
        Pathlb(self._features_path).mkdir(parents=True, exist_ok=True)
        data.to_parquet(os.path.join(self._features_path, f"{name}_{self._combination}.parquet"))
            
    def loading_COA_handcrafted(self, COAs:np.ndarray) -> pd.DataFrame:
        try:
            COA_handcrafted = pd.read_parquet(os.path.join(self._features_path, f"COA_handcrafted_{self._combination}.parquet"))   
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
    def extraxting_COP_handcrafted(self, COPs:np.ndarray) -> pd.DataFrame:
        COP_handcrafted = list()
        for _, sample in COPs.iterrows():
            sample = sample.values.reshape(3,100)

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

            COP_handcrafted.append(np.concatenate((MDIST, RDIST, TOTEX, MVELO, RANGE, [AREACC], [AREACE], [AREASW], MFREQ, FDPD, [FDCC], [FDCE]), axis = 0))
            
        COP_handcrafted = pd.DataFrame(np.array(COP_handcrafted), columns=self.COX_feature_name) 
        self.saving_dataframe(COP_handcrafted, "COP_handcrafted")
        return COP_handcrafted
            
    def loading_COP_handcrafted(self, COPs:np.ndarray) -> pd.DataFrame:
        try:
            COP_handcrafted = pd.read_parquet(os.path.join(self._features_path, f"COP_handcrafted_{self._combination}.parquet"))
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
    def extraxting_GRF_handcrafted(self, GRFs:np.ndarray) -> pd.DataFrame:
        GRF_handcrafted = list()
        for _, sample in GRFs.iterrows():
            GRF_handcrafted.append(self.computeGRFfeatures(sample))
               
        GRF_handcrafted = pd.DataFrame(np.array(GRF_handcrafted), columns=self.GRF_feature_name)
        self.saving_dataframe(GRF_handcrafted, "GRF_handcrafted")
        return GRF_handcrafted
            
    def loading_GRF_handcrafted(self, GRFs:np.ndarray) -> pd.DataFrame:
        try:
            GRF_handcrafted = pd.read_parquet(os.path.join(self._features_path, f"GRF_handcrafted_{self._combination}.parquet"))
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
    def extraxting_GRF_WPT(self, GRFs:np.ndarray) -> pd.DataFrame:
        GRF_WPT = list()
        for _, sample in GRFs.iterrows():
            GRF_WPT.append(self.wt_feature(sample))
               
        GRF_WPT = pd.DataFrame(np.array(GRF_WPT), columns=["GRF_WPT_"+str(i) for i in range(np.array(GRF_WPT).shape[1])]) 
        self.saving_dataframe(GRF_WPT, "GRF_WPT")

        return GRF_WPT

    def loading_GRF_WPT(self, GRFs:np.ndarray) -> pd.DataFrame:
        try:
            GRF_WPT = pd.read_parquet(os.path.join(self._features_path, f"GRF_WPT_{self._combination}.parquet"))
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
    def extraxting_COP_WPT(self, COPs:np.ndarray) -> pd.DataFrame:
        COP_WPT = list()
        for _, sample in COPs.iterrows():
            sample = sample.values.reshape(3,100)
            wt_COA_RD = self.wt_feature(sample[0,:])
            wt_COA_AP = self.wt_feature(sample[1,:])
            wt_COA_ML = self.wt_feature(sample[2,:])
            COP_WPT.append(np.concatenate((wt_COA_RD, wt_COA_AP, wt_COA_ML), axis = 0))
               
        COP_WPT = pd.DataFrame(np.array(COP_WPT), columns=["COP_WPT_"+str(i) for i in range(np.array(COP_WPT).shape[1])])  
        self.saving_dataframe(COP_WPT, "COP_WPT")
        return COP_WPT
        
    def loading_COP_WPT(self, COPs:np.ndarray) -> pd.DataFrame:
        try:
            COP_WPT = pd.read_parquet(os.path.join(self._features_path, f"COP_WPT_{self._combination}.parquet"))
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
    def extraxting_COA_WPT(self, COAs:np.ndarray) -> pd.DataFrame:
        COA_WPT = list()
        for _, sample in COAs.iterrows():
            sample = sample.values.reshape(3,100)
            wt_COA_RD = self.wt_feature(sample[0,:])
            wt_COA_AP = self.wt_feature(sample[1,:])
            wt_COA_ML = self.wt_feature(sample[2,:])
            COA_WPT.append(np.concatenate((wt_COA_RD, wt_COA_AP, wt_COA_ML), axis = 0))
               
        COA_WPT = pd.DataFrame(np.array(COA_WPT), columns=["COA_WPT_"+str(i) for i in range(np.array(COA_WPT).shape[1])]) 
        self.saving_dataframe(COA_WPT, "COA_WPT")
        return COA_WPT
        
    def loading_COA_WPT(self, COAs:np.ndarray) -> pd.DataFrame:
        try:
            COA_WPT = pd.read_parquet(os.path.join(self._features_path, f"COA_WPT_{self._combination}.parquet"))
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
        images = tf.image.grayscale_to_rgb(tf.expand_dims(images, -1))
        images = tf.image.resize(images, (224, 224))
        return images, labels

    def extraxting_deep_features(self, data:tuple, pre_image_name:str, CNN_base_model:str) -> pd.DataFrame:
        # self._CNN_base_model = CNN_base_model
        
        try:
            logger.info(f"Loading { CNN_base_model } model...")
            base_model = eval(f"tf.keras.applications.{CNN_base_model}(weights='{self._CNN_weights}', include_top={self._CNN_include_top})")
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


        input = tf.keras.layers.Input(shape= (224, 224, 3), dtype = tf.float64, name="original_img")
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

            if (Deep_features.shape[0]-1) % 256 == 0:
                logger.info(f" ->>> ({os.getpid()}) completed images: " + str(Deep_features.shape[0]))


        Deep_features = Deep_features[1:, :]
        logger.info(f"Deep features shape: {Deep_features.shape}")

        # time = int(timeit.default_timer() * 1_000_000)
        exec(f"deep_{pre_image_name}_{CNN_name} = pd.DataFrame(Deep_features, columns=['deep_{pre_image_name}_{CNN_name}_'+str(i) for i in range(Deep_features.shape[1])])")

        self.saving_deep_features(eval(f"deep_{pre_image_name}_{CNN_name}"), pre_image_name, CNN_name)

        return eval(f"deep_{pre_image_name}_{CNN_name}")

    def normalizing_pre_image(self, pre_images:np.ndarray, pre_image_name:str) -> np.ndarray:
        if not pre_image_name in self._pre_image_names:
            raise Exception("Invalid pre image name!!!")

        i = self._pre_image_names.index(pre_image_name)
        maxvalues = np.max(pre_images[..., i])
        return pre_images[..., i]/maxvalues

    def saving_deep_features(self, data:pd.DataFrame, pre_image_name:str, CNN_name:str) -> None:
        Pathlb(self._features_path).mkdir(parents=True, exist_ok=True)
        exec(f"data.to_parquet(os.path.join(self._features_path, f'deep_{pre_image_name}_{CNN_name}_{self._combination}.parquet'))")
                   
    def loading_deep_features(self, data:tuple, pre_image_name:str, CNN_base_model:str) -> pd.DataFrame:
        CNN_name = CNN_base_model.split(".")[0]
        if not pre_image_name in self._pre_image_names:
            raise Exception("Invalid pre image name!!!")
        try:
            logger.info(f"loading deep features from {pre_image_name}!!!")
            exec(f"df = pd.read_parquet(os.path.join(self._features_path, f'deep_{pre_image_name}_{CNN_name}_{self._combination}.parquet'))")

        except Exception as e: 
            logger.error(e) 
            logger.info(f"extraxting deep features from {pre_image_name}!!!")
            exec(f'df = self.extraxting_deep_features(data, pre_image_name, CNN_base_model)')

        self._features_set[f'deep_{pre_image_name}_{CNN_name}'] = {
            "columns": eval('df.columns'),
            "number_of_features": eval('df.shape[1]'), 
            "number_of_samples": eval('df.shape[0]'),           
        }
        return eval('df')

    def loading_deep_features_from_list(self, data:tuple, pre_image_names:list, CNN_base_model:str) -> pd.DataFrame:
        """loading deep features from a list of image features"""
        CNN_name = CNN_base_model.split(".")[0]
        sss = []
        for pre_image_name in pre_image_names:
            if not pre_image_name in self._pre_image_names:
                raise Exception("Invalid pre image name!!!")
            try:
                exec(f"{pre_image_name} = pd.read_parquet(os.path.join(self._features_path, f'deep_{pre_image_name}_{CNN_name}_{self._combination}.parquet'))")
                logger.info(f"loading deep features from {pre_image_name}!!!")

            except Exception as e: 
                logger.error(e) 
                logger.info(f"extraxting deep features  from {pre_image_name}!!!")
                exec(f"{pre_image_name} = self.extraxting_deep_features(data, pre_image_name, CNN_base_model)")

            self._features_set[f'deep_{pre_image_name}_{CNN_name}'] = {
                "columns": eval(f"{pre_image_name}.columns"),
                "number_of_features": eval(f"{pre_image_name}.shape[1]"), 
                "number_of_samples": eval(f"{pre_image_name}.shape[0]"),           
            }
        
            sss.append(eval(f"{pre_image_name}"))
        return sss

    ## images
    def extraxting_pre_image(self, pre_images:np.ndarray, pre_image_name:str) -> pd.DataFrame:
        if not pre_image_name in self._pre_image_names:
            raise Exception("Invalid pre image name!!!")

        pre_image = list()
        for idx in range(pre_images.shape[0]):

            sample = pre_images[idx,..., self._pre_image_names.index(pre_image_name)]
            sample = sample.reshape(-1)
            pre_image.append(sample)
               
        exec(f"I = pd.DataFrame(np.array(pre_image), columns=['{pre_image_name}_pixel_'+str(i) for i in range(np.array(pre_image).shape[1])]) ")
        exec(f"self.saving_pre_image(I, '{pre_image_name}')")

        return eval("I")

    def saving_pre_image(self, data, pre_image_name:str) -> None:
        Pathlb(self._features_path).mkdir(parents=True, exist_ok=True)
        exec(f"data.to_parquet(os.path.join(self._features_path, '{pre_image_name}_{self._combination}.parquet'))")

    def loading_pre_image(self, pre_images:np.ndarray, pre_image_name: str) -> pd.DataFrame:
        """loading a pre image from a excel file."""
        if not pre_image_name in self._pre_image_names:
            raise Exception("Invalid pre image name!!!")
        try:
            exec(f"I = pd.read_parquet(os.path.join(self._features_path, '{pre_image_name}_{self._combination}.parquet'))")
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

    def loading_pre_image_from_list(self, pre_images:np.ndarray, list_pre_image:list) -> list:
        """loading multiple pre image features from list"""
        sss = []
        for pre_image_name in list_pre_image:
            if not pre_image_name in self._pre_image_names:
                raise Exception("Invalid pre image name!!!")

            try:
                exec(f"{pre_image_name} = pd.read_parquet(os.path.join(self._features_path, '{pre_image_name}_{self._combination}.parquet'))")
                logger.info(f"loading {pre_image_name} features!!!")

            except Exception as e: 
                logger.error(e) 
                logger.info(f"extraxting {pre_image_name} features!!!")
                exec(f"{pre_image_name} = self.extraxting_pre_image(pre_images, pre_image_name)")
                
            self._features_set[f"{pre_image_name}"] = {
                "columns": eval(f"{pre_image_name}.columns"),
                "number_of_features": eval(f"{pre_image_name}.shape[1]"), 
                "number_of_samples": eval(f"{pre_image_name}.shape[0]"),           
            }
        
            sss.append(eval(f"{pre_image_name}"))
        return sss

    def pack(self, list_features:list, labels:pd.DataFrame) -> pd.DataFrame:
        """
        list of features=[
            [GRFs, COAs, COPs, 
            COA_handcrafted, COP_handcrafted, GRF_handcrafted,
            deep_P100_resnet50, deep_P80_resnet50, deep_P90_resnet50,
            P50, P60, P70, 
            COA_WPT, COP_WPT, GRF_WPT]
        """
        return pd.concat( list_features + [labels], axis=1)

    def filtering_subjects_and_samples(self, DF_features_all:pd.DataFrame) -> pd.DataFrame:
        subjects, samples = np.unique(DF_features_all["ID"].values, return_counts=True)

        ss = [a[0] for a in list(zip(subjects, samples)) if a[1]>=self._min_number_of_sample]

        if self._known_imposter + self._unknown_imposter > len(ss):
            raise Exception("Invalid _known_imposter and _unknown_imposter!!!")

        self._known_imposter_list   = ss[:self._known_imposter] 
        self._unknown_imposter_list = ss[-self._unknown_imposter:] 


       
        DF_unknown_imposter =  DF_features_all[DF_features_all["ID"].isin(self._unknown_imposter_list)]
        DF_known_imposter =    DF_features_all[DF_features_all["ID"].isin(self._known_imposter_list)]

        DF_unknown_imposter = DF_unknown_imposter.groupby('ID', group_keys=False).apply(lambda x: x.sample(frac=self._number_of_unknown_imposter_samples, replace=False, random_state=self._random_state))
        
        return DF_known_imposter, DF_unknown_imposter

    ## fine_tuning
    def FT_deep_features(self, data:tuple, training_data:str, pre_image_name:str, CNN_base_model:str) -> pd.DataFrame:

        logger.info("fine_tuning")

        S = Features(training_data)

        pre_images, labels = S.loading_pre_features_image(training_data)
        pre_image_norm = S.normalizing_pre_image(pre_images, pre_image_name)




        # # ##################################################################
        # #                phase 3: processing labels
        # # ##################################################################
        le = sk_preprocessing.LabelEncoder()
        le.fit(labels['ID'])

        logger.info(f"Number of subjects: {len(np.unique(labels['ID']))}")

        transfered_labels = le.transform(labels['ID'])

        # labels = tf.keras.utils.to_categorical(labels, num_classes=len(np.unique(indices)))

        logger.info(f"features shape: {pre_image_norm.shape}")
        logger.info(f"features shape: {transfered_labels.shape}")

        self._val_size = .2 #todo: change this value
        X_train, X_val, y_train, y_val = model_selection.train_test_split(pre_image_norm, transfered_labels, test_size=self._val_size, random_state=self._random_state, stratify=transfered_labels)


        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(self.resize_images).shuffle(1000)
        train_ds = train_ds.batch(self._CNN_batch_size)
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).map(self.resize_images).shuffle(1000)
        val_ds = val_ds.batch(self._CNN_batch_size)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # # ##################################################################
        # #                phase 4: loading CNN
        # # ##################################################################
        try:
            IMG_SHAPE = (224, 224, 3)
            logger.info(f"Loading { CNN_base_model } model...")
            base_model = eval(f"tf.keras.applications.{CNN_base_model}(input_shape=IMG_SHAPE, weights='{self._CNN_weights}', include_top={self._CNN_include_top})")
            logger.info("Successfully loaded base model and model...")
            base_model.trainable = False
            CNN_name = CNN_base_model.split(".")[0]
            logger.info(f"MaduleName: {CNN_name}")

        except Exception as e: 
            base_model = None
            logger.error("The base model could NOT be loaded correctly!!!")
            logger.error(e)
            sys.exit()

        
        for images, labels in train_ds.take(1):
            print(images.shape)

            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i])
                # plt.title(labels[i])
                plt.axis("off")
                print(images[i].shape)


        data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'),tf.keras.layers.RandomRotation(0.2),])
        for image, _ in train_ds.take(1):
            plt.figure(figsize=(10, 10))
            first_image = image[0]
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
                plt.imshow(augmented_image[0])
                plt.axis('off')
        # breakpoint()

        input = tf.keras.layers.Input(shape=IMG_SHAPE, dtype = tf.float64, name="original_img")
        x = tf.cast(input, tf.float32)
        x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
        x = tf.keras.layers.RandomRotation(0.2)(x)
        x = tf.keras.layers.RandomZoom(0.1)(x)
        x = eval("tf.keras.applications." + CNN_name + ".preprocess_input(x)")
        x = base_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Flatten()(x)
        # x = tf.keras.layers.Dense(512,  activation='relu', name="last_dense-2")(x) # kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        # x = tf.keras.layers.Dense(256,  activation='relu', name="last_dense-1")(x) # kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        # # x = tf.keras.layers.Dropout(0.2)(x)
        # x = tf.keras.layers.Dense(128,  activation='relu', name="last_dense")(x) # kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        # # x = tf.keras.layers.Dropout(0.2)(x)
        output = tf.keras.layers.Dense(256, name="prediction")(x) # activation='softmax',

        model = tf.keras.models.Model(inputs=input, outputs=output, name=CNN_name)
        
        # breakpoint()

        # # ##################################################################
        # #                phase 7: training CNN
        # # ##################################################################

        model.compile(
            optimizer=tf.keras.optimizers.Adam(), #learning_rate=0.001
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #if softmaxt then from_logits=False otherwise True
            # metrics=["sparse_categorical_accuracy"]
            metrics=METRICS
            )

        time = int(timeit.timeit()*1_000_000)
        # TensorBoard_logs =  os.path.join( configs["paths"]["TensorBoard_logs"], "_".join(("FT", SLURM_JOBID, CNN_name, configs["features"]["image_feature_name"], str(time)) )  )
        # path = configs["CNN"]["saving_path"] + "_".join(( "FT", SLURM_JOBID, CNN_name, configs["features"]["image_feature_name"], "best.h5" ))

        checkpoint = [
                tf.keras.callbacks.ModelCheckpoint(    './best.h5', save_best_only=True, monitor="val_loss"),
                tf.keras.callbacks.ReduceLROnPlateau(  monitor="val_loss", factor=0.5, patience=30, min_lr=0.00001),
                tf.keras.callbacks.EarlyStopping(      monitor="val_loss", patience=90, verbose=1),
                # tf.keras.callbacks.TensorBoard(        log_dir=TensorBoard_logs)   
            ]    


        history = model.fit(
            train_ds,    
            batch_size=self._CNN_batch_size,
            callbacks=[checkpoint],
            epochs=self._CNN_epochs,
            validation_data=val_ds,
            verbose=self._verbose,
        )

        # path = configs["CNN"]["saving_path"] + "_".join(( "FT", SLURM_JOBID, CNN_name, configs["features"]["image_feature_name"], str(int(np.round(test_acc*100)))+"%" + ".h5" ))
        # model.save(path)
        plt.plot(history.history['accuracy'], label='accuracy')
        # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()
        breakpoint()

        # # ##################################################################
        # #                phase 5: loading pre image features
        # # ##################################################################
        plt.show()
        breakpoint()

        pre_image_norm = self.normalizing_pre_image(data[0], pre_image_name)

        train_ds = tf.data.Dataset.from_tensor_slices((pre_image_norm, data[1])) 
        train_ds = train_ds.batch(self._CNN_batch_size)
        logger.info(f"batch_size: {self._CNN_batch_size}")
        train_ds = train_ds.map(self.resize_images)


        
        x = eval("tf.keras.applications." + CNN_name + ".preprocess_input(x)")
        x = base_model(x)
        output = tf.keras.layers.GlobalMaxPool2D()(x)

        model = tf.keras.Model(input, output, name=CNN_name)

        return history


class Classifier(Features):
    
    def __init__(self, dataset_name, classifier_name):
        super().__init__(dataset_name)
        self._classifier_name=classifier_name

    def binarize_labels(self, DF_known_imposter, DF_unknown_imposter, subject):
        DF_known_imposter_binariezed = DF_known_imposter.copy().drop(["side"], axis=1)
        DF_known_imposter_binariezed["ID"] = DF_known_imposter_binariezed["ID"].map(lambda x: 1.0 if x==subject else 0.0)


        DF_unknown_imposter_binariezed = DF_unknown_imposter.copy().drop(["side"], axis=1)
        DF_unknown_imposter_binariezed["ID"] = DF_unknown_imposter_binariezed["ID"].map(lambda x: 1.0 if x==subject else 0.0)
        
        return DF_known_imposter_binariezed, DF_unknown_imposter_binariezed
        
    def down_sampling(self, DF):

        number_of_negatives =DF[ DF["ID"]==0 ].shape[0]
        if self._ratio == True:
            self._n_training_samples = int(self._p_training_samples*self._train_ratio)
            if number_of_negatives < (self._p_training_samples*self._train_ratio):
                self._n_training_samples = int(number_of_negatives)
        
        else:
            self._n_training_samples = self._train_ratio
            if number_of_negatives < self._train_ratio:
                self._n_training_samples = int(number_of_negatives)
        
            
        DF_positive_samples_train = DF[ DF["ID"]==1 ].sample(n=self._p_training_samples, replace=False, random_state=self._random_state)
        DF_negative_samples_train = DF[ DF["ID"]==0 ].sample(n=self._n_training_samples, replace=False, random_state=self._random_state)

        return pd.concat([DF_positive_samples_train, DF_negative_samples_train], axis=0)
    
    def scaler(self, df_train, df_test, df_test_U):

        if self._normilizing == "minmax":
            scaling = sk_preprocessing.MinMaxScaler()

        elif self._normilizing == "z-score":
            scaling = sk_preprocessing.StandardScaler()

        else:
            raise KeyError(self._normilizing)


        Scaled_train = scaling.fit_transform(df_train.iloc[:, :-1])
        Scaled_test = scaling.transform(df_test.iloc[:, :-1])            

        Scaled_train = pd.DataFrame(np.concatenate((Scaled_train, df_train.iloc[:, -1:].values), axis = 1), columns=df_train.columns)
        Scaled_test  = pd.DataFrame(np.concatenate((Scaled_test,  df_test.iloc[:, -1:].values),  axis = 1), columns=df_test.columns)
        
        Scaled_test_U = pd.DataFrame(columns=df_test_U.columns)
       
        if df_test_U.shape[0] != 0:
            Scaled_test_U = scaling.transform(df_test_U.iloc[:, :-1])
            Scaled_test_U  = pd.DataFrame(np.concatenate((Scaled_test_U,  df_test_U.iloc[:, -1:].values),  axis = 1), columns=df_test_U.columns)

        return Scaled_train, Scaled_test, Scaled_test_U
        
    def projector(self, df_train, df_test, df_test_U, listn):
        # elif persentage != 1.0:
        #     principal = PCA(svd_solver="full")
        #     PCA_out_train = principal.fit_transform(df_train.iloc[:,:-1])
        #     PCA_out_test = principal.transform(df_test.iloc[:,:-1])

        #     variance_ratio = np.cumsum(principal.explained_variance_ratio_)
        #     high_var_PC = np.zeros(variance_ratio.shape)
        #     high_var_PC[variance_ratio <= persentage] = 1

        #     num_pc = int(np.sum(high_var_PC))

        #     columnsName = ["PC"+str(i) for i in list(range(1, num_pc+1))] + ["ID"]

        #     df_train_pc = pd.DataFrame(np.concatenate((PCA_out_train[:,:num_pc], df_train.iloc[:, -1:].values), axis = 1), columns = columnsName)
        #     df_test_pc  = pd.DataFrame(np.concatenate(( PCA_out_test[:,:num_pc],  df_test.iloc[:, -1:].values), axis = 1), columns = columnsName)

        #     self._num_pc = num_pc

        #     return df_train_pc, df_test_pc

        if self._persentage == 1.0:
            num_pc = df_train.shape[1]-1
            columnsName = ["PC"+str(i) for i in list(range(1, num_pc+1))] + ["ID"]

            df_train.columns = columnsName
            df_test.columns = columnsName
            df_test_U.columns = columnsName

            self._num_pc = num_pc

            return df_train, df_test, df_test_U

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

                N.append( int(np.sum(high_var_PC))  )
                columnsName = [listn[ind]+"_PC"+str(i) for i in list(range(1, N[ind]+1))]

                exec( f"df_train_pc_{ind} = pd.DataFrame(PCA_out_train[:,:N[ind]], columns = columnsName)" )
                exec( f"df_test_pc_{ind} = pd.DataFrame(PCA_out_test[:,:N[ind]], columns = columnsName)" )
                
                if df_test_U.shape[0] != 0:
                    PCA_out_test_U = principal.transform(df_test_U.loc[:, col])
                    exec( f"df_test_U_pc_{ind} = pd.DataFrame(PCA_out_test_U[:,:N[ind]], columns = columnsName)" )
           
            tem = [("df_train_pc_"+str(i)) for i in range(len(listn))] + ['df_train["ID"]']
            exec( f"df_train_pc = pd.concat({tem}, axis=1)".replace("'",""))
            tem = [("df_test_pc_"+str(i)) for i in range(len(listn))] + ['df_test["ID"]']
            exec( f"df_test_pc = pd.concat({tem}, axis=1)".replace("'",""))

            exec( f"df_test_U_pc = pd.DataFrame(columns = columnsName)".replace("'",""))
            if df_test_U.shape[0] != 0:
                tem = [("df_test_U_pc_"+str(i)) for i in range(len(listn))] + ['df_test_U["ID"]']
                exec( f"df_test_U_pc = pd.concat({tem}, axis=1)".replace("'",""))

            self._num_pc = np.sum(N)

            return eval("df_train_pc"), eval("df_test_pc"), eval("df_test_U_pc")

    def FXR_calculater(self, x_train, y_pred):
        FRR = list()
        FAR = list()

        for tx in self._THRESHOLDs:
            E1 = np.zeros((y_pred.shape))
            E1[y_pred >= tx] = 1

            e = pd.DataFrame([x_train.values, E1]).T
            e.columns = ["y", "pred"]
            e['FAR'] = e.apply(lambda x: 1 if x['y'] < x['pred'] else 0, axis=1)
            e['FRR'] = e.apply(lambda x: 1 if x['y'] > x['pred'] else 0, axis=1)
            
            a1 = e.sum()
            N = e.shape[0]-a1["y"]
            P = a1["y"]
            FRR.append(a1['FRR']/P)
            FAR.append(a1['FAR']/N)

        return FRR, FAR

    def balancer(self, DF, method="random", ratio=1): # None, DEND, MDIST, Random
        pos_samples = DF[DF["ID"]==1]
        n = pos_samples.shape[0]
        neg_samples = DF[DF["ID"]==0]#.sample()#, random_state=cfg.config["Pipeline"]["random_state"])
        neg_samples = self.template_selection(neg_samples, 
                                        method=method, 
                                        k_cluster=n*ratio, 
                                        verbose=False)
        DF_balanced = pd.concat([pos_samples, neg_samples])
        return DF_balanced, pos_samples.shape[0]

    @staticmethod
    def compute_eer(FAR, FRR):
        """ Returns equal error rate (EER) and the corresponding threshold. """
        abs_diffs = np.abs(np.subtract(FRR, FAR)) 
        min_index = np.argmin(abs_diffs)
        min_index = 99 - np.argmin(abs_diffs[::-1])
        eer = np.mean((FAR[min_index], FRR[min_index]))
        
        return eer, min_index

    def template_selection(self, DF, method, k_cluster, verbose=True):
        if DF.shape[0]<k_cluster:
            k_cluster=DF.shape[0]
    
        if method == "DEND":
            kmeans = KMeans(n_clusters=k_cluster, random_state=self._random_state )
            kmeans.fit(DF.iloc[:, :-2].values)
            clusters = np.unique(kmeans.labels_)
            col = DF.columns

            DF1 = DF.copy().reset_index(drop=True)
            for i, r in DF.reset_index(drop=True).iterrows():
                
                DF1.loc[i,"dist"] = distance.euclidean(kmeans.cluster_centers_[kmeans.labels_[i]], r[:-2].values)
                DF1.loc[i,"label"] = kmeans.labels_[i]
            DF_clustered = list()

            for cluster in clusters:
                mean_cluster = DF1[DF1["label"] == cluster].sort_values(by=['dist'], )
                DF_clustered.append(mean_cluster.iloc[0,:-2])

            DF_clustered  = pd.DataFrame(DF_clustered, columns=col)
            if verbose: 
                logger.info(f"Clustered data shape: {DF_clustered.shape}")
                logger.info(f"original data shape: {DF.shape}")

        elif method == "MDIST":
            A = distance.squareform(distance.pdist(DF.iloc[:, :-2].values)).mean(axis=1)
            i = np.argsort(A)[:k_cluster]
            DF_clustered = DF.iloc[i, :]
            DF_clustered  = pd.DataFrame(np.concatenate((DF_clustered,  DF.iloc[:, -2:].values),  axis = 1), columns=DF.columns)

        elif method == "None":
            DF_clustered  = pd.DataFrame(DF, columns=DF.columns)

        elif method == "Random":
            DF_clustered  = pd.DataFrame(DF, columns=DF.columns).sample(n=k_cluster)

        return DF_clustered

    def ML_classifier(self, x_train, x_test, x_test_U, a):
        
        if self._classifier_name=="knn":
            classifier = knn(n_neighbors=self._KNN_n_neighbors, metric=self._KNN_metric, weights=self._KNN_weights, n_jobs=-1)

            best_model = classifier.fit(x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values)
            y_pred_tr = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]
            FRR_t, FAR_t = self.FXR_calculater(x_train["ID"], y_pred_tr)
            EER, t_idx = self.compute_eer(FRR_t, FAR_t)
            TH = self._THRESHOLDs[t_idx]

        elif self._classifier_name=="TM":
            positives = x_train[x_train["ID"]== 1.0] 
            negatives = x_train[x_train["ID"]== 0.0] 
            similarity_matrix_positives, similarity_matrix_negatives = self.compute_score_matrix(positives, negatives)
            client_scores, imposter_scores = self.compute_scores(similarity_matrix_positives, similarity_matrix_negatives, criteria="min")
            y_pred_tr = np.append(client_scores.data, imposter_scores.data)

            # classifier = knn(n_neighbors=1, metric=self._KNN_metric, weights=self._KNN_weights, n_jobs=-1)
            # best_model = classifier.fit(x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values)
            # y_pred_tr = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]
            FRR_t, FAR_t = self.FXR_calculater(x_train["ID"], y_pred_tr)
            self.plot_eer(FRR_t, FAR_t)
            EER, t_idx = self.compute_eer(FRR_t, FAR_t)
            TH = self._THRESHOLDs[t_idx]
        
        elif self._classifier_name=="svm":
            classifier = svm.SVC(kernel=self._SVM_kernel , probability=True, random_state=self._random_state)

            best_model = classifier.fit(x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values)
            y_pred_tr = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]
            FRR_t, FAR_t = self.FXR_calculater(x_train["ID"], y_pred_tr)
            EER, t_idx = self.compute_eer(FRR_t, FAR_t)
            TH = self._THRESHOLDs[t_idx]
        else:
            raise Exception(f"_classifier_name ({self._classifier_name}) is not valid!!")

        

        acc = list()
        CMM = list()
        BACC = list()
        for _ in range(self._random_runs):
            DF_temp, pos_number = self.balancer(x_test, method="Random")

            if self._classifier_name=="TM":
                positives = x_train[x_train["ID"]== 1.0]
                similarity_matrix_positives, similarity_matrix_negatives = self.compute_score_matrix(positives, DF_temp)
                client_scores, imposter_scores = self.compute_scores(similarity_matrix_positives, similarity_matrix_negatives, criteria="min")
                y_pred = imposter_scores.data

            else:
                y_pred = best_model.predict_proba(DF_temp.iloc[:, :-1].values)[:, 1]

            
            y_pred[y_pred >= TH ] = 1.
            y_pred[y_pred <  TH ] = 0.

            acc.append( accuracy_score(DF_temp.iloc[:,-1].values, y_pred)*100 )
            CM = confusion_matrix(DF_temp.iloc[:,-1].values, y_pred)
            spec = (CM[0,0]/(CM[0,1]+CM[0,0]+1e-33))*100
            sens = (CM[1,1]/(CM[1,0]+CM[1,1]+1e-33))*100
            BACC.append( (spec + sens)/2 )
            CMM.append(CM)
        
        ACC_bd = np.mean(acc)
        CM_bd = np.array(CMM).sum(axis=0) 
        BACC_bd = np.mean(BACC)
        FAR_bd = CM_bd[0,1]/CM_bd[0,:].sum()
        FRR_bd = CM_bd[1,0]/CM_bd[1,:].sum()
        
        
        
        if self._classifier_name=="TM":
            positives = x_train[x_train["ID"]== 1.0]
            similarity_matrix_positives, similarity_matrix_negatives = self.compute_score_matrix(positives, x_test)
            client_scores, imposter_scores = self.compute_scores(similarity_matrix_positives, similarity_matrix_negatives, criteria="min")
            y_pred = imposter_scores.data

        else:
            y_pred = best_model.predict_proba(x_test.iloc[:, :-1].values)[:, 1]
        
        y_pred1 = y_pred.copy()
        y_pred[y_pred >= TH ] = 1
        y_pred[y_pred <  TH ] = 0


        ACC_ud = accuracy_score(x_test["ID"].values, y_pred)*100 
        CM_ud = confusion_matrix(x_test.iloc[:,-1].values, y_pred)
        spec = (CM_ud[0,0]/(CM_ud[0,1]+CM_ud[0,0]+1e-33))*100
        sens = (CM_ud[1,1]/(CM_ud[1,0]+CM_ud[1,1]+1e-33))*100
        BACC_ud = (spec + sens)/2 
        FAR_ud = CM_ud[0,1]/CM_ud[0,:].sum()
        FRR_ud = CM_ud[1,0]/CM_ud[1,:].sum()

        AUS, FAU = 100, 0
        AUS_All, FAU_All = 100, 0


        if x_test_U.shape[0] != 0:

            AUS, FAU = [], []
            for _ in range(self._random_runs):
                numbers = x_test_U.shape[0] if x_test_U.shape[0]<60 else 60
                temp = x_test_U.sample(n=numbers)

                if self._classifier_name=="TM":
                    positives = x_train[x_train["ID"]== 1.0]
                    similarity_matrix_positives, similarity_matrix_negatives = self.compute_score_matrix(positives, temp)
                    client_scores, imposter_scores = self.compute_scores(similarity_matrix_positives, similarity_matrix_negatives, criteria="min")
                    y_pred = imposter_scores.data

                else:
                    y_pred = best_model.predict_proba(temp.iloc[:, :-1].values)[:, 1]

                y_pred_U = y_pred
                y_pred_U[y_pred_U >= TH ] = 1.
                y_pred_U[y_pred_U <  TH ] = 0.

                AUS.append(accuracy_score(temp["ID"].values, y_pred_U)*100 )
                FAU.append(np.where(y_pred_U==1)[0].shape[0])
            AUS = np.mean(AUS)
            FAU = np.mean(FAU)

            
            if self._classifier_name=="TM":
                positives = x_train[x_train["ID"]== 1.0]
                similarity_matrix_positives, similarity_matrix_negatives = self.compute_score_matrix(positives, x_test_U)
                client_scores, imposter_scores = self.compute_scores(similarity_matrix_positives, similarity_matrix_negatives, criteria="min")
                y_pred_U = imposter_scores.data

            else:
                y_pred_U = best_model.predict_proba(x_test_U.iloc[:, :-1].values)[:, 1]

            y_pred_U1 = y_pred_U.copy()

            y_pred_U[y_pred_U >= TH ] = 1.
            y_pred_U[y_pred_U <  TH ] = 0.
            AUS_All = accuracy_score(x_test_U["ID"].values, y_pred_U)*100 
            FAU_All = np.where(y_pred_U==1)[0].shape[0]

        # breakpoint()
        # #todo
        PP = f"./C/{self._classifier_name}/{str(a)}/{str(self._unknown_imposter)}/1/"
        PP1 = f"./C/{self._classifier_name}/{str(a)}/{str(self._unknown_imposter)}/2/"
        PP2 = f"./C/{self._classifier_name}/{str(a)}/{str(self._unknown_imposter)}/3/"
        PP3 = f"./C/{self._classifier_name}/{str(a)}/{str(self._unknown_imposter)}/4/"
        Pathlb(PP).mkdir(parents=True, exist_ok=True)
        Pathlb(PP1).mkdir(parents=True, exist_ok=True)
        Pathlb(PP2).mkdir(parents=True, exist_ok=True)
        Pathlb(PP3).mkdir(parents=True, exist_ok=True)
        # breakpoint()

        plt.figure()
        SS = pd.DataFrame(y_pred_tr,x_train['ID'].values).reset_index()
        SS.columns = ["Labels","train scores"]
        sns.histplot(data=SS, x="train scores", hue="Labels", bins=100)
        plt.plot([TH,TH],[0,13], 'r--', linewidth = 2)
        plt.title(f"Number of known Imposters: {str(self._known_imposter)} \n EER: {round(EER,2)}      Threshold: {round(TH,2)}")
        plt.savefig(PP+f"{str(self._known_imposter)}.png")


        plt.figure()
        SS = pd.DataFrame(y_pred1,x_test['ID'].values).reset_index()
        SS.columns = ["Labels","test scores"]
        sns.histplot(data=SS, x="test scores", hue="Labels", bins=100)
        plt.plot([TH,TH],[0,13], 'r--', linewidth = 2)
        plt.title(f"known Imposters: {str(self._known_imposter)},   ACC: {round(ACC_ud,2)},    BACC: {round(BACC_ud,2)},   CM: {CM_ud}")
        plt.savefig(PP1+f"{str(self._known_imposter)}.png")


        plt.figure()
        sns.histplot(y_pred_U1, bins=100)
        plt.plot([TH,TH],[0,13], 'r--', linewidth = 2)
        plt.xlabel("unknown imposter scores")
        plt.title(f"Number of known Imposters: {str(self._known_imposter)},\n AUS: {round(AUS_All,2)},       FAU: {round(FAU_All,2)}")
        plt.savefig(PP2+f"{str(self._known_imposter)}.png")

        plt.figure()
        plt.scatter(x_train.iloc[:, 0].values, x_train.iloc[:, 1].values, c ="red", marker ="s", label="train", s = x_train.iloc[:, -1].values*22+1)
        plt.scatter(x_test.iloc[:, 0].values, x_test.iloc[:, 1].values,  c ="blue", marker ="*", label="test", s = x_test.iloc[:, -1].values*22+1)
        plt.scatter(x_test_U.iloc[:, 0].values, x_test_U.iloc[:, 1].values, c ="green", marker ="o", label="u", s = 5)
        plt.title(f'# training positives: {x_train[x_train["ID"]== 1.0].shape[0]},       # training negatives: {x_train[x_train["ID"]== 0.0].shape[0]} \n # test positives: {x_test[x_test["ID"]== 1.0].shape[0]},       # test negatives: {x_test[x_test["ID"]== 0.0].shape[0]}               # test_U : {x_test_U.shape[0]}')

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.savefig(PP3+f"{str(self._known_imposter)}.png")
        # plt.close('all')
        print(self._known_imposter_list) 
        print(self._unknown_imposter_list)

        # plt.show()
        # breakpoint()  


        results = [EER, TH, ACC_bd, BACC_bd, FAR_bd, FRR_bd, ACC_ud, BACC_ud, FAR_ud, FRR_ud, AUS, FAU, x_test_U.shape[0], AUS_All, FAU_All]

        return results, CM_bd, CM_ud      

    @staticmethod
    def compute_score_matrix(positive_samples, negative_samples):
        """ Returns score matrix of trmplate matching"""
        positive_model = np.zeros((positive_samples.shape[0], positive_samples.shape[0]))
        negative_model = np.zeros((positive_samples.shape[0], negative_samples.shape[0]))

        for i in range(positive_samples.shape[0]):
            for j in range(positive_samples.shape[0]):
                positive_model[i, j] = distance.euclidean(positive_samples.iloc[i, :-1], positive_samples.iloc[j, :-1])
            for j in range(negative_samples.shape[0]):
                negative_model[i, j] = distance.euclidean(positive_samples.iloc[i, :-1], negative_samples.iloc[j, :-1])
        
        return np.power(positive_model+1, -1), np.power(negative_model+1, -1), 

    @staticmethod
    def compute_scores(similarity_matrix_positives, similarity_matrix_negatives, criteria = "min"):
        if criteria == "average":
            client_scores = np.mean(np.ma.masked_where(similarity_matrix_positives==1,similarity_matrix_positives), axis = 0)
            client_scores = np.expand_dims(client_scores,-1)
            
            imposter_scores = (np.sum(similarity_matrix_negatives, axis = 0))/(similarity_matrix_positives.shape[1])
            imposter_scores = np.expand_dims(imposter_scores, -1)
                
        elif criteria == "min":
            client_scores = np.max(np.ma.masked_where(similarity_matrix_positives==1,similarity_matrix_positives), axis = 0)
            client_scores = np.expand_dims(client_scores,-1)
            
            imposter_scores = np.max(np.ma.masked_where(similarity_matrix_negatives==1,similarity_matrix_negatives), axis = 0)
            imposter_scores = np.expand_dims(imposter_scores, -1)
                    
        elif criteria == "median":
            client_scores = np.median(similarity_matrix_positives, axis = 0)
            client_scores = np.expand_dims(client_scores,-1)            

            imposter_scores = np.median(similarity_matrix_negatives, axis = 0)
            imposter_scores = np.expand_dims(imposter_scores, -1)

        return client_scores, imposter_scores

    @staticmethod
    def plot_eer(FAR, FRR):
        """ Returns equal error rate (EER) and the corresponding threshold. """
        abs_diffs = np.abs(np.subtract(FRR, FAR)) 
        
        min_index = np.argmin(abs_diffs)
        # breakpoint()
        min_index = 99 - np.argmin(abs_diffs[::-1])
        plt.figure(figsize=(5,5))
        eer = np.mean((FAR[min_index], FRR[min_index]))
        plt.plot( np.linspace(0, 1, 100), FRR, label = "FRR")
        plt.plot( np.linspace(0, 1, 100), FAR, label = "FAR")
        plt.plot(np.linspace(0, 1, 100)[min_index], eer, "r*",label = "EER")
        plt.legend()
        # plt.savefig(path, bbox_inches='tight')

        # plt.show()


class Pipeline(Classifier):
  
    _col = ["test_id",
        "subject", 
        "combination", 
        "classifier_name", 
        "normilizing", 
        "persentage", 
        "num_pc",
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

        self._CNN_weights = 'imagenet'
        self._CNN_include_top = False
        self._verbose = False
        self._CNN_batch_size = 32
        self._CNN_epochs = 10
        self._CNN_optimizer = 'adam'
        self._val_size = 0.2

        #####################################################
        self._CNN_class_numbers = 97
        self._CNN_epochs = 10
        self._CNN_image_size = (60, 40, 3)

        self._min_number_of_sample = 30
        self._known_imposter = 5
        self._unknown_imposter = 30
        self._number_of_unknown_imposter_samples = 1.0  # Must be less than 1


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

    def run(self, listn:list, label:pd.DataFrame, feature_set_names:list):
        DF_features_all = self.pack(listn, label)

        
        DF_known_imposter, DF_unknown_imposter = self.filtering_subjects_and_samples(DF_features_all)

        results = list()
        for idx, subject in enumerate(self._known_imposter_list):
            if subject not in [4,5,6,7]: #todo: remove this  5, 6, 7
                break


            if self._verbose == True:
                logger.info(f"     Subject number: {idx} out of {len(self._known_imposter_list)} (subject ID is {subject})")
            DF_known_imposter_binariezed, DF_unknown_imposter_binariezed = self.binarize_labels(DF_known_imposter, DF_unknown_imposter, subject)

            CV = model_selection.StratifiedKFold(n_splits=self._KFold, random_state=self._random_state, shuffle=True)
            X = DF_known_imposter_binariezed
            U = DF_unknown_imposter_binariezed

            cv_results = list()
          
            ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default=multiprocessing.cpu_count()))
            pool = multiprocessing.Pool(processes=ncpus)


            for fold, (train_index, test_index) in enumerate(CV.split(X.iloc[:,:-1], X.iloc[:,-1])):

                pool.apply_async(self.fold_calculating, args=(feature_set_names, subject, X, U, train_index, test_index, fold), callback=cv_results.append)
                break #todo: remove this

            pool.close()
            pool.join()

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

        result.append([
            self._test_id,
            subject, 
            self._combination, 
            self._classifier_name, 
            self._normilizing, 
            self._persentage, 
            self._num_pc, 
            # configs["classifier"][CLS], 
        ])

        result.append(np.array(results).mean(axis=0))
        # result.append([np.array(CM_bd).mean(axis=0), np.array(CM_ud).mean(axis=0)])
        

        # _CNN_weights = 'imagenet'
        # _CNN_base_model = ""

        result.append([
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
        ])

        return [val for sublist in result for val in sublist]

    def fold_calculating(self, feature_set_names:list, subject:int, X, U, train_index, test_index, fold):
        logger.info(f"\t   Fold number: {fold} out of {self._KFold} ({os.getpid()})")
        df_train = X.iloc[train_index, :]
        df_test = X.iloc[test_index, :]
        df_train = self.down_sampling(df_train)

        df_train, df_test, df_test_U = self.scaler(df_train, df_test, U)
        df_train, df_test, df_test_U = self.projector(df_train, df_test, df_test_U, feature_set_names)
        result, CM_bd, CM_ud = self.ML_classifier(df_train, df_test, df_test_U, subject)
        
        return result + CM_ud.reshape(1,-1).tolist()[0] + CM_bd.reshape(1,-1).tolist()[0]

    def collect_results(self, result: pd.DataFrame, pipeline_name: str) -> None:
        result['pipeline'] = pipeline_name
        test = os.environ.get('SLURM_JOB_NAME', default=pipeline_name)
        excel_path = os.path.join(os.getcwd(), "results", f"Result__{test}.xlsx")

        if os.path.isfile(excel_path):
            Results_DF = pd.read_excel(excel_path, index_col = 0)
        else:
            Results_DF = pd.DataFrame(columns=self._col)

        Results_DF = Results_DF.append(result)
        try:
            Results_DF.to_excel(excel_path)
        except Exception as e: 
            logger.error(e) 
            Results_DF.to_excel(excel_path[:-5]+str(self._test_id)+'.xlsx') 


class Deep_network(Pipeline):

    def __init__(self, dataset_name, classifier_name, kwargs):
        super().__init__(dataset_name, classifier_name, kwargs)
        
    
    def label_encoding(self):

        indices = self._labels["ID"]
        logger.info("metadata shape: {}".format(indices.shape))

        indices = self._labels["ID"]
        le = sk_preprocessing.LabelEncoder()
        le.fit(indices)

        logger.info(f"Number of subjects: {len(np.unique(indices))}")

        return le.transform(indices)


    def deep_model_1(self, image_size):
        CNN_name = "from_scratch"
        Number_of_subjects = len(np.unique(self._labels["ID"]))

        input = tf.keras.layers.Input(shape=image_size, dtype = tf.float64, name="original_img")
        x = tf.cast(input, tf.float32)
        x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
        x = tf.keras.layers.RandomRotation(0.2)(x)
        x = tf.keras.layers.RandomZoom(0.1)(x)
        x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128,  activation='relu', name="last_dense")(x) # kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        x = tf.keras.layers.Dropout(0.2)(x)
        output = tf.keras.layers.Dense(Number_of_subjects, name="prediction")(x) # activation='softmax',

        ## The CNN Model
        return tf.keras.models.Model(inputs=input, outputs=output, name=CNN_name)


    def deep_training_1(self, pre_image_name):
        encode_label = self.label_encoding()
        pre_images_norm = self.normalizing_pre_image(pre_image_name)

        pre_images_norm = pre_images_norm[...,tf.newaxis]
        pre_images_norm = np.concatenate((pre_images_norm, pre_images_norm, pre_images_norm), axis=-1)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(pre_images_norm, encode_label, test_size=0.15, random_state=42, stratify=encode_label)
        X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000)
        train_ds = train_ds.batch(self._CNN_batch_size)
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(1000)
        val_ds = val_ds.batch(self._CNN_batch_size)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.batch(self._CNN_batch_size)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        image_size = X_train.shape[1:]
        model = self.deep_model_1(image_size)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(), #learning_rate=0.001
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            metrics=["Accuracy"]
            )

        TensorBoard_logs =  os.path.join( os.getcwd(), "logs", "TensorBoard_logs", "_".join(("FS", str(os.getpid()), pre_image_name, str(self._test_id)) )  )
        path = os.path.join( os.getcwd(), "results", "deep_model", "_".join( ("FS", str(os.getpid()), pre_image_name, "best.h5") ))

        checkpoint = [
                tf.keras.callbacks.ModelCheckpoint(    path, save_best_only=True, monitor="val_loss"),
                tf.keras.callbacks.ReduceLROnPlateau(  monitor="val_loss", factor=0.5, patience=30, min_lr=0.00001),
                tf.keras.callbacks.EarlyStopping(      monitor="val_loss", patience=90, verbose=1),
                tf.keras.callbacks.TensorBoard(        log_dir=TensorBoard_logs+str(self._test_id))   
            ]    

        history = model.fit(
            train_ds,    
            batch_size=self._CNN_batch_size,
            callbacks=[checkpoint],
            epochs=self._CNN_epochs,
            validation_data=val_ds,
            verbose=self._verbose,
        )

        test_loss, test_acc = model.evaluate(test_ds, verbose=2)

        path = os.path.join( os.getcwd(), "results", "deep_model", "_".join( ("FS", str(os.getpid()), pre_image_name, str(int(np.round(test_acc*100)))+"%.h5") ))
        model.save(path)
        
        # plt.plot(history.history['accuracy'], label='accuracy')
        # # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.show()


        logger.info(f"test_loss: {np.round(test_loss)}, test_acc: {int(np.round(test_acc*100))}%")


    def deep_model_2(self, image_size):
        CNN_name = "Omar-2017"
        Number_of_subjects = len(np.unique(self._labels["ID"]))

        input = tf.keras.layers.Input(shape=image_size, dtype = tf.float64, name="original_img")
        x = tf.cast(input, tf.float32)

        x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
        x = tf.keras.layers.RandomRotation(0.2)(x)
        x = tf.keras.layers.RandomZoom(0.1)(x)

        x = tf.keras.layers.Conv2D(20, kernel_size=(7, 7), activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(20, kernel_size=(7, 7), activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(x)

        # x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Flatten()(x)
        output = tf.keras.layers.Dense(127, activation='softmax', name="last_dense")(x) # kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        # x = tf.keras.layers.Dropout(0.2)(x)
        # output = tf.keras.layers.Dense(Number_of_subjects, activation='softmax', name="prediction")(x) # 

        ## The CNN Model
        return tf.keras.models.Model(inputs=input, outputs=output, name=CNN_name)


    def deep_training_2(self, pre_image_name):
        self._CNN_batch_size = 16
        encode_label = self.label_encoding()
        pre_images_norm = self.normalizing_pre_image(pre_image_name)

        pre_images_norm = pre_images_norm[...,tf.newaxis]
        pre_images_norm = np.concatenate((pre_images_norm, pre_images_norm, pre_images_norm), axis=-1)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(pre_images_norm, encode_label, test_size=0.15, random_state=42, stratify=encode_label)
        X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000)
        train_ds = train_ds.batch(self._CNN_batch_size)
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(1000)
        val_ds = val_ds.batch(self._CNN_batch_size)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.batch(self._CNN_batch_size)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        image_size = X_train.shape[1:]
        model = self.deep_model_2(image_size)

        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), #learning_rate=0.001
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            metrics=["Accuracy"]
            )

        TensorBoard_logs =  os.path.join( os.getcwd(), "logs", "TensorBoard_logs", "_".join(("FS", str(os.getpid()), pre_image_name, str(self._test_id)) )  )
        path = os.path.join( os.getcwd(), "results", "deep_model", "_".join( ("FS", str(os.getpid()), pre_image_name, "best.h5") ))

        checkpoint = [
                tf.keras.callbacks.ModelCheckpoint(    path, save_best_only=True, monitor="val_loss"),
                tf.keras.callbacks.ReduceLROnPlateau(  monitor="val_loss", factor=0.5, patience=10, min_lr=0.00001),
                tf.keras.callbacks.EarlyStopping(      monitor="val_loss", patience=10, verbose=1),
                tf.keras.callbacks.TensorBoard(        log_dir=TensorBoard_logs+str(self._test_id))   
            ]    

        history = model.fit(
            train_ds,    
            batch_size=self._CNN_batch_size,
            callbacks=[checkpoint],
            epochs=self._CNN_epochs,
            validation_data=val_ds,
            verbose=self._verbose,
        )

        test_loss, test_acc = model.evaluate(test_ds, verbose=2)

        path = os.path.join( os.getcwd(), "results", "deep_model", "_".join( ("FS", str(os.getpid()), pre_image_name, str(int(np.round(test_acc*100)))+"%.h5") ))
        model.save(path)
        breakpoint()
        plt.plot(history.history['accuracy'], label='accuracy')
        # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()


        logger.info(f"test_loss: {np.round(test_loss)}, test_acc: {int(np.round(test_acc*100))}%")


    def deep_testing(self):
        pass


    def fine_tuning(self, CNN:str, dataset:str, pre_image_name:list):
        logger.info("fine_tuning")

        self.loading_pre_image_from_list(['P100', 'P80'])



        # # ##################################################################
        # #                phase 3: processing labels
        # # ##################################################################
        metadata = np.load(configs["paths"]["stepscan_image_label.npy"])
        logger.info("metadata shape: {}".format(metadata.shape))



        indices = metadata[:,0]
        le = preprocessing.LabelEncoder()
        le.fit(indices)

        logger.info(f"Number of subjects: {len(np.unique(indices))}")

        labels = le.transform(indices)

        # labels = tf.keras.utils.to_categorical(labels, num_classes=len(np.unique(indices)))





        # # ##################################################################
        # #                phase 4: Loading Image features
        # # ##################################################################
        features = np.load(configs["paths"]["stepscan_image_feature.npy"])
        logger.info("features shape: {}".format(features.shape))


        # #CD, PTI, Tmax, Tmin, P50, P60, P70, P80, P90, P100
        logger.info("batch_size: {}".format(configs["CNN"]["batch_size"]))

        maxvalues = [np.max(features[...,ind]) for ind in range(len(cfg.image_feature_name))]

        for i in range(len(cfg.image_feature_name)):
            features[..., i] = features[..., i]/maxvalues[i]


        if configs['CNN']["image_feature"]=="tile":
            images = tile(features)

        else:
            image_feature_name = dict(zip(cfg.image_feature_name, range(len(cfg.image_feature_name))))
            ind = image_feature_name[configs['CNN']["image_feature"]]
            
            images = features[...,ind]
            images = images[...,tf.newaxis]
            images = np.concatenate((images, images, images), axis=-1)


class Specificity(tf.keras.metrics.Metric):
    def __init__(self, name='specificity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tn = tf.keras.metrics.TrueNegatives()
        self.fp = tf.keras.metrics.FalsePositives()
        self.specificity = self.add_weight(
            name='specificity', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        tn = self.tn(y_true, y_pred)
        fp = self.fp(y_true, y_pred)
        self.specificity.assign((tn) / (fp + tn + 1e-6))

    def result(self):
        return self.specificity

    def reset_states(self):
        self.tn.reset_states()
        self.fp.reset_states()
        self.specificity.assign(0)


class Sensitivity(tf.keras.metrics.Metric):
    def __init__(self, name='sensitivity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = tf.keras.metrics.TruePositives()
        self.fn = tf.keras.metrics.FalseNegatives()
        self.sensitivity = self.add_weight(
            name='sensitivity', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        tp = self.tp(y_true, y_pred)
        fn = self.fn(y_true, y_pred)
        # since f1 is a variable, we use assign
        self.sensitivity.assign((tp) / (tp + fn + 1e-6))

    def result(self):
        return self.sensitivity

    def reset_states(self):
        self.tp.reset_states()
        self.fn.reset_states()
        self.sensitivity.assign(0.0)


class F1_Score(tf.keras.metrics.Metric):

    def __init__(self, name='fscore', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='fscore', initializer='zeros')
        self.precision_fn = tf.keras.metrics.Precision(thresholds=0.5)
        self.recall_fn = tf.keras.metrics.Recall(thresholds=0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        # since f1 is a variable, we use assign
        self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))

    def result(self):
        return self.f1

    def reset_states(self):
        self.precision_fn.reset_states()
        self.recall_fn.reset_states()
        self.f1.assign(0)


class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='bac', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = tf.keras.metrics.TruePositives()
        self.tn = tf.keras.metrics.TrueNegatives()
        self.fp = tf.keras.metrics.FalsePositives()
        self.fn = tf.keras.metrics.FalseNegatives()
        self.bac = self.add_weight(name='bac', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        tp = self.tp(y_true, y_pred)
        tn = self.tn(y_true, y_pred)
        fp = self.fp(y_true, y_pred)
        fn = self.fn(y_true, y_pred)
        spec = ((tn) / (fp + tn + 1e-6))
        sen = ((tp) / (tp + fn + 1e-6))
        self.bac.assign((sen + spec)/2)

    def result(self):
        return self.bac

    def reset_states(self):
        self.tp.reset_states()
        self.tn.reset_states()
        self.fp.reset_states()
        self.fn.reset_states()
        self.bac.assign(0)


METRICS = [
    tf.keras.metrics.SparseCategoricalCrossentropy(name='accuracy'),
    # BalancedAccuracy(),
    # Kappa(),
    # tf.keras.metrics.Precision(name='precision'),
    # tf.keras.metrics.Recall(name='recall'),
    # F1_Score(),
    # Specificity(),
    # Sensitivity(),
    # tf.keras.metrics.TruePositives(name='tp'),
    # tf.keras.metrics.FalsePositives(name='fp'),
    # tf.keras.metrics.TrueNegatives(name='tn'),
    # tf.keras.metrics.FalseNegatives(name='fn'),
    # tf.keras.metrics.AUC(name='auc'),

]


def Participant_Count():
    setting = {

        "dataset_name": 'casia',

        "_classifier_name": 'TM',
        "_combination": True,

        "_CNN_weights": 'imagenet',
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": '',

        "_min_number_of_sample": 30,
        "_known_imposter": 3,
        "_unknown_imposter": 5,
        "_number_of_unknown_imposter_samples": 1.0,  # Must be less than 1


        "_waveletname": 'coif1',
        "_pywt_mode": 'constant',
        "_wavelet_level": 4,


        "_p_training_samples": 27,
        "_train_ratio": 1000,
        "_ratio": False,


        "_KNN_n_neighbors": 5,
        "_KNN_metric": 'euclidean',
        "_KNN_weights": 'uniform',
        "_SVM_kernel": 'linear',


        "_KFold": 10,
        "_random_runs": 10,
        "_persentage": 0.95,
        "_normilizing": 'z-score',

    }
    
    A = Pipeline(setting)

    GRFs, COPs, COAs, pre_images, labels = A.loading_pre_features('casia')
    COA_handcrafted = A.loading_COA_handcrafted(COAs)
    COP_handcrafted = A.loading_COP_handcrafted(COPs)
    GRF_handcrafted = A.loading_GRF_handcrafted(GRFs)
    COA_WPT = A.loading_COA_WPT(COAs)
    COP_WPT = A.loading_COP_WPT(COPs)
    GRF_WPT= A.loading_GRF_WPT(GRFs)

    deep_features_list = A.loading_deep_features_from_list((pre_images, labels), ['P100', 'P80'], 'resnet50.ResNet50')
    image_from_list = A.loading_pre_image_from_list(pre_images, ['P80', 'P100'])
    # P70 = A.loading_pre_image(pre_images, 'P70')
    # P90 = A.loading_deep_features((pre_images, labels), 'P90', 'resnet50.ResNet50')
    

    feature_set_names = ['COP_handcrafted', 'COPs', 'COP_WPT', 'GRF_handcrafted', 'GRFs', 'GRF_WPT']
    feature_set =[]
    for i in feature_set_names:
        feature_set.append(eval(f"{i}"))

    p0 = [5, 30]
    p1 = [5, 10, 15, 20, 25, 30]
    p2 = ['TM']#, 'svm']

    space = list(product(p0, p1, p2))
    space = space[:]

    for idx, parameters in enumerate(space):
        
        A._known_imposter     = parameters[1]
        A._unknown_imposter   = parameters[0]
        A._classifier_name    = parameters[2]

        tic = timeit.default_timer()
        A.collect_results(A.run( feature_set, labels, feature_set_names), 'COP+GRF')
        toc = timeit.default_timer()

        logger.info(f'[step {idx+1} out of {len(space)}], parameters: {parameters}, process time: {round(toc-tic, 2)}')


def Feature_Count():
    setting = {

        "dataset_name": 'casia',

        "_classifier_name": 'TM',
        "_combination": True,

        "_CNN_weights": 'imagenet',
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": '',

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
    
    A = Pipeline(setting)

    GRFs, COPs, COAs, pre_images, labels = A.loading_pre_features('casia')
    COA_handcrafted = A.loading_COA_handcrafted(COAs)
    COP_handcrafted = A.loading_COP_handcrafted(COPs)
    GRF_handcrafted = A.loading_GRF_handcrafted(GRFs)
    COA_WPT = A.loading_COA_WPT(COAs)
    COP_WPT = A.loading_COP_WPT(COPs)
    GRF_WPT= A.loading_GRF_WPT(GRFs)

    deep_features_list = A.loading_deep_features_from_list((pre_images, labels), ['P100', 'P80'], 'resnet50.ResNet50')
    image_from_list = A.loading_pre_image_from_list(pre_images, ['P80', 'P100'])
    # P70 = A.loading_pre_image(pre_images, 'P70')
    # P90 = A.loading_deep_features((pre_images, labels), 'P90', 'resnet50.ResNet50')
    

    feature_set_names = ['COP_handcrafted', 'COPs', 'COP_WPT', 'GRF_handcrafted', 'GRFs', 'GRF_WPT']
    feature_set =[]
    for i in feature_set_names:
        feature_set.append(eval(f"{i}"))

    p0 = [5, 30]
    p1 = [5, 10, 15, 20, 25, 30]
    p1 = [5, 30]

    space = list(product(p0, p1))
    space = space[:]

    for idx, parameters in enumerate(space):
        
        A._known_imposter     = parameters[0]
        A._unknown_imposter   = parameters[1]

        tic = timeit.default_timer()
        A.collect_results(A.run( feature_set, labels, feature_set_names), 'COP+GRF')
        toc = timeit.default_timer()

        logger.info(f'[step {idx+1} out of {len(space)}], parameters: {parameters}, process time: {round(toc-tic, 2)}')


def template_Count():

    setting = {

        "dataset_name": 'casia',

        "_classifier_name": 'TM',
        "_combination": True,

        "_CNN_weights": 'imagenet',
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": '',

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
    
    A = Pipeline(setting)

    GRFs, COPs, COAs, pre_images, labels = A.loading_pre_features('casia')
    COA_handcrafted = A.loading_COA_handcrafted(COAs)
    COP_handcrafted = A.loading_COP_handcrafted(COPs)
    GRF_handcrafted = A.loading_GRF_handcrafted(GRFs)
    COA_WPT = A.loading_COA_WPT(COAs)
    COP_WPT = A.loading_COP_WPT(COPs)
    GRF_WPT= A.loading_GRF_WPT(GRFs)

    deep_features_list = A.loading_deep_features_from_list((pre_images, labels), ['P100', 'P80'], 'resnet50.ResNet50')
    image_from_list = A.loading_pre_image_from_list(pre_images, ['P80', 'P100'])
    # P70 = A.loading_pre_image(pre_images, 'P70')
    # P90 = A.loading_deep_features((pre_images, labels), 'P90', 'resnet50.ResNet50')
    

    feature_set_names = ['COP_handcrafted', 'COPs', 'COP_WPT', 'GRF_handcrafted', 'GRFs', 'GRF_WPT']
    feature_set =[]
    for i in feature_set_names:
        feature_set.append(eval(f"{i}"))

    p0 = [5, 30]
    p1 = [5, 10, 15, 20, 25, 30]
    p1 = [5, 30]

    space = list(product(p0, p1))
    space = space[:]

    for idx, parameters in enumerate(space):
        
        A._known_imposter     = parameters[0]
        A._unknown_imposter   = parameters[1]

        tic = timeit.default_timer()
        A.collect_results(A.run( feature_set, labels, feature_set_names), 'COP+GRF')
        toc = timeit.default_timer()

        logger.info(f'[step {idx+1} out of {len(space)}], parameters: {parameters}, process time: {round(toc-tic, 2)}')


def FT():

    setting = {

        "dataset_name": 'casia',

        "_classifier_name": 'TM',
        "_combination": True,

        "_CNN_weights": 'imagenet',
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": '',

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
    
    A = Pipeline(setting)

   

    GRFs, COPs, COAs, pre_images, labels = A.loading_pre_features('casia')
    COA_handcrafted = A.loading_COA_handcrafted(COAs)
    COP_handcrafted = A.loading_COP_handcrafted(COPs)
    GRF_handcrafted = A.loading_GRF_handcrafted(GRFs)
    COA_WPT = A.loading_COA_WPT(COAs)
    COP_WPT = A.loading_COP_WPT(COPs)
    GRF_WPT= A.loading_GRF_WPT(GRFs)

    deep_features_list = A.loading_deep_features_from_list((pre_images, labels), ['P100', 'P80'], 'resnet50.ResNet50')
    image_from_list = A.loading_pre_image_from_list(pre_images, ['P80', 'P100'])
    # P70 = A.loading_pre_image(pre_images, 'P70')
    # P90 = A.loading_deep_features((pre_images, labels), 'P90', 'resnet50.ResNet50')
    
    A.FT_deep_features((pre_images, labels), 'stepscan', 'P100', 'resnet50.ResNet50')

    feature_set_names = ['COP_handcrafted', 'COPs', 'COP_WPT', 'GRF_handcrafted', 'GRFs', 'GRF_WPT']
    feature_set =[]
    for i in feature_set_names:
        feature_set.append(eval(f"{i}"))

    p0 = [5, 30]
    p1 = [5, 10, 15, 20, 25, 30]
    p1 = [5, 30]

    space = list(product(p0, p1))
    space = space[:]

    for idx, parameters in enumerate(space):
        
        A._known_imposter     = parameters[0]
        A._unknown_imposter   = parameters[1]

        tic = timeit.default_timer()
        A.collect_results(A.run( feature_set, labels, feature_set_names), 'COP+GRF')
        toc = timeit.default_timer()

        logger.info(f'[step {idx+1} out of {len(space)}], parameters: {parameters}, process time: {round(toc-tic, 2)}')


def main():

    setting = {
        "_classifier_name": 'TM',
        "_combination": True,

        "_CNN_weights": 'imagenet',
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": '',

        "_min_number_of_sample": 30,
        "_known_imposter": 50,
        "_unknown_imposter": 0,
        "_number_of_unknown_imposter_samples": 1.0,  # Must be less than 1

        "_waveletname": 'coif1',
        "_pywt_mode": 'constant',
        "_wavelet_level": 4,

        "_p_training_samples": 27,
        "_train_ratio": 300,
        "_ratio": False,

        "_KNN_n_neighbors": 5,
        "_KNN_metric": 'euclidean',
        "_KNN_weights": 'uniform',
        "_SVM_kernel": 'linear',

        "_KFold": 10,
        "_random_runs": 10,
        "_persentage": 0.95,
        "_normilizing": 'z-score',
    }
    

    P = Pipeline("casia", "TM", setting)
    P.t = "Aim1_P3-resnet50"

    # P.loading_pre_features_image()
    P.loading_pre_features_GRF()
    P.loading_pre_features_COP()

    # P.loading_pre_image_from_list(['P100', 'P80'])
    P.loading_GRF_handcrafted()
    P.loading_GRF_WPT()
    P.loading_COP_handcrafted()
    P.loading_COP_WPT()

 

    ######################################################################################################################
    ######################################################################################################################
    test = os.environ.get('SLURM_JOB_NAME', default= P.t )
    logger.info(f'test name: {test}')

    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default=4))
    pool = multiprocessing.Pool(processes=ncpus)
    logger.info(f'CPU count: {ncpus}')

    # p0 = [9, 10, 11, 12, 13, 14, 15, 18]
    # p1 = [3, 21, 27, 30, 45, 60, 90, 120, 150, 180, 210]
    p0 = ["svm"]
    p1 = ["resnet50.ResNet50"]#["vgg16.VGG16", "resnet50.ResNet50", "efficientnet.EfficientNetB0", "mobilenet.MobileNet"]


    space = list(product(p0, p1))
    space = space[:]

    for idx, parameters in enumerate(space):


        P._classifier_name = parameters[0]
        P.loading_deep_features_from_list(['P100', 'P80'], CNN_base_model=parameters[1])
 

 

        # P.collect_results(P.pipeline_test())
        # P._classifier_name = 'TM'
        # P.collect_results(P.pipeline_1(), "Pipeline_1") 
        # P.collect_results(P.pipeline_2(['P100', 'P80']), "Pipeline_2") 
        # P.collect_results(P.pipeline_4('P100'), "Pipeline_4") 
        # P._classifier_name = 'svm'
        P.collect_results(P.pipeline_3(['P100', 'P80']), "Pipeline_3") 


        toc = timeit.default_timer()
        logger.info(f'[step {idx+1} out of {len(space)}], parameters: {parameters}, process time: {round(toc-tic, 2)}')



if __name__ == "__main__":
    logger.info("Starting !!!")
    tic1 = timeit.default_timer()

    # main()
    # Participant_Count()

    FT()

    toc1 = timeit.default_timer()
    logger.info("Done ({:2.2f} process time)!!!\n\n\n".format(toc1-tic1))




