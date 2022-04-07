from xml.sax.handler import feature_string_interning
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
    formatter_colored = logging.Formatter(blue + '[%(asctime)s]-' + yellow + '[%(name)s @%(lineno)d]' + reset + blue + '-[%(levelname)s]' + reset + bold_red + '\t\t%(message)s' + reset, datefmt='%m/%d/%Y %I:%M:%S %p ')
    formatter = logging.Formatter('[%(asctime)s]-[%(name)s @%(lineno)d]-[%(levelname)s]\t\t%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p ')
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
    def __init__(self, dataset_name, combination=True):
        self.dataset_name = dataset_name
        self._combination = combination
        self.set_dataset_path()

    def load_H5():
        pass

    def print_paths(self):
        logger.info(self._h5_path)
        logger.info(self._data_path)
        logger.info(self._meta_path)

    def set_dataset_path(self):
        if self.dataset_name == "casia":
            self._h5_path = os.path.join(os.getcwd(), "Datasets", "Casia-D", "footpressures_align.h5")
            self._data_path = os.path.join(os.getcwd(), "Datasets", "Casia-D", "Data-barefoot.npy")
            self._meta_path = os.path.join(os.getcwd(), "Datasets", "Casia-D", "Metadata-barefoot.npy")
            self._pre_features_path = os.path.join(os.getcwd(), "Datasets", "Casia-D", "pre_features")
            self._features_path = os.path.join(os.getcwd(), "Datasets", "Casia-D", "features")


        elif self.dataset_name == "stepscan":
            self._h5_path = os.path.join(os.getcwd(), "Datasets", "stepscan", "footpressures_align.h5")
            self._data_path = os.path.join(os.getcwd(), "Datasets", "stepscan", "Data-barefoot.npy")
            self._meta_path = os.path.join(os.getcwd(), "Datasets", "stepscan", "Metadata-barefoot.npy")
            self._pre_features_path = os.path.join(os.getcwd(), "Datasets", "stepscan", "pre_features")
            self._features_path = os.path.join(os.getcwd(), "Datasets", "stepscan", "features")

        elif self.dataset_name == "sfootbd":
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

    def extracting_labels(self):
        if self.dataset_name == "casia":
            self._labels = self._meta[:,0:2]

        elif self.dataset_name == "stepscan":
            self._labels = self._meta[:,0:2]

        elif self.dataset_name == "sfootbd":
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

            self._labels = pd.DataFrame(lst, columns=["ID"])


        else:
            logger.error("The name is not valid!!")

    def print_dataset(self):
        logger.info("Data shape: {}".format(self._data.shape))
        logger.info("Metadata shape: {}".format(self._meta.shape))

    def mat_to_numpy(self):
        g = glob.glob(self._mat_path + "\*.mat", recursive=True)
        data = list()
        label = list()
        for i in g:
            if i.endswith(".mat"):
                print(i.split("\\")[-1] + " is loading...")
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

    def loaddataset(self):
        self._data = np.load(self._data_path, allow_pickle=True)
        self._meta = np.load(self._meta_path, allow_pickle=True)
        self._sample_size = self._data.shape[1:]
        self._samples = self._data.shape[0]
        self.extracting_labels()


class PreFeatures(Database):
    def __init__( self, dataset_name):
        super().__init__(dataset_name)
        self._test_id = int(timeit.default_timer() * 1_000_000)
        
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
        return prefeatures: [x, y, 10] (CD, PTI, Tmin, Tmax, P50, P60, P70, P80, P90, P100)

        If The 30th percentile of a is 24.0: This means that 30% of values fall below 24.
        
        """

        prefeaturesl = list()
            
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

        prefeaturesl = np.stack((CD, PTI, Tmin, Tmax, P50, P60, P70, P80, P90, P100), axis = -1)

        return prefeaturesl
    
    def extracting_pre_features(self):
        self.loaddataset()
        GRFs = list()
        COPs = list()
        COAs = list()
        pre_images = list()
        for sample, label in zip(self._data, self._meta):
    
            if self._combination==True and label[1]==0:
                sample = np.fliplr(sample)

            COA = self.computeCOATimeSeries(sample, Binarize = "simple", Threshold = 0)
            COA = COA.flatten()

            GRF = self.computeGRF(sample)

            COP = self.computeCOPTimeSeries(sample)
            COP = COP.flatten()

            pre_image = self.pre_image(sample)

            COPs.append(COP)
            COAs.append(COA)
            GRFs.append(GRF)
            pre_images.append(pre_image)

        self._GRFs = pd.DataFrame(np.array(GRFs), columns=["GRF_"+str(i) for i in range(np.array(GRFs).shape[1])])
        self._COPs = pd.DataFrame(np.array(COPs), columns=["COP_"+str(i) for i in range(np.array(COPs).shape[1])]) 
        self._COAs = pd.DataFrame(np.array(COAs), columns=["COA_"+str(i) for i in range(np.array(COAs).shape[1])]) 
        self._pre_images = np.array(pre_images)

        self.saving_pre_features()

    def saving_pre_features(self):
        pd.DataFrame(self._labels, columns=["ID", "side",]).to_excel(os.path.join(self._pre_features_path, "label.xlsx"))

        if self._combination==True:
            self._GRFs.to_excel(os.path.join(self._pre_features_path, "GRF_c.xlsx"))
            self._COAs.to_excel(os.path.join(self._pre_features_path, "COA_c.xlsx"))
            self._COPs.to_excel(os.path.join(self._pre_features_path, "COP_c.xlsx"))
            np.save(os.path.join(self._pre_features_path, "pre_images_c.npy"), self._pre_images)
        else:
            self._GRFs.to_excel(os.path.join(self._pre_features_path, "GRF.xlsx"))
            self._COAs.to_excel(os.path.join(self._pre_features_path, "COA.xlsx"))
            self._COPs.to_excel(os.path.join(self._pre_features_path, "COP.xlsx"))
            np.save(os.path.join(self._pre_features_path, "pre_images.npy"), self._pre_images)

    def loading_pre_features_COP(self):
        self.loaddataset()

        try:
            logger.info("loading pre features!!!")

            self._labels = pd.read_excel(os.path.join(self._pre_features_path, "label.xlsx"), index_col = 0)

            if self._combination==True:
                self._COPs = pd.read_excel(os.path.join(self._pre_features_path, "COP_c.xlsx"), index_col = 0)
            else:
                self._COPs = pd.read_excel(os.path.join(self._pre_features_path, "COP.xlsx"), index_col = 0)
        except:
            logger.info("extraxting pre features!!!")
            self.extracting_pre_features()

        self._features_set["COPs"] = {
            "columns": self._COPs.columns,
            "number_of_features": self._COPs.shape[1], 
            "number_of_samples": self._COPs.shape[0],           
        }  

    def loading_pre_features_COA(self):
        self.loaddataset()

        try:
            logger.info("loading pre features!!!")

            self._labels = pd.read_excel(os.path.join(self._pre_features_path, "label.xlsx"), index_col = 0)

            if self._combination==True:
                self._COAs = pd.read_excel(os.path.join(self._pre_features_path, "COA_c.xlsx"), index_col = 0)
                
            else:
                self._COAs = pd.read_excel(os.path.join(self._pre_features_path, "COA.xlsx"), index_col = 0)
                
        except:
            logger.info("extraxting pre features!!!")
            self.extracting_pre_features()

        self._features_set["COAs"] = {
            "columns": self._COAs.columns,
            "number_of_features": self._COAs.shape[1], 
            "number_of_samples": self._COAs.shape[0],           
        }  

    def loading_pre_features_GRF(self):
        self.loaddataset()

        try:
            logger.info("loading pre features!!!")

            self._labels = pd.read_excel(os.path.join(self._pre_features_path, "label.xlsx"), index_col = 0)

            if self._combination==True:
                self._GRFs = pd.read_excel(os.path.join(self._pre_features_path, "GRF_c.xlsx"), index_col = 0)
                
            else:
                self._GRFs = pd.read_excel(os.path.join(self._pre_features_path, "GRF.xlsx"), index_col = 0)
                
        except:
            logger.info("extraxting pre features!!!")
            self.extracting_pre_features()

        self._features_set["GRFs"] = {
            "columns": self._GRFs.columns,
            "number_of_features": self._GRFs.shape[1], 
            "number_of_samples": self._GRFs.shape[0],           
        }  

    def loading_pre_features_image(self):
        self.loaddataset()

        try:
            logger.info("loading pre features!!!")

            self._labels = pd.read_excel(os.path.join(self._pre_features_path, "label.xlsx"), index_col = 0)

            if self._combination==True:
                self._pre_images = np.load(os.path.join(self._pre_features_path, "pre_images_c.npy"))
            else:
                self._pre_images = np.load(os.path.join(self._pre_features_path, "pre_images.npy"))
        except:
            logger.info("extraxting pre features!!!")
            self.extracting_pre_features()
    
    def loading_pre_features(self):
        self.loaddataset()

        try:
            logger.info("loading pre features!!!")

            self._labels = pd.read_excel(os.path.join(self._pre_features_path, "label.xlsx"), index_col = 0)

            if self._combination==True:
                self._GRFs = pd.read_excel(os.path.join(self._pre_features_path, "GRF_c.xlsx"), index_col = 0)
                self._COAs = pd.read_excel(os.path.join(self._pre_features_path, "COA_c.xlsx"), index_col = 0)
                self._COPs = pd.read_excel(os.path.join(self._pre_features_path, "COP_c.xlsx"), index_col = 0)
                self._pre_images = np.load(os.path.join(self._pre_features_path, "pre_images_c.npy"))
            else:
                self._GRFs = pd.read_excel(os.path.join(self._pre_features_path, "GRF.xlsx"), index_col = 0)
                self._COAs = pd.read_excel(os.path.join(self._pre_features_path, "COA.xlsx"), index_col = 0)
                self._COPs = pd.read_excel(os.path.join(self._pre_features_path, "COP.xlsx"), index_col = 0)
                self._pre_images = np.load(os.path.join(self._pre_features_path, "pre_images.npy"))
        except:
            logger.info("extraxting pre features!!!")
            self.extracting_pre_features()

        self._features_set["GRFs"] = {
            "columns": self._GRFs.columns,
            "number_of_features": self._GRFs.shape[1], 
            "number_of_samples": self._GRFs.shape[0],           
        } 

        self._features_set["COAs"] = {
            "columns": self._COAs.columns,
            "number_of_features": self._COAs.shape[1], 
            "number_of_samples": self._COAs.shape[0],           
        } 

        self._features_set["COPs"] = {
            "columns": self._COPs.columns,
            "number_of_features": self._COPs.shape[1], 
            "number_of_samples": self._COPs.shape[0],           
        } 
        self._CNN_image_size = self._pre_images.shape
        

class Features(PreFeatures):

    def __init__(self, dataset_name, waveletname="coif1", pywt_mode="constant", wavelet_level=4):
        super().__init__(dataset_name)
        self._waveletname=waveletname
        self._pywt_mode=pywt_mode
        self._wavelet_level=wavelet_level

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
    def extraxting_COA_handcrafted(self):
        COA_handcrafted = list()
        for idx, sample in self._COAs.iterrows():
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
            
            
        self._COA_handcrafted = pd.DataFrame(np.array(COA_handcrafted), columns=self.COX_feature_name)

        self.saving_COA_handcrafted()

    def saving_COA_handcrafted(self):
        Pathlb(self._features_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._labels, columns=["ID", "side",]).to_excel(os.path.join(self._features_path, "label.xlsx"))

        if self._combination==True:
            self._COA_handcrafted.to_excel(os.path.join(self._features_path, "COA_handcrafted_c.xlsx"))
            
        else:
            self._COA_handcrafted.to_excel(os.path.join(self._features_path, "COA_handcrafted.xlsx"))
            
    def loading_COA_handcrafted(self):
        try:
            logger.info("loading COA features!!!")

            self._labels = pd.read_excel(os.path.join(self._features_path, "label.xlsx"), index_col = 0)

            if self._combination==True:
                self._COA_handcrafted = pd.read_excel(os.path.join(self._features_path, "COA_handcrafted_c.xlsx"), index_col = 0)

            else:
                self._COA_handcrafted = pd.read_excel(os.path.join(self._features_path, "COA_handcrafted.xlsx"), index_col = 0)

        except:
            logger.info("extraxting COA features!!!")
            self.extraxting_COA_handcrafted()

        self._features_set["COA_handcrafted"] = {
            "columns": self._COA_handcrafted.columns,
            "number_of_features": self._COA_handcrafted.shape[1], 
            "number_of_samples": self._COA_handcrafted.shape[0],           
        }

    ## COP
    def extraxting_COP_handcrafted(self):
        COP_handcrafted = list()
        for idx, sample in self._COPs.iterrows():
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
            
            
        self._COP_handcrafted = pd.DataFrame(np.array(COP_handcrafted), columns=self.COX_feature_name) 

        self.saving_COP_handcrafted()

    def saving_COP_handcrafted(self):
        Pathlb(self._features_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._labels, columns=["ID", "side",]).to_excel(os.path.join(self._features_path, "label.xlsx"))

        if self._combination==True:
            self._COP_handcrafted.to_excel(os.path.join(self._features_path, "COP_handcrafted_c.xlsx"))
            
        else:
            self._COP_handcrafted.to_excel(os.path.join(self._features_path, "COP_handcrafted.xlsx"))
            
    def loading_COP_handcrafted(self):
        try:
            logger.info("loading COP features!!!")

            self._labels = pd.read_excel(os.path.join(self._features_path, "label.xlsx"), index_col = 0)

            if self._combination==True:
                self._COP_handcrafted = pd.read_excel(os.path.join(self._features_path, "COP_handcrafted_c.xlsx"), index_col = 0)

            else:
                self._COP_handcrafted = pd.read_excel(os.path.join(self._features_path, "COP_handcrafted.xlsx"), index_col = 0)

        except:
            logger.info("extraxting COP features!!!")
            self.extraxting_COP_handcrafted()
        
        self._features_set["COP_handcrafted"] = {
            "columns": self._COP_handcrafted.columns,
            "number_of_features": self._COP_handcrafted.shape[1], 
            "number_of_samples": self._COP_handcrafted.shape[0],           
        }

    ## GRF
    def extraxting_GRF_handcrafted(self):
        GRF_handcrafted = list()
        for idx, sample in self._GRFs.iterrows():
            GRF_handcrafted.append(self.computeGRFfeatures(sample))
               
        self._GRF_handcrafted = pd.DataFrame(np.array(GRF_handcrafted), columns=self.GRF_feature_name)
        self.saving_GRF_handcrafted()

    def saving_GRF_handcrafted(self):
        Pathlb(self._features_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._labels, columns=["ID", "side",]).to_excel(os.path.join(self._features_path, "label.xlsx"))

        if self._combination==True:
            self._GRF_handcrafted.to_excel(os.path.join(self._features_path, "GRF_handcrafted_c.xlsx"))
            
        else:
            self._GRF_handcrafted.to_excel(os.path.join(self._features_path, "GRF_handcrafted.xlsx"))
            
    def loading_GRF_handcrafted(self):
        try:
            logger.info("loading GRF features!!!")

            self._labels = pd.read_excel(os.path.join(self._features_path, "label.xlsx"), index_col = 0)

            if self._combination==True:
                self._GRF_handcrafted = pd.read_excel(os.path.join(self._features_path, "GRF_handcrafted_c.xlsx"), index_col = 0)

            else:
                self._GRF_handcrafted = pd.read_excel(os.path.join(self._features_path, "GRF_handcrafted.xlsx"), index_col = 0)

        except:
            logger.info("extraxting GRF features!!!")
            self.extraxting_GRF_handcrafted()

        self._features_set["GRF_handcrafted"] = {
            "columns": self._GRF_handcrafted.columns,
            "number_of_features": self._GRF_handcrafted.shape[1], 
            "number_of_samples": self._GRF_handcrafted.shape[0],           
        }

    ## GRF WPT
    def extraxting_GRF_WPT(self):
        GRF_WPT = list()
        for idx, sample in self._GRFs.iterrows():
            GRF_WPT.append(self.wt_feature(sample))
               
        self._GRF_WPT = pd.DataFrame(np.array(GRF_WPT), columns=["GRF_WPT_"+str(i) for i in range(np.array(GRF_WPT).shape[1])]) 
        self.saving_GRF_WPT()

    def saving_GRF_WPT(self):
        Pathlb(self._features_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._labels, columns=["ID", "side",]).to_excel(os.path.join(self._features_path, "label.xlsx"))

        if self._combination==True:
            self._GRF_WPT.to_excel(os.path.join(self._features_path, "GRF_WPT_c.xlsx"))
            
        else:
            self._GRF_WPT.to_excel(os.path.join(self._features_path, "GRF_WPT.xlsx"))
            
    def loading_GRF_WPT(self):
        try:
            logger.info("loading GRF features!!!")

            self._labels = pd.read_excel(os.path.join(self._features_path, "label.xlsx"), index_col = 0)

            if self._combination==True:
                self._GRF_WPT = pd.read_excel(os.path.join(self._features_path, "GRF_WPT_c.xlsx"), index_col = 0)

            else:
                self._GRF_WPT = pd.read_excel(os.path.join(self._features_path, "GRF_WPT.xlsx"), index_col = 0)

        except:
            logger.info("extraxting GRF features!!!")
            self.extraxting_GRF_WPT()

        self._features_set["GRF_WPT"] = {
            "columns": self._GRF_WPT.columns,
            "number_of_features": self._GRF_WPT.shape[1], 
            "number_of_samples": self._GRF_WPT.shape[0],           
        }

    ## COP WPT
    def extraxting_COP_WPT(self):
        COP_WPT = list()
        for idx, sample in self._COPs.iterrows():
            sample = sample.values.reshape(3,100)
            wt_COA_RD = self.wt_feature(sample[0,:])
            wt_COA_AP = self.wt_feature(sample[1,:])
            wt_COA_ML = self.wt_feature(sample[2,:])
            COP_WPT.append(np.concatenate((wt_COA_RD, wt_COA_AP, wt_COA_ML), axis = 0))
               
        self._COP_WPT = pd.DataFrame(np.array(COP_WPT), columns=["COP_WPT_"+str(i) for i in range(np.array(COP_WPT).shape[1])])  
        self.saving_COP_WPT()

    def saving_COP_WPT(self):
        Pathlb(self._features_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._labels, columns=["ID", "side",]).to_excel(os.path.join(self._features_path, "label.xlsx"))

        if self._combination==True:
            self._COP_WPT.to_excel(os.path.join(self._features_path, "COP_WPT_c.xlsx"))
            
        else:
            self._COP_WPT.to_excel(os.path.join(self._features_path, "COP_WPT.xlsx"))
            
    def loading_COP_WPT(self):
        try:
            logger.info("loading COP features!!!")

            self._labels = pd.read_excel(os.path.join(self._features_path, "label.xlsx"), index_col = 0)

            if self._combination==True:
                self._COP_WPT = pd.read_excel(os.path.join(self._features_path, "COP_WPT_c.xlsx"), index_col = 0)

            else:
                self._COP_WPT = pd.read_excel(os.path.join(self._features_path, "COP_WPT.xlsx"), index_col = 0)

        except:
            logger.info("extraxting COP features!!!")
            self.extraxting_COP_WPT()

        self._features_set["COP_WPT"] = {
            "columns": self._COP_WPT.columns,
            "number_of_features": self._COP_WPT.shape[1], 
            "number_of_samples": self._COP_WPT.shape[0],           
        }

    ## COA WPT
    def extraxting_COA_WPT(self):
        COA_WPT = list()
        for idx, sample in self._COAs.iterrows():
            sample = sample.values.reshape(3,100)
            wt_COA_RD = self.wt_feature(sample[0,:])
            wt_COA_AP = self.wt_feature(sample[1,:])
            wt_COA_ML = self.wt_feature(sample[2,:])
            COA_WPT.append(np.concatenate((wt_COA_RD, wt_COA_AP, wt_COA_ML), axis = 0))
               
        self._COA_WPT = pd.DataFrame(np.array(COA_WPT), columns=["COA_WPT_"+str(i) for i in range(np.array(COA_WPT).shape[1])]) 
        self.saving_COA_WPT()

    def saving_COA_WPT(self):
        Pathlb(self._features_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._labels, columns=["ID", "side",]).to_excel(os.path.join(self._features_path, "label.xlsx"))

        if self._combination==True:
            self._COA_WPT.to_excel(os.path.join(self._features_path, "COA_WPT_c.xlsx"))
            
        else:
            self._COA_WPT.to_excel(os.path.join(self._features_path, "COA_WPT.xlsx"))
            
    def loading_COA_WPT(self):
        try:
            logger.info("loading COA features!!!")

            self._labels = pd.read_excel(os.path.join(self._features_path, "label.xlsx"), index_col = 0)

            if self._combination==True:
                self._COA_WPT = pd.read_excel(os.path.join(self._features_path, "COA_WPT_c.xlsx"), index_col = 0)

            else:
                self._COA_WPT = pd.read_excel(os.path.join(self._features_path, "COA_WPT.xlsx"), index_col = 0)

        except:
            logger.info("extraxting COA features!!!")
            self.extraxting_COA_WPT()

        self._features_set["COA_WPT"] = {
            "columns": self._COA_WPT.columns,
            "number_of_features": self._COA_WPT.shape[1], 
            "number_of_samples": self._COA_WPT.shape[0],           
        }

    ## deep
    def extraxting_deep_features(self, pre_image_name, CNN_base_model="resnet50.ResNet50"):
        self._CNN_base_model = CNN_base_model
        
        try:
            logger.info(f"Loading { CNN_base_model } model...")
            base_model = eval(f"tf.keras.applications.{CNN_base_model}(weights='{self._CNN_weights}', include_top={self._CNN_include_top})")
            logger.info("Successfully loaded base model and model...")

        except Exception as e: 
            base_model = None
            logger.error("The base model could NOT be loaded correctly!!!")
            print(e)


        base_model.trainable = False

        CNN_name = CNN_base_model.split(".")[0]
        logger.info("MaduleName: {}\n".format(CNN_name))
        
        
        input = tf.keras.layers.Input(shape= (60, 40, 3), dtype = tf.float64, name="original_img") # todo image size
        x = tf.cast(input, tf.float32)
        x = eval("tf.keras.applications." + CNN_name + ".preprocess_input(x)")
        x = base_model(x)
        output = tf.keras.layers.GlobalMaxPool2D()(x)

        model = tf.keras.Model(input, output, name=CNN_name)

        if self._verbose==True:
            model.summary() 
            tf.keras.utils.plot_model(model, to_file=CNN_name + ".png", show_shapes=True)


        logger.info("batch_size: {}".format(self._CNN_batch_size))


        pre_images_norm = self.normalizing_pre_image(pre_image_name)


        train_ds = tf.data.Dataset.from_tensor_slices((pre_images_norm, self._labels["ID"] ))
        train_ds = train_ds.batch(self._CNN_batch_size)

        Deep_features = np.zeros((1, model.layers[-1].output_shape[1]))

        deep_features = list()

        for image_batch, labels_batch in train_ds:
           
            images = image_batch[...,tf.newaxis]
            images = np.concatenate((images, images, images), axis=-1)

            feature = model(images)
            Deep_features = np.append(Deep_features, feature, axis=0)

            if (Deep_features.shape[0]-1) % 256 == 0:
                logger.info(f" ->>> ({os.getpid()}) completed images: " + str(Deep_features.shape[0]))


        Deep_features = Deep_features[1:, :]
        logger.info(f"Deep features shape: {Deep_features.shape}")

        time = int(timeit.default_timer() * 1_000_000)
        self._deep_features = pd.DataFrame(Deep_features, columns=['deep_'+str(i) for i in range(Deep_features.shape[1])])


        self.saving_deep_features()

    def normalizing_pre_image(self, pre_image_name):
        if not pre_image_name in self._pre_image_names:
            raise Exception("Invalid pre image name!!!")


        i = self._pre_image_names.index(pre_image_name)
        maxvalues = np.max(self._pre_images[..., i])
        return self._pre_images[..., i]/maxvalues

    def saving_deep_features(self):
        
        Pathlb(self._features_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._labels, columns=["ID", "side",]).to_excel(os.path.join(self._features_path, "label.xlsx"))

        if self._combination==True:
            self._deep_features.to_excel(os.path.join(self._features_path, 'deep_features_c.xlsx'))
            
        else:
            self._deep_features.to_excel(os.path.join(self._features_path, 'deep_features.xlsx'))
            
    def loading_deep_features(self, pre_image_name, CNN_base_model="resnet50.ResNet50"):
        if not pre_image_name in self._pre_image_names:
            raise Exception("Invalid pre image name!!!")
        try:
            logger.info("loading COA features!!!")

            self._labels = pd.read_excel(os.path.join(self._features_path, "label.xlsx"), index_col = 0)

            if self._combination==True:
                self._deep_features = pd.read_excel(os.path.join(self._features_path, 'deep_features_c.xlsx'), index_col = 0)

            else:
                self._deep_features = pd.read_excel(os.path.join(self._features_path, 'deep_features.xlsx'), index_col = 0)        

        except:
            logger.info("extraxting COA features!!!")
            self.extraxting_deep_features(pre_image_name, CNN_base_model)

        self._features_set["deep_features"] = {
            "columns": self._deep_features.columns,
            "number_of_features": self._deep_features.shape[1], 
            "number_of_samples": self._deep_features.shape[0],           
        }

    ## images
    def extraxting_pre_image(self, pre_image_name):
        if not pre_image_name in self._pre_image_names:
            raise Exception("Invalid pre image name!!!")

        pre_image = list()
        for idx in range(self._pre_images.shape[0]):

            sample = self._pre_images[idx,..., self._pre_image_names.index(pre_image_name)]
            sample = sample.reshape(-1)
            pre_image.append(sample)
               
        
        exec(f"self._{pre_image_name} = pd.DataFrame(np.array(pre_image), columns=['pixel_'+str(i) for i in range(np.array(pre_image).shape[1])]) ")
        self.saving_pre_image(pre_image_name)

    def saving_pre_image(self, pre_image_name):
        Pathlb(self._features_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._labels, columns=["ID", "side",]).to_excel(os.path.join(self._features_path, "label.xlsx"))

        if self._combination==True:
            exec(f"self._{pre_image_name}.to_excel(os.path.join(self._features_path, '{pre_image_name}_c.xlsx'))")
            
        else:
            exec(f"self._{pre_image_name}.to_excel(os.path.join(self._features_path, '{pre_image_name}.xlsx'))")
            
    def loading_pre_image(self, pre_image_name):
        if not pre_image_name in self._pre_image_names:
            raise Exception("Invalid pre image name!!!")
        try:
            logger.info(f"loading {pre_image_name} features!!!")

            self._labels = pd.read_excel(os.path.join(self._features_path, "label.xlsx"), index_col = 0)

            if self._combination==True:
                exec(f"self._{pre_image_name}= pd.read_excel(os.path.join(self._features_path, '{pre_image_name}_c.xlsx'), index_col = 0)")

            else:
                exec(f"self._{pre_image_name}= pd.read_excel(os.path.join(self._features_path, '{pre_image_name}.xlsx'), index_col = 0)")
    
            

        except:
            logger.info(f"extraxting {pre_image_name} features!!!")
            self.extraxting_pre_image(pre_image_name)
            
        self._features_set[f"{pre_image_name}"] = {
            "columns": eval(f"self._{pre_image_name}.columns"),
            "number_of_features": eval(f"self._{pre_image_name}.shape[1]"), 
            "number_of_samples": eval(f"self._{pre_image_name}.shape[0]"),           
        }
   
    def pack(self, list_features):
        """
        list of features=[
            ["COA_handcrafted", "COAs", "COA_WPT",     
             "COP_handcrafted", "COPs", "COP_WPT",    
             "GRF_handcrafted", "GRFs", "GRF_WPT",
             "P100", "CD", "PTI", "Tmin", "Tmax", "P50", "P60", "P70", "P80", "P90", "P100",
             "deep_features",]
        # ]"""
        C = ["self._"+i for i in list_features]
        exec(f"DF_features_all = pd.concat({C} + [self._labels], axis=1)".replace("'", ""))
        return eval("DF_features_all")

    def filtering_subjects_and_samples(self, DF_features_all):
        subjects, samples = np.unique(DF_features_all["ID"].values, return_counts=True)

        ss = [a[0] for a in list(zip(subjects, samples)) if a[1]>=self._min_number_of_sample]

        if self._known_imposter + self._unknown_imposter > len(ss):
            raise Exception("Invalid _known_imposter and _unknown_imposter!!!")


        self._known_imposter_list   = ss[:self._known_imposter] 
        self._unknown_imposter_list = ss[self._known_imposter : self._known_imposter + self._unknown_imposter] 

        DF_unknown_imposter =  DF_features_all[DF_features_all["ID"].isin(self._unknown_imposter_list)]
        DF_known_imposter =    DF_features_all[DF_features_all["ID"].isin(self._known_imposter_list)]

        DF_unknown_imposter = DF_unknown_imposter.groupby('ID', group_keys=False).apply(lambda x: x.sample(frac=self._number_of_unknown_imposter_samples, replace=False, random_state=self._random_state))
        
        return DF_known_imposter, DF_unknown_imposter


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


        return FRR,FAR

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

    def ML_classifier(self, x_train, x_test, x_test_U):
        
        if self._classifier_name=="knn":
            classifier = knn(n_neighbors=self._KNN_n_neighbors, metric=self._KNN_metric, weights=self._KNN_weights, n_jobs=-1)
        elif self._classifier_name=="TM":
            classifier = knn(n_neighbors=1, metric=self._KNN_metric, weights=self._KNN_weights, n_jobs=-1)
        elif self._classifier_name=="svm":
            classifier = svm.SVC(kernel=self._SVM_kernel , probability=True, random_state=self._random_state)
        else:
            raise Exception("_classifier_name is not valid!!")

        best_model = classifier.fit(x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values)
        y_pred_tr = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]
        FRR_t, FAR_t = self.FXR_calculater(x_train["ID"], y_pred_tr)
        EER, t_idx = self.compute_eer(FRR_t, FAR_t)
        TH = self._THRESHOLDs[t_idx]


        acc = list()
        CMM = list()
        BACC = list()
        for _ in range(self._random_runs):
            DF_temp, pos_number = self.balancer(x_test, method="Random")

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
        
        
        
        y_pred = best_model.predict_proba(x_test.iloc[:, :-1].values)[:, 1]
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
                y_pred_U = best_model.predict_proba(temp.iloc[:, :-1].values)[:, 1]
                y_pred_U[y_pred_U >= TH ] = 1.
                y_pred_U[y_pred_U <  TH ] = 0.

                AUS.append(accuracy_score(temp["ID"].values, y_pred_U)*100 )
                FAU.append(np.where(y_pred_U==1)[0].shape[0])
            AUS = np.mean(AUS)
            FAU = np.mean(FAU)

            y_pred_U = best_model.predict_proba(x_test_U.iloc[:, :-1].values)[:, 1]
            y_pred_U[y_pred_U >= TH ] = 1.
            y_pred_U[y_pred_U <  TH ] = 0.
            AUS_All = accuracy_score(x_test_U["ID"].values, y_pred_U)*100 
            FAU_All = np.where(y_pred_U==1)[0].shape[0]



        results = [EER, TH, ACC_bd, BACC_bd, FAR_bd, FRR_bd, ACC_ud, BACC_ud, FAR_ud, FRR_ud, AUS, FAU, x_test_U.shape[0], AUS_All, FAU_All]

        return results, CM_bd, CM_ud      

    def compacting_results(self, results, CM_bd, CM_ud, subject):
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
        result.append([np.array(CM_bd).mean(axis=0), np.array(CM_ud).mean(axis=0)])
        

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


        result = [val for sublist in result for val in sublist]
        
        # results.append(result)
        return result


class Pipeline(Classifier):
    _features_set = dict()

    COX_feature_name = ['MDIST_RD', 'MDIST_AP', 'MDIST_ML', 'RDIST_RD', 'RDIST_AP', 'RDIST_ML', 
        'TOTEX_RD', 'TOTEX_AP', 'TOTEX_ML', 'MVELO_RD', 'MVELO_AP', 'MVELO_ML', 
        'RANGE_RD', 'RANGE_AP', 'RANGE_ML', 'AREA_CC',  'AREA_CE',  'AREA_SW', 
        'MFREQ_RD', 'MFREQ_AP', 'MFREQ_ML', 'FDPD_RD',  'FDPD_AP',  'FDPD_ML', 
        'FDCC',     'FDCE']

    GRF_feature_name = ["max_value_1", "max_value_1_ind", "max_value_2", "max_value_2_ind", 
        "min_value", "min_value_ind", "mean_value", "std_value", "sum_value"]

    _pre_image_names = ["CD", "PTI", "Tmin", "Tmax", "P50", "P60", "P70", "P80", "P90", "P100"]

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
        "CM_bd", 
        "CM_ud",
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

    def __init__(self, dataset_name, classifier_name, kwargs):
        super().__init__(dataset_name, classifier_name)
        
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
        self._CNN_base_model = ""
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

    def pipeline_tem(self, ):
        ###############
        ##  block 1  ##
        ###############
        # F.loading_pre_features_GRF()
        # F.loading_pre_features_image()
        # F.loading_pre_features_COP()
        # F.loading_pre_features_COA()


        ###############
        ##  block 2  ##
        ###############
        # F.loading_deep_features("P100")

        # F.loading_pre_image("P100")

        # F.loading_COA_handcrafted()
        # F.loading_COA_WPT()

        self.loading_GRF_handcrafted()
        # F.loading_GRF_WPT()

        self.loading_COP_handcrafted()
        # F.loading_COP_WPT()



        ###############
        ##  block 2  ##
        ###############
        # ["COA_handcrafted", "COAs", "COA_WPT",     
        # "COP_handcrafted", "COPs", "COP_WPT",    
        # "GRF_handcrafted", "GRFs", "GRF_WPT",
        # "P100", "CD", "PTI", "Tmin", "Tmax", "P50", "P60", "P70", "P80", "P90", "P100",
        # "deep_features",]
            
        listn = ["COP_handcrafted", "GRF_handcrafted"]
        DF_features_all = self.pack(listn)     

        return self.run(DF_features_all, listn)

    def pipeline_test(self, ):
        ###############
        ##  block 1  ##
        ###############
        if self._GRFs.empty:
            self.loading_pre_features_GRF()
        # F.loading_pre_features_image()
        # F.loading_pre_features_COP()
        # F.loading_pre_features_COA()


        ###############
        ##  block 2  ##
        ###############
        # F.loading_deep_features("P100")

        # F.loading_pre_image("P100")

        # F.loading_COA_handcrafted()
        # F.loading_COA_WPT()
        if self._GRF_handcrafted.empty:
            self.loading_GRF_handcrafted()
        # F.loading_GRF_WPT()

        # F.loading_COP_handcrafted()
        # F.loading_COP_WPT()



        ###############
        ##  block 2  ##
        ###############
        # ["COA_handcrafted", "COAs", "COA_WPT",     
        # "COP_handcrafted", "COPs", "COP_WPT",    
        # "GRF_handcrafted", "GRFs", "GRF_WPT",
        # "P100", "CD", "PTI", "Tmin", "Tmax", "P50", "P60", "P70", "P80", "P90", "P100",
        # "deep_features",]
            
        listn = ["GRF_handcrafted"]
        DF_features_all = self.pack(listn)     

        return self.run(DF_features_all, listn)

    def pipeline_1(self, ):
        """GRF+COP"""
        # self.t = "P1"
        ###############
        ##  block 1  ##
        ###############
        # if self._GRFs.empty or self._COPs.empty:
        #     self.loading_pre_features_GRF()
        #     self.loading_pre_features_COP()


        # ###############
        # ##  block 2  ##
        # ###############
        # if self._GRF_handcrafted.empty or self._GRF_WPT.empty:
        #     self.loading_GRF_handcrafted()
        #     self.loading_GRF_WPT()

        # if self._COP_handcrafted.empty or self._COP_WPT.empty:
        #     self.loading_COP_handcrafted()
        #     self.loading_COP_WPT()


        ###############
        ##  block 2  ##
        ###############           
        listn = ["COP_handcrafted", "COPs", "COP_WPT", "GRF_handcrafted", "GRFs", "GRF_WPT",]
        DF_features_all = self.pack(listn)     

        return self.run(DF_features_all, listn)
        
    def pipeline_2(self, Image_feature_name):
        """image"""
        # self.t = "P2"

        ###############
        ##  block 1  ##
        ###############
        # if self._pre_images.empty or eval(f'self._{Image_feature_name}.empty'):
        #     self.loading_pre_features_image()


        #     ###############
        #     ##  block 2  ##
        #     ###############
        #     self.loading_pre_image(Image_feature_name)


        ###############
        ##  block 2  ##
        ###############
        listn = [Image_feature_name]
        DF_features_all = self.pack(listn)     

        return self.run(DF_features_all, listn)

    def pipeline_3(self, Image_feature_name):
        """deep features"""
        # self.t = "P3"

        ###############
        ##  block 1  ##
        ###############
        # if self._pre_images.empty or eval(f'self._{Image_feature_name}.empty'):

        #     self.loading_pre_features_image()


        #     ###############
        #     ##  block 2  ##
        #     ###############
        #     self.loading_deep_features(Image_feature_name)

        
        ###############
        ##  block 2  ##
        ###############         
 
        listn = ["deep_features"]
        DF_features_all = self.pack(listn)     

        return self.run(DF_features_all, listn)

    def pipeline_4(self, Image_feature_name):
        """All Handcrafted"""
        # self.t = "P4"

        ###############
        ##  block 1  ##
        ###############
        # breakpoint()
        # if self._pre_images.empty or eval(f'self._{Image_feature_name}.empty'):

        #     self.loading_pre_features_GRF()
        #     self.loading_pre_features_image()
        #     self.loading_pre_features_COP()
        #     # self.loading_pre_features_COA()


        #     ###############
        #     ##  block 2  ##
        #     ###############
        #     self.loading_pre_image(Image_feature_name)

        #     # self.loading_COA_handcrafted()
        #     # self.loading_COA_WPT()

        #     self.loading_GRF_handcrafted()
        #     self.loading_GRF_WPT()

        #     self.loading_COP_handcrafted()
        #     self.loading_COP_WPT()


        ###############
        ##  block 2  ##
        ###############           
        listn = ["COP_handcrafted", "COPs", "COP_WPT", "GRF_handcrafted", "GRFs", "GRF_WPT", Image_feature_name]
        DF_features_all = self.pack(listn)     

        return self.run(DF_features_all, listn)

    def run(self, DF_features_all, listn):
        # pool = multiprocessing.Pool(processes=ncpus)
        DF_known_imposter, DF_unknown_imposter = self.filtering_subjects_and_samples(DF_features_all)


        results = list()
        for subject in self._known_imposter_list:
            if self._verbose == True:
                print(f"Subject number: {subject} out of {len(self._known_imposter_list)} ")
            DF_known_imposter_binariezed, DF_unknown_imposter_binariezed = self.binarize_labels(DF_known_imposter, DF_unknown_imposter, subject)

            CV = model_selection.StratifiedKFold(n_splits=self._KFold, random_state=None, shuffle=False)
            X = DF_known_imposter_binariezed
            U = DF_unknown_imposter_binariezed


            cv_results = list()
            cv_CM_u = list()
            cv_CM_b = list()
            for train_index, test_index in CV.split(X.iloc[:,:-1], X.iloc[:,-1]):

                df_train = X.iloc[train_index, :]
                df_test = X.iloc[test_index, :]
                # df_test = pd.concat([X.iloc[test_index, :], DF_unknown_imposter_binariezed])

                df_train = self.down_sampling(df_train)

                df_train, df_test, df_test_U = self.scaler(df_train, df_test, U)

                df_train, df_test, df_test_U = self.projector(df_train, df_test, df_test_U, listn)

                result, CM_bd, CM_ud = self.ML_classifier(df_train, df_test, df_test_U)

                cv_results.append(result)
                cv_CM_u.append(CM_ud)
                cv_CM_b.append(CM_bd)

            result = self.compacting_results(cv_results, cv_CM_b, cv_CM_u, subject)
            results.append(result)

        return pd.DataFrame(results, columns=self._col)

    def collect_results(self, result: pd.DataFrame, pipeline_name: str) -> None:
        result['pipeline'] = pipeline_name
        test = os.environ.get('SLURM_JOB_NAME', default=self.t)
        excel_path = os.path.join(os.getcwd(), "results", f"Result__{test}.xlsx")

        if os.path.isfile(excel_path):
            Results_DF = pd.read_excel(excel_path, index_col = 0)
        else:
            Results_DF = pd.DataFrame(columns=self._col)

        Results_DF = Results_DF.append(result)
        try:
            Results_DF.to_excel(excel_path)
        except:
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


def main():

    setting = {
        "_classifier_name": 'TM',
        "_combination": True,

        "_CNN_weights": 'imagenet',
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": '',

        "_min_number_of_sample": 30,
        "_known_imposter": 3,
        "_unknown_imposter": 1,
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
    P.t = "Participant_Count"

    P.loading_pre_features_GRF()
    P.loading_pre_features_image()
    P.loading_pre_features_COP()

    P.loading_pre_image('P100')
    P.loading_GRF_handcrafted()
    P.loading_GRF_WPT()
    P.loading_COP_handcrafted()
    P.loading_COP_WPT()

    P.loading_deep_features('P100')
 

    ######################################################################################################################
    ######################################################################################################################
    test = os.environ.get('SLURM_JOB_NAME', default= P.t )
    logger.info(f'test name: {test}')

    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default=4))
    pool = multiprocessing.Pool(processes=ncpus)
    logger.info(f'CPU count: {ncpus}')

    # p0 = [9, 10, 11, 12, 13, 14, 15, 18]
    # p1 = [3, 21, 27, 30, 45, 60, 90, 120, 150, 180, 210]
    p0 = [5, 10, 15, 20, 25, 30]
    p1 = [10, 0, 5, 15, 20, 25, 30]

    space = list(product(p0, p1))
    space = space[:]

    for idx, parameters in enumerate(space):

        # P._p_training_samples = parameters[0]
        P._known_imposter     = parameters[0]
        P._unknown_imposter   = parameters[1]

        # P.collect_results(P.pipeline_test())
        P._classifier_name = 'TM'
        P.collect_results(P.pipeline_1(), "Pipeline_1") 
        P.collect_results(P.pipeline_2('P100'), "Pipeline_2") 
        P.collect_results(P.pipeline_4('P100'), "Pipeline_4") 
        P._classifier_name = 'svm'
        P.collect_results(P.pipeline_3('P100'), "Pipeline_3") 


        toc = timeit.default_timer()
        logger.info(f'[step {idx+1} out of {len(space)}], parameters: {parameters}, process time: {round(toc-tic, 2)}')


def main_test():
    setting = {
        "_classifier_name": 'TM',
        "_combination": True,

        "_CNN_weights": 'imagenet',
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": '',

        "_min_number_of_sample": 30,
        "_known_imposter": 3,
        "_unknown_imposter": 1,
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
    A = Deep_network("casia", "TM", setting)
    A.loading_pre_features_image()
    A.deep_training_2("P100")


if __name__ == "__main__":
    logger.info("Starting !!!")
    tic = timeit.default_timer()
    main()

    # main_test()

    toc = timeit.default_timer()
    logger.info("Done ({:2.2f} process time)!!!\n\n\n".format(toc-tic))




