import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import cdist

from pathlib import Path as Pathlb
import pywt
import os
import Butterworth
import convertGS2BW 


class Pipeline:

    def __init__(self):
        pass



class PreFeatures(Pipeline):
    _labels = 0

    _combination = 0
    _GRFs = 0
    _COAs = 0
    _COPs = 0
    _pre_images = 0
    _data_path = ""

    def __init__(cls, dataset_name, combination=True):
        cls.dataset_name = dataset_name
        cls._combination = combination
        cls.set_dataset_path()


    def load_H5():
        pass
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
    
    def print_paths(cls):
        print(cls._h5_path)
        print(cls._data_path)
        print(cls._meta_path)

    def set_dataset_path(cls):
        if cls.dataset_name == "casia":
            cls._h5_path = os.path.join(os.getcwd(), "Datasets", "Casia-D", "footpressures_align.h5")
            cls._data_path = os.path.join(os.getcwd(), "Datasets", "Casia-D", "Data-barefoot.npy")
            cls._meta_path = os.path.join(os.getcwd(), "Datasets", "Casia-D", "Metadata-barefoot.npy")
            cls._pre_features_path = os.path.join(os.getcwd(), "Datasets", "Casia-D", "pre_features")
            cls._features_path = os.path.join(os.getcwd(), "Datasets", "Casia-D", "features")


        elif cls.dataset_name == "stepscan":
            cls._h5_path = os.path.join(os.getcwd(), "Datasets", "stepscan", "footpressures_align.h5")
            cls._data_path = os.path.join(os.getcwd(), "Datasets", "stepscan", "Data-barefoot.npy")
            cls._meta_path = os.path.join(os.getcwd(), "Datasets", "stepscan", "Metadata-barefoot.npy")
            cls._pre_features_path = os.path.join(os.getcwd(), "Datasets", "stepscan", "pre_features")
            cls._features_path = os.path.join(os.getcwd(), "Datasets", "Casia-D", "features")

        else:
            print("The name is not valid!!")

    def extracting_labels(cls):
        if cls.dataset_name == "casia":
            cls._labels = cls._meta[:,0:2]


        elif cls.dataset_name == "stepscan":
            cls._labels = cls._meta[:,0:2]


        else:
            print("The name is not valid!!")

    def print_dataset(cls):
        print("Data shape: {}".format(cls._data.shape))
        print("Metadata shape: {}".format(cls._meta.shape))

    def loaddataset(cls):
        
        cls._data = np.load(cls._data_path)
        cls._meta = np.load(cls._meta_path)
        cls._image_size = cls._data.shape[1:]
        cls._samples = cls._data.shape[0]
        cls.extracting_labels()
        

    def extracting_pre_features(cls):
        cls.loaddataset()
        GRFs = list()
        COPs = list()
        COAs = list()
        pre_images = list()
        for sample, label in zip(cls._data, cls._meta):
    
            if cls._combination==True and label[1]==0:
                sample = np.fliplr(sample)

            COA = cls.computeCOATimeSeries(sample, Binarize = "simple", Threshold = 0)
            COA = COA.flatten()

            GRF = cls.computeGRF(sample)

            COP = cls.computeCOPTimeSeries(sample)
            COP = COP.flatten()

            pre_image = cls.pre_image(sample)

            COPs.append(COP)
            COAs.append(COA)
            GRFs.append(GRF)
            pre_images.append(pre_image)

        cls._GRFs = pd.DataFrame(np.array(GRFs), columns=["GRF_"+str(i) for i in range(cls._GRFs[0].shape[1])])
        cls._COPs = pd.DataFrame(np.array(COPs), columns=["COP_"+str(i) for i in range(cls._COPs[0].shape[1])]) 
        cls._COAs = pd.DataFrame(np.array(COAs), columns=["COA_"+str(i) for i in range(cls._COAs[0].shape[1])]) 
        cls._pre_images = np.array(pre_images)

        cls.saving_pre_features()

    def saving_pre_features(cls):
        Pathlb(cls._pre_features_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cls._labels, columns=["ID", "side",]).to_excel(os.path.join(cls._pre_features_path, "label.xlsx"))

        if cls._combination==True:
            cls._GRFs.to_excel(os.path.join(cls._pre_features_path, "GRF_c.xlsx"))
            cls._COAs.to_excel(os.path.join(cls._pre_features_path, "COA_c.xlsx"))
            cls._COPs.to_excel(os.path.join(cls._pre_features_path, "COP_c.xlsx"))
            np.save(os.path.join(cls._pre_features_path, "pre_images_c.npy"), cls._pre_images)
        else:
            cls._GRFs.to_excel(os.path.join(cls._pre_features_path, "GRF.xlsx"))
            cls._COAs.to_excel(os.path.join(cls._pre_features_path, "COA.xlsx"))
            cls._COPs.to_excel(os.path.join(cls._pre_features_path, "COP.xlsx"))
            np.save(os.path.join(cls._pre_features_path, "pre_images.npy"), cls._pre_images)

    def loading_pre_features(cls):
        cls.loaddataset()

        try:
            print("loading pre features!!!")

            cls._labels = pd.read_excel(os.path.join(cls._pre_features_path, "label.xlsx"), index_col = 0)

            if cls._combination==True:
                cls._GRFs = pd.read_excel(os.path.join(cls._pre_features_path, "GRF_c.xlsx"), index_col = 0)
                cls._COAs = pd.read_excel(os.path.join(cls._pre_features_path, "COA_c.xlsx"), index_col = 0)
                cls._COPs = pd.read_excel(os.path.join(cls._pre_features_path, "COP_c.xlsx"), index_col = 0)
                cls._pre_images = np.load(os.path.join(cls._pre_features_path, "pre_images_c.npy"))
            else:
                cls._GRFs = pd.read_excel(os.path.join(cls._pre_features_path, "GRF.xlsx"), index_col = 0)
                cls._COAs = pd.read_excel(os.path.join(cls._pre_features_path, "COA.xlsx"), index_col = 0)
                cls._COPs = pd.read_excel(os.path.join(cls._pre_features_path, "COP.xlsx"), index_col = 0)
                cls._pre_images = np.load(os.path.join(cls._pre_features_path, "pre_images.npy"))
        except:
            print("extraxting pre features!!!")
            cls.extracting_pre_features()


class Features(PreFeatures):
    COX_feature_name = ['MDIST_RD', 'MDIST_AP', 'MDIST_ML', 'RDIST_RD', 'RDIST_AP', 'RDIST_ML', 
        'TOTEX_RD', 'TOTEX_AP', 'TOTEX_ML', 'MVELO_RD', 'MVELO_AP', 'MVELO_ML', 
        'RANGE_RD', 'RANGE_AP', 'RANGE_ML', 'AREA_CC',  'AREA_CE',  'AREA_SW', 
        'MFREQ_RD', 'MFREQ_AP', 'MFREQ_ML', 'FDPD_RD',  'FDPD_AP',  'FDPD_ML', 
        'FDCC',     'FDCE']

    GRF_feature_name = ["max_value_1", "max_value_1_ind", "max_value_2", "max_value_2_ind", 
        "min_value", "min_value_ind", "mean_value", "std_value", "sum_value"]

    _pre_image_names = ["CD", "PTI", "Tmin", "Tmax", "P50", "P60", "P70", "P80", "P90", "P100"]

    def __init__(cls, dataset_name, waveletname="coif1", pywt_mode="constant", wavelet_level=4):
        super().__init__(dataset_name)
        cls._waveletname=waveletname
        cls._pywt_mode=pywt_mode
        cls._wavelet_level=wavelet_level

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
        RANGE.append(np.max(cdist(COPTS[1:2,:].T, COPTS[1:2,:].T)))
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

    def computeAREACC(cls, COPTS):
        """
        computeAREACC(COPTS)
        AREA-CC : 95% Confidence Circle Area
        COPTS [3,t] : RD (AP, ML) COP time series
        return AREACC [1] : AREA-CC
        """
        
        MDIST = cls.computeMDIST(COPTS)
        RDIST = cls.computeRDIST(COPTS)
        z05 = 1.645 # z.05 = the z statistic at the 95% confidence level
        SRD = np.sqrt((RDIST[0]**2)-(MDIST[0]**2)) #the standard deviation of the RD time series
        
        AREACC = np.pi*((MDIST[0]+(z05*SRD))**2)
        return AREACC

    def computeAREACE(cls, COPTS):
        """
        computeAREACE(COPTS)
        AREA-CE : 95% Confidence Ellipse Area
        COPTS [3,t] : (RD,) AP, ML COP time series
        return AREACE [1] : AREA-CE
        """
        
        F05 = 3
        RDIST = cls.computeRDIST(COPTS)
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

    def computeMFREQ(cls, COPTS, T=1):
        """
        computeMFREQ(COPTS, T)
        MFREQ : Mean Frequency
        COPTS [3,t] : RD, AP, ML COP time series
        T : the period of time selected for analysis (CASIA-D = 1s)
        return MFREQ [3] : MFREQ_RD, MFREQ_AP, MFREQ_ML
        """
        
        TOTEX = cls.computeTOTEX(COPTS)
        MDIST = cls.computeMDIST(COPTS)

        MFREQ = list()
        MFREQ.append( TOTEX[0]/(2*np.pi*T*MDIST[0]) )
        MFREQ.append( TOTEX[1]/(4*np.sqrt(2)*T*MDIST[1]))
        MFREQ.append( TOTEX[2]/(4*np.sqrt(2)*T*MDIST[2]))

        return MFREQ

    def computeFDPD(cls, COPTS):
        """
        computeFDPD(COPTS)
        FD-PD : Fractal Dimension based on the Plantar Diameter of the Curve
        COPTS [3,t] : RD, AP, ML COP time series
        return FDPD [3] : FD-PD_RD, FD-PD_AP, FD-PD_ML
        """

        N = COPTS.shape[1]
        TOTEX = cls.computeTOTEX(COPTS)
        d = cls.computeRANGE(COPTS)
        Nd = [elemt*N for elemt in d]
        dev = [i / j for i, j in zip(Nd, TOTEX)]
        
        
        FDPD = np.log(N)/np.log(dev)
        # sys.exit()
        return FDPD

    def computeFDCC(cls, COPTS):
        """
        computeFDCC(COPTS)
        FD-CC : Fractal Dimension based on the 95% Confidence Circle
        COPTS [3,t] : RD, (AP, ML) COP time series
        return FDCC [1] : FD-CC_RD
        """
        
        N = COPTS.shape[1]
        MDIST = cls.computeMDIST(COPTS)    
        RDIST = cls.computeRDIST(COPTS)
        z05 = 1.645; # z.05 = the z statistic at the 95% confidence level
        SRD = np.sqrt((RDIST[0]**2)-(MDIST[0]**2)) #the standard deviation of the RD time series

        d = 2*(MDIST[0]+z05*SRD)
        TOTEX = cls.computeTOTEX(COPTS)
        
        FDCC = np.log(N)/np.log((N*d)/TOTEX[0])
        return FDCC

    def computeFDCE(cls, COPTS):
        """
        computeFDCE(COPTS)
        FD-CE : Fractal Dimension based on the 95% Confidence Ellipse
        COPTS [3,t] : (RD,) AP, ML COP time series
        return FDCE [2] : FD-CE_AP, FD-CE_ML
        """
        
        
        N = COPTS.shape[1]
        F05 = 3; 
        RDIST = cls.computeRDIST(COPTS)
        SAPML = np.mean(COPTS[2,:]*COPTS[1,:])
        
        d = np.sqrt(8*F05*np.sqrt(((RDIST[1]**2)*(RDIST[2]**2))-(SAPML**2)))
        TOTEX = cls.computeTOTEX(COPTS)

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

    def wt_feature(cls, signal):
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

        dwt_coeff = pywt.wavedec(signal, cls._waveletname, mode=cls._pywt_mode, level=cls._wavelet_level)
        dwt_coeff = np.concatenate(dwt_coeff).ravel()

        return dwt_coeff

    
    
    ## COA
    def extraxting_COA_handcrafted(cls):
        COA_handcrafted = list()
        for idx, sample in cls._COAs.iterrows():
            sample = sample.values.reshape(3,100)

            MDIST = cls.computeMDIST(sample)    
            RDIST = cls.computeRDIST(sample)
            TOTEX = cls.computeTOTEX(sample)
            MVELO = cls.computeMVELO(sample)
            RANGE = cls.computeRANGE(sample)
            AREACC = cls.computeAREACC(sample)
            AREACE = cls.computeAREACE(sample)
            AREASW = cls.computeAREASW(sample)
            MFREQ = cls.computeMFREQ(sample)
            FDPD = cls.computeFDPD(sample)
            FDCC = cls.computeFDCC(sample)
            FDCE = cls.computeFDCE(sample)

            COA_handcrafted.append(np.concatenate((MDIST, RDIST, TOTEX, MVELO, RANGE, [AREACC], [AREACE], [AREASW], MFREQ, FDPD, [FDCC], [FDCE]), axis = 0))
            
            
        cls._COA_handcrafted = pd.DataFrame(np.array(COA_handcrafted), columns=cls.COX_feature_name)

        cls.saving_COA_handcrafted()

    def saving_COA_handcrafted(cls):
        Pathlb(cls._features_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cls._labels, columns=["ID", "side",]).to_excel(os.path.join(cls._features_path, "label.xlsx"))

        if cls._combination==True:
            cls._COA_handcrafted.to_excel(os.path.join(cls._features_path, "COA_handcrafted_c.xlsx"))
            
        else:
            cls._COA_handcrafted.to_excel(os.path.join(cls._features_path, "COA_handcrafted.xlsx"))
            
    def loading_COA_handcrafted(cls):
        try:
            print("loading COA features!!!")

            cls._labels = pd.read_excel(os.path.join(cls._features_path, "label.xlsx"), index_col = 0)

            if cls._combination==True:
                cls._COA_handcrafted = pd.read_excel(os.path.join(cls._features_path, "COA_handcrafted_c.xlsx"), index_col = 0)

            else:
                cls._COA_handcrafted = pd.read_excel(os.path.join(cls._features_path, "COA_handcrafted.xlsx"), index_col = 0)

        except:
            print("extraxting COA features!!!")
            cls.extraxting_COA_handcrafted()



    ## COP
    def extraxting_COP_handcrafted(cls):
        COP_handcrafted = list()
        for idx, sample in cls._COPs.iterrows():
            sample = sample.values.reshape(3,100)

            MDIST = cls.computeMDIST(sample)    
            RDIST = cls.computeRDIST(sample)
            TOTEX = cls.computeTOTEX(sample)
            MVELO = cls.computeMVELO(sample)
            RANGE = cls.computeRANGE(sample)
            AREACC = cls.computeAREACC(sample)
            AREACE = cls.computeAREACE(sample)
            AREASW = cls.computeAREASW(sample)
            MFREQ = cls.computeMFREQ(sample)
            FDPD = cls.computeFDPD(sample)
            FDCC = cls.computeFDCC(sample)
            FDCE = cls.computeFDCE(sample)

            COP_handcrafted.append(np.concatenate((MDIST, RDIST, TOTEX, MVELO, RANGE, [AREACC], [AREACE], [AREASW], MFREQ, FDPD, [FDCC], [FDCE]), axis = 0))
            
            
        cls._COP_handcrafted = pd.DataFrame(np.array(COP_handcrafted), columns=cls.COX_feature_name) 

        cls.saving_COP_handcrafted()

    def saving_COP_handcrafted(cls):
        Pathlb(cls._features_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cls._labels, columns=["ID", "side",]).to_excel(os.path.join(cls._features_path, "label.xlsx"))

        if cls._combination==True:
            cls._COP_handcrafted.to_excel(os.path.join(cls._features_path, "COP_handcrafted_c.xlsx"))
            
        else:
            cls._COP_handcrafted.to_excel(os.path.join(cls._features_path, "COP_handcrafted.xlsx"))
            
    def loading_COP_handcrafted(cls):
        try:
            print("loading COP features!!!")

            cls._labels = pd.read_excel(os.path.join(cls._features_path, "label.xlsx"), index_col = 0)

            if cls._combination==True:
                cls._COP_handcrafted = pd.read_excel(os.path.join(cls._features_path, "COP_handcrafted_c.xlsx"), index_col = 0)

            else:
                cls._COP_handcrafted = pd.read_excel(os.path.join(cls._features_path, "COP_handcrafted.xlsx"), index_col = 0)

        except:
            print("extraxting COP features!!!")
            cls.extraxting_COP_handcrafted()
        


    ## GRF
    def extraxting_GRF_handcrafted(cls):
        GRF_handcrafted = list()
        for idx, sample in cls._GRFs.iterrows():
            GRF_handcrafted.append(cls.computeGRFfeatures(sample))
               
        cls._GRF_handcrafted = pd.DataFrame(np.array(GRF_handcrafted), columns=cls.GRF_feature_name)
        cls.saving_GRF_handcrafted()

    def saving_GRF_handcrafted(cls):
        Pathlb(cls._features_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cls._labels, columns=["ID", "side",]).to_excel(os.path.join(cls._features_path, "label.xlsx"))

        if cls._combination==True:
            cls._GRF_handcrafted.to_excel(os.path.join(cls._features_path, "GRF_handcrafted_c.xlsx"))
            
        else:
            cls._GRF_handcrafted.to_excel(os.path.join(cls._features_path, "GRF_handcrafted.xlsx"))
            
    def loading_GRF_handcrafted(cls):
        try:
            print("loading GRF features!!!")

            cls._labels = pd.read_excel(os.path.join(cls._features_path, "label.xlsx"), index_col = 0)

            if cls._combination==True:
                cls._GRF_handcrafted = pd.read_excel(os.path.join(cls._features_path, "GRF_handcrafted_c.xlsx"), index_col = 0)

            else:
                cls._GRF_handcrafted = pd.read_excel(os.path.join(cls._features_path, "GRF_handcrafted.xlsx"), index_col = 0)

        except:
            print("extraxting GRF features!!!")
            cls.extraxting_GRF_handcrafted()



    ## GRF WPT
    def extraxting_GRF_WPT(cls):
        GRF_WPT = list()
        for idx, sample in cls._GRFs.iterrows():
            GRF_WPT.append(cls.wt_feature(sample))
               
        cls._GRF_WPT = pd.DataFrame(np.array(GRF_WPT), columns=["GRF_WPT_"+str(i) for i in range(cls._GRF_WPT.shape[1])]) 
        cls.saving_GRF_WPT()

    def saving_GRF_WPT(cls):
        Pathlb(cls._features_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cls._labels, columns=["ID", "side",]).to_excel(os.path.join(cls._features_path, "label.xlsx"))

        if cls._combination==True:
            cls._GRF_WPT.to_excel(os.path.join(cls._features_path, "GRF_WPT_c.xlsx"))
            
        else:
            cls._GRF_WPT.to_excel(os.path.join(cls._features_path, "GRF_WPT.xlsx"))
            
    def loading_GRF_WPT(cls):
        try:
            print("loading GRF features!!!")

            cls._labels = pd.read_excel(os.path.join(cls._features_path, "label.xlsx"), index_col = 0)

            if cls._combination==True:
                cls._GRF_WPT = pd.read_excel(os.path.join(cls._features_path, "GRF_WPT_c.xlsx"), index_col = 0)

            else:
                cls._GRF_WPT = pd.read_excel(os.path.join(cls._features_path, "GRF_WPT.xlsx"), index_col = 0)

        except:
            print("extraxting GRF features!!!")
            cls.extraxting_GRF_WPT()



    ## COP WPT
    def extraxting_COP_WPT(cls):
        COP_WPT = list()
        for idx, sample in cls._COPs.iterrows():
            sample = sample.values.reshape(3,100)
            wt_COA_RD = cls.wt_feature(sample[0,:])
            wt_COA_AP = cls.wt_feature(sample[1,:])
            wt_COA_ML = cls.wt_feature(sample[2,:])
            COP_WPT.append(np.concatenate((wt_COA_RD, wt_COA_AP, wt_COA_ML), axis = 0))
               
        cls._COP_WPT = pd.DataFrame(np.array(COP_WPT), columns=["COP_WPT_"+str(i) for i in range(cls._COP_WPT.shape[1])])  
        cls.saving_COP_WPT()

    def saving_COP_WPT(cls):
        Pathlb(cls._features_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cls._labels, columns=["ID", "side",]).to_excel(os.path.join(cls._features_path, "label.xlsx"))

        if cls._combination==True:
            cls._COP_WPT.to_excel(os.path.join(cls._features_path, "COP_WPT_c.xlsx"))
            
        else:
            cls._COP_WPT.to_excel(os.path.join(cls._features_path, "COP_WPT.xlsx"))
            
    def loading_COP_WPT(cls):
        try:
            print("loading COP features!!!")

            cls._labels = pd.read_excel(os.path.join(cls._features_path, "label.xlsx"), index_col = 0)

            if cls._combination==True:
                cls._COP_WPT = pd.read_excel(os.path.join(cls._features_path, "COP_WPT_c.xlsx"), index_col = 0)

            else:
                cls._COP_WPT = pd.read_excel(os.path.join(cls._features_path, "COP_WPT.xlsx"), index_col = 0)

        except:
            print("extraxting COP features!!!")
            cls.extraxting_COP_WPT()



    ## COA WPT
    def extraxting_COA_WPT(cls):
        COA_WPT = list()
        for idx, sample in cls._COAs.iterrows():
            sample = sample.values.reshape(3,100)
            wt_COA_RD = cls.wt_feature(sample[0,:])
            wt_COA_AP = cls.wt_feature(sample[1,:])
            wt_COA_ML = cls.wt_feature(sample[2,:])
            COA_WPT.append(np.concatenate((wt_COA_RD, wt_COA_AP, wt_COA_ML), axis = 0))
               
        cls._COA_WPT = pd.DataFrame(np.array(COA_WPT), columns=["COA_WPT_"+str(i) for i in range(cls._COA_WPT.shape[1])]) 
        cls.saving_COA_WPT()

    def saving_COA_WPT(cls):
        Pathlb(cls._features_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cls._labels, columns=["ID", "side",]).to_excel(os.path.join(cls._features_path, "label.xlsx"))

        if cls._combination==True:
            cls._COA_WPT.to_excel(os.path.join(cls._features_path, "COA_WPT_c.xlsx"))
            
        else:
            cls._COA_WPT.to_excel(os.path.join(cls._features_path, "COA_WPT.xlsx"))
            
    def loading_COA_WPT(cls):
        try:
            print("loading COA features!!!")

            cls._labels = pd.read_excel(os.path.join(cls._features_path, "label.xlsx"), index_col = 0)

            if cls._combination==True:
                cls._COA_WPT = pd.read_excel(os.path.join(cls._features_path, "COA_WPT_c.xlsx"), index_col = 0)

            else:
                cls._COA_WPT = pd.read_excel(os.path.join(cls._features_path, "COA_WPT.xlsx"), index_col = 0)

        except:
            print("extraxting COA features!!!")
            cls.extraxting_COA_WPT()


    ## deep

    ## images
    def extraxting_pre_image(cls, pre_image_name):
        if not pre_image_name in cls._pre_image_names:
            raise Exception("Invalid pre image name!!!")

        pre_image = list()
        for idx in range(cls._pre_images.shape[0]):

            sample = cls._pre_images[idx,..., cls._pre_image_names.index(pre_image_name)]
            sample = sample.reshape(-1)
            pre_image.append(sample)
               
        
        exec(f"cls._{pre_image_name} = pd.DataFrame(np.array(pre_image), columns=['Pixel_'+str(i) for i in range(pre_image[0].shape[0])]) ")
        cls.saving_pre_image(pre_image_name)

    def saving_pre_image(cls, pre_image_name):
        Pathlb(cls._features_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cls._labels, columns=["ID", "side",]).to_excel(os.path.join(cls._features_path, "label.xlsx"))

        if cls._combination==True:
            exec(f"cls._{pre_image_name}.to_excel(os.path.join(cls._features_path, '{pre_image_name}_c.xlsx'))")
            
        else:
            exec(f"cls._{pre_image_name}.to_excel(os.path.join(cls._features_path, '{pre_image_name}.xlsx'))")
            
    def loading_pre_image(cls, pre_image_name):
        if not pre_image_name in cls._pre_image_names:
            raise Exception("Invalid pre image name!!!")
        try:
            print("loading COA features!!!")

            cls._labels = pd.read_excel(os.path.join(cls._features_path, "label.xlsx"), index_col = 0)

            if cls._combination==True:
                exec(f"cls._{pre_image_name}= pd.read_excel(os.path.join(cls._features_path, '{pre_image_name}_c.xlsx'), index_col = 0)")

            else:
                exec(f"cls._{pre_image_name}= pd.read_excel(os.path.join(cls._features_path, '{pre_image_name}.xlsx'), index_col = 0)")

        except:
            print("extraxting COA features!!!")
            cls.extraxting_COA_WPT()


    def pack(cls, list_features):
        return pd.concat(list_features + [cls._labels], axis=1)




# D = PreFeatures("casia", combination=True)

# D.loading()

F = Features("casia")
F.loading_pre_features()
F.extraxting_pre_image("CD")
breakpoint()

F.loading_COA_WPT()
F.loading_COP_WPT()
F.loading_GRF_WPT()
F.loading_GRF_handcrafted()
F.loading_COP_handcrafted()
F.loading_COA_handcrafted()
X = F.pack([F._COP_handcrafted, F._COPs, F._COP_WPT, F._GRF_handcrafted, F._GRFs, F._GRF_WPT])   # F._COA_handcrafted, F._COAs, F._COA_WPT,     F._COP_handcrafted, F._COPs, F._COP_WPT,    F._GRF_handcrafted, F._GRFs, F._GRF_WPT,
print(X)
breakpoint()

A = Pipeline()


