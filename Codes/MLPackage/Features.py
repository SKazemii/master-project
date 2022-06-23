from .PreFeatures import *

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

    def extraxting_deep_features( self, data: tuple, pre_image_name: str, CNN_base_model: str ) -> pd.DataFrame:
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
            pd.read_parquet
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

    def filtering_subjects_and_samples(self, DF_features_all: pd.DataFrame) -> pd.DataFrame:
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

