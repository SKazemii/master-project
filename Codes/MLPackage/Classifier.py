from .Features import *
import pyod.models as od

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

    def FXR_calculater(self, x_train, y_pred, THRESHOLD=None):
        FRR = list()
        FAR = list()
        # breakpoint()
        
        x = self._THRESHOLDs
        if THRESHOLD.any() != None:
            x = THRESHOLD


        for tx in x:
            E1 = np.zeros((y_pred.shape))
            E1[y_pred >= tx] = 1

            e = pd.DataFrame([x_train, E1]).T# todo x_train.values
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

    @staticmethod
    def compute_EER(y, score, metric):

        fpr, tpr, threshold = roc_curve(y, score, pos_label=1)
        fnr = 1 - tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)

        if metric == 'eer':
            EER = np.mean((fpr[min_index], fnr[min_index]))
            TH = threshold[min_index]

        elif metric == 'gmeans':
            gmeans = sqrt(tpr * (1-fpr))
            ix = argmax(gmeans)
            EER = gmeans[ix]
            TH = threshold[ix]

        elif metric == 'zero_fpr':
            ix = np.max(np.where(fpr==0))
            TH = threshold[ix]
            EER = 0

        elif metric == 'zero_frr':
            ix = np.min(np.where(tpr==1))
            TH = threshold[ix]
            EER = 0
        
        return EER, fpr, tpr, TH


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

            classifier = knn(n_neighbors=int(kwargs["params"]['n_neighbors']))
            
            best_model = classifier.fit( x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values )
            score = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]
            EER, _, _, TH = self.compute_EER(x_train.iloc[:, -1].values, score, metric='eer')

            y_pred = best_model.predict_proba(x_test.iloc[:, :-1].values)[:, 1]

        elif self._classifier_name == "tm":
            positives = x_train[x_train["ID"] == 1.0]
            negatives = x_train[x_train["ID"] == 0.0]
            similarity_matrix_positives, similarity_matrix_negatives = self.compute_score_matrix(positives, negatives)
            client_scores, imposter_scores = self.compute_scores(similarity_matrix_positives, similarity_matrix_negatives, criteria="min")
            score = np.append(client_scores.data, imposter_scores.data)

            EER, _, _, TH = self.compute_EER(x_train.iloc[:, -1].values, score, metric='eer')

            similarity_matrix_positives, similarity_matrix_negatives = self.compute_score_matrix(positives, x_test)
            client_scores, imposter_scores = self.compute_scores( similarity_matrix_positives, similarity_matrix_negatives, criteria="min")
            y_pred = imposter_scores.data

        elif self._classifier_name == "svm-linear":
            classifier = svm.SVC(kernel='linear', probability=True, random_state=self._random_state, C=10 ** kwargs["params"]['logC'] )
              
            best_model = classifier.fit(x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values)
            score = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]

            EER, _, _, TH = self.compute_EER(x_train.iloc[:, -1].values, score, metric='eer')

            y_pred = best_model.predict_proba(x_test.iloc[:, :-1].values)[:, 1]

        elif self._classifier_name == "svm-rbf":
            classifier = svm.SVC(kernel='rbf', probability=True, random_state=self._random_state, C=10 ** kwargs["params"]['logC'], gamma=10 ** kwargs["params"]['logGamma'] )

            best_model = classifier.fit(x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values)
            score = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]

            EER, _, _, TH = self.compute_EER(x_train.iloc[:, -1].values, score, metric='eer')

            y_pred = best_model.predict_proba(x_test.iloc[:, :-1].values)[:, 1]

        elif self._classifier_name == "svm-poly":
            classifier = svm.SVC(kernel='poly', probability=True, random_state=self._random_state, C=10 ** kwargs["params"]['logC'], degree=kwargs["params"]['degree'], coef0=kwargs["params"]['coef0'] )
            
            best_model = classifier.fit(x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values)
            score = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]

            EER, _, _, TH = self.compute_EER(x_train.iloc[:, -1].values, score, metric='eer')

            y_pred = best_model.predict_proba(x_test.iloc[:, :-1].values)[:, 1]

        elif self._classifier_name == "lda":
            classifier = LinearDiscriminantAnalysis()
            best_model = classifier.fit(x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values)
            score = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]

            EER, _, _, TH = self.compute_EER(x_train.iloc[:, -1].values, score, metric='eer')

            y_pred = best_model.predict_proba(x_test.iloc[:, :-1].values)[:, 1]

        elif self._classifier_name == "rf":
            classifier = RandomForestClassifier(random_state=self._random_state)
            best_model = classifier.fit(x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values)
            score = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]

            EER, _, _, TH = self.compute_EER(x_train.iloc[:, -1].values, score, metric='eer')

            y_pred = best_model.predict_proba(x_test.iloc[:, :-1].values)[:, 1]

        elif self._classifier_name == "nb":
            classifier = GaussianNB()
            best_model = classifier.fit(x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values)
            score = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]

            EER, _, _, TH = self.compute_EER(x_train.iloc[:, -1].values, score, metric='eer')

            y_pred = best_model.predict_proba(x_test.iloc[:, :-1].values)[:, 1]

        elif self._classifier_name == "if":
            
            # classifier = IsolationForest(random_state=self._random_state)
            # best_model = classifier.fit(x_train.iloc[:, :-1].values)
            # EER = 0
            # TH = 0

            # y_pred = best_model.predict(x_test.iloc[:, :-1].values)
            from pyod.models.iforest import IForest
            classifier = IForest(n_estimators=100, max_samples='auto', contamination=0.1, max_features=1.0, bootstrap=False, n_jobs=-1, behaviour='old', random_state=self._random_state, verbose=0)
            best_model = classifier.fit(x_train.iloc[:, :-1].values)

            # EER, _, _, TH = self.compute_EER(x_train.iloc[:, -1].values, score, metric='eer')
            EER = 0
            TH = 0

            y_pred = best_model.predict(x_test.iloc[:, :-1].values)
            
        elif self._classifier_name == "ocsvm":
            # classifier = OneClassSVM(kernel='linear',nu=kwargs["params"]['nu'])
            
            # best_model = classifier.fit(x_train.iloc[:, :-1].values)
            # EER = 0
            # TH = 0

            # y_pred = best_model.predict(x_test.iloc[:, :-1].values)

            from pyod.models.ocsvm import OCSVM
            classifier = OCSVM(kernel='linear', degree=3, gamma='auto', coef0=0.0, tol=0.001, nu=kwargs["params"]['nu'], shrinking=True, cache_size=200, verbose=False, max_iter=-1, contamination=0.1) 
            best_model = classifier.fit(x_train.iloc[:, :-1].values)


            EER = 0
            TH = 0

            y_pred = best_model.predict(x_test.iloc[:, :-1].values)

        elif self._classifier_name == "svdd":

            from pyod.models.ocsvm import OCSVM
            classifier = OCSVM(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, nu=kwargs["params"]['nu'], shrinking=True, cache_size=200, verbose=False, max_iter=-1, contamination=0.1) 
            best_model = classifier.fit(x_train.iloc[:, :-1].values)


            EER = 0
            TH = 0

            y_pred = best_model.predict(x_test.iloc[:, :-1].values)

        elif self._classifier_name == "ocknn":

            from pyod.models.knn import KNN as OCKNN
            classifier = OCKNN(contamination=0.1, n_neighbors=5, method='largest', radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=-1)
            best_model = classifier.fit(x_train.iloc[:, :-1].values)


            EER = 0
            TH = 0

            y_pred = best_model.predict(x_test.iloc[:, :-1].values)

        elif self._classifier_name == "cblof":

            from pyod.models.cblof import CBLOF
            classifier = CBLOF(n_clusters=8, contamination=0.1, clustering_estimator=None, alpha=0.9, beta=5, use_weights=False, check_estimator=False, random_state=self._random_state, n_jobs=-1)
            best_model = classifier.fit(x_train.iloc[:, :-1].values)


            EER = 0
            TH = 0

            y_pred = best_model.predict(x_test.iloc[:, :-1].values)

        elif self._classifier_name == "hbos":

            from pyod.models.hbos import HBOS
            classifier = HBOS(n_bins=10, alpha=0.1, tol=0.5, contamination=0.1)
            best_model = classifier.fit(x_train.iloc[:, :-1].values)


            EER = 0
            TH = 0

            y_pred = best_model.predict(x_test.iloc[:, :-1].values)

        elif self._classifier_name == "anogan":

            from pyod.models.anogan import AnoGAN
            classifier = AnoGAN(activation_hidden='tanh', dropout_rate=0.2, latent_dim_G=2, G_layers=[20, 10, 3, 10, 20], verbose=0, D_layers=[20, 10, 5], index_D_layer_for_recon_error=1, epochs=500, preprocessing=False, learning_rate=0.001, learning_rate_query=0.01, epochs_query=20, batch_size=32, output_activation=None, contamination=0.1)
            best_model = classifier.fit(x_train.iloc[:, :-1].values)


            EER = 0
            TH = 0

            y_pred = best_model.predict(x_test.iloc[:, :-1].values)

        elif self._classifier_name == "deep_svdd":

            from pyod.models.deep_svdd import DeepSVDD

            classifier = DeepSVDD(c=None, use_ae=False, hidden_neurons=None, hidden_activation='relu', output_activation='sigmoid', optimizer='adam', epochs=100, batch_size=32, dropout_rate=0.2, l2_regularizer=0.1, validation_size=0.1, preprocessing=True, verbose=1, random_state=None, contamination=0.1)
            best_model = classifier.fit(x_train.iloc[:, :-1].values)


            EER = 0
            TH = 0

            y_pred = best_model.predict(x_test.iloc[:, :-1].values)

        else:
            raise Exception(f"_classifier_name ({self._classifier_name}) is not valid!!")

        # "deep_svdd", "anogan", "hbos", "cblof", "ocknn", "if", "ocsvm", "svdd"

               
        if self._classifier_name not in ["deep_svdd", "anogan", "hbos", "cblof", "ocknn", "if", "ocsvm", "svdd"]:
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

    def subject_optimizer(self, x_train, num_particles, num_generations, search):
         
        # CV = model_selection.StratifiedKFold( n_splits=self._KFold, shuffle=False)
        # folds = [[list(test) for train, test in CV.split(x_train.iloc[:, :-1], x_train.iloc[:, -1])]]

        
        # x_train, y_train, x_test, y_test = model_selection.train_test_split(x=x_train.iloc[:, :-1].values, y=x_train.iloc[:, -1].values, test_size=0.20, random_state=self._random_state)
        # @optunity.cross_validated(x=x_train.iloc[:, :-1].values, y=x_train.iloc[:, -1].values, folds=folds, num_folds=self._KFold)
        @optunity.hold_out_validated(x=x_train.iloc[:, :-1].values, y=x_train.iloc[:, -1].values, test_size=20)
        def performance(x_train, y_train, x_test, y_test, 
                        n_neighbors=None, metric=None, weights=None,
                        logC=None, logGamma=None, degree=None, coef0=None,
                        n_estimators=None, max_features=None,):
            if   self._classifier_name == 'knn':
                if int(n_neighbors) < 1:
                    return 0
                else:
                    classifier = knn( n_neighbors=int(n_neighbors))#, metric=metric, weights=weights, )
                    best_model = classifier.fit(x_train, y_train)
                    y_pred_tr = best_model.predict_proba(x_train)[:, 1]

                    FRR_t, FAR_t = self.FXR_calculater(y_train, y_pred_tr)
                    EER, t_idx = self.compute_eer(FRR_t, FAR_t)
                    TH = self._THRESHOLDs[t_idx]

                    y_pred = best_model.predict_proba(x_test)[:, 1]
                    
                    y_pred[y_pred >= TH] = 1
                    y_pred[y_pred < TH] = 0
            elif self._classifier_name == 'svm-linear':
                classifier = svm.SVC(kernel='linear', probability=True, random_state=self._random_state, C=10 ** logC)
                best_model = classifier.fit(x_train, y_train)
                y_pred_tr = best_model.predict_proba(x_train)[:, 1]

                FRR_t, FAR_t = self.FXR_calculater(y_train, y_pred_tr)
                EER, t_idx = self.compute_eer(FRR_t, FAR_t)
                TH = self._THRESHOLDs[t_idx]

                y_pred = best_model.predict_proba(x_test)[:, 1]
                
                y_pred[y_pred >= TH] = 1
                y_pred[y_pred < TH] = 0
            elif self._classifier_name == 'svm-poly':
                classifier = svm.SVC(kernel='poly', probability=True, random_state=self._random_state , C=10 ** logC, degree=degree, coef0=coef0)
                best_model = classifier.fit(x_train, y_train)
                y_pred_tr = best_model.predict_proba(x_train)[:, 1]

                FRR_t, FAR_t = self.FXR_calculater(y_train, y_pred_tr)
                EER, t_idx = self.compute_eer(FRR_t, FAR_t)
                TH = self._THRESHOLDs[t_idx]

                y_pred = best_model.predict_proba(x_test)[:, 1]
                
                y_pred[y_pred >= TH] = 1
                y_pred[y_pred < TH] = 0
            elif self._classifier_name == 'svm-rbf':
                classifier = svm.SVC(kernel='rbf', probability=True, random_state=self._random_state, C=10 ** logC, gamma=10 ** logGamma)
                best_model = classifier.fit(x_train, y_train)
                y_pred_tr = best_model.predict_proba(x_train)[:, 1]

                FRR_t, FAR_t = self.FXR_calculater(y_train, y_pred_tr)
                EER, t_idx = self.compute_eer(FRR_t, FAR_t)
                TH = self._THRESHOLDs[t_idx]

                y_pred = best_model.predict_proba(x_test)[:, 1]
                
                y_pred[y_pred >= TH] = 1
                y_pred[y_pred < TH] = 0
            elif self._classifier_name == "rf":
                classifier = RandomForestClassifier(n_estimators=int(n_estimators), max_features=int(max_features))
                best_model = classifier.fit(x_train, y_train)
                y_pred_tr = best_model.predict_proba(x_train)[:, 1]

                FRR_t, FAR_t = self.FXR_calculater(y_train, y_pred_tr)
                EER, t_idx = self.compute_eer(FRR_t, FAR_t)
                TH = self._THRESHOLDs[t_idx]

                y_pred = best_model.predict_proba(x_test)[:, 1]
                
                y_pred[y_pred >= TH] = 1
                y_pred[y_pred < TH] = 0
            elif self._classifier_name == "nb":
                pass
            elif self._classifier_name == "if":
                classifier = IsolationForest(n_estimators=int(n_estimators), max_features=int(max_features), random_state=self._random_state)
                best_model = classifier.fit(x_train)
                EER = 0
                TH = 0
                y_pred = best_model.predict(x_test)
            elif self._classifier_name == "ocsvm":
                pass
            elif self._classifier_name == "svdd":
                pass
            elif self._classifier_name == "lda":
                pass
            elif self._classifier_name == "tm":
                pass
            else:
                raise(f'Unknown algorithm: {self._classifier_name}')

            return optunity.metrics.bacc(y_test, y_pred, 1)
            
        pmap8 = optunity.parallel.create_pmap(8)
        solver = optunity.solvers.ParticleSwarm(num_particles=num_particles, num_generations=num_generations, **search)
        optimum, stats = optunity.optimize(solver, performance, maximize=True) # , pmap=pmap8
        
        # breakpoint()
        # params.update(fixed_params)

        # A = optunity.make_solver('particle swarm', num_particles=10, num_generations=10)
        # param, info, solver = optunity.maximize(performance, solver_name='particle swarm', num_evals=num_evals, **search)

        return optimum, stats

 