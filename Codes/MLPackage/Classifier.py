from .Features import *

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
        # breakpoint()
        for tx in self._THRESHOLDs:
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
        x_test,
        x_test_U
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

            if "params" in kwargs.keys():
                classifier = knn(n_neighbors=int(kwargs["params"]['n_neighbors']))
            else:
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

        elif self._classifier_name == "svm-linear":
            if "params" in kwargs.keys():
                classifier = svm.SVC(kernel='linear', probability=True, random_state=self._random_state, C=10 ** kwargs["params"]['logC'] )
            else:
                classifier = svm.SVC(kernel='linear', probability=True, random_state=self._random_state, )
               
            best_model = classifier.fit(x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values)
            y_pred_tr = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]

            FRR_t, FAR_t = self.FXR_calculater(x_train["ID"], y_pred_tr)
            EER, t_idx = self.compute_eer(FRR_t, FAR_t)
            TH = self._THRESHOLDs[t_idx]

            y_pred = best_model.predict_proba(x_test.iloc[:, :-1].values)[:, 1]

        elif self._classifier_name == "svm-rbf":
            if "params" in kwargs.keys():
                classifier = svm.SVC(kernel='rbf', probability=True, random_state=self._random_state, C=10 ** kwargs["params"]['logC'], gamma=10 ** kwargs["params"]['logGamma'] )
            else:
                classifier = svm.SVC(kernel='rbf', probability=True, random_state=self._random_state, )


            best_model = classifier.fit(x_train.iloc[:, :-1].values, x_train.iloc[:, -1].values)
            y_pred_tr = best_model.predict_proba(x_train.iloc[:, :-1].values)[:, 1]

            FRR_t, FAR_t = self.FXR_calculater(x_train["ID"], y_pred_tr)
            EER, t_idx = self.compute_eer(FRR_t, FAR_t)
            TH = self._THRESHOLDs[t_idx]

            y_pred = best_model.predict_proba(x_test.iloc[:, :-1].values)[:, 1]

        elif self._classifier_name == "svm-poly":
            if "params" in kwargs.keys():
                classifier = svm.SVC(kernel='poly', probability=True, random_state=self._random_state, C=10 ** kwargs["params"]['logC'], degree=kwargs["params"]['degree'], coef0=kwargs["params"]['coef0'] )
            else:
                classifier = svm.SVC(kernel='poly', probability=True, random_state=self._random_state, )
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
            if "params" in kwargs.keys():
                classifier = OneClassSVM(kernel='linear',nu=kwargs["params"]['nu'])
            else:
                classifier = OneClassSVM(kernel='linear',nu=0.1)
            
            best_model = classifier.fit(x_train.iloc[:, :-1].values)
            EER = 0
            TH = 0

            y_pred = best_model.predict(x_test.iloc[:, :-1].values)

        elif self._classifier_name == "svdd":
            if "params" in kwargs.keys():
                classifier = OneClassSVM(kernel='rbf',nu=kwargs["params"]['nu'])
            else:
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

 