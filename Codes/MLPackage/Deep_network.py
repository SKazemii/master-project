from .Classifier import *

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
            # callbacks=[
            #     calculating_threshold( train=([pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:]), validation=([pairval[:, 0], pairval[:, 1]], labelval[:]),)
            # ],
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


