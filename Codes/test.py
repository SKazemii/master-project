from MLPackage.Stepscan import * 

def Participant_Count():
    setting = {
        "dataset_name": "casia",
        "_classifier_name": "TM",
        "_combination": True,
        "_CNN_weights": "imagenet",
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": "",
        "_min_number_of_sample": 30,
        "_known_imposter": 3,
        "_unknown_imposter": 5,
        "_number_of_unknown_imposter_samples": 1.0,  # Must be less than 1
        "_waveletname": "coif1",
        "_pywt_mode": "constant",
        "_wavelet_level": 4,
        "_p_training_samples": 27,
        "_train_ratio": 1000,
        "_ratio": False,
        "_KNN_n_neighbors": 5,
        "_KNN_metric": "euclidean",
        "_KNN_weights": "uniform",
        "_SVM_kernel": "linear",
        "_KFold": 10,
        "_random_runs": 10,
        "_persentage": 0.95,
        "_normilizing": "z-score",
    }

    A = Pipeline(setting)
    DF_feature_all, feature_set_names = A.extracting_feature_set1("casia")

    A.collect_results(A.run(DF_feature_all, feature_set_names), "COP1+DEND")


def Participant_Count_shod():
    setting = {
        "dataset_name": "casia",
        "_classifier_name": "TM",
        "_combination": True,
        "_CNN_weights": "imagenet",
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": "",
        "_min_number_of_sample": 30,
        "_known_imposter": 3,
        "_unknown_imposter": 5,
        "_number_of_unknown_imposter_samples": 1.0,  # Must be less than 1
        "_waveletname": "coif1",
        "_pywt_mode": "constant",
        "_wavelet_level": 4,
        "_p_training_samples": 27,
        "_train_ratio": 1000,
        "_ratio": False,
        "_KNN_n_neighbors": 5,
        "_KNN_metric": "euclidean",
        "_KNN_weights": "uniform",
        "_SVM_kernel": "linear",
        "_KFold": 10,
        "_random_runs": 10,
        "_persentage": 0.95,
        "_normilizing": "z-score",
    }

    A = Pipeline(setting)
    DF_feature_all, feature_set_names = A.extracting_feature_set1("casia-shod")

    p0 = [5, 10]
    p1 = [5, 10]  # , 20, 25, 30]
    p2 = ["TM", "svm"]

    space = list(product(p0, p1, p2))
    space = space[:]

    for idx, parameters in enumerate(space):
        if parameters[1] + parameters[0] > 15:
            continue
        logger.info(
            f"Starting [step {idx+1} out of {len(space)}], parameters: {parameters}"
        )

        A._known_imposter = parameters[1]
        A._unknown_imposter = parameters[0]
        A._classifier_name = parameters[2]

        tic = timeit.default_timer()
        A.collect_results(A.run(DF_feature_all, feature_set_names), "COP+shod+DEND")
        toc = timeit.default_timer()

        logger.info(
            f"ending [step {idx+1} out of {len(space)}], parameters: {parameters}, process time: {round(toc-tic, 2)}"
        )


def Feature_Count():
    setting = {
        "dataset_name": "casia",
        "_classifier_name": "TM",
        "_combination": True,
        "_CNN_weights": "imagenet",
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": "",
        "_min_number_of_sample": 30,
        "_known_imposter": 3,
        "_unknown_imposter": 5,
        "_number_of_unknown_imposter_samples": 1.0,  # Must be less than 1
        "_waveletname": "coif1",
        "_pywt_mode": "constant",
        "_wavelet_level": 4,
        "_p_training_samples": 11,
        "_train_ratio": 34,
        "_ratio": False,
        "_KNN_n_neighbors": 5,
        "_KNN_metric": "euclidean",
        "_KNN_weights": "uniform",
        "_SVM_kernel": "linear",
        "_KFold": 10,
        "_random_runs": 20,
        "_persentage": 0.95,
        "_normilizing": "z-score",
    }

    A = Pipeline(setting)

    GRFs, COPs, COAs, pre_images, labels = A.loading_pre_features("casia")
    COA_handcrafted = A.loading_COA_handcrafted(COAs)
    COP_handcrafted = A.loading_COP_handcrafted(COPs)
    GRF_handcrafted = A.loading_GRF_handcrafted(GRFs)
    COA_WPT = A.loading_COA_WPT(COAs)
    COP_WPT = A.loading_COP_WPT(COPs)
    GRF_WPT = A.loading_GRF_WPT(GRFs)

    deep_features_list = A.loading_deep_features_from_list(
        (pre_images, labels), ["P100", "P80"], "resnet50.ResNet50"
    )
    image_from_list = A.loading_pre_image_from_list(pre_images, ["P80", "P100"])
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

    p0 = [5, 30]
    p1 = [5, 10, 15, 20, 25, 30]
    p1 = [5, 30]

    space = list(product(p0, p1))
    space = space[:]

    for idx, parameters in enumerate(space):

        A._known_imposter = parameters[0]
        A._unknown_imposter = parameters[1]

        tic = timeit.default_timer()
        A.collect_results(A.run(feature_set, labels, feature_set_names), "COP+GRF")
        toc = timeit.default_timer()

        logger.info(
            f"[step {idx+1} out of {len(space)}], parameters: {parameters}, process time: {round(toc-tic, 2)}"
        )


def template_Count():

    setting = {
        "dataset_name": "casia",
        "_classifier_name": "TM",
        "_combination": True,
        "_CNN_weights": "imagenet",
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": "",
        "_min_number_of_sample": 30,
        "_known_imposter": 3,
        "_unknown_imposter": 5,
        "_number_of_unknown_imposter_samples": 1.0,  # Must be less than 1
        "_waveletname": "coif1",
        "_pywt_mode": "constant",
        "_wavelet_level": 4,
        "_p_training_samples": 11,
        "_train_ratio": 34,
        "_ratio": False,
        "_KNN_n_neighbors": 5,
        "_KNN_metric": "euclidean",
        "_KNN_weights": "uniform",
        "_SVM_kernel": "linear",
        "_KFold": 10,
        "_random_runs": 20,
        "_persentage": 0.95,
        "_normilizing": "z-score",
    }

    A = Pipeline(setting)

    GRFs, COPs, COAs, pre_images, labels = A.loading_pre_features("casia")
    COA_handcrafted = A.loading_COA_handcrafted(COAs)
    COP_handcrafted = A.loading_COP_handcrafted(COPs)
    GRF_handcrafted = A.loading_GRF_handcrafted(GRFs)
    COA_WPT = A.loading_COA_WPT(COAs)
    COP_WPT = A.loading_COP_WPT(COPs)
    GRF_WPT = A.loading_GRF_WPT(GRFs)

    deep_features_list = A.loading_deep_features_from_list(
        (pre_images, labels), ["P100", "P80"], "resnet50.ResNet50"
    )
    image_from_list = A.loading_pre_image_from_list(pre_images, ["P80", "P100"])
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

    p0 = [5, 30]
    p1 = [5, 10, 15, 20, 25, 30]
    p1 = [5, 30]

    space = list(product(p0, p1))
    space = space[:]

    for idx, parameters in enumerate(space):

        A._known_imposter = parameters[0]
        A._unknown_imposter = parameters[1]

        tic = timeit.default_timer()
        A.collect_results(A.run(feature_set, labels, feature_set_names), "COP+GRF")
        toc = timeit.default_timer()

        logger.info(
            f"[step {idx+1} out of {len(space)}], parameters: {parameters}, process time: {round(toc-tic, 2)}"
        )


def lightweight():

    setting = {
        "dataset_name": "casia",
        "_classifier_name": "TM",
        "_combination": True,
        "_CNN_weights": "imagenet",
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": "",
        "_CNN_epochs": 500,
        "_CNN_optimizer": "adam",
        "_val_size": 0.2,
        "_min_number_of_sample": 30,
        "_known_imposter": 3,
        "_unknown_imposter": 5,
        "_number_of_unknown_imposter_samples": 1.0,  # Must be less than 1
        "_waveletname": "coif1",
        "_pywt_mode": "constant",
        "_wavelet_level": 4,
        "_p_training_samples": 11,
        "_train_ratio": 34,
        "_ratio": False,
        "_KNN_n_neighbors": 5,
        "_KNN_metric": "euclidean",
        "_KNN_weights": "uniform",
        "_SVM_kernel": "linear",
        "_KFold": 10,
        "_random_runs": 20,
        "_persentage": 0.95,
        "_normilizing": "z-score",
    }

    A = Pipeline(setting)

    A._CNN_batch_size = 64
    A._CNN_epochs = 666
    A._CNN_optimizer = tf.keras.optimizers.Adadelta()
    A._val_size = 0.2

    A._known_imposter = 32

    image_feature_name = [
        "P100"
    ]  # ["CD", "PTI", "Tmin", "Tmax", "P50", "P60", "P70", "P80", "P90", "P100"]
    dataset_name = "casia"
    CNN_name = "lightweight_CNN"

    # model = A.train_deep_CNN(dataset_name, image_feature_name, CNN_name, update=True)
    # model = A.train_Seamese_model(image_feature_name, dataset_name, update=True)

    path = os.path.join(os.getcwd(), "results", CNN_name, "best.h5")
    logger.info("best_model")
    model = load_model(path)
    model.summary()

    # breakpoint()
    pre_images, labels = A.loading_pre_features_image(dataset_name)
    pre_image = A.loading_image_features_from_list(pre_images, image_feature_name)
    known_imposters, _ = A.filtering_subjects_and_samples_deep((pre_image, labels))

    # pre_image, labels = known_imposters[0], known_imposters[1]

    data = (
        pre_image[~labels["ID"].isin(A._known_imposter_list)],
        labels[~labels["ID"].isin(A._known_imposter_list)],
    )

    deep_features = A.loading_deep_feature_from_model(
        model, "last_dense", data, image_feature_name
    )

    A._known_imposter = 32
    A._unknown_imposter = 0

    DF_feature_all = pd.concat([deep_features, data[1].reset_index(drop=True)], axis=1)

    feature_set_names = ["deep_P100_lightweight_CNN"]

    p0 = [5, 10, 15, 20, 25, 30]
    p1 = [5, 10, 15, 20, 25, 30]
    p2 = ["TM"]

    space = list(product(p0, p1, p2))
    space = space[:]
    # i=0
    for idx, parameters in enumerate(space):
        if parameters[1] + parameters[0] > 32:
            continue
        logger.info(
            f"Starting [step {idx+1} out of {len(space)}], parameters: {parameters}"
        )

        A._known_imposter = parameters[1]
        A._unknown_imposter = parameters[0]
        A._classifier_name = parameters[2]

        tic = timeit.default_timer()
        # print(parameters, i)
        # i += 1
        A.collect_results(A.run(DF_feature_all, feature_set_names), "LWCNN")
        toc = timeit.default_timer()

        logger.info(
            f"ending [step {idx+1} out of {len(space)}], parameters: {parameters}, process time: {round(toc-tic, 2)}"
        )

    breakpoint()


def FT():
    setting = {
        "dataset_name": "casia",
        "_classifier_name": "TM",
        "_combination": True,
        "_CNN_weights": "imagenet",
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": "",
        "_CNN_epochs": 500,
        "_CNN_optimizer": "adam",
        "_val_size": 0.2,
        "_min_number_of_sample": 30,
        "_known_imposter": 3,
        "_unknown_imposter": 5,
        "_number_of_unknown_imposter_samples": 1.0,  # Must be less than 1
        "_waveletname": "coif1",
        "_pywt_mode": "constant",
        "_wavelet_level": 4,
        "_p_training_samples": 11,
        "_train_ratio": 34,
        "_ratio": False,
        "_KNN_n_neighbors": 5,
        "_KNN_metric": "euclidean",
        "_KNN_weights": "uniform",
        "_SVM_kernel": "linear",
        "_KFold": 10,
        "_random_runs": 20,
        "_persentage": 0.95,
        "_normilizing": "z-score",
    }

    A = Pipeline(setting)

    A._CNN_batch_size = 64
    A._CNN_epochs = 1000
    A._CNN_optimizer = tf.keras.optimizers.Adadelta()
    A._val_size = 0.2

    A._known_imposter = 32

    image_feature_name = [
        "P100"
    ]  # ["CD", "PTI", "Tmin", "Tmax", "P50", "P60", "P70", "P80", "P90", "P100"]
    dataset_name = "casia"
    CNN_name = "ResNet50"

    model = A.train_deep_CNN(dataset_name, image_feature_name, CNN_name, update=True)
    breakpoint()


def test_all_pipelines():
    setting = {
        "dataset_name": "casia",
        "_classifier_name": "TM",
        "_combination": True,
        "_CNN_weights": "imagenet",
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": "",
        "_CNN_epochs": 500,
        "_CNN_optimizer": "adam",
        "_val_size": 0.2,
        "_min_number_of_sample": 30,
        "_known_imposter": 32,
        "_unknown_imposter": 32,
        "_number_of_unknown_imposter_samples": 1.0,  # Must be less than 1
        "_waveletname": "coif1",
        "_pywt_mode": "constant",
        "_wavelet_level": 4,
        "_p_training_samples": 11,
        "_train_ratio": 34,
        "_ratio": False,
        "_KNN_n_neighbors": 5,
        "_KNN_metric": "euclidean",
        "_KNN_weights": "uniform",
        "_SVM_kernel": "linear",
        "_KFold": 10,
        "_random_runs": 20,
        "_persentage": 0.95,
        "_normilizing": "z-score",
    }

    A = Pipeline(setting)

    A._known_imposter = 23
    A._unknown_imposter = 10
    A._classifier_name = "knn"
    nam = "All"

    image_feature_name = [
        "P100"
    ]  # ["CD", "PTI", "Tmin", "Tmax", "P50", "P60", "P70", "P80", "P90", "P100"]
    dataset_name = "casia"

    GRFs, COPs, COAs, pre_images, labels = A.loading_pre_features(dataset_name)

    ####################################################################################################################
    # pipeline 1: COP and GRf
    COP_handcrafted = A.loading_COP_handcrafted(COPs)
    GRF_handcrafted = A.loading_GRF_handcrafted(GRFs)
    COP_WPT = A.loading_COP_WPT(COPs)
    GRF_WPT = A.loading_GRF_WPT(GRFs)

    feature_set_names = [
        "COP_handcrafted",
        "COPs",
        "COP_WPT",
        "GRF_handcrafted",
        "GRFs",
        "GRF_WPT",
    ]
    DF_feature_all = pd.concat(
        [COP_handcrafted, COPs, COP_WPT, GRF_handcrafted, GRFs, GRF_WPT, labels], axis=1
    )

    result = get_results(A, feature_set_names, DF_feature_all)
    result["pipeline"] = "pipeline 1: COP and GRf"
    A.collect_results(result, nam)

    # breakpoint()

    ####################################################################################################################
    # pipeline 2: P100 and P80
    image_from_list = A.loading_pre_image_from_list(pre_images, image_feature_name)
    feature_set_names = ["P100"]
    DF_feature_all = pd.concat([i for i in image_from_list] + [labels], axis=1)

    result = get_results(A, feature_set_names, DF_feature_all)
    result["pipeline"] = "pipeline 2: P100"
    A.collect_results(result, nam)

    # breakpoint()

    ####################################################################################################################
    # pipeline 3: pre_trained CNN (Resnet50)
    deep_features_list = A.loading_deep_features_from_list(
        (pre_images, labels), image_feature_name, "resnet50.ResNet50"
    )
    feature_set_names = ["deep_P100_resnet50"]
    DF_feature_all = pd.concat([i for i in deep_features_list] + [labels], axis=1)

    result = get_results(A, feature_set_names, DF_feature_all)
    result["pipeline"] = "pipeline 3: pre_trained CNN"
    A.collect_results(result, nam)

    # breakpoint()

    ##################################################################################################################
    # pipeline 4: lightweight CNN

    # model = A.train_deep_CNN(dataset_name, image_feature_name, CNN_name, update=True)
    # A._persentage= 1.0
    CNN_name = "lightweight_CNN"
    path = os.path.join(os.getcwd(), "results", CNN_name, "best.h5")
    model = load_model(path)

    pre_image = A.loading_image_features_from_list(pre_images, image_feature_name)

    data = pre_image, labels

    deep_features = A.loading_deep_feature_from_model(
        model, "last_dense", data, image_feature_name
    )
    DF_feature_all = pd.concat([deep_features, data[1].reset_index(drop=True)], axis=1)
    feature_set_names = ["deep_P100_lightweight_CNN_trained"]

    result = get_results(A, feature_set_names, DF_feature_all)
    result["pipeline"] = "pipeline 4: lightweight CNN"
    A.collect_results(result, nam)

    # breakpoint()

    ####################################################################################################################
    # pipeline 5: Fine-tuning Resnet50

    path = os.path.join(os.getcwd(), "results", "ResNet50_FT", "best.h5")
    model = load_model(path)

    pre_image = A.loading_image_features_from_list(pre_images, image_feature_name)

    data = pre_image, labels

    CNN_name = "ResNet50_FT"
    deep_features = A.loading_deep_feature_from_model(
        model, "last_dense", data, image_feature_name
    )
    DF_feature_all = pd.concat([deep_features, data[1].reset_index(drop=True)], axis=1)
    feature_set_names = ["deep_P100_ResNet50_trained"]

    result = get_results(A, feature_set_names, DF_feature_all)
    result["pipeline"] = "pipeline 5: Fine-tuning Resnet50"
    A.collect_results(result, nam)

    breakpoint()


def get_results(A, feature_set_names, DF_feature_all):
    subjects, samples = np.unique(DF_feature_all["ID"].values, return_counts=True)

    ss = [a[0] for a in list(zip(subjects, samples)) if a[1] >= A._min_number_of_sample]

    known_imposter_list = ss[32 : 32 + A._known_imposter]
    unknown_imposter_list = ss[-A._unknown_imposter :]

    DF_unknown_imposter = DF_feature_all[
        DF_feature_all["ID"].isin(unknown_imposter_list)
    ]
    DF_known_imposter = DF_feature_all[DF_feature_all["ID"].isin(known_imposter_list)]

    results = list()
    for idx, subject in enumerate(DF_known_imposter["ID"].unique()):
        # if idx not in [0, 1]: #todo: remove this block to run for all subjects.
        #     break
        logger.info(
            f"   Subject number: {idx} out of {len(known_imposter_list)} (subject ID is {subject})"
        )
        X, U = A.binarize_labels(DF_known_imposter, DF_unknown_imposter, subject)

        CV = model_selection.StratifiedKFold(
            n_splits=A._KFold, shuffle=False
        )  # random_state=self._random_state,

        cv_results = list()

        ncpus = int(
            os.environ.get("SLURM_CPUS_PER_TASK", default=multiprocessing.cpu_count())
        )
        pool = multiprocessing.Pool(processes=ncpus)
        # breakpoinqt()
        for fold, (train_index, test_index) in enumerate(
            CV.split(X.iloc[:, :-1], X.iloc[:, -1])
        ):
            # breakpoint()
            res = pool.apply_async(
                A.fold_calculating,
                args=(
                    feature_set_names,
                    subject,
                    X,
                    U,
                    train_index,
                    test_index,
                    fold,
                ),
                callback=cv_results.append,
            )
            # print(res.get())  # this will raise an exception if it happens within func

            # cv_results.append(A.fold_calculating(feature_set_names, subject, X, U, train_index, test_index, fold)) #todo: comment this line to run all folds
            # break #todo: comment this line to run all folds

        pool.close()
        pool.join()

        result = A.compacting_results(cv_results, subject)
        results.append(result)
        # breakpoint()
    return pd.DataFrame(results, columns=A._col)


def train_e2e_CNN():
    setting = {
        "dataset_name": "casia",
        "_classifier_name": "TM",
        "_combination": True,
        "_CNN_weights": "imagenet",
        "_verbose": False,
        "_CNN_batch_size": 32,
        "_CNN_base_model": "",
        "_CNN_epochs": 500,
        "_CNN_optimizer": "adam",
        "_val_size": 0.2,
        "_min_number_of_sample": 30,
        "_known_imposter": 32,
        "_unknown_imposter": 32,
        "_number_of_unknown_imposter_samples": 1.0,  # Must be less than 1
        "_waveletname": "coif1",
        "_pywt_mode": "constant",
        "_wavelet_level": 4,
        "_p_training_samples": 11,
        "_train_ratio": 34,
        "_ratio": False,
        "_KNN_n_neighbors": 5,
        "_KNN_metric": "euclidean",
        "_KNN_weights": "uniform",
        "_SVM_kernel": "linear",
        "_KFold": 10,
        "_random_runs": 20,
        "_persentage": 0.95,
        "_normilizing": "z-score",
    }

    A = Pipeline(setting)

    A._known_imposter = 23
    A._unknown_imposter = 10
    A._classifier_name = "svm"
    nam = "All"
    A._CNN_epochs = 2

    image_feature_name = ["P100"]  # ["CD", "PTI", "Tmin", "Tmax", "P50", "P60", "P70", "P80", "P90", "P100"]
    dataset_name = "casia"
    CNN_name = "lightweight_CNN"

    GRFs, COPs, COAs, pre_images, labels = A.loading_pre_features(dataset_name)
    pre_image1 = A.loading_image_features_from_list(pre_images, image_feature_name)

    pre_image1 = A.normalizing_image_features(pre_image1)
    pre_image1 = pre_image1/255

    subjects, samples = np.unique(labels["ID"].values, return_counts=True)

    ss = [a[0] for a in list(zip(subjects, samples)) if a[1] >= A._min_number_of_sample]

    known_imposter_list = ss[32 : 32 + A._known_imposter]
    unknown_imposter_list = ss[-A._unknown_imposter :]

    label = labels[labels["ID"].isin(known_imposter_list)]
    pre_image = pre_image1[label.index, :, :, :]

    # DF_unknown_imposter =  DF_feature_all[DF_feature_all["ID"].isin(unknown_imposter_list)]
    U_label = labels[labels["ID"].isin(unknown_imposter_list)]
    U_pre_image = pre_image1[U_label.index, :, :, :]

    results = list()
    for idx, subject in enumerate(known_imposter_list):
        if idx <= 13:
            continue
        logger.info(f"   Subject number: {idx} out of {len(known_imposter_list)} (subject ID is {subject})")

        label_binariezed = tf.keras.utils.to_categorical(A.label_encoding(label))

        label_ = np.expand_dims(label_binariezed[:, idx], axis=1)


        X_train, X_test, y_train, y_test = model_selection.train_test_split( pre_image, label_, test_size=0.2, random_state=A._random_state, stratify=label_, )
        X_train, X_val, y_train, y_val = model_selection.train_test_split( X_train, y_train,  test_size=0.2, random_state=A._random_state, stratify=y_train, )  # todo

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000)
        train_ds = train_ds.batch(A._CNN_batch_size)
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(1000)
        val_ds = val_ds.batch(A._CNN_batch_size)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.batch(A._CNN_batch_size)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        U_ds = tf.data.Dataset.from_tensor_slices((U_pre_image, np.zeros((U_label.shape[0], 1))))
        U_ds = U_ds.batch(A._CNN_batch_size)
        U_ds = U_ds.cache().prefetch(buffer_size=AUTOTUNE)

        total = y_train.shape[0]
        pos = y_train.sum()
        neg = total - pos

        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)

        class_weight = {0: weight_for_0, 1: weight_for_1}


        model = A.train_e2e(train_ds, val_ds, test_ds, CNN_name, subject, class_weight, update=False )
        results = A.test_e2e( train_ds, val_ds, test_ds, CNN_name, subject, U_data=U_ds)

        results.update({
            "training_samples": y_train.shape[0],
            "pos_training_samples": y_train.sum(),
            "validation_samples": y_val.shape[0],
            "pos_validation_samples": y_val.sum(),
            "testing_samples": y_test.shape[0],
            "pos_testing_samples": y_test.sum(),
            "unknown samples": U_label.shape[0],

        })

        for i in results:
            try:
                results_dict[i].append(results[i])
            except UnboundLocalError:
                results_dict = {i: [] for i in results.keys()}
                results_dict[i].append(results[i])

    results = pd.DataFrame.from_dict(results_dict)
    path = os.path.join(os.getcwd(), "results", "e2e", "result.xlsx")
    results.to_excel(path)
    return results


def Toon_p100():
    setting = {
        "dataset_name": "casia",
        "_classifier_name": "TM",
        "_combination": True,
        "_CNN_weights": "imagenet",
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": "",
        "_CNN_epochs": 500,
        "_CNN_optimizer": "adam",
        "_val_size": 0.2,
        "_min_number_of_sample": 30,
        "_known_imposter": 32,
        "_unknown_imposter": 32,
        "_number_of_unknown_imposter_samples": 1.0,  # Must be less than 1
        "_waveletname": "coif1",
        "_pywt_mode": "constant",
        "_wavelet_level": 4,
        "_p_training_samples": 11,
        "_train_ratio": 34,
        "_ratio": False,
        "_KNN_n_neighbors": 5,
        "_KNN_metric": "euclidean",
        "_KNN_weights": "uniform",
        "_SVM_kernel": "linear",
        "_KFold": 10,
        "_random_runs": 20,
        "_persentage": 0.99,
        "_normilizing": "z-mean",
    }

    A = Pipeline(setting)

    A._known_imposter = 55
    A._unknown_imposter = 10
    A._classifier_name = "TM"
    nam = "All"

    image_feature_name = [
        "P100"
    ]  # ["CD", "PTI", "Tmin", "Tmax", "P50", "P60", "P70", "P80", "P90", "P100"]
    dataset_name = "casia"

    GRFs, COPs, COAs, pre_images, labels = A.loading_pre_features(dataset_name)

    # pipeline 2: P100 and P80
    image_from_list = A.loading_pre_image_from_list(pre_images, image_feature_name)
    feature_set_names = ["P100"]
    DF_feature_all = pd.concat([i for i in image_from_list] + [labels], axis=1)

    subjects, samples = np.unique(DF_feature_all["ID"].values, return_counts=True)

    ss = [a[0] for a in list(zip(subjects, samples)) if a[1] >= A._min_number_of_sample]

    known_imposter_list = ss[0 : 0 + A._known_imposter]
    unknown_imposter_list = ss[-A._unknown_imposter :]

    DF_unknown_imposter = DF_feature_all[
        DF_feature_all["ID"].isin(unknown_imposter_list)
    ]
    DF_known_imposter = DF_feature_all[DF_feature_all["ID"].isin(known_imposter_list)]

    results = list()
    for idx, subject in enumerate(DF_known_imposter["ID"].unique()):

        logger.info(
            f"   Subject number: {idx} out of {len(known_imposter_list)} (subject ID is {subject})"
        )

        X, U = A.binarize_labels(DF_known_imposter, DF_unknown_imposter, subject)

        X_train, X_test = model_selection.train_test_split(
            X, test_size=0.2, random_state=A._random_state, stratify=X.iloc[:, -1]
        )
        # X_train, X_val = model_selection.train_test_split(X_train, test_size=0.2, random_state=A._random_state, stratify=X_train.iloc[:,-1])#todo
        # breakpoint()

        df_train, df_test, df_test_U = A.scaler(X_train, X_test, U)

        df_train, df_test, df_test_U, num_pc = A.projector(
            df_train, df_test, df_test_U, feature_set_names
        )
        result, CM_bd, CM_ud = A.ML_classifier(df_train, df_test, df_test_U, subject)

        cv_results = (
            result
            + CM_ud.reshape(1, -1).tolist()[0]
            + CM_bd.reshape(1, -1).tolist()[0]
            + [num_pc]
        )

        result = list()

        result.append(
            [
                A._test_id,
                subject,
                A._combination,
                A._classifier_name,
                A._normilizing,
                A._persentage,
                # configs["classifier"][CLS],
            ]
        )

        result.append(cv_results)
        # result.append([np.array(CM_bd).mean(axis=0), np.array(CM_ud).mean(axis=0)])

        # _CNN_weights = 'imagenet'
        # _CNN_base_model = ""

        result.append(
            [
                A._KFold,
                A._p_training_samples,
                A._train_ratio,
                A._ratio,
                # pos_te_samples,
                # neg_te_samples,
                A._known_imposter,
                A._unknown_imposter,
                A._min_number_of_sample,
                A._number_of_unknown_imposter_samples,
                X_train.shape[0],
                X_train.iloc[:, -1].sum(),
                "-",
                "-",
                X_test.shape[0],
                X_test.iloc[:, -1].sum(),
            ]
        )

        results.append([val for sublist in result for val in sublist])

    return pd.DataFrame(results, columns=A._col)


def second_retrain():
    setting = {
        "dataset_name": "casia",
        "_classifier_name": "knn",
        "_combination": True,
        "_CNN_weights": "imagenet",
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": "",
        "_CNN_epochs": 500,
        "_CNN_optimizer": "adam",
        "_val_size": 0.2,
        "_min_number_of_sample": 30,
        "_known_imposter": 3,
        "_unknown_imposter": 5,
        "_number_of_unknown_imposter_samples": 1.0,  # Must be less than 1
        "_waveletname": "coif1",
        "_pywt_mode": "constant",
        "_wavelet_level": 4,
        "_p_training_samples": 11,
        "_train_ratio": 34,
        "_ratio": False,
        "_KNN_n_neighbors": 5,
        "_KNN_metric": "euclidean",
        "_KNN_weights": "uniform",
        "_SVM_kernel": "linear",
        "_KFold": 10,
        "_random_runs": 20,
        "_persentage": 1.0,
        "_normilizing": "z-score",
    }

    A = Pipeline(setting)

    A._CNN_batch_size = 64
    A._CNN_epochs = 400
    A._CNN_optimizer = tf.keras.optimizers.Adadelta()
    A._val_size = 0.2

    A._known_imposter = 23
    A._unknown_imposter = 10

    image_feature_name = ["P100"]  # ["CD", "PTI", "Tmin", "Tmax", "P50", "P60", "P70", "P80", "P90", "P100"]
    dataset_name = "casia"
    CNN_name = "lightweight_CNN"

    GRFs, COPs, COAs, pre_images, labels = A.loading_pre_features(dataset_name)
    pre_image1 = A.loading_image_features_from_list(pre_images, image_feature_name)

    images_feat_norm = pre_image1 / 255

    subjects, samples = np.unique(labels["ID"].values, return_counts=True)

    ss = [a[0] for a in list(zip(subjects, samples)) if a[1] >= A._min_number_of_sample]

    known_imposter_list = ss[32 : 32 + A._known_imposter]
    unknown_imposter_list = ss[-A._unknown_imposter :]

    label = labels[labels["ID"].isin(known_imposter_list)]
    pre_image = images_feat_norm[label.index, :, :, :]

    U_label = labels[labels["ID"].isin(unknown_imposter_list)]
    U_pre_image = images_feat_norm[U_label.index, :, :, :]

    path = os.path.join(os.getcwd(), "results", "results", CNN_name, "best.h5")
    model = load_model(path)

    label_binariezed = tf.keras.utils.to_categorical(A.label_encoding(label))

    results = list()
    for idx, subject in enumerate(known_imposter_list):
        # if idx < 11:
        #     continue

        logger.info(f"   Subject number: {idx} out of {len(known_imposter_list)} (subject ID is {subject})")

        label_ = np.expand_dims(label_binariezed[:, idx], axis=1)

        X_train, X_test, y_train, y_test = model_selection.train_test_split( pre_image, label_, test_size=0.2, random_state=A._random_state, stratify=label_, )
        X_train, X_val, y_train, y_val = model_selection.train_test_split(  X_train,  y_train, test_size=0.2,  random_state=A._random_state,  stratify=y_train, )  # todo

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000)
        train_ds = train_ds.batch(A._CNN_batch_size)
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(1000)
        val_ds = val_ds.batch(A._CNN_batch_size)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.batch(A._CNN_batch_size)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        U_ds = tf.data.Dataset.from_tensor_slices((U_pre_image, np.zeros((U_label["ID"].shape[0], 1))))
        U_ds = U_ds.batch(A._CNN_batch_size)
        U_ds = U_ds.cache().prefetch(buffer_size=AUTOTUNE)

        np.zeros((U_label["ID"].shape[0], 1))

        total = y_train.shape[0]
        pos = y_train.sum()
        neg = total - pos
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}

        
        # binary_model = A.second_training(model, subject, train_ds, val_ds, test_ds, class_weight, update=False)
        # logger.info(f"   Subject number: {idx} out of {len(known_imposter_list)} (subject ID is {subject})")
        # sys.exit()

        path = os.path.join(os.getcwd(), "results", "results", "second_train", model.name, str(subject), "best.h5")
        binary_model = load_model(path, custom_objects={"BalancedAccuracy": BalancedAccuracy()})

        x = binary_model.layers[-2].output
        binary_model1 = Model(inputs=binary_model.input, outputs=x)
        # breakpoint()

        train_features = A.extract_deep_features(train_ds, binary_model1)
        val_features = A.extract_deep_features(val_ds, binary_model1)
        test_features = A.extract_deep_features(test_ds, binary_model1)
        U_features = A.extract_deep_features(U_ds, binary_model1)

        train_features, val_features, test_features, U_features = A.scaler( train_features, val_features, test_features, U_features)

        train_features, val_features, test_features, U_features, num_pc = A.projector(["deep_second_trained"], train_features,val_features, test_features, U_features,)

        result = A.ML_classifier(subject, x_train=train_features, x_val=val_features, x_test=test_features, x_test_U=U_features)
        result["num_pc"] = num_pc

        results = {
            "test_id": A._test_id,
            "subject": subject,
            "combination": A._combination,
            "classifier_name": A._classifier_name,
            "normilizing": A._normilizing,
            "persentage": A._persentage,
            "KFold": "-",
            "known_imposter": A._known_imposter,
            "unknown_imposter": A._unknown_imposter,
            "min_number_of_sample": A._min_number_of_sample,
            "training_samples": y_train.shape[0],
            "pos_training_samples": y_train.sum(),
            "validation_samples": y_val.shape[0],
            "pos_validation_samples": y_val.sum(),
            "testing_samples": y_test.shape[0],
            "pos_testing_samples": y_test.sum(),
        }
        results.update(result)

        for i in results:
            try:
                results_dict[i].append(results[i])
            except UnboundLocalError:
                results_dict = {i: [] for i in results.keys()}
                results_dict[i].append(results[i])
        # breakpoint()

    results = pd.DataFrame.from_dict(results_dict)
    return results


def new_image():
    setting = {
        "dataset_name": "casia",
        "_classifier_name": "TM",
        "_combination": True,
        "_CNN_weights": "imagenet",
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": "",
        "_CNN_epochs": 500,
        "_CNN_optimizer": "adam",
        "_val_size": 0.2,
        "_min_number_of_sample": 30,
        "_known_imposter": 32,
        "_unknown_imposter": 32,
        "_number_of_unknown_imposter_samples": 1.0,  # Must be less than 1
        "_waveletname": "coif1",
        "_pywt_mode": "constant",
        "_wavelet_level": 4,
        "_p_training_samples": 11,
        "_train_ratio": 34,
        "_ratio": False,
        "_KNN_n_neighbors": 5,
        "_KNN_metric": "euclidean",
        "_KNN_weights": "uniform",
        "_SVM_kernel": "linear",
        "_KFold": 10,
        "_random_runs": 20,
        "_persentage": 0.95,
        "_normilizing": "z-score",
    }

    A = Pipeline(setting)

    A._known_imposter = 23
    A._unknown_imposter = 10
    A._classifier_name = "knn"
    nam = "All"

    image_feature_name = [
        "P100"
    ]  # ["CD", "PTI", "Tmin", "Tmax", "P50", "P60", "P70", "P80", "P90", "P100"]
    dataset_name = "casia"

    GRFs, COPs, COAs, pre_images, labels = A.loading_pre_features(dataset_name)

    ####################################################################################################################
    # pipeline 2: P100 and P80
    image_from_list = A.loading_pre_image_from_list(pre_images, image_feature_name)
    feature_set_names = ["P100"]
    DF_feature_all = pd.concat([i for i in image_from_list] + [labels], axis=1)

    subjects, samples = np.unique(DF_feature_all["ID"].values, return_counts=True)

    ss = [
        a[0] for a in list(zip(subjects, samples)) if a[1] >= 30
    ]  # A._min_number_of_sample]

    known_imposter_list = ss[32 : 32 + A._known_imposter]
    unknown_imposter_list = ss[-A._unknown_imposter :]

    DF_unknown_imposter = DF_feature_all[
        DF_feature_all["ID"].isin(unknown_imposter_list)
    ]
    DF_known_imposter = DF_feature_all[DF_feature_all["ID"].isin(known_imposter_list)]

    results = list()
    for idx, subject in enumerate(DF_known_imposter["ID"].unique()):
        # if idx not in [0, 1]: #todo: remove this block to run for all subjects.
        #     break
        logger.info(
            f"   Subject number: {idx} out of {len(known_imposter_list)} (subject ID is {subject})"
        )

        X, U = A.binarize_labels(DF_known_imposter, DF_unknown_imposter, subject)

        breakpoint()
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X.iloc[:, :-1],
            X.iloc[:, -1],
            test_size=0.2,
            stratify=X.iloc[:, -1],
            random_state=A._random_state,
        )

        cv_results = list()

        ncpus = int(
            os.environ.get("SLURM_CPUS_PER_TASK", default=multiprocessing.cpu_count())
        )
        pool = multiprocessing.Pool(processes=ncpus)
        # breakpoinqt()
        for fold, (train_index, test_index) in enumerate(
            CV.split(X.iloc[:, :-1], X.iloc[:, -1])
        ):
            # breakpoint()
            res = pool.apply_async(
                A.fold_calculating,
                args=(
                    feature_set_names,
                    subject,
                    X,
                    U,
                    train_index,
                    test_index,
                    fold,
                ),
                callback=cv_results.append,
            )
            # print(res.get())  # this will raise an exception if it happens within func

            # cv_results.append(A.fold_calculating(feature_set_names, subject, X, U, train_index, test_index, fold)) #todo: comment this line to run all folds
            # break #todo: comment this line to run all folds

        pool.close()
        pool.join()

        result = A.compacting_results(cv_results, subject)
        results.append(result)

    result = get_results(A, feature_set_names, DF_feature_all)
    result["pipeline"] = "pipeline 2: P100"
    A.collect_results(result, nam)


def two_stage():
    setting = {
        "dataset_name": "casia",
        "_classifier_name": "TM",
        "_combination": True,
        "_CNN_weights": "imagenet",
        "_verbose": True,
        "_CNN_batch_size": 32,
        "_CNN_base_model": "",
        "_CNN_epochs": 500,
        "_CNN_optimizer": "adam",
        "_val_size": 0.2,
        "_min_number_of_sample": 30,
        "_known_imposter": 32,
        "_unknown_imposter": 32,
        "_number_of_unknown_imposter_samples": 1.0,  # Must be less than 1
        "_waveletname": "coif1",
        "_pywt_mode": "constant",
        "_wavelet_level": 4,
        "_p_training_samples": 11,
        "_train_ratio": 34,
        "_ratio": False,
        "_KNN_n_neighbors": 5,
        "_KNN_metric": "euclidean",
        "_KNN_weights": "uniform",
        "_SVM_kernel": "linear",
        "_KFold": 10,
        "_random_runs": 20,
        "_persentage": 0.95,
        "_normilizing": "z-score",
    }

    A = Pipeline(setting)

    A._known_imposter = 23
    A._unknown_imposter = 10
    A._classifier_name = "knn"
    nam = "All"

    dataset_name = "casia"

    GRFs, COPs, COAs, pre_images, labels = A.loading_pre_features(dataset_name)

    ####################################################################################################################
    # pipeline 1: COP and GRf
    COP_handcrafted = A.loading_COP_handcrafted(COPs)
    GRF_handcrafted = A.loading_GRF_handcrafted(GRFs)
    COP_WPT = A.loading_COP_WPT(COPs)
    GRF_WPT = A.loading_GRF_WPT(GRFs)

    feature_set_names = [
        "COP_handcrafted",
        "COPs",
        "COP_WPT",
        "GRF_handcrafted",
        "GRFs",
        "GRF_WPT",
    ]
    DF_feature_all = pd.concat(
        [COP_handcrafted, COPs, COP_WPT, GRF_handcrafted, GRFs, GRF_WPT, labels], axis=1
    )

    subjects, samples = np.unique(DF_feature_all["ID"].values, return_counts=True)

    ss = [a[0] for a in list(zip(subjects, samples)) if a[1] >= A._min_number_of_sample]

    known_imposter_list = ss[32 : 32 + A._known_imposter]
    unknown_imposter_list = ss[-A._unknown_imposter :]

    DF_unknown_imposter = DF_feature_all[
        DF_feature_all["ID"].isin(unknown_imposter_list)
    ]
    DF_known_imposter = DF_feature_all[DF_feature_all["ID"].isin(known_imposter_list)]

    results = list()
    for idx, subject in enumerate(DF_known_imposter["ID"].unique()):
        # if idx not in [0, 1]: #todo: remove this block to run for all subjects.
        #     break
        logger.info(
            f"   Subject number: {idx} out of {len(known_imposter_list)} (subject ID is {subject})"
        )
        X, U = A.binarize_labels(DF_known_imposter, DF_unknown_imposter, subject)

        X_train, X_test = model_selection.train_test_split(
            X, test_size=0.2, random_state=A._random_state, stratify=X["ID"]
        )
        # X_train, X_val  = model_selection.train_test_split(X_train, test_size=0.2, random_state=A._random_state, stratify=X_train['ID'])#todo

        # X_test = pd.concat([X_test, U], axis=0)

        X_train, X_test, U = A.scaler(X_train, X_test, U)

        X_train, X_test, U, num_pc = A.projector(X_train, X_test, U, feature_set_names)

        classifier = svm.SVC(
            kernel=A._SVM_kernel, probability=True, random_state=A._random_state
        )

        best_model_svm = classifier.fit(
            X_train.iloc[:, :-1].values, X_train.iloc[:, -1].values
        )
        y_pred_tr = best_model_svm.predict_proba(X_train.iloc[:, :-1].values)[:, 1]
        FRR_t, FAR_t = A.FXR_calculater(X_train["ID"], y_pred_tr)
        EER_svm, t_idx = A.compute_eer(FRR_t, FAR_t)
        TH_svm = A._THRESHOLDs[t_idx]

        positives = X_train[X_train["ID"] == 1.0]
        negatives = X_train[X_train["ID"] == 0.0]
        (
            similarity_matrix_positives,
            similarity_matrix_negatives,
        ) = A.compute_score_matrix(positives, negatives)
        client_scores, imposter_scores = A.compute_scores(
            similarity_matrix_positives, similarity_matrix_negatives, criteria="min"
        )
        y_pred_tr = np.append(client_scores.data, imposter_scores.data)

        FRR_t, FAR_t = A.FXR_calculater(X_train["ID"], y_pred_tr)
        EER_tm, t_idx = A.compute_eer(FRR_t, FAR_t)
        TH_tm = A._THRESHOLDs[t_idx]

        breakpoint()
        y_pred = best_model_svm.predict_proba(X_test.iloc[:, :-1].values)[:, 1]

        y_pred1 = y_pred.copy()
        y_pred[y_pred >= TH_svm] = 1
        y_pred[y_pred < TH_svm] = 0

        y_pred_U = best_model_svm.predict_proba(U.iloc[:, :-1].values)[:, 1]
        y_pred_U[y_pred_U >= TH_svm] = 1.0
        y_pred_U[y_pred_U < TH_svm] = 0.0

        AUS, FAU = [], []

        if self._classifier_name == "TM":
            positives = x_train[x_train["ID"] == 1.0]
            (
                similarity_matrix_positives,
                similarity_matrix_negatives,
            ) = self.compute_score_matrix(positives, x_test_U)
            client_scores, imposter_scores = self.compute_scores(
                similarity_matrix_positives, similarity_matrix_negatives, criteria="min"
            )
            y_pred_U = imposter_scores.data

        else:

            y_pred_U1 = y_pred_U.copy()

            y_pred_U[y_pred_U >= TH] = 1.0
            y_pred_U[y_pred_U < TH] = 0.0
            AUS_All = accuracy_score(x_test_U["ID"].values, y_pred_U) * 100
            FAU_All = np.where(y_pred_U == 1)[0].shape[0]

        if self._classifier_name == "TM":
            positives = x_train[x_train["ID"] == 1.0]
            (
                similarity_matrix_positives,
                similarity_matrix_negatives,
            ) = A.compute_score_matrix(positives, x_test)
            client_scores, imposter_scores = A.compute_scores(
                similarity_matrix_positives, similarity_matrix_negatives, criteria="min"
            )
            y_pred = imposter_scores.data

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

        # #todo
        result, CM_bd, CM_ud = A.ML_classifier(X_train, X_test, U, subject)

        return (
            result
            + CM_ud.reshape(1, -1).tolist()[0]
            + CM_bd.reshape(1, -1).tolist()[0]
            + [num_pc]
        )

        ncpus = int(
            os.environ.get("SLURM_CPUS_PER_TASK", default=multiprocessing.cpu_count())
        )
        pool = multiprocessing.Pool(processes=ncpus)
        # breakpoinqt()
        for fold, (train_index, test_index) in enumerate(
            CV.split(X.iloc[:, :-1], X.iloc[:, -1])
        ):
            # breakpoint()
            res = pool.apply_async(
                A.fold_calculating,
                args=(
                    feature_set_names,
                    subject,
                    X,
                    U,
                    train_index,
                    test_index,
                    fold,
                ),
                callback=cv_results.append,
            )
            # print(res.get())  # this will raise an exception if it happens within func

            # cv_results.append(A.fold_calculating(feature_set_names, subject, X, U, train_index, test_index, fold)) #todo: comment this line to run all folds
            # break #todo: comment this line to run all folds

        pool.close()
        pool.join()

        result = A.compacting_results(cv_results, subject)
        results.append(result)
        # breakpoint()
    return pd.DataFrame(results, columns=A._col)

    result = get_results(A, feature_set_names, DF_feature_all)
    result["pipeline"] = "pipeline 1: COP and GRf"
    A.collect_results(result, nam)


if __name__ == "__main__":
    logger.info("Starting !!!")
    tic1 = timeit.default_timer()

    # main()
    # Participant_Count()

    # lightweight()
    # FT()
    # test_all_pipelines()
    # new_image()

    # dd = second_retrain()
    # path = os.path.join( os.getcwd(), "results", "results", "second_train", "lightweight_CNN", "best.xlsx")
    # dd.to_excel(path)
    # breakpoint()

    # aa = Toon_p100()
    # path = os.path.join( os.getcwd(), "results", "e2e", "TM.xlsx")
    # aa.to_excel(path)
    # breakpoint()

    train_e2e_CNN()
    # breakpoint()

    toc1 = timeit.default_timer()
    logger.info("Done ({:2.2f} process time)!!!\n\n\n".format(toc1 - tic1))
