
import logging
import multiprocessing
import os
import pprint
import sys
import timeit
from pathlib import Path as Pathlb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn import preprocessing

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_curve)

pd.options.mode.chained_assignment = None 

import tensorflow as tf


from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedKFold,  train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn

from MLPackage import config as cfg
from MLPackage import Features as feat

# global variables

# Pipeline = cfg.configs["Pipeline"]
# CNN = cfg.configs["CNN"]
# Template_Matching = cfg.configs["Template_Matching"]
# SVM = cfg.configs["SVM"]
# KNN = cfg.configs["KNN"]
# paths = cfg.configs["paths"]

# num_pc = 0
columnsname = ["testID", "subject ID", "direction", "clasifier", "PCA", "num_pc", "classifier_parameters", "normilizing", "feature_type", "test_ratio", "mean(EER)", "t_idx", "mean(acc)", "mean(f1)", "# positive samples training", "# positive samples test", "# negative samples test", "len(FAR)", "len(FRR)"] + ["FAR_" + str(i) for i in range(100)] + ["FRR_" + str(i) for i in range(100)] 
time = int(timeit.default_timer() * 1_000_000)



SLURM_JOBID = str(os.environ.get('SLURM_JOBID',default=os.getpid()))




log_path = os.path.join(cfg.configs["paths"]["project_dir"], 'logs')
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
    # file_handler = logging.FileHandler( os.path.join(log_path, loggerName + '_loger.log'), mode = 'w')
    file_handler = logging.FileHandler( os.path.join(log_path, f"{SLURM_JOBID}_" + loggerName + '_loger.log'), mode = 'w')

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    stream_handler.setFormatter(formatter_colored)


    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
logger = create_logger(logging.DEBUG)



def knn_classifier(**kwargs):
    global time
    num_pc = kwargs["num_pc"]
    configs = kwargs["configs"]

    pos_samples = kwargs["pos_train"].shape
    temp = kwargs["neg_train"].sample(n = pos_samples[0])


    kwargs["pos_train"]["left(0)/right(1)"] = kwargs["pos_train"]["left(0)/right(1)"].map(lambda x: 1)
    temp["left(0)/right(1)"] = temp["left(0)/right(1)"].map(lambda x: 0)
    x_train = kwargs["pos_train"].append(temp)


    kwargs["pos_train"]["left(0)/right(1)"] = kwargs["pos_train"]["left(0)/right(1)"].map(lambda x: 1)
    kwargs["neg_train"]["left(0)/right(1)"] = kwargs["neg_train"]["left(0)/right(1)"].map(lambda x: 0)
    x_train = kwargs["pos_train"].append(kwargs["neg_train"])

    # cv = StratifiedKFold(n_splits=5, shuffle=True)


    clf = knn(n_neighbors=configs["KNN"]["n_neighbors"], metric=configs["KNN"]["metric"], weights=configs["KNN"]["weights"])


    # space = KNN

    # # define search
    # search = GridSearchCV(
    #     clf,
    #     space,
    #     scoring="accuracy",
    #     n_jobs=-1,
    #     cv=cv,
    #     refit=True,
    # )


    # execute search
    result = clf.fit(x_train.iloc[:, :-2].values, x_train.iloc[:, -1], )
    best_model = result#.best_estimator_

    y_pred = best_model.predict_proba(x_train.iloc[:, :-2].values)[:, 1]
    

    FAR, tpr, _ = roc_curve(x_train.iloc[:, -1], y_pred, pos_label=1)
    FRR = 1 - tpr

    EER, t_idx = compute_eer(FAR, FRR)
    
    


    acc = list()
    f1 = list()
    for _ in range(configs["KNN"]["random_runs"]):
        pos_samples = kwargs["pos_test"].shape
        temp = kwargs["neg_test"].sample(n = pos_samples[0])

        DF_temp = pd.concat([kwargs["pos_test"], temp])
        DF_temp["subject ID"] = DF_temp["subject ID"].map(lambda x: 1 if x == kwargs["sub"] else 0)

        y_pred = best_model.predict(DF_temp.iloc[:, :-2].values)

        acc.append( accuracy_score(DF_temp.iloc[:,-2].values, y_pred)*100 )
        f1.append(  f1_score(DF_temp.iloc[:,-2].values, y_pred)*100 )
      
    
    results = list()

    results.append([time, kwargs["sub"], kwargs["dir"], "KNN", configs["Pipeline"]["persentage"], num_pc, configs["KNN"], configs["Pipeline"]["normilizing"], kwargs["feature_type"], configs["Pipeline"]["test_ratio"]])

    results.append([EER, t_idx, np.mean(acc), np.mean(f1), kwargs["pos_train"].shape[0], kwargs["pos_test"].shape[0], kwargs["neg_test"].shape[0], len(FAR), len(FRR)])

    results.append(FAR)
    results.append(FRR)

    results = [val for sublist in results for val in sublist]
    return results



def svm_classifier(**kwargs):
    global time
    num_pc = kwargs["num_pc"]
    configs = kwargs["configs"]

    pos_samples = kwargs["pos_train"].shape
    temp = kwargs["neg_train"].sample(n = 100)#pos_samples[0])
    

    kwargs["pos_train"]["left(0)/right(1)"] = kwargs["pos_train"]["left(0)/right(1)"].map(lambda x: 1)
    temp["left(0)/right(1)"] = temp["left(0)/right(1)"].map(lambda x: 0)
    x_train = kwargs["pos_train"].append(temp)


    clf = svm.SVC(kernel=configs["SVM"]["kernel"] , probability=True)
    clf.fit(x_train.iloc[:, :-2], x_train.iloc[:, -1])


    y_pred = clf.predict_proba(x_train.iloc[:, :-2])[:, 1]
    

    # for _ in range(SVM["random_runs"]):
    FAR, tpr, threshold = roc_curve(x_train.iloc[:, -1], y_pred, pos_label=1)
    FRR = 1 - tpr

    EER, t_idx = compute_eer(FAR, FRR)
    
    
    

    acc = list()
    f1 = list()
    for _ in range(configs["SVM"]["random_runs"]):
        pos_samples = kwargs["pos_test"].shape
        temp = kwargs["neg_test"].sample(n = pos_samples[0])

        DF_temp = pd.concat([kwargs["pos_test"], temp])
        DF_temp["subject ID"] = DF_temp["subject ID"].map(lambda x: 1 if x == kwargs["sub"] else 0)

        y_pred = clf.predict(DF_temp.iloc[:, :-2])

        acc.append( accuracy_score(DF_temp.iloc[:,-2].values, y_pred)*100 )
        f1.append(  f1_score(DF_temp.iloc[:,-2].values, y_pred)*100 )
      
    
    results = list()

    results.append([time, kwargs["sub"], kwargs["dir"], "SVM", configs["Pipeline"]["persentage"], num_pc, configs["SVM"], configs["Pipeline"]["normilizing"], kwargs["feature_type"], configs["Pipeline"]["test_ratio"]])

    results.append([EER, t_idx, np.mean(acc), np.mean(f1), kwargs["pos_train"].shape[0], kwargs["pos_test"].shape[0], kwargs["neg_test"].shape[0], len(FAR), len(FRR)])

    results.append(FAR)
    results.append(FRR)

    results = [val for sublist in results for val in sublist]
    return results



def Template_Matching_classifier(**kwargs):
    global time
    configs = kwargs["configs"]
    num_pc = kwargs["num_pc"]
    FAR = list()
    FRR = list()
    EER = list()
    TH  = list()
    for _ in range(configs["Template_Matching"]["random_runs"]):
        pos_samples = kwargs["pos_train"].shape
        temp = kwargs["neg_train"].sample(n = pos_samples[0])


        distModel1, distModel2 = compute_score_matrix(kwargs["pos_train"].iloc[:, :-2].values,
                                                    temp.iloc[:, :-2].values,
                                                    mode = configs["Template_Matching"]["mode"], score = configs["Template_Matching"]["score"])

        Model_client, Model_imposter = model(distModel1,
                                                distModel2, 
                                                criteria=configs["Template_Matching"]["criteria"], 
                                                score=configs["Template_Matching"]["score"] )



        FAR_temp, FRR_temp = calculating_fxr(Model_client, Model_imposter, distModel1, distModel2, configs["Pipeline"]["THRESHOLDs"], configs["Template_Matching"]["score"])
        EER_temp = compute_eer(FAR_temp, FRR_temp)

        FAR.append(FAR_temp)
        FRR.append(FRR_temp)
        EER.append(EER_temp[0])
        TH.append(EER_temp[1])

    
    
    acc = list()
    f1 = list()
    t_idx = int(np.ceil(np.mean(TH)))

    for _ in range(configs["Template_Matching"]["random_runs"]):
        pos_samples = kwargs["pos_test"].shape
        temp = kwargs["neg_test"].sample(n = pos_samples[0])

        DF_temp = pd.concat([kwargs["pos_test"], temp])
        DF_temp["subject ID"] = DF_temp["subject ID"].map(lambda x: 1 if x == kwargs["sub"] else 0)


        distModel1 , distModel2 = compute_score_matrix(kwargs["pos_train"].iloc[:, :-2].values, DF_temp.iloc[:, :-2].values, mode=configs["Template_Matching"]["mode"], score=configs["Template_Matching"]["score"])
        Model_client, Model_test = model(distModel1, distModel2, criteria=configs["Template_Matching"]["criteria"], score=configs["Template_Matching"]["score"])


        y_pred = np.zeros((Model_test.shape))


        y_pred[Model_test > configs["Pipeline"]["THRESHOLDs"][t_idx]] = 1


        acc.append( accuracy_score(DF_temp.iloc[:,-2].values, y_pred)*100 )
        f1.append(  f1_score(DF_temp.iloc[:,-2].values, y_pred)*100 )
    
    results = list()

    results.append([time, kwargs["sub"], kwargs["dir"], "Template_Matching", configs["Pipeline"]["persentage"], num_pc, configs["Template_Matching"], configs["Pipeline"]["normilizing"], kwargs["feature_type"], configs["Pipeline"]["test_ratio"]])

    results.append([np.mean(EER), t_idx, np.mean(acc), np.mean(f1), kwargs["pos_train"].shape[0], kwargs["pos_test"].shape[0], kwargs["neg_test"].shape[0], len(FAR[0]), len(FRR[0])])
    results.append(list(np.mean(FAR, axis=0)))
    results.append(list(np.mean(FRR, axis=0)))

    results = [val for sublist in results for val in sublist]
    return results

    

def pipeline(configs):

    # set_sonfig(configs)
    classifier = configs["Pipeline"]["classifier"]

    if configs["Pipeline"]["category"]=="deep":
        if configs["Pipeline"]["type"]=="PT":
            feature_path = os.path.join(configs["paths"]["casia_deep_feature"], configs["CNN"]["base_model"].split(".")[0]+'_'+configs["CNN"]["image_feature"]+'_features.xlsx')
            DF_features_all = pd.read_excel(feature_path, index_col = 0)
        elif configs["Pipeline"]["type"]=="FS":
            feature_path = os.path.join(configs["paths"]["casia_deep_feature"], 'FS_'+configs["CNN"]["image_feature"]+'_features.xlsx')
            DF_features_all = pd.read_excel(feature_path, index_col = 0)
        elif configs["Pipeline"]["type"]=="FT":
            feature_path = os.path.join(configs["paths"]["casia_deep_feature"], 'FT_resnet50_'+configs["CNN"]["image_feature"]+'_features.xlsx')
            DF_features_all = pd.read_excel(feature_path, index_col = 0)
    elif configs["Pipeline"]["category"]=="hand_crafted":
        feature_path = configs["paths"]["casia_all_feature.xlsx"]
        DF_features_all = pd.read_excel(feature_path, index_col = 0)
    elif configs["Pipeline"]["category"]=="image":
        feature_path = configs["paths"]["casia_image_feature.npy"]
        image_features = np.load(feature_path)
        image_feature_name_dict = dict(zip(cfg.image_feature_name, range(len(cfg.image_feature_name))))
        image_features = image_features[..., image_feature_name_dict[configs["CNN"]["image_feature"]]]
        image_features = image_features.reshape(2851, 2400 ,1).squeeze()

        
        meta = np.load(configs["paths"]["casia_dataset-meta.npy"])

        DF_features_all = pd.DataFrame(np.concatenate((image_features, meta[:,0:2]), axis=1 ), columns=["pixel_"+str(i) for i in range(image_features.shape[1])]+cfg.label)



 
     



    subjects = DF_features_all["subject ID"].unique()
    
    persentage = configs["Pipeline"]["persentage"]
    normilizing = configs["Pipeline"]["normilizing"]
    test_ratio = configs["Pipeline"]["test_ratio"]

    if configs["Pipeline"]["category"]=="deep":
        feature_type = configs["CNN"]["base_model"].split(".")[0]
    elif configs["Pipeline"]["category"]=="hand_crafted":
        feature_type = configs["Pipeline"]["feature_type"]
    elif configs["Pipeline"]["category"]=="image":
        feature_type = "image"
    
    DF_features = extracting_features(DF_features_all, feature_type, configs)
    tic=timeit.default_timer()


    logger.info("Start [pipeline]:   +++   {}".format(classifier))


    results = list()
    if configs["Pipeline"]["Debug"]==True: subjects = [4, 5,]

    for subject in subjects:
        if (subject % 86) == 0: continue
        
        
        if (subject % 10) == 0 and configs["Pipeline"]["verbose"] is True:
            logger.info("--------------- Subject Number: {}".format(subject))
        

        for idx, direction in enumerate(["left_0", "right_1"]):
            logger.info(f"-->> Model {subject},\t {direction} \t\t PID: {os.getpid()}")    


            DF_side = DF_features[DF_features["left(0)/right(1)"] == idx]
        
            DF_positive_samples = DF_side[DF_side["subject ID"] == subject]
            DF_negative_samples = DF_side[DF_side["subject ID"] != subject]

                
            DF_positive_samples_test = DF_positive_samples.sample(frac = test_ratio, 
                                                                    replace = False, 
                                                                    random_state = 2)
            DF_positive_samples_train = DF_positive_samples.drop(DF_positive_samples_test.index)

            DF_negative_samples_test = DF_negative_samples.sample(frac = test_ratio,
                                                                    replace = False, 
                                                                    random_state = 2)
            DF_negative_samples_train = DF_negative_samples.drop(DF_negative_samples_test.index)
            
            
            df_train = pd.concat([DF_positive_samples_train, DF_negative_samples_train])
            df_test = pd.concat([DF_positive_samples_test, DF_negative_samples_test])

            Scaled_train, Scaled_test = scaler(normilizing, df_train, df_test)
            
            # logger.info("direction: {}".format(direction))
            # logger.info("subject: {}".format(subject))
            # breakpoint()

            (DF_positive_samples_test, 
            DF_positive_samples_train, 
            DF_negative_samples_test, 
            DF_negative_samples_train,
            num_pc) = projector(persentage, 
                                feature_type, 
                                subject, 
                                df_train, 
                                df_test, 
                                Scaled_train, 
                                Scaled_test,
                                configs)

            # logger.debug("DF_positive_samples_train.shape {}".format(DF_positive_samples_train.shape))    
            # logger.debug("DF_negative_samples_train.shape {}".format(DF_negative_samples_train.shape))   
            # logger.debug("DF_positive_samples_test.shape {}".format(DF_positive_samples_test.shape))    
            # logger.debug("DF_negative_samples_test.shape {}".format(DF_negative_samples_test.shape))    
                


            # DF_positive_samples_train = template_selection(DF_positive_samples_train, 
            #                                                method=Pipeline["template_selection_method"], 
            #                                                k_cluster=Pipeline["template_selection_k_cluster"], 
            #                                                verbose=Pipeline["verbose"])
            # DF_negative_samples_train = template_selection(DF_negative_samples_train, 
            #                                                method="MDIST", 
            #                                                k_cluster=200, 
            #                                                verbose=Pipeline["verbose"])
            if configs["Pipeline"]["category"]=="deep" or configs["Pipeline"]["category"]=="image" :
                temp1="_".join((feature_type, configs["CNN"]["image_feature"], configs["Pipeline"]["type"]))
            result = eval(classifier)(pos_train=DF_positive_samples_train, 
                                neg_train=DF_negative_samples_train, 
                                pos_test=DF_positive_samples_test, 
                                neg_test=DF_negative_samples_test, 
                                sub=subject, 
                                dir=direction,
                                num_pc=num_pc,
                                feature_type=temp1, 
                                configs=configs)

            result = np.pad(result, (0, len(columnsname) - len(result)), 'constant')
                    
            results.append(result)
            # print(results)
        

    toc=timeit.default_timer()
    logger.info("End   [pipeline]:     ---    {}, \t\t Process time: {:.2f}  seconds".format(feature_type, toc - tic)) 

    return pd.DataFrame(results, columns=columnsname)



def extracting_features(DF_features_all, feature_type, configs):
    if configs["Pipeline"]["category"]=="deep" or configs["Pipeline"]["category"]=="image":
        return DF_features_all
    elif feature_type == "all": #"all", "GRF_HC", "COA_HC", "GRF", "COA", "wt_GRF", "wt_COA"
        DF_features = DF_features_all.drop(columns=cfg.wt_GRF).copy()
    elif feature_type == "GRF_HC":
        DF_features = DF_features_all.loc[:, cfg.GRF_HC + cfg.label]
    elif feature_type == "COA_HC":
        DF_features = DF_features_all.loc[:, cfg.COA_HC + cfg.label]
    elif feature_type == "GRF":
        DF_features = DF_features_all.loc[:, cfg.GRF + cfg.label]
    elif feature_type == "COA":
        DF_features = DF_features_all.loc[:, cfg.COA_RD + cfg.COA_AP + cfg.COA_ML + cfg.label]
    elif feature_type == "wt_GRF":
        DF_features = DF_features_all.loc[:, cfg.wt_GRF + cfg.label]
    elif feature_type == "wt_COA":
        DF_features = DF_features_all.loc[:, cfg.wt_COA_RD + cfg.wt_COA_AP + cfg.wt_COA_ML + cfg.label]
    else:
        raise("Could not find the feature_type")
    return DF_features



def projector(persentage, feature_type, subject, df_train, df_test, Scaled_train, Scaled_test, configs):
    if persentage == 1.0:
        num_pc = Scaled_train.shape[1]
                

        columnsName = ["PC"+str(i) for i in list(range(1, num_pc+1))] + ["subject ID", "left(0)/right(1)"]
        DF_features_PCA_train = pd.DataFrame(np.concatenate((Scaled_train[:,:num_pc],df_train.iloc[:, -2:].values), axis = 1), columns = columnsName)
        DF_features_PCA_test = pd.DataFrame(np.concatenate((Scaled_test[:,:num_pc],df_test.iloc[:, -2:].values), axis = 1), columns = columnsName)

        DF_positive_samples_train = DF_features_PCA_train[DF_features_PCA_train["subject ID"] == subject]
        DF_negative_samples_train = DF_features_PCA_train[DF_features_PCA_train["subject ID"] != subject]
                
                
        DF_positive_samples_test = DF_features_PCA_test[DF_features_PCA_test["subject ID"] == subject]   
        DF_negative_samples_test = DF_features_PCA_test[DF_features_PCA_test["subject ID"] != subject]

    elif persentage != 1.0 and (feature_type in ["image", "GRF_HC", "COA_HC", "GRF", "wt_GRF", configs["CNN"]["base_model"].split(".")[0]]):
        principal = PCA(svd_solver="full")
        PCA_out_train = principal.fit_transform(Scaled_train)
        PCA_out_test = principal.transform(Scaled_test)

        variance_ratio = np.cumsum(principal.explained_variance_ratio_)
        high_var_PC = np.zeros(variance_ratio.shape)
        high_var_PC[variance_ratio <= persentage] = 1

        loadings = principal.components_
        num_pc = int(np.sum(high_var_PC))




        columnsName = ["PC"+str(i) for i in list(range(1, num_pc+1))] + ["subject ID", "left(0)/right(1)"]
        DF_features_PCA_train = (pd.DataFrame(np.concatenate((PCA_out_train[:,:num_pc],df_train.iloc[:, -2:].values), axis = 1), columns = columnsName))
        DF_features_PCA_test = (pd.DataFrame(np.concatenate((PCA_out_test[:,:num_pc],df_test.iloc[:, -2:].values), axis = 1), columns = columnsName))

        DF_positive_samples_train = DF_features_PCA_train[DF_features_PCA_train["subject ID"] == subject]
        DF_negative_samples_train = DF_features_PCA_train[DF_features_PCA_train["subject ID"] != subject]
                
                
        DF_positive_samples_test = DF_features_PCA_test[DF_features_PCA_test["subject ID"] == subject]   
        DF_negative_samples_test = DF_features_PCA_test[DF_features_PCA_test["subject ID"] != subject]
            
    elif persentage != 1.0 and (feature_type in ["all", "COA", "wt_COA"]):
        tempa = []
        tempb = []
        if feature_type in "all":
            tempx = ["COA_RD", "COA_AP", "COA_ML", "COA_HC", "GRF", "GRF_HC", "wt_COA_RD", "wt_COA_AP", "wt_COA_ML"] # wt_GRF
        elif feature_type in "COA":
            tempx = ["COA_RD", "COA_AP", "COA_ML"]
        elif feature_type in "wt_COA":
            tempx = ["wt_COA_RD", "wt_COA_AP", "wt_COA_ML"]



        for i in range(len(tempx)):
            principal = PCA(svd_solver="full")
            # logger.info("feature type and shape: {} - {}".format(tempx[i], Scaled_train.loc[:, eval("cfg." + tempx[i])].shape))
            
            
            PCA_out_train = principal.fit_transform(Scaled_train.loc[:, eval("cfg." + tempx[i])])
            PCA_out_test = principal.transform(Scaled_test.loc[:, eval("cfg." + tempx[i])])

                    

            variance_ratio = np.cumsum(principal.explained_variance_ratio_)
            high_var_PC = np.zeros(variance_ratio.shape)
            high_var_PC[variance_ratio <= persentage] = 1

            loadings = principal.components_
            num_pc = int(np.sum(high_var_PC))



            tempa.append(PCA_out_train[:,:num_pc])
            tempb.append(PCA_out_test[:,:num_pc])

            del principal

                   
        for i in range(len(tempx)-1):
            tempa[len(tempx)-1] = np.concatenate((tempa[len(tempx)-1],tempa[i]), axis=1)
            tempb[len(tempx)-1] = np.concatenate((tempb[len(tempx)-1],tempb[i]), axis=1)

        num_pc = tempa[len(tempx)-1].shape[1]

        columnsName = ["PC_" + str(i) for i in list(range(1, num_pc+1))] + ["subject ID", "left(0)/right(1)"]

        DF_features_PCA_train = pd.DataFrame(np.concatenate((tempa[len(tempx)-1],df_train.iloc[:, -2:].values), axis = 1), columns = columnsName)
        DF_features_PCA_test = pd.DataFrame(np.concatenate((tempb[len(tempx)-1],df_test.iloc[:, -2:].values), axis = 1), columns = columnsName)

        DF_positive_samples_train = DF_features_PCA_train[DF_features_PCA_train["subject ID"] == subject]
        DF_negative_samples_train = DF_features_PCA_train[DF_features_PCA_train["subject ID"] != subject]
                
                
        DF_positive_samples_test = DF_features_PCA_test[DF_features_PCA_test["subject ID"] == subject]   
        DF_negative_samples_test = DF_features_PCA_test[DF_features_PCA_test["subject ID"] != subject]

    return DF_positive_samples_test,DF_positive_samples_train,DF_negative_samples_test,DF_negative_samples_train, num_pc



def scaler(normilizing, df_train, df_test):
    if normilizing == "minmax":
        scaling = preprocessing.MinMaxScaler()
        Scaled_train = scaling.fit_transform(df_train.iloc[:, :-2])
        Scaled_test = scaling.transform(df_test.iloc[:, :-2])

    elif normilizing == "z-score":
        scaling = preprocessing.StandardScaler()
        Scaled_train = scaling.fit_transform(df_train.iloc[:, :-2])
        Scaled_test = scaling.transform(df_test.iloc[:, :-2])

    Scaled_train = pd.DataFrame(Scaled_train, columns=df_train.columns[:-2])
    Scaled_test = pd.DataFrame(Scaled_test, columns=df_test.columns[:-2])
    return Scaled_train, Scaled_test



def compute_score_matrix(positive_samples, negative_samples, mode="dist", score = None):
    """ Returns score matrix of trmplate matching"""

    positive_model = np.zeros((positive_samples.shape[0], positive_samples.shape[0]))
    negative_model = np.zeros((positive_samples.shape[0], negative_samples.shape[0]))

    if mode == "dist":

        for i in range(positive_samples.shape[0]):
            for j in range(positive_samples.shape[0]):
                positive_model[i, j] = distance.euclidean(
                    positive_samples[i, :], positive_samples[j, :]
                )
            for j in range(negative_samples.shape[0]):
                negative_model[i, j] = distance.euclidean(
                    positive_samples[i, :], negative_samples[j, :]
                )
        if score != None:
            return compute_similarity(positive_model, score), compute_similarity(negative_model, score)
        elif score == None:
            return positive_model, negative_model

    elif mode == "corr":

        for i in range(positive_samples.shape[0]):
            for j in range(positive_samples.shape[0]):
                positive_model[i, j] = abs(np.corrcoef(
                    positive_samples[i, :], positive_samples[j, :]
                )[0,1])

            for j in range(negative_samples.shape[0]):
                negative_model[i, j] = abs(np.corrcoef(
                    positive_samples[i, :], negative_samples[j, :]
                )[0,1])
        return positive_model, negative_model



def model(distModel1, distModel2, criteria = "average", score = None ):
    if score is None:
        if criteria == "average":
            # model_client = (np.sum(distModel1, axis = 0))/(distModel1.shape[1]-1)
            model_client = np.mean(np.ma.masked_where(distModel1==0,distModel1), axis = 0)
            model_client = np.expand_dims(model_client,-1)
            
            model_imposter = (np.sum(distModel2, axis = 0))/(distModel1.shape[1])
            model_imposter = np.expand_dims(model_imposter, -1)
                
        elif criteria == "min":

            model_client = np.min(np.ma.masked_where(distModel1==0,distModel1), axis = 0)
            model_client = np.expand_dims(model_client,-1)
            
            model_imposter = np.min(np.ma.masked_where(distModel2==0,distModel2), axis = 0)
            model_imposter = np.expand_dims(model_imposter, -1)
                    
        elif criteria == "median":
            model_client = np.median(distModel1, axis = 0)
            model_client = np.expand_dims(model_client,-1)
            

            model_imposter = np.median(distModel2, axis = 0)
            model_imposter = np.expand_dims(model_imposter, -1)

    if score is not None:
        if criteria == "average":
            model_client = np.mean(np.ma.masked_where(distModel1==1,distModel1), axis = 0)
            model_client = np.expand_dims(model_client,-1)
            
            model_imposter = (np.sum(distModel2, axis = 0))/(distModel1.shape[1])
            model_imposter = np.expand_dims(model_imposter, -1)
                
        elif criteria == "min":

            model_client = np.max(np.ma.masked_where(distModel1==1,distModel1), axis = 0)
            model_client = np.expand_dims(model_client,-1)
            
            model_imposter = np.max(np.ma.masked_where(distModel2==1,distModel2), axis = 0)
            model_imposter = np.expand_dims(model_imposter, -1)
                    
        elif criteria == "median":
            model_client = np.median(distModel1, axis = 0)
            model_client = np.expand_dims(model_client,-1)            

            model_imposter = np.median(distModel2, axis = 0)
            model_imposter = np.expand_dims(model_imposter, -1)
    

    return model_client, model_imposter



def calculating_fxr(Model_client, Model_imposter, distModel1, distModel2, THRESHOLDs, score):
    """ Returns FAR and FRR our model """

    FRR_temp = list()
    FAR_temp = list()

    if score is not None:
        for tx in THRESHOLDs:
            E1 = np.zeros((Model_client.shape))
            E1[Model_client < tx] = 1
            FRR_temp.append(np.sum(E1)/(distModel1.shape[1]))


            E2 = np.zeros((Model_imposter.shape))
            E2[Model_imposter > tx] = 1
            FAR_temp.append(np.sum(E2)/distModel2.shape[1])

    elif score is None:
        for tx in THRESHOLDs:
            E1 = np.zeros((Model_client.shape))
            E1[Model_client > tx] = 1
            FRR_temp.append(np.sum(E1)/(distModel1.shape[1]))


            E2 = np.zeros((Model_imposter.shape))
            E2[Model_imposter < tx] = 1
            FAR_temp.append(np.sum(E2)/distModel2.shape[1])
    return FAR_temp, FRR_temp
    


def compute_eer(fpr, fnr):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    abs_diffs = np.abs(np.subtract(fpr, fnr)) 
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))

    return eer, min_index



def compute_similarity(distance, mode = "A"):
    """change distance score to similarity score."""

    distance = np.array(distance)
    if mode == "A":
        return np.power(distance+1, -1) 
    elif mode =="B":
        return 1/np.exp(distance)



def template_selection(DF_positive_samples_train,  method, k_cluster, verbose=1):
    if method == "DEND":
        kmeans = KMeans(n_clusters = k_cluster)
        kmeans.fit(DF_positive_samples_train.iloc[:, :-2].values)
        clusters = np.unique(kmeans.labels_)

        for i, r in DF_positive_samples_train.reset_index(drop=True).iterrows():
            DF_positive_samples_train.loc[i,"dist"] = distance.euclidean(kmeans.cluster_centers_[kmeans.labels_[i]], r[:-2].values)
            DF_positive_samples_train.loc[i,"label"] = kmeans.labels_[i]
        DF_positive_samples_train_clustered = pd.DataFrame(np.empty((k_cluster,DF_positive_samples_train.shape[1]-2)))
        for cluster in clusters:
            mean_cluster = DF_positive_samples_train[DF_positive_samples_train["label"] == cluster].sort_values(by=['dist'])
            DF_positive_samples_train_clustered.iloc[cluster, :] = mean_cluster.iloc[0,:-2]
        if verbose: 
            logger.info(f"Clustered data size: { DF_positive_samples_train_clustered.shape}")
        DF_positive_samples_train_clustered.columns = DF_positive_samples_train.columns.values[:-2]

    elif method == "MDIST":
        A = distance.squareform(distance.pdist(DF_positive_samples_train.iloc[:, :-2].values)).mean(axis=1)
        i = np.argsort(A)[:k_cluster]
        DF_positive_samples_train_clustered = DF_positive_samples_train.iloc[i, :]
        DF_positive_samples_train_clustered.columns = DF_positive_samples_train.columns.values

    elif method == "None":
        DF_positive_samples_train_clustered = DF_positive_samples_train
        DF_positive_samples_train_clustered.columns = DF_positive_samples_train.columns.values

    return DF_positive_samples_train_clustered



def tile(samples):
    '''return a tile of different images'''
    sample_size = samples.shape
    tile_image = list()
    batch = sample_size[0]

    for i in range(batch):
        sample = np.array(samples[i,...])
        sample = sample.transpose((2, 0, 1))

        total_image = sample[0,:,:]
        total_image1 = sample[5,:,:]

        for i in range(1,5):
            total_image = np.concatenate((total_image, sample[i,:,:]), axis=1)
            total_image1 = np.concatenate((total_image1, sample[i+5,:,:]), axis=1)

        total_image = np.concatenate((total_image, total_image1), axis=0)
        total_image = total_image[:,:, np.newaxis]
        total_image = np.concatenate((total_image, total_image, total_image), axis=2)
        tile_image.append(total_image)

    return np.array(tile_image)



def fine_tuning(configs):
    logger.info("func: fine_tuning")


    if configs['CNN']["image_feature"]=="tile":
        configs['CNN']["image_size"] = (120, 80, 3)


    # configs['CNN']["image_size"] = (32, 32, 3)
    # configs['CNN']["class_numbers"] = 10




    # # ##################################################################
    # #                phase 1: Reading image
    # # ##################################################################
    # logger.info("Reading stepscan dataset....")
    # with h5py.File(cfg.configs["paths"]["stepscan_dataset.h5"], "r") as hdf:
    #     barefoots = hdf.get("/barefoot/data")[:]
    #     metadata = hdf.get("/barefoot/metadata")[:]

    # data = barefoots.transpose(0,2,3,1)

    # np.save(cfg.configs["paths"]["stepscan_data.npy"], data)
    # np.save(cfg.configs["paths"]["stepscan_meta.npy"], metadata)





    # ##################################################################
    #                phase 2: extracting image features
    # ##################################################################
    # metadata = np.load(cfg.configs["paths"]["stepscan_meta.npy"])
    # data = np.load(cfg.configs["paths"]["stepscan_data.npy"])

    # logger.info(f"barefoots.shape: {data.shape}")
    # logger.info(f"metadata.shape: {metadata.shape}")


    # # plt.imshow(data[1,:,:,:].sum(axis=2))
    # # plt.show()




    # ## Extracting Image Features
    # features = list()
    # labels = list()

    # for label, sample in zip(metadata, data):
    #     try:
    #         B = sample.sum(axis=1).sum(axis=0)
    #         A = np.trim_zeros(B)

    #         aa = np.where(B == A[0])
    #         bb = np.where(B == A[-1])

    #         if aa[0][0]<bb[0][0]:
    #             features.append(feat.prefeatures(sample[10:70, 10:50, aa[0][0]:bb[0][0]]))
    #             labels.append(label)
    #         else:
    #             # print(aa[0][0],bb[0][0])
    #             k=sample
    #             l=label
        
    #     except Exception as e:
    #         logger.error(e)
    #         continue
        

    # logger.info(f"len prefeatures: {len(features)}")
    # logger.info(f"prefeatures.shape: {features[0].shape}")
    # logger.info(f"labels.shape: {labels[0].shape}")

    # np.save(cfg.configs["paths"]["stepscan_image_feature.npy"], features)
    # np.save(cfg.configs["paths"]["stepscan_image_label.npy"], labels)






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


    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # # Normalize pixel values to be between 0 and 1
    # train_images, test_images = train_images / 255.0, test_images / 255.0

    # images = train_images 
    # labels = train_labels
    # # labels = tf.keras.utils.to_categorical(train_labels,10)

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    logger.info(f"images: {images.shape}")
    logger.info(f"labels: {labels.shape}")


    # # ##################################################################
    # #                phase 5: Making tf.dataset object
    # # ##################################################################

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.15, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)




    AUTOTUNE = tf.data.AUTOTUNE


    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000)
    train_ds = train_ds.batch(configs['CNN']["batch_size"])
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(1000)
    val_ds = val_ds.batch(configs['CNN']["batch_size"])
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(configs['CNN']["batch_size"])
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)




    # # ##################################################################
    # #                phase 6: Making Base Model
    # # ##################################################################
    try:
        logger.info(f"Loading { configs['CNN']['base_model'] } model...")
        base_model = eval("tf.keras.applications." + configs["CNN"]["base_model"] + "(weights=cfg.configs['CNN']['weights'], include_top=cfg.configs['CNN']['include_top'])")
        logger.info("Successfully loaded base model and model...")

    except Exception as e: 
        base_model = None
        logger.error("The base model could NOT be loaded correctly!!!")
        logger.error(e)


    base_model.trainable = False

    CNN_name = configs["CNN"]["base_model"].split(".")[0]

    input = tf.keras.layers.Input(shape=cfg.configs["CNN"]["image_size"], dtype = tf.float64, name="original_img")
    x = tf.cast(input, tf.float32)
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    x = tf.keras.layers.RandomRotation(0.2)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    x = eval("tf.keras.applications." + CNN_name + ".preprocess_input(x)")
    x = base_model(x)
    x = tf.keras.layers.GlobalMaxPool2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128,  activation='relu', name="last_dense")(x) # kernel_regularizer=tf.keras.regularizers.l2(0.0001),
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(configs['CNN']['class_numbers'], name="prediction")(x) # activation='softmax',

    ## The CNN Model
    model = tf.keras.models.Model(inputs=input, outputs=output, name=configs['CNN']['base_model'])

    # Freeze the layers 
    for layer in model.layers[-2:]:
        layer.trainable = True


    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name,layer.trainable)

    # model.summary() 
    # tf.keras.utils.plot_model(model, to_file=cfg.configs['CNN']['base_model'] + ".png", show_shapes=True)
    # plt.show()



    # # ##################################################################
    # #                phase 7: training CNN
    # # ##################################################################

    model.compile(
        optimizer=tf.keras.optimizers.Adam(), #learning_rate=0.001
        loss=tf.keras.losses.SparseCategoricalCrossentropy (from_logits=True), 
        metrics=["Accuracy"]
        )

    time = int(timeit.timeit()*1_000_000)
    TensorBoard_logs =  os.path.join( configs["paths"]["TensorBoard_logs"], "_".join(("FT", SLURM_JOBID, CNN_name, configs["CNN"]["image_feature"], str(time)) )  )
    path = configs["CNN"]["saving_path"] + "_".join(( "FT", SLURM_JOBID, CNN_name, configs["CNN"]["image_feature"], "best.h5" ))

    checkpoint = [
            tf.keras.callbacks.ModelCheckpoint(    path, save_best_only=True, monitor="val_loss"),
            tf.keras.callbacks.ReduceLROnPlateau(  monitor="val_loss", factor=0.5, patience=30, min_lr=0.00001),
            tf.keras.callbacks.EarlyStopping(      monitor="val_loss", patience=90, verbose=1),
            tf.keras.callbacks.TensorBoard(        log_dir=TensorBoard_logs)   
        ]    


    history = model.fit(
        train_ds,    
        batch_size=configs["CNN"]["batch_size"],
        callbacks=[checkpoint],
        epochs=configs["CNN"]["epochs"],
        validation_data=val_ds,
        verbose=configs["CNN"]["verbose"],
    )
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)

    path = configs["CNN"]["saving_path"] + "_".join(( "FT", SLURM_JOBID, CNN_name, configs["CNN"]["image_feature"], str(int(np.round(test_acc*100)))+"%" + ".h5" ))
    model.save(path)
    # plt.plot(history.history['accuracy'], label='accuracy')
    # # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.show()

    print(" test_loss ", test_loss, " test_acc ", test_acc)
    return history



def from_scratch(configs):
    logger.info("func: from_scratch")
    
    if configs['CNN']["image_feature"]=="tile":
        configs['CNN']["image_size"] = (120, 80, 3)


    # configs['CNN']["image_size"] = (32, 32, 3)
    # configs['CNN']["class_numbers"] = 10




    # # ##################################################################
    # #                phase 1: Reading image
    # # ##################################################################
    # logger.info("Reading dataset....")
    # with h5py.File(cfg.configs["paths"]["stepscan_dataset.h5"], "r") as hdf:
    #     barefoots = hdf.get("/barefoot/data")[:]
    #     metadata = hdf.get("/barefoot/metadata")[:]

    # data = barefoots.transpose(0,2,3,1)

    # np.save(cfg.configs["paths"]["stepscan_data.npy"], data)
    # np.save(cfg.configs["paths"]["stepscan_meta.npy"], metadata)





    # # ##################################################################
    # #                phase 2: extracting image features
    # # ##################################################################
    # metadata = np.load(cfg.configs["paths"]["stepscan_meta.npy"])
    # data = np.load(cfg.configs["paths"]["stepscan_data.npy"])

    # logger.info(f"barefoots.shape: {data.shape}")
    # logger.info(f"metadata.shape: {metadata.shape}")


    # # plt.imshow(data[1,:,:,:].sum(axis=2))
    # # plt.show()




    # ## Extracting Image Features
    # features = list()
    # labels = list()

    # for label, sample in zip(metadata, data):
    #     try:
    #         B = sample.sum(axis=1).sum(axis=0)
    #         A = np.trim_zeros(B)

    #         aa = np.where(B == A[0])
    #         bb = np.where(B == A[-1])

    #         if aa[0][0]<bb[0][0]:
    #             features.append(feat.prefeatures(sample[10:70, 10:50, aa[0][0]:bb[0][0]]))
    #             labels.append(label)
    #         else:
    #             print(aa[0][0],bb[0][0])
    #             k=sample
    #             l=label
        
    #     except Exception as e:
    #         print(e)
    #         continue
        

    # logger.info(f"len prefeatures: {len(features)}")
    # logger.info(f"prefeatures.shape: {features[0].shape}")
    # logger.info(f"labels.shape: {labels[0].shape}")

    # np.save(cfg.configs["paths"]["stepscan_image_feature.npy"], features)
    # np.save(cfg.configs["paths"]["stepscan_image_label.npy"], labels)






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



    logger.info(f"images: {images.shape}")
    logger.info(f"labels: {labels.shape}")


    # # ##################################################################
    # #                phase 5: Making tf.dataset object
    # # ##################################################################

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.15, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)




    AUTOTUNE = tf.data.AUTOTUNE


    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000)
    train_ds = train_ds.batch(configs['CNN']["batch_size"])
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(1000)
    val_ds = val_ds.batch(configs['CNN']["batch_size"])
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(configs['CNN']["batch_size"])
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)




    # # ##################################################################
    # #                phase 6: Making Base Model
    # # ##################################################################


    CNN_name = "from_scratch"

    input = tf.keras.layers.Input(shape=configs["CNN"]["image_size"], dtype = tf.float64, name="original_img")
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
    output = tf.keras.layers.Dense(configs['CNN']['class_numbers'], name="prediction")(x) # activation='softmax',

    ## The CNN Model
    model = tf.keras.models.Model(inputs=input, outputs=output, name=CNN_name)


    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name,layer.trainable)

    # model.summary() 
    # tf.keras.utils.plot_model(model, to_file=cfg.configs['CNN']['base_model'] + ".png", show_shapes=True)
    # plt.show()



    # # ##################################################################
    # #                phase 7: training CNN
    # # ##################################################################

    model.compile(
        optimizer=tf.keras.optimizers.Adam(), #learning_rate=0.001
        loss=tf.keras.losses.SparseCategoricalCrossentropy (from_logits=True), 
        metrics=["Accuracy"]
        )


    time = int(timeit.timeit()*1_000_000)
    TensorBoard_logs =  os.path.join( configs["paths"]["TensorBoard_logs"], "_".join(("FS", SLURM_JOBID, configs["CNN"]["image_feature"], str(time)) )  )
    path = configs["CNN"]["saving_path"] + "_".join(( "FS", SLURM_JOBID, configs["CNN"]["image_feature"], "best.h5" ))
    logger.info(f"TensorBoard_logs: {TensorBoard_logs}")
    logger.info(f"path: {path}")


    checkpoint = [
            tf.keras.callbacks.ModelCheckpoint(    path, save_best_only=True, monitor="val_loss"),
            tf.keras.callbacks.ReduceLROnPlateau(  monitor="val_loss", factor=0.5, patience=30, min_lr=0.00001),
            tf.keras.callbacks.EarlyStopping(      monitor="val_loss", patience=90, verbose=1),
            tf.keras.callbacks.TensorBoard(        log_dir=TensorBoard_logs)   
        ]    


    history = model.fit(
        train_ds,    
        batch_size=configs["CNN"]["batch_size"],
        callbacks=[checkpoint],
        epochs=configs["CNN"]["epochs"],
        validation_data=val_ds,
        verbose=configs["CNN"]["verbose"],
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=2)

    path = configs["CNN"]["saving_path"] + "_".join(( "FS", SLURM_JOBID, configs["CNN"]["image_feature"], str(int(np.round(test_acc*100)))+"%" + ".h5" ))
    model.save(path)
    # plt.plot(history.history['accuracy'], label='accuracy')
    # # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.show()


    logger.info(f"test_loss: {np.round(test_loss,3)}, test_acc: {int(np.round(test_acc*100))}%")
    return history




def collect_results(result):
    global columnsname, time
    excel_path = cfg.configs["paths"]["results_dir"]

    if os.path.isfile(os.path.join(excel_path, 'Results.xlsx')):
        Results_DF = pd.read_excel(os.path.join(excel_path, 'Results.xlsx'), index_col = 0)
    else:
        Results_DF = pd.DataFrame(columns=columnsname)

    Results_DF = Results_DF.append(result)
    try:
        Results_DF.to_excel(os.path.join(excel_path, 'Results.xlsx'), columns=columnsname)
    except:
        Results_DF.to_excel(os.path.join(excel_path, 'Results'+str(time)+'.xlsx'), columns=columnsname)


# def set_sonfig(configs):
#     global paths, Pipeline, CNN, Template_Matching, SVM, KNN

#     Pipeline = configs["Pipeline"]
#     CNN = configs["CNN"]
#     Template_Matching = configs["Template_Matching"]
#     SVM = configs["SVM"]
#     KNN = configs["KNN"]
#     paths = configs["paths"]
    
def main():
    configs = cfg.configs
    # configs["Pipeline"]["classifier"] = "knn_classifier"


    fine_tuning(configs)  

    # collect_results(z)

    # 
    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    # for ii in test_ratios:
    #     folder = str(0.95) + "_z-score_" + str(-3) + "_dist_median_" +  str(ii)      
    #     pool.apply_async(fcn, args=(DF_features_all, folder, features_excel, 2, "None"), callback=collect_results)
        
    # pool.close()
    # pool.join()


if __name__ == "__main__":
    logger.info("Starting [main] !!!")
    tic = timeit.default_timer()
    main()
    toc = timeit.default_timer()
    logger.info("Done [main] ({:2.2f} process time)!!!\n\n\n".format(toc-tic))


