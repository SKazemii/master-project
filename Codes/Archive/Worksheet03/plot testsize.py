import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, os
from pathlib import Path as Pathlb




from scipy.stats import shapiro, ttest_ind, mannwhitneyu

import ws3 as perf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MLPackage import FS 
from MLPackage import stat 


plt.rcParams["font.size"] = 13
plt.tight_layout()

print("[INFO] Setting directories")
project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "Manuscripts", "src", "figures")
tbl_dir = os.path.join(project_dir, "Manuscripts", "src", "tables")
data_dir = os.path.join(project_dir, "Archive", "results WS3")

Pathlb(fig_dir).mkdir(parents=True, exist_ok=True)
Pathlb(tbl_dir).mkdir(parents=True, exist_ok=True)





THRESHOLDs = perf.THRESHOLDs
feature_names = perf.feature_names

color = ['darkorange', 'navy', 'red', 'greenyellow', 'lightsteelblue', 'lightcoral', 'olive', 'mediumpurple', 'khaki', 'hotpink', 'blueviolet']


Results_DF = pd.read_excel(os.path.join(data_dir, 'crite.xlsx'), index_col = 0).sort_values(by="Criteria")
# Results_DF.columns = perf.cols



Results_DF["Mode"] = Results_DF["Mode"].map(lambda x: "Correlation" if x == "corr" else "Euclidean distance")
Results_DF["Normalizition"] = Results_DF["Normalizition"].map(lambda x: "Z-score algorithm" if x == "z-score" else "Minmax algorithm")
Results_DF["PCA"] = Results_DF["PCA"].map(lambda x: "All PCs" if x == 1.0 else "keeping {}% variance".format(x*100))
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "afeatures-simple" if x == "afeatures_simple" else x)
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "afeatures-otsu" if x == "afeatures_otsu" else x)
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "COAs-otsu" if x == "COAs_otsu" else x)
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "COAs-simple" if x == "COAs_simple" else x)

print(Results_DF)
       
Results_DF_group = Results_DF.groupby(["Criteria"])
values = Results_DF["Criteria"].sort_values().unique()
print(values)
FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()        
for value in values:


    
    DF = Results_DF_group.get_group((value))

    FAR_L.append(DF[["FAR_L_" + str(i) for i in range(100)]].values)
    FRR_L.append(DF[["FRR_L_" + str(i) for i in range(100)]].values)
    FAR_R.append(DF[["FAR_R_" + str(i) for i in range(100)]].values)
    FRR_R.append(DF[["FRR_R_" + str(i) for i in range(100)]].values)



perf.plot_ROC(FAR_L, FRR_L, FAR_R, FRR_R, values)
plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, "WS3_testsize.png"))
# plt.close('all')
plt.show()

Results_DF = pd.read_excel(os.path.join(data_dir, 'PCA.xlsx'), index_col = 0).sort_values(by="PCA")
# Results_DF.columns = perf.cols



Results_DF["Mode"] = Results_DF["Mode"].map(lambda x: "Correlation" if x == "corr" else "Euclidean distance")
Results_DF["Normalizition"] = Results_DF["Normalizition"].map(lambda x: "Z-score algorithm" if x == "z-score" else "Minmax algorithm")
Results_DF["PCA"] = Results_DF["PCA"].map(lambda x: "All PCs" if x == 1.0 else "keeping {}% variance".format(x*100))
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "afeatures-simple" if x == "afeatures_simple" else x)
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "afeatures-otsu" if x == "afeatures_otsu" else x)
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "COAs-otsu" if x == "COAs_otsu" else x)
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "COAs-simple" if x == "COAs_simple" else x)

print(Results_DF)
       
Results_DF_group = Results_DF.groupby(["PCA", "Criteria"])
values = Results_DF["PCA"].sort_values().unique()
print(values)
FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()        
for value in values:


    
    DF = Results_DF_group.get_group((value, "median"))

    FAR_L.append(DF[["FAR_L_" + str(i) for i in range(100)]].values)
    FRR_L.append(DF[["FRR_L_" + str(i) for i in range(100)]].values)
    FAR_R.append(DF[["FAR_R_" + str(i) for i in range(100)]].values)
    FRR_R.append(DF[["FRR_R_" + str(i) for i in range(100)]].values)



perf.plot_ROC(FAR_L, FRR_L, FAR_R, FRR_R, values)
plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, "WS3_testsize.png"))
# plt.close('all')
plt.show()

sys.exit()
Results_DF = pd.read_excel(os.path.join(data_dir, 'DFtest.xlsx'), index_col = 0)
# Results_DF.columns = perf.cols



Results_DF["Mode"] = Results_DF["Mode"].map(lambda x: "Correlation" if x == "corr" else "Euclidean distance")
Results_DF["Normalizition"] = Results_DF["Normalizition"].map(lambda x: "Z-score algorithm" if x == "z-score" else "Minmax algorithm")
Results_DF["PCA"] = Results_DF["PCA"].map(lambda x: "All PCs" if x == 1.0 else "keeping 95% variance")
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "afeatures-simple" if x == "afeatures_simple" else x)
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "afeatures-otsu" if x == "afeatures_otsu" else x)
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "COAs-otsu" if x == "COAs_otsu" else x)
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "COAs-simple" if x == "COAs_simple" else x)


       
Results_DF_group = Results_DF.groupby(["Test_Size"])
values = Results_DF["Test_Size"].sort_values().unique()
X = pd.DataFrame(index=values , columns=["Accuracy", "F1-score", "EER"])
Y = pd.DataFrame(index=values , columns=[ "EER Left", "EER Right"])
FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()        
for value in values:


    
    DF = Results_DF_group.get_group((value))
    X.loc[value, "Accuracy"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(
        (DF["Mean_Acc_L"].mean()+ DF["Mean_Acc_R"].mean())/2, 
        (DF["Mean_Acc_L"].std() + DF["Mean_Acc_R"].min())/2,
        (DF["Mean_Acc_L"].min() + DF["Mean_Acc_R"].min())/2, 
        (DF["Mean_Acc_L"].max() + DF["Mean_Acc_R"].max())/2)
    X.loc[value, "EER"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(
        (DF["Mean_EER_L_te"].mean()+ DF["Mean_EER_R_te"].mean())/2,  
        (DF["Mean_EER_L_te"].std() + DF["Mean_EER_R_te"].min())/2,
        (DF["Mean_EER_L_te"].min() + DF["Mean_EER_R_te"].min())/2, 
        (DF["Mean_EER_L_te"].max() + DF["Mean_EER_R_te"].max())/2)
    X.loc[value, "F1-score"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(
        (DF["Mean_f1_L"].mean()+ DF["Mean_f1_R"].mean())/2,  
        (DF["Mean_f1_L"].std() + DF["Mean_f1_R"].min())/2,
        (DF["Mean_f1_L"].min() + DF["Mean_f1_R"].min())/2, 
        (DF["Mean_f1_L"].max() + DF["Mean_f1_R"].max())/2)


    


    FAR_L.append(DF[["FAR_L_" + str(i) for i in range(100)]].values)
    FRR_L.append(DF[["FRR_L_" + str(i) for i in range(100)]].values)
    FAR_R.append(DF[["FAR_R_" + str(i) for i in range(100)]].values)
    FRR_R.append(DF[["FRR_R_" + str(i) for i in range(100)]].values)


print(FRR_L)
print(len(FRR_L))
print(FRR_L[0][0, :].shape)
perf.plot_ROC(FAR_L, FRR_L, FAR_R, FRR_R, [str(x*100)+" %" for x in values])
plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, "WS3_testsize.png"))
# plt.close('all')
plt.show()


with open(os.path.join(tbl_dir, "WS3_testsize.tex"), "w") as tf:
    tf.write(X.to_latex())

with pd.ExcelWriter(os.path.join(tbl_dir,  "WS3_testsize.xlsx")) as writer:  
    X.to_excel(writer, sheet_name='Sheet_name_1')





