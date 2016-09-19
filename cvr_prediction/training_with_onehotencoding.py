import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from math import sqrt
from lr_model import LR_predict,LR_fit_country
from NB_model import NB_predict
import scipy as sp
from scipy.sparse import coo_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.datasets import load_svmlight_file

import pickle
from read_sparse_matrix import convert_to_sparse_matrix
from kmeans_cluster import Run_Kmeans
from sklearn.metrics import roc_curve, auc, roc_auc_score, mean_squared_error


from sklearn.preprocessing import OneHotEncoder



train = pd.read_csv("../Data/initial-parse/train_1519.txt", header=0,sep=',',index_col=False
                    )

test = pd.read_csv("../Data/initial-parse/test_2021.txt", header=0, sep=',', index_col=False
                   )
#[["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
#  "idcampaign", "idcat", "idaffiliate", "aff_type", "purchase", "idcampaign_diff_cvr_1", "user_id_diff_cvr_1",
#  "decay_purchase_delta", "decay_delta", "decay_mean"]]



train_df = train[["hour", "weekday", "country_code", "idoperator","idhardware", "idbrowser", "idos",
      "idcampaign", "idcat","idaffiliate", "aff_type","purchase", "idcampaign_diff_cvr_1", "user_id_diff_cvr_1","user_id"]].dropna()
test_df = test[["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
                "idcampaign", "idcat","idaffiliate", "aff_type","purchase", "idcampaign_diff_cvr_1", "user_id_diff_cvr_1","user_id"]].dropna()


print train_df.shape
train_df = train_df.replace([-1],[2])
test_df = test_df.replace([-1],[2])







train_features = train_df[["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
                              "idcampaign", "idcat", "idaffiliate", "aff_type"]].values
test_features = test_df[["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
                            "idcampaign", "idcat", "idaffiliate", "aff_type"]].values


#flag_train = sp.sparse.csr_matrix(train_df[["decay_purchase_delta","decay_delta","decay_mean"]].values)
#flag_test = sp.sparse.csr_matrix(test_df[["decay_purchase_delta","decay_delta","decay_mean"]].values)



label_train = train_df[['purchase']].values
label_test = test_df[['purchase']].values

#train_df = train_df.astype(int)
#test_df = test_df.astype(int)
wholeset = np.concatenate((train_features,test_features),axis = 0)
#wholeset = np.asmatrix(wholeset)

#print type(wholeset)

print "started encoding"


#encoded_features = pd.get_dummies(train_df)
#print encoded_features.shape
enc = OneHotEncoder(dtype=int)
encoded_features = enc.fit_transform(wholeset)


from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
selected = selector.fit_transform(encoded_features)




mtx_train = selected[:train_df.shape[0],]
mtx_test = selected[train_df.shape[0]:,]


#mtx_train = hstack([mtx_train,flag_train])
#mtx_test = hstack([mtx_test,flag_test])


import timeit


#pCVR, predict_CVR, auc_score, lg_rmse = LR_predict(mtx_train,label_train,mtx_test,label_test)
#rate = float(predict_CVR)/cvr
#print ("LR model with AUC: %.4f, RMSE: %.4f, CVR_ratio: %.4f" %(auc_score, lg_rmse, rate))

lg = LogisticRegression(random_state=44, penalty='l2')
label_train = np.ravel(label_train)

start = timeit.default_timer()
clf_lg = lg.fit(mtx_train, label_train)
stop = timeit.default_timer()
time_interval = stop - start
print ("predict time for LR is %f" %time_interval)


# start = timeit.default_timer()
pCVR = clf_lg.predict_proba(mtx_test)
# stop = timeit.default_timer()
# time_interval = stop - start
# print ("predict time is %f" %time_interval)


####### Evaluation
# fpr,tpr,thresholds = roc_curve(label_test,pCVR[:,1])
# roc_auc = auc(fpr,tpr)
predict_CVR = np.mean(pCVR[:, 1])
# print("LR predicted CVR is %.5f" % predict_CVR)
auc_score = roc_auc_score(label_test, pCVR[:, 1])
# print("ROC AUC score for LR is %.4f" % auc_score)
lg_rmse = sqrt(mean_squared_error(label_test, pCVR[:, 1]))

print ("LR model with AUC: %.4f, RMSE: %.4f" %(auc_score, lg_rmse))





pCVR, predict_CVR, auc_score, nb_rmse = NB_predict(train_features,label_train,test_features,label_test)
#rate = float(predict_CVR)/cvr
#print ("LR model with AUC: %.4f, RMSE: %.4f, CVR_ratio: %.4f" %(auc_score, lg_rmse, rate))

print ("GaussianNB model with AUC: %.4f, RMSE: %.4f" %(auc_score, nb_rmse))




from sklearn.ensemble import RandomForestClassifier
from math import sqrt


clf_RF = RandomForestClassifier(n_estimators=20, max_depth=50, min_samples_split=1, random_state=0, class_weight="balanced")
label_train = np.ravel(label_train)
start = timeit.default_timer()
clf_lg = clf_RF.fit(mtx_train,label_train)
stop = timeit.default_timer()
time_interval = stop - start
print ("predict time is %f" %time_interval)
pCVR = clf_lg.predict_proba(mtx_test)
#predict_CVR = np.mean(pCVR[:,1])
auc_score = roc_auc_score(label_test,pCVR[:,1])
lg_rmse = sqrt(mean_squared_error(label_test, pCVR[:,1]))
print ("RF model with 500 trees, AUC: %.4f, RMSE: %.4f" %(auc_score, lg_rmse))

