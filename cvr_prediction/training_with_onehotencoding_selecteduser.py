import numpy as np
import pandas as pd
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
      "idcampaign", "idcat","idaffiliate", "aff_type","purchase", "idcampaign_diff_cvr_1", "user_id_diff_cvr_1","user_id","decay_purchase_delta", "decay_delta", "decay_mean"]].dropna()
test_df = test[["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
                "idcampaign", "idcat","idaffiliate", "aff_type","purchase", "idcampaign_diff_cvr_1", "user_id_diff_cvr_1","user_id","decay_purchase_delta", "decay_delta", "decay_mean"]].dropna()


print train_df.shape
train_df = train_df.replace([-1],[2])
test_df = test_df.replace([-1],[2])





train_features = train_df[["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
                              "idcampaign", "idcat", "idaffiliate", "aff_type","idcampaign_diff_cvr_1", "user_id_diff_cvr_1"]].values
test_features = test_df[["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
                            "idcampaign", "idcat", "idaffiliate", "aff_type","idcampaign_diff_cvr_1", "user_id_diff_cvr_1"]].values


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
mtx_train = encoded_features[:train_df.shape[0],]
mtx_test = encoded_features[train_df.shape[0]:,]


#mtx_train = hstack([mtx_train,flag_train])
#mtx_test = hstack([mtx_test,flag_test])



pCVR, predict_CVR, auc_score, lg_rmse = LR_predict(mtx_train,label_train,mtx_test,label_test)
#rate = float(predict_CVR)/cvr
#print ("LR model with AUC: %.4f, RMSE: %.4f, CVR_ratio: %.4f" %(auc_score, lg_rmse, rate))

print ("LR model with AUC: %.4f, RMSE: %.4f" %(auc_score, lg_rmse))



pCVR, predict_CVR, auc_score, nb_rmse = NB_predict(train_features,label_train,test_features,label_test)
#rate = float(predict_CVR)/cvr
#print ("LR model with AUC: %.4f, RMSE: %.4f, CVR_ratio: %.4f" %(auc_score, lg_rmse, rate))

print ("GaussianNB model with AUC: %.4f, RMSE: %.4f" %(auc_score, nb_rmse))



