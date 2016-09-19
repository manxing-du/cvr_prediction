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


train_df = train[["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
                  "idcampaign", "idcat", "idaffiliate", "aff_type", "purchase", "idcampaign_diff_cvr_1",
                  "user_id_diff_cvr_1", "uniform_price", "date_added_full", "user_id"]].dropna()
test_df = test[["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
                "idcampaign", "idcat", "idaffiliate", "aff_type", "purchase", "idcampaign_diff_cvr_1",
                "user_id_diff_cvr_1", "uniform_price", "date_added_full", "user_id"]].dropna()


day_start = 2015071500
training_window = []
test_window = []
def get_sliding_window_index(df,day_start):

    for j in range(0,6):
        if j == 0:
            sliding_window_start = day_start
        else:
            sliding_window_start = day_start + 100 * j

        for i in range(0,18):
            start = df.loc[df['date_added_full'] == sliding_window_start].index.tolist()[0]
            stop = df.loc[df['date_added_full'] == sliding_window_start+6].index.tolist()[0]
            training_window.append((start,stop))
            sliding_window_start +=  1
        for i in range(19,24):
            start = df.loc[df['date_added_full'] == sliding_window_start].index.tolist()[0]
            stop =  df.loc[df['date_added_full'] == sliding_window_start + 82].index.tolist()[0]
            training_window.append((start, stop))
            sliding_window_start +=1
    return training_window

training_window = get_sliding_window_index()












print train_df.shape
train_df = train_df.replace([-1],[2])
test_df = test_df.replace([-1],[2])


hawkes = [100,281,324,324,2500,8414]
hawkes_campaign = [4755, 4398, 4755, 4848, 4848, 4755]


top_U_revenue = top_users_revenue.drop(['user_id'], axis = 1 , errors= 'ignore')
top_U_revenue = top_U_revenue.reset_index()

#user_campaign_revenue = user_campaign_revenuelist.drop(['user_id','idcampaign'], axis = 1,errors= 'ignore')
#user_campaign_revenue = user_campaign_revenue.reset_index()

samples = zip(hawkes, hawkes_campaign)
samples = pd.DataFrame(samples, columns=['user_id', 'idcampaign'])

result = pd.merge(samples, user_campaign_revenue, how='inner')


train_df.loc[~(train_df['user_id'].isin()&(train_df))


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



