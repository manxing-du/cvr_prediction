
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from lr_model import LR_predict,LR_fit_country
from sklearn.cluster import KMeans
from sklearn.datasets import load_svmlight_file

import pickle
from read_sparse_matrix import convert_to_sparse_matrix
from kmeans_cluster import Run_Kmeans
from sklearn.metrics import roc_curve, auc, roc_auc_score, mean_squared_error


# In[2]:

def cluster_lr_model(mtx_train,mtx_test,num_cluster):
    cluster_train, cluster_test = Run_Kmeans(mtx_train, mtx_test, num_cluster)

    pCVR_all = []
    label_reset = []
    for i in np.unique(cluster_train):
        # if i == 2:
        #    break
        # Get training data from each cluster
        cluster_index = np.where(cluster_train == i)[0]
        cluster_label = label_train[cluster_index]

        cluster_label = np.ravel(cluster_label)
        cluster_features = mtx_train[cluster_index, :]

        ###cluster in test set
        test_cluster_index = np.where(cluster_test == i)[0]
        test_cluster_label = label_test[test_cluster_index]
        test_cluster_label = np.ravel(test_cluster_label)
        test_cluster_features = mtx_test[test_cluster_index, :]
        
        ##check label in clusters
        if len(set(cluster_label))==1 :
            if cluster_label[0] == 1:
                print("Cluster %d has eCVR: 1 " % (i))
                pCVR = np.ones((len(test_cluster_label),1))
            else:
                print("Cluster %d has eCVR: 0" % (i))
                pCVR = np.zeros((len(test_cluster_label),0))
            
        else:
            pCVR, predict_CVR, auc_score, lg_rmse = LR_predict(cluster_features, cluster_label,
                                                           test_cluster_features, test_cluster_label)
            unique, counts = np.unique(test_cluster_label, return_counts=True)
            eCVR = float(counts[1]) / float(sum(counts))
            print("Cluster %d ROC AUC score for LR is %.4f, eCVR is %.4f, RMSE is %.4f" % (i, auc_score, eCVR, lg_rmse))

        pCVR_all.extend(pCVR[:, 1])
        label_reset.extend(test_cluster_label)
        
    overall_auc = roc_auc_score(label_reset, pCVR_all)
    print ("Overall auc is %.4f" % overall_auc)
    print np.sqrt(mean_squared_error(label_reset, pCVR_all))


# In[3]:

'''

def country_lr_model(mtx_train,train_country_list):
    print "###total country"
    print len(np.unique(train_country_list))
    for country in np.unique(train_country_list):
        country_group_index = np.where(np.array(train_country_list) == str(country))[0]
        country_group_label = label_train[country_group_index]
        country_group_label = np.ravel(country_group_label)
        country_features = mtx_train[country_group_index, :]
        
        
        #print "#### group index"
        #print len(country_group_index)
        country_purchase = len(set(country_group_label))
        if country_purchase == 1:
            continue
        else:
            model = LR_fit_country(country_features, country_group_label)
            if country not in country_model:
                country_model[country] = model
    print "total country"
    print len(country_model.keys())
    return country_model


def country_lr_model_prediction(mtx_test, test_country_list, country_model):
    for country in np.unique(test_country_list):
        ###cluster in test set
        test_country_index = np.where(np.array(test_country_list) == country)[0]
        test_country_label = label_test[test_country_index]
        test_country_label = np.ravel(test_country_label)
        test_country_features = mtx_test[test_country_index, :]
        
        
        ## TODO: If country has a model
        if country in country_model:
            model = pickle.loads(country_model[country])
            pCVR = model.predict_proba(test_country_features)
            unique, counts = np.unique(test_country_label, return_counts=True)
            if unique == 1:
                eCVR == 0.0
                
            else:
                eCVR = float(counts[1]) / float(sum(counts))
            predict_CVR = np.mean(pCVR[:, 1])
            auc_score = roc_auc_score(test_country_label, pCVR[:, 1])
            lg_rmse = sqrt(mean_squared_error(test_country_label, pCVR[:, 1]))
        else:
            pCVR.append(0.0)
            eCVR = None
            predict_CVR = None
            auc_score = None
            lg_rmse = None
        
        return country, pCVR, eCVR, predict_CVR, auc_score, lg_rmse
        
        
'''


# In[4]:


if __name__ == '__main__':
    # Reading training and testing data
    train_data = file('../../../iPinyou/ipinyou-data-addflag/train.flag.libsvm.txt').readlines()
    test_data = file('../../../iPinyou/ipinyou-data-addflag/test.flag.libsvm.txt').readlines()
    
    #train_data = file('../../../iPinyou/make-ipinyou-data-master/3476/train.yzx.txt').readlines()
    #test_data = file('../../../iPinyou/make-ipinyou-data-master/3476/test.yzx.txt').readlines()
    
    

    #mtx_train, label_train, train_country_list = convert_to_sparse_matrix(train_data)
    #mtx_test, label_test, test_country_list= convert_to_sparse_matrix(test_data)
    mtx_train, label_train = convert_to_sparse_matrix(train_data)
    mtx_test, label_test = convert_to_sparse_matrix(test_data)
    

    ##### calculate empirical cvr
    elements, repeats = np.unique(label_test, return_counts=True)
    cvr = float(repeats[1]) / float(repeats.sum())
    print cvr

    '''
    ##### One Logistic Regression
    pCVR, predict_CVR, auc_score, lg_rmse = LR_predict(mtx_train,label_train,mtx_test,label_test)
    print("LR predicted CVR is %.5f" % predict_CVR)
    print("ROC AUC score for LR is %.4f" % auc_score)
    print("rmse is %.4f" % lg_rmse)
    '''


# In[5]:

print mtx_test.shape
print mtx_train.shape


# In[6]:

### CTR/CVR predicion One LR model
pCVR, predict_CVR, auc_score, lg_rmse = LR_predict(mtx_train,label_train,mtx_test,label_test)
rate = float(predict_CVR)/cvr
print ("LR model with AUC: %.4f, RMSE: %.4f, CVR_ratio: %.4f" %(auc_score, lg_rmse, rate))


# In[7]:

##### Clustering + LR per cluster
#cluster_lr_model(mtx_train,mtx_test,16)
# Testing


# In[8]:

'''
# LR per country
country_model = {}
country_model = country_lr_model(mtx_train,train_country_list)
country, pCVR, eCVR, predict_CVR, auc_score, lg_rmse = country_lr_model_prediction(mtx_test, test_country_list, country_model)
print ("country %s with eCVR %.4f, auc_score is %.4f, lg_rmse is %.4f" % (country, eCVR,auc_score,lg_rmse))
'''

