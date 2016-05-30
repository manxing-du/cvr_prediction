from math import sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score, mean_squared_error
import numpy as np
import timeit
import pickle



def LR_predict(mtx_train,label_train,mtx_test,label_test):
    lg = LogisticRegression(random_state=44,penalty='l2')
    label_train = np.ravel(label_train)

    #start = timeit.default_timer()
    clf_lg = lg.fit(mtx_train,label_train)
    #stop = timeit.default_timer()
    #time_interval = stop - start
    #print ("predict time is %f" %time_interval)


    #start = timeit.default_timer()
    pCVR = clf_lg.predict_proba(mtx_test)
    #stop = timeit.default_timer()
    #time_interval = stop - start
    #print ("predict time is %f" %time_interval)


    ####### Evaluation
    #fpr,tpr,thresholds = roc_curve(label_test,pCVR[:,1])
    #roc_auc = auc(fpr,tpr)
    predict_CVR = np.mean(pCVR[:,1])
    #print("LR predicted CVR is %.5f" % predict_CVR)
    auc_score = roc_auc_score(label_test,pCVR[:,1])
    #print("ROC AUC score for LR is %.4f" % auc_score)
    lg_rmse = sqrt(mean_squared_error(label_test, pCVR[:,1]))
    #print("rmse is %.4f" % lg_rmse)

    return pCVR, predict_CVR, auc_score, lg_rmse



def LR_fit_country(mtx_train,label_train):
    lg = LogisticRegression(random_state=44, penalty='l2')
    label_train = np.ravel(label_train)
    clf_lg = lg.fit(mtx_train, label_train)
    model = pickle.dumps(clf_lg)
    return model
