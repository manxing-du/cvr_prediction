from sklearn.naive_bayes import GaussianNB
from math import sqrt
from sklearn.metrics import roc_curve, auc, roc_auc_score, mean_squared_error
import numpy as np



def NB_predict(mtx_train,label_train,mtx_test,label_test):
    G_NB = GaussianNB()
    label_train = np.ravel(label_train)

    #start = timeit.default_timer()
    clf_nb = G_NB.fit(mtx_train,label_train)
    #stop = timeit.default_timer()
    #time_interval = stop - start
    #print ("predict time is %f" %time_interval)


    #start = timeit.default_timer()
    pCVR = clf_nb.predict_proba(mtx_test)
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
