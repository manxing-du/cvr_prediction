
# coding: utf-8

# In[1]:

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

train['uniform_price'].fillna(0,inplace=True)
test['uniform_price'].fillna(0,inplace=True)


train_df = train[["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
                  "idcampaign", "idcat", "idaffiliate", "aff_type", "purchase", "idcampaign_diff_cvr_1",
                  "user_id_diff_cvr_1", "uniform_price", "date_added_full", "user_id"]].dropna()
test_df = test[["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
                "idcampaign", "idcat", "idaffiliate", "aff_type", "purchase", "idcampaign_diff_cvr_1",
                "user_id_diff_cvr_1", "uniform_price", "date_added_full", "user_id"]].dropna()




merged_df = pd.concat([train_df,test_df], axis=0)


# In[ ]:

print merged_df.tail()


# In[33]:

merged_df = merged_df.reset_index(drop=True)


# In[37]:

merged_df = merged_df.replace([-1],[2])
all_features = merged_df[["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
                            "idcampaign", "idcat", "idaffiliate", "aff_type","idcampaign_diff_cvr_1", "user_id_diff_cvr_1"]].values

print "started encoding"
#encoded_features = pd.get_dummies(train_df)
#print encoded_features.shape
enc = OneHotEncoder(dtype=int)
encoded_features = enc.fit_transform(all_features)


# In[38]:

print encoded_features.shape


# In[218]:

day_start = 2015071500
###### change window size 
auc_dict = {}
def get_sliding_window_index(df,day_start,window_size):
    training_window = []
    test_window = []
    for j in range(0,6):
        if j == 0:
            sliding_window_start = day_start
        else:
            sliding_window_start = day_start + 100 * j
        next_day = sliding_window_start + 100
            
        breaking_point = window_size*(24/window_size -1) -1
        
        for i in range(0,breaking_point):
            start = df.loc[df['date_added_full'] == sliding_window_start].index.tolist()[0]
            stop = df.loc[df['date_added_full'] == sliding_window_start+window_size].index.tolist()[0]
            training_window.append((start,stop))

            start_test = stop
            stop_test = df.loc[df['date_added_full'] == sliding_window_start+window_size+1].index.tolist()[0]
            test_window.append((start_test,stop_test))
           
            sliding_window_start +=  1
            
        for i in range(breaking_point,24):
            if i == breaking_point:
                start = df.loc[df['date_added_full'] == sliding_window_start].index.tolist()[0]
                stop =  df.loc[df['date_added_full'] == sliding_window_start + window_size].index.tolist()[0]
                training_window.append((start, stop))
                
                start_test = stop
                stop_test = df.loc[df['date_added_full'] == next_day].index.tolist()[0]
                test_window.append((start_test,stop_test))
                
            else:
                print sliding_window_start
                start = df.loc[df['date_added_full'] == sliding_window_start].index.tolist()[0]
                stop =  df.loc[df['date_added_full'] == next_day + i - breaking_point - 1].index.tolist()[0]
                training_window.append((start, stop))
                
                start_test = stop
                stop_test = df.loc[df['date_added_full'] == next_day + i - breaking_point].index.tolist()[0]
                test_window.append((start_test,stop_test))
            sliding_window_start +=1
        
    return training_window,test_window
    

window_size = [6]    
for wsize in window_size:
    training_window, test_window = get_sliding_window_index(merged_df,day_start,wsize)

    
    AUC_list = []
    for item in zip(training_window,test_window):
    
        mtx_train = encoded_features[item[0][0]:item[0][1],]
        mtx_test = encoded_features[item[1][0]:item[1][1],]
        label_train = merged_df[['purchase']].values[item[0][0]:item[0][1]]
        label_test = merged_df[['purchase']].values[item[1][0]:item[1][1]]
        pCVR, predict_CVR, auc_score, lg_rmse = LR_predict(mtx_train,label_train,mtx_test,label_test)
        AUC_list.append(auc_score)
    
    

    #plt.ioff()
    #fig = plt.figure()
    # Create an axes instance
    #ax = fig.add_subplot(111)
    #ax.plot(AUC_list,color='k')
    #ax.set_xlim([0,150])
    #plt.xlabel("hours")
    #plt.ylabel("AUC")
    #plt.grid()
    #filename = "AUC_list_" + str(wsize) + ".pdf"
    #plt.savefig('../Figures/0715-19/'+ filename)
    #plt.close(fig)
    
    if wsize not in auc_dict:
        auc_dict[wsize] = AUC_list


# In[217]:

print auc_dict.keys()


# In[57]:

for key, item in auc_dict.iteritems():
    item[0:0] = [0] * (int(key)-1)
    plt.ioff()
    fig = plt.figure()
    # Create an axes instance
    ax = fig.add_subplot(111)
    ax.plot(item,color='k')
    ax.set_xlim([0,150+11])
    ax.set_ylim([0.70,0.85])
    plt.xlabel("hours")
    plt.ylabel("AUC_value")
    plt.grid()
    filename = "AUC_list_" + str(key) + ".pdf"
    plt.savefig('../Figures/0715-19/'+ filename)
    plt.close(fig)


# In[164]:

backup = auc_dict


# In[ ]:

print backup.keys()
print auc_dict


# In[166]:

###insert 0 for plot CAN ONLY RUN ONCE!!!!! otherwise will insert multiple 0's
for key in sorted(backup.iterkeys()):
    backup[key][0:0] = [0] * (int(key)-1)


# In[168]:

plt.ioff()
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, sharey=True)
for key, ax_value in zip(sorted(auc_dict.iterkeys()),[ax1,ax2,ax3,ax4,ax5]):
    print key
    item = auc_dict[key]
    ax_value.plot(item,color='k')
    ax_value.set_xlim([0,150+11])
    ax_value.set_ylim([0.70,0.85])
    ax_value.set_yticks(np.arange(0.70,0.82,0.05))
    if key == 12:
        ax_value.set_xlabel = ("hour")
        
f.subplots_adjust(hspace=0)
#plt.set_xlim([0,150+11])
#plt.set_ylim([0.70,0.85])
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

f.text(0.5, 0.04, 'hour', ha='center')
f.text(0.04, 0.5, 'AUC', va='center', rotation='vertical')
filename = "AUC_combine.pdf"
plt.savefig('../Figures/0715-19/'+ filename)
plt.close(fig)


# In[175]:

y=[]
for key in sorted(auc_dict.iterkeys()):
    y.append(auc_dict[key])
    
plt.ioff()
fig = plt.figure()
# Create an axes instance
ax = fig.add_subplot(111)
bp = ax.boxplot(y)
plt.setp(bp['boxes'], color='k')
plt.setp(bp['whiskers'], color='k')
ax.set_ylim([0.7,0.85])
ax.set_ylim([0.7,0.85])
plt.xticks([1,2,3,4,5], [1,4,6,8,12])
ax.set_xlabel("Window size")
ax.set_ylabel("AUC")
plt.savefig('../Figures/0715-19/sliding_window_boxplot.pdf',bbox_inches='tight')
plt.close(fig)


# In[161]:

##### Get average auc
y,z = [],[]
for key, item in auc_dict.iteritems():
    a = [x for x in item if x != 0]
    print y
    y.append(np.mean(a))
    z.append(np.std(a))


plt.ioff()
fig = plt.figure()
# Create an axes instance
ax = fig.add_subplot(111)
x = [1,2,3,4,5]
ax.plot(x,y,'ko--')
#ax.set_xlim([0,150+11])
#ax.set_ylim([0.70,0.85])

ax.set_xlabel("Window size")
ax.set_ylabel("mean AUC")
plt.grid()
ax.set_xlim([0,6])
plt.xticks([1,2,3,4,5], [1,4,6,8,12])
filename = "AUC_list_average.pdf"
plt.savefig('../Figures/0715-19/'+ filename)
plt.close(fig)

plt.ioff()
fig = plt.figure()
# Create an axes instance
ax = fig.add_subplot(111)
x = [1,2,3,4,5]
ax.plot(x,z,'ko--')
#ax.set_xlim([0,150+11])
#ax.set_ylim([0.70,0.85])

ax.set_xlabel("Window size")
ax.set_ylabel("mean AUC")
plt.grid()
ax.set_xlim([0,6])
plt.xticks([1,2,3,4,5], [1,4,6,8,12])
filename = "AUC_list_std.pdf"
plt.savefig('../Figures/0715-19/'+ filename)
plt.close(fig)


# In[194]:

####### Fix the training window, changing the test window

###### change window size 
auc_dict_change_test = {}
def get_sliding_window_index(df,day_start,window_size):
    training_window = []
    test_window = []
    for j in range(0,6):
        if j == 0:
            sliding_window_start = day_start
        else:
            sliding_window_start = day_start + 100 * j
        next_day = sliding_window_start + 100
        
        
        if ((24-(6+window_size)) % window_size) != 0:
            breaking_point = window_size*((24-(6+window_size))/window_size + 1)
        
        
        else:
            breaking_point = window_size*((24-(6+window_size))/window_size)
            
            
            
        
        for i in range(0,breaking_point,window_size):
            start = df.loc[df['date_added_full'] == sliding_window_start].index.tolist()[0]
            stop = df.loc[df['date_added_full'] == sliding_window_start + 6].index.tolist()[0]
            training_window.append((start,stop))
            
            '''
            if (start,stop) not in test_every_hour:
                test_every_hour[(start,stop)] = []

            '''
           
            
            start_test = stop
            stop_test = df.loc[df['date_added_full'] == sliding_window_start + 6 + window_size].index.tolist()[0]
            test_window.append((start_test,stop_test))
            
            '''
            for j in range(1,window_size+1):
                if j == 1:
                    start_hour = start_test
                else:
                    start_hour = stop_hour
                stop_hour = df.loc[df['date_added_full'] == sliding_window_start + 6 + j].index.tolist()[0]
                test_every_hour[(start,stop)].append((start_hour,stop_hour))

            '''
            
           
            sliding_window_start +=  window_size
            
        for i in range(breaking_point,24,window_size):
            if i == breaking_point:
                start = df.loc[df['date_added_full'] == sliding_window_start].index.tolist()[0]
                stop =  df.loc[df['date_added_full'] == sliding_window_start + 6].index.tolist()[0]
                training_window.append((start, stop))
                
            
                start_test = stop
                stop_test = df.loc[df['date_added_full'] == next_day + (window_size-(24-breaking_point-6))].index.tolist()[0]
                test_window.append((start_test,stop_test))
                
                
                
                
                
                
                
            else:
                print sliding_window_start
                start = df.loc[df['date_added_full'] == sliding_window_start].index.tolist()[0]
                stop =  df.loc[df['date_added_full'] == next_day + (6-(24-i))].index.tolist()[0]
                training_window.append((start, stop))
                
        
                start_test = stop
                stop_test = df.loc[df['date_added_full'] == next_day + (6-(24-i)) + window_size].index.tolist()[0]
                test_window.append((start_test,stop_test))
                
          
                
            
            sliding_window_start += window_size
        
    return training_window,test_window
    

window_size = [2,4,6,8,12]    
for wsize in window_size:
    training_window, test_window = get_sliding_window_index(merged_df,day_start,wsize)

    
    AUC_list = []
    for item in zip(training_window,test_window):
    
        mtx_train = encoded_features[item[0][0]:item[0][1],]
        mtx_test = encoded_features[item[1][0]:item[1][1],]
        label_train = merged_df[['purchase']].values[item[0][0]:item[0][1]]
        label_test = merged_df[['purchase']].values[item[1][0]:item[1][1]]
        pCVR, predict_CVR, auc_score, lg_rmse = LR_predict(mtx_train,label_train,mtx_test,label_test)
        AUC_list.append(auc_score)
    
    

    #plt.ioff()
    #fig = plt.figure()
    # Create an axes instance
    #ax = fig.add_subplot(111)
    #ax.plot(AUC_list,color='k')
    #ax.set_xlim([0,150])
    #plt.xlabel("hours")
    #plt.ylabel("AUC")
    #plt.grid()
    #filename = "AUC_list_" + str(wsize) + ".pdf"
    #plt.savefig('../Figures/0715-19/'+ filename)
    #plt.close(fig)
    
    if wsize not in auc_dict_change_test:
        auc_dict_change_test[wsize] = AUC_list


# In[195]:

for key, item in auc_dict_change_test.iteritems():
    item[0:0] = [0] * 6
    plt.ioff()
    fig = plt.figure()
    # Create an axes instance
    ax = fig.add_subplot(111)
    ax.plot(item,color='k')
    ax.set_xlim([0,160/key])
    ax.set_ylim([0.74,0.85])
    ax.set_xlabel("hours")
    ax.set_ylabel("AUC_value")
    plt.grid()
    filename = "change_test_window_AUC_list_" + str(key) + ".pdf"
    plt.savefig('../Figures/0715-19/'+ filename)
    plt.close(fig)
    


# In[215]:

y=[]
for key in sorted(auc_dict_change_test.iterkeys()):
    
    a = [x for x in auc_dict_change_test[key] if x != 0]
    y.append(a)
    
plt.ioff()
fig = plt.figure()
# Create an axes instance
ax = fig.add_subplot(111)
bp = ax.boxplot(y)
plt.setp(bp['boxes'], color='k')
plt.setp(bp['whiskers'], color='k')

ax.set_ylim([0.74,0.85])
plt.xticks([1,2,3,4,5], [str(float(2)/float(6)*100)[:-8]+"%",str(float(4)/float(6)*100)[:-8]+"%",str(float(6)/float(6)*100)[:-2]+"%"
                         ,str(float(8)/float(6)*100)[:-8]+"%",str(float(12)/float(6)*100)[:-2]+"%"])
ax.set_xlabel("The ratio of test window size over training window size")
ax.set_ylabel("AUC")
plt.savefig('../Figures/0715-19/sliding_window_boxplot_testset.pdf',bbox_inches='tight')
plt.close(fig)


# In[199]:

print auc_dict_change_test[12]


# In[216]:

print len(z)

