
# coding: utf-8

# In[25]:

import pandas as pd

df = pd.read_csv("../../../iPinyou/make-ipinyou-data-master/all/test.log.txt",header=0,sep='\t',index_col=False)


# In[26]:

print df[["click","weekday","hour","advertiser"]].head()


# In[27]:

###############Campaign cvr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


df_sub_cvr =  df[["click","weekday","hour","advertiser"]]
df_add_features = pd.DataFrame(df_sub_cvr.groupby(['advertiser','weekday','hour','click']).size().reset_index(name='count'))
print df_add_features.head()


# In[28]:

#i = 0
for name, group in df_add_features.groupby(['advertiser','weekday','hour']): 
    #i += 1
    #print name[1]

    counts = group['count'].values
    #print group
    #print group['purchase'].values
    
    if ((len(counts) == 1) & (group['click'].values[0] == 0)):
        cvr_h = 0
    elif ((len(counts) == 1) & (group['click'].values[0] == 1)):
        cvr_h = 1
    else:
        cvr_h = float(counts[1])/float(counts.sum())
    
    df.ix[(df.advertiser==name[0]) & (df.weekday == name[1])
              & (df.hour == name[2]), 'cvr_h'] = cvr_h

    #if i == 10:
    #    break


# In[29]:

# compare the previous two hours (t-1, t-2) cvr
from itertools import izip
from math import log
import sys
print df.shape[0]

cvr_diff_df = df.drop_duplicates(['advertiser','weekday','hour'])[['advertiser','weekday','hour','cvr_h']]
print cvr_diff_df.shape[0]
print cvr_diff_df.head()

  
epsilon = np.finfo(float).eps

def myfunc(group):
    group = group.sort_values(['weekday', 'hour'], ascending=[True, True])
    for i in range(1,2):
        diff_cvr = []
        diff_cvr.extend(np.zeros(i+1,dtype=np.int))
        cvr_col = group['cvr_h'].values
             
        for x, y in zip(cvr_col,cvr_col[1:]):
            if y - x > 0.0:
                diff_cvr.append(1)
            elif y - x == 0.0:
                diff_cvr.append(0)
            else: 
                diff_cvr.append(-1)
                
        col_name = 'diff_cvr_' + str(i)
        group[col_name] = pd.Series(diff_cvr[:-i], index=group.index)
        #print diff_cvr
        #group['cvr_ratio'] = pd.Series(ratio_cvr[:-1], index=group.index)
    #print group
    return group


cvr_diff_result = cvr_diff_df.groupby(['advertiser']).apply(myfunc)
print cvr_diff_result.tail()
#print cvr_diff_result[cvr_diff_result['cvr_ratio'] != 1].head()
#print cvr_diff_result[cvr_diff_result['idcampaign'] == 90]


# In[30]:

df_merge = pd.merge(df,cvr_diff_result[['advertiser','weekday','hour','diff_cvr_1']], 
                    how='left', on=['advertiser','weekday','hour'])
print df_merge.tail()
print df_merge.shape[0]


# In[31]:

df_merge.to_csv('../../../iPinyou/ipinyou-data-addflag/ipinyou_testall.flag.csv',index=False, sep="\t")

