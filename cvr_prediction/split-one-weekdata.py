
# coding: utf-8

# In[14]:

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
from statsmodels.distributions.empirical_distribution import ECDF


df = pd.read_csv("../Data/add_cvr_flag/campaign_testmode_week2_uflag_cflag.csv",header=0,sep=',',index_col=0)


# In[ ]:

print df['hour'].unique()


# In[6]:

df = df.reset_index(drop=True)
df['uniform_price'].fillna(0,inplace=True)

print df.head(1)
####Exclude hour/weekday/aff_type = None
df = df[["hour", "weekday", "country_code", "idoperator","idhardware", "idbrowser", "idos",
      "idcampaign", "idcat","idaffiliate", "aff_type","purchase", "idcampaign_diff_cvr_1", "user_id_diff_cvr_1","uniform_price","date_added_full","user_id",'iddevice']].dropna()



test_start_index = df.loc[df.date_added_full == 2015072000].index[0]
print test_start_index
training_set = df[["hour", "weekday", "country_code", "idoperator","idhardware", "idbrowser", "idos",
      "idcampaign", "idcat","idaffiliate", "aff_type","purchase", "idcampaign_diff_cvr_1", "user_id_diff_cvr_1","uniform_price","user_id",'iddevice']][:test_start_index]
test_set = df[["hour", "weekday", "country_code", "idoperator","idhardware", "idbrowser", "idos",
      "idcampaign", "idcat","idaffiliate", "aff_type","purchase", "idcampaign_diff_cvr_1", "user_id_diff_cvr_1","uniform_price","user_id",'iddevice']][test_start_index:]


# In[7]:

print ("number of user profiles per week is %d" %df['user_id'].nunique())
df[['country_code','idoperator', 'iddevice',  'idhardware',  'idbrowser','idos','idcampaign','idcat',
    'idaffiliate','aff_type','user_id']].apply(pd.Series.nunique)


# In[13]:

print df['hour'].unique()


# In[39]:

####Write to file
training_set.to_csv("train_1519.txt",index=False)
test_set.to_csv("test_2021.txt",index=False)


# In[ ]:

####

