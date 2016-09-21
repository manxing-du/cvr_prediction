
# coding: utf-8

# In[ ]:

import csv
import glob
import sys
import os.path
import re
import datetime
import collections
import pandas as pd
import numpy as np

import scipy as sp

#from unbalanced_dataset import SMOTE
from sklearn import tree
import pytz
from pytz import timezone
import math


# In[91]:

# read test mode data (sample with 1 million rows)
parse_time = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

df = pd.read_csv("../Original_Data/campaign_testmode_0715_0721.txt",header=None,sep=',',index_col=None,names=['id','idwebsite','date_added',
                                                                      'country_code','idoperator','idcampaign','idurl','iddevice','idhardware','idbrowser',
                                                                      'iddevicegroup','idos','flag_test_mode','idaffiliate','date_added_full','multiple_sales'],
date_parser=parse_time, parse_dates = [2])



# In[92]:

### Add purchase or not
df_transaction = pd.read_csv("../Original_Data/transaction07_withdate_complete.txt", index_col=None,header=None,sep='\t',names=['price','currency','idclick','date_added_full'])
df_transaction['date_added_full'] = df_transaction['date_added_full'].astype(str)


# In[93]:

#df_transaction['idclick']=df_transaction['idclick'].map(lambda x:int(x.split('-')[0]))
df['purchase'] = (df.id.isin(df_transaction.idclick)).astype(int)
df['date_added_full'] = df['date_added_full'].astype(str)



df_dynamic = pd.read_csv("../Original_Data/dynamic_price_with_default_currency.txt",header=0, index_col=0, sep=',')
df_dynamic['date_added_full'] = df_dynamic['date_added_full'].astype(str)

# In[97]:

### Merge dynamic price
df = pd.merge(df, df_dynamic, how='left', left_on=['id','date_added_full'], right_on=['idclick','date_added_full'])
#print df[df["purchase"] == 1].head()
#print df_transaction.head()


df_default = pd.read_csv("../Original_Data/default_price_currency.txt",header=0, index_col=0, sep=',')
df_default['date_added'] = df_default['date_added'].astype(str)
## When update the price info in the campaign table, problem is, the price may stay the same for a few days and is updated later.
## some campaign's price may be 0 if there's no purchase

def match_price(row):
    if row['uniform_price'] == 0:
        current_campaign = row['idcampaign']
        compare_date = row['date_added_full'][:-2]
        sub_df = df_default.loc[(df_default['idcampaign'] == current_campaign) & (df_default['date_added']<= compare_date)]
        if sub_df.shape[0] != 0:
            index = sub_df.shape[0] - 1
            #print sub_df.iloc[[0]]['uniform_price']
            #print sub_df.iloc[0]
            return float("{0:.6f}".format(sub_df.iloc[index]['uniform_price']))
        else:
            print "cannot find default price"
            return 0
    else:
        return row['uniform_price']

df['uniform_price'] = df.apply(lambda row: match_price(row),axis = 1)



print "after merge dynamic price"
### Read country code list
df_countrylist = pd.read_csv("../Original_Data/compare_country_code/country_list.csv",header=None,index_col=None,names=['code_string','country_code'])

### Converting time zone
#localize the date_added to France time
tz_fr = timezone('Europe/Paris')
df['date_added'] = df['date_added'].map(lambda x: tz_fr.localize(x))

list_no_timeinfo = []



def localize_time(a, b):
    if a != "N/A" and isinstance(a, str):
        try:
            tz = pytz.country_timezones[a]
            tz_fr = timezone('Europe/Paris')
            tz_format = timezone(tz[0])
            local_time = tz_format.normalize(b.astimezone(tz_format))
        except:
            ##In 0715 data, best mode contains records with country_code= AP (Asia/Pacific)
            ##which cannot be mapped to a specific local time, thus, set it to None
            print a
            list_no_timeinfo.append(a)
            local_time = None
    else:
        #if math.isnan(a):
        #print "NO_INFO"
        local_time = None
    return local_time

#Match the country_code id to country code
df = df.merge(df_countrylist,on='country_code',how='left')
#df_debug = df[pd.isnull(df['code_string'])]['country_code']
#df.loc[df.country_code == 155,['code_string']] = "NA"

#df['code_string'] = df['code_string'].apply(lambda x: x if x!= "UK" else "GB")
df['local_time'] = df.apply(lambda row: localize_time(row['code_string'],row['date_added']), axis=1)

fmt = '%Y-%m-%d %H:%M:%S'
df['local_time_string'] = df['local_time'].apply(lambda x: x.strftime(fmt) if x != None else None)
df['local_time_string'] = df['local_time_string'].apply(lambda x: datetime.datetime.strptime(x, fmt) if x != None else None)

#print df[df['local_time_string'].isnull()].count()
#print set(list_no_timeinfo)

#find if any purchase is from China
#print df[(df['purchase'] == 1) & (df['code_string'] == "CN")].count()


# Extract hour feature
df['hour'] = df['local_time_string'].map(lambda x:x.hour)
df['weekday'] = df['local_time_string'].apply(lambda x:x.weekday())



### Add vertical type
df_webVType = pd.read_csv("../Original_Data/websiteverticleidapp.csv",header=None,index_col=None,names=['idwebsite','idVtype','idApp'],sep='\t')
df_app = pd.read_csv("../Original_Data/apps.csv",header=None, index_col=None,names=['idApp', 'idcategory', 'type', 'end_date', 'change_date', 'date_added'],sep='\t')


df_verti = df_webVType.merge(df_app,on='idApp',how='left')


def update_category(row):
    if row['idApp'] == 0:
        return row['idVtype']
    else:
        return row['idcategory']
df_verti['idcat'] = df_verti.apply(update_category, axis=1)
df_merge = df.merge(df_verti,on='idwebsite',how='left')


###Add affiliate id and type
df_affiliate_account = pd.read_csv("../Original_Data/campaign/accounts.tsv", header=None,
                         sep='\t', index_col=None,
                         names=['idaffiliate', 'aff_type', 'language', 'idcurrency', 'timezone', 'timezone_city'])


df_merge = pd.merge(df_merge,df_affiliate_account[['idaffiliate','aff_type']], how='left', on='idaffiliate')



#####Generate output
df_merge = df_merge[['hour','weekday','country_code','idoperator','iddevice','idhardware',
                     'idbrowser','idos','idcampaign','idcat','idaffiliate','aff_type','uniform_price','purchase','date_added_x','date_added_full']]
df_merge.fillna("NA", inplace=True)

df_merge.to_csv("../Data/initial-parse/campaign_testmode_week2_fullinfo.txt")




