
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
import gzip

# In[91]:

# read test mode data (sample with 1 million rows)
parse_time = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


df = pd.read_csv("head_manxing_09.tsv",header=None,sep='\t',index_col=None,names=['id','idwebsite','click_id','date_added','idmarket','country_code','idgateway',
                                                                 'idoperator','idcampaign','idurl','traffic_type','iddevice','idhardware','idbrowser',
                                                                 'idtemplate','iddevicegroup','idos','price','flag_test_mode','idcurrency','date_added_full',
                                                                 'multiple_sales','identifier'],

date_parser=parse_time, parse_dates = [3])
print df.head()

##################
### Read country code list
df_countrylist = pd.read_csv("../../GitRepo/data/compare_country_code/country_list.csv",header=None,index_col=None,names=['code_string','country_code'])

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
print set(list_no_timeinfo)

#find if any purchase is from China
#print df[(df['purchase'] == 1) & (df['code_string'] == "CN")].count()


# Extract hour feature
df['hour'] = df['local_time_string'].map(lambda x:x.hour)
df['weekday'] = df['local_time_string'].apply(lambda x:x.weekday())

print df.head()