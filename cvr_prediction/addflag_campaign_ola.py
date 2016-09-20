
# coding: utf-8

# In[3]:

import pandas as pd

df = pd.read_csv("campaign_testmode_week1_fullinfo.txt",header=0,sep=',',index_col=0)
df_test = pd.read_csv("campaign_testmode_week2_fullinfo.txt", header=0, sep=',', index_col=0)
#sort_campaign = pd.value_counts(df['idcampaign'].values, sort=True)


# In[5]:

df_2days = pd.concat([df, df_test])


# In[6]:

#####user profile encoding
df_user_profile = df_2days[['idbrowser','idos','idoperator','country_code']].drop_duplicates().reset_index()
df_user_profile.head()


# In[7]:

df_user_profile.rename(columns={'index':'user_id'}, inplace=True)


# In[8]:

df_add_userid = pd.merge(df_2days, df_user_profile, on=['country_code','idoperator','idbrowser','idos'], how='left')


# In[9]:

###############Campaign cvr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

#df_sub_cvr = df_add_userid.ix[:,df_add_userid.columns != 'date_added_full']
df_sub_cvr = df_add_userid
flag_type = ["idcampaign","user_id"]


def cvr_h(group):
    counts = group['count'].values
    if ((len(counts) == 1) & (group['purchase'].values[0] == 0)):
        cvr_h = 0
    elif ((len(counts) == 1) & (group['purchase'].values[0] == 1)):
        cvr_h = 1
    else:
        cvr_h = float(counts[1])/float(counts.sum())
    group['cvr_h'] = cvr_h
    return group    


def add_cvr_change_flag(col):
    df_add_features = pd.DataFrame(df_sub_cvr.groupby([col,'weekday','hour','purchase']).size().reset_index(name='count'))
    df_add_features = df_add_features.groupby([col,'weekday','hour']).apply(cvr_h)
    print df_add_features.shape[0]
    df_add_userid = pd.merge(df_sub_cvr, df_add_features[[col,'weekday','hour','cvr_h']].drop_duplicates(), how ='left',
                             on=[col,'weekday','hour'])
    # compare the cvr_difference for each vertical type


    cat_cvr_diff_df = df_add_userid.drop_duplicates([col,'weekday','hour'])[[col,'weekday','hour','cvr_h']]
    #print catcvr_diff_df.shape[0]
    #print cvr_diff_df.head()
  
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
            col_name = col +'_diff_cvr_' + str(i)
            group[col_name] = pd.Series(diff_cvr[:-i], index=group.index)
            #group['cvr_ratio'] = pd.Series(ratio_cvr[:-1], index=group.index)
    
        return group


    cat_cvr_diff_result = cat_cvr_diff_df.groupby([col]).apply(myfunc)
    print cat_cvr_diff_result.head()
    return cat_cvr_diff_result
    
    


for item in flag_type:
    cat_cvr_diff_result = add_cvr_change_flag(item)
    df_sub_cvr = pd.merge(df_sub_cvr, cat_cvr_diff_result[[item,'weekday','hour', item + '_diff_cvr_1']], 
                    how='left', on=[item,'weekday','hour'])


# In[10]:

#print df_add_features.loc[(df_add_features.user_id == 1) & (df_add_features.hour == 17)]
print df_sub_cvr.head()


# In[11]:

print df_sub_cvr.tail()
print df_sub_cvr.shape[0]

df_print_train = df_sub_cvr[:df.shape[0]]
df_print_test = df_sub_cvr[df.shape[0]:]


# In[12]:

print df_print_train.shape[0]
print df_print_test.shape[0]


# In[13]:

df_print_train.to_csv('ola_dataset/campaign_testmode_week1_uflag_cflag.csv')
df_print_test.to_csv('ola_dataset/campaign_testmode_week2_uflag_cflag.csv')

