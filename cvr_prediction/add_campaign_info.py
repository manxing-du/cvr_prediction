
# coding: utf-8

# In[45]:

import pandas as pd

#df = pd.read_csv("../../../iPinyou/make-ipinyou-data-master/all/test.log.txt",header=0,sep='\t',index_col=False)
campaign_info = pd.read_csv("../../GitRepo/data/campaign/cds_singletransactions_campaigns_segments.tsv", 
                            header=None, sep='\t',index_col=0, names=['id','idcampaign','campaign_type','id_partner',
                                                                  'id_application','id_vertical_type', 'id_country', 
                                                                  'id_operator','id_os','id_browser','exclude', 'id_group'])
df = pd.read_csv("../../GitRepo/format_data/campaign_testmode_0715_withprice_correct.txt", header=0, sep=',',index_col=0)



# In[30]:

print df.shape[0]


# In[46]:

campaign_info_delete = pd.read_csv("../../GitRepo/data/campaign/cds_singletransactions_campaigns_segments_deleted.csv", 
                            header=None, sep=',',index_col=0, names=['id','code','name','idapplication','idpartner',
                                                                     'idcountry','idhardware','operators','device_groups',
                                                                     'os','test_results', 'behaviour', 'status','filter_details',
                                                                     'flag_sensitive','billing_type','optin_type', 'flag_exclusive',
                                                                     'flag_exclusive_type','flag_exclusive_affiliate_type','flag_leads_type',
                                                                     'leads_number','enabled_hours','disable_date','multiple_sales','deny_country',
                                                                     'campaign_type','change_date','segments_type','original_id','geoedge_frequency'])


# In[61]:

a = campaign_info_delete.reset_index()
a = a[['id','idapplication','idpartner']].drop_duplicates()
a.columns = ['idcampaign','id_application','id_partner']
print a.head()


# In[37]:

print campaign_info_delete.loc[campaign_info_delete.index == 4755]


# In[78]:

print df["idcampaign"].nunique()


# In[48]:

campaign_list = campaign_info[['idcampaign','id_partner','id_application']].drop_duplicates()


# In[40]:

print campaign_list.head()


# In[64]:

df_merge = pd.merge(df, campaign_list[['idcampaign','id_partner','id_application']], how="left", on = ['idcampaign'])


# In[77]:

print df_merge.head()
set1 = df_merge[np.isnan(df_merge['id_partner'])]['idcampaign'].unique()
df_merge_1 = pd.merge(df, a[['idcampaign','id_partner','id_application']], how="left", on = ['idcampaign'])
set2 = df_merge_1[np.isnan(df_merge_1['id_partner'])]['idcampaign'].unique()

print len(set1)
i=0
for x in set1:
    if x in set2:
        i += 1
print i


# In[66]:

import numpy as np
df_merge_delete = pd.merge(df_merge, a[['idcampaign','id_application','id_partner']], how="left", on = ['idcampaign'],left_index=True, right_index=True)
print df_merge_delete.head()
df_merge_delete[np.isnan(df_merge_delete['id_partner'])]


# In[ ]:

df_merge.to_csv("campaign_tesmost_0715_withmorecampaigninfo.csv")

