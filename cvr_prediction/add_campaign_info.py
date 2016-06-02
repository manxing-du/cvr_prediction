
# coding: utf-8

# In[1]:

import pandas as pd

#df = pd.read_csv("../../../iPinyou/make-ipinyou-data-master/all/test.log.txt",header=0,sep='\t',index_col=False)
campaign_info = pd.read_csv("../../GitRepo/data/campaign/cds_singletransactions_campaigns_segments.tsv", 
                            header=0, sep='\t',index_col=0, names=['id','idcampaign','campaign_type','id_partner',
                                                                  'id_application','id_vertical_type', 'id_country', 
                                                                  'id_operator','id_os','id_browser','exclude', 'id_group'])

df = pd.read_csv("../../GitRepo/format_data/campaign_testmode_0715_withprice_correct.txt", header=0, sep=',',index_col=0)



# In[6]:

print df.head()
print campaign_info.head()
print campaign_info.shape


# In[11]:

print campaign_info[campaign_info["idcampaign"] == 4044]


# In[9]:

print campaign_info["id_application"].nunique()


# In[ ]:

df_merge = pd.merge(df, campaign_info[['idcampaign','id_partner','id_application','id_country','id_operator','id_os',
                                     'id_browser','exclude']], how="left", on = ['idcampaign'])


# In[ ]:

df_merge.to_csv("campaign_tesmost_0715_withmorecampaigninfo.csv")


# In[ ]:



