
# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
from statsmodels.distributions.empirical_distribution import ECDF


# In[9]:

def get_list(group):
    counts = group['count'].values
    if ((len(counts) == 1) & (group['purchase'].values[0] == 0)):
        cvr_h = 0
        clicks = counts[0]
        purchase_total = 0
    elif ((len(counts) == 1) & (group['purchase'].values[0] == 1)):
        cvr_h = 1
        clicks = counts[0]
        purchase_total = counts[0]
    else:
        cvr_h = float(counts[1])/float(counts.sum())
        clicks = counts.sum()
        purchase_total = counts[1]
    
    group['cvr_h'] = cvr_h
    group['purchase_total'] = purchase_total
    group['clicks'] = clicks    
    return group



def get_top_user_list(df):
    user_clicks_list = pd.value_counts(df['user_id'].values, sort=True)

    #######top profile ranked by cvr 
    df_sub_cvr = df[['user_id','purchase']]
    df_add_pur_count = pd.DataFrame(df_sub_cvr.groupby(['user_id','purchase']).size().reset_index(name='count'))

    user_click_purchase_cvr_list = df_add_pur_count.groupby(['user_id']).apply(get_list)
    user_profile_index = user_click_purchase_cvr_list[['user_id','cvr_h', 'purchase_total', 'clicks']].drop_duplicates()
    print user_profile_index.head()
    #purchase_only = user_profile_index.loc[user_profile_index.purchase_total!=0]
    return user_profile_index


 
def plot_topuser(df,plot_list,flag):
    i = 1
    for plot_type in plot_list:
        ######## user profile based click cdf
        sort_users = df.sort_values(plot_type,ascending=False)
        sort_users = sort_users.reset_index(drop=True)
    

        sort_users['cum_sum'] = sort_users[plot_type].cumsum()
        sort_users['cum_perc'] = sort_users.cum_sum/sort_users[plot_type].sum()

        index_list = map(float,sort_users.index.values)

        #sort_users['cum_sum_u'] = index_list.cumsum()
        sort_users['cum_perc_u'] = map(lambda x:x/float(len(index_list)), index_list)

        ##x: percentage of number of users, y: percentage of clicks
        x = sort_users['cum_perc_u']
        y = sort_users['cum_perc']
   
        if i == 1:
            plt.ioff()
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            line = ax1.plot(x,y,'.')
            xvalues = line[0].get_xdata()
            yvalues = line[0].get_ydata()
            idx = np.where(yvalues >= 0.8)
            #ax1.set_ylabel("fraction of clicks")
            sort_users_1 = sort_users
            i += 1
        else:
            ax2 = fig.add_subplot(122)
            line = ax2.plot(x,y,'.')
            xvalues_2 = line[0].get_xdata()
            yvalues_2 = line[0].get_ydata()
            idx_2 = np.where(yvalues_2 >= 0.8)
            #ax2.set_ylabel("fraction of purchase")
    
    plt.xlabel("fraction of user profiles")
    plt.grid(True)
    plt.savefig('../Figures/0715-19/user_click_purchase_cdf.pdf')
    #if plot_type == "purchase_total":
    #    plt.ylabel("fraction of purchase")
    #else:
    #    plt.ylabel("fraction of " + plot_type)
    
    
    return (xvalues[idx][0],xvalues_2[idx_2][0], sort_users_1, sort_users)
    
    '''
    line = plt.plot(x,y,'.')
    xvalues = line[0].get_xdata()
    yvalues = line[0].get_ydata()
    idx = np.where(yvalues >= 0.8)
    
    print ('%.4f%% of users generates 80%% of %s' %(xvalues[idx][0]*100,plot_type)) 

    #idy = np.where(xvalues >= 0.2)
    #print yvalues[idy][0]

    plt.xlabel("fraction of user profiles")
    plt.grid(True)
    if plot_type == "purchase_total":
        plt.ylabel("fraction of purchase")
    else:
        plt.ylabel("fraction of " + plot_type)
    
    if flag == 1:
        plt.savefig('../Figures/0715-19/user_' + "purchase_only" + '_cdf.pdf')
    else:
        plt.savefig('../Figures/0715-19/user_' + plot_type + '_cdf.pdf')
    
    plt.close(fig)
    
    sort_users.to_csv("../Data/top_users" + plot_type + ".txt")
  
    return (xvalues[idx][0],sort_users)
    '''


# In[ ]:




# In[ ]:

#user_no_purchase = user_click_purchase_cvr_list.loc[user_click_purchase_cvr_list.purchase_total == 0]['user_id']


# In[ ]:

#print user_click_purchase_cvr_list.head()
#user_profile_index = user_click_purchase_cvr_list[['user_id','cvr_h', 'purchase_total', 'clicks']].drop_duplicates()
#print user_profile_index.head()
#purchase_only = user_profile_index.loc[user_profile_index.purchase_total!=0]


# In[ ]:

####
#user_clicks_without_purchase = df[df['user_id'].isin(user_no_purchase)]
#print ("The users without any purchase generate %.4f %% of clicks" %(float(user_clicks_without_purchase.shape[0])/float(df.shape[0])))
#user_clicks_without_purchase_profile = user_clicks_without_purchase[['user_id','idbrowser','idos','idoperator','country_code']].drop_duplicates()
#print user_clicks_without_purchase_profile['user_id'].nunique()


# In[ ]:

#a, b = plot_topuser(purchase_only,"purchase_total", 1)


# In[ ]:

### Get top user profile
#top_user_profile = pd.merge(top_users_click,df,on=['user_id'],how='left')
#top_users_profile_purchase = top_user_profile[['user_id','idbrowser','idos','idoperator','country_code']].drop_duplicates()

#print b.head()
#print b.loc[b.purchase_total>=10].tail(20)
#print top_user_withoutany_purchase


# In[10]:



#### get top cvr users
sub_cvr = user_profile_index.loc[user_profile_index.cvr_h != 0]
sort_users_cvr = sub_cvr.sort_values('cvr_h',ascending=False)
sort_users_cvr = sort_users_cvr.reset_index(drop=True)
idx = np.where(sort_users_cvr['cvr_h'] >= 0.004)
top_cvr_users = sort_users_cvr['user_id'][:idx[0][-1]]


top_users_index = int(top_users_click.user_id.nunique() * percentage_users)
top_users_id_click = top_users_click.user_id[:top_users_index]

top_users_index_purchase = int(top_users_purchase.user_id.nunique() * percentage_users_pur)
top_users_id_purchase = top_users_purchase.user_id[:top_users_index_purchase]




#print ("top user profiles with 80%% clicks: %d" %top_users)
#top_users_id_click = top_user_list.user_id[:top_users]
#top_users_id_click = top_users_id_click.to_frame()



##sub_df = top user with most clicks without purchase
top_6perc_users = top_users_click.iloc[:top_users_index]
top_users_without_purchase = top_6perc_users.loc[top_6perc_users.purchase_total == 0].shape[0]
perc_top_without_purchase = float(top_users_without_purchase)/float(top_6perc_users.shape[0])
perc_clicks_without_purchase = float(top_6perc_users.loc[top_6perc_users.purchase_total == 0].clicks.values.sum())/float(user_profile_index['clicks'].values.sum())


#####print top user without purchase profile
top_user_without_purchase = top_6perc_users.loc[top_6perc_users.purchase_total == 0]
#top_user_without_purchase_profile = df[df['user_id'].isin(top_user_without_purchase)][['user_id','idbrowser','idos','idoperator','country_code']].drop_duplicates()
top_user_without_purchase = pd.merge(top_user_without_purchase, df[['user_id','idbrowser','idos','idoperator','country_code']], on =['user_id'], how="left",left_index=True)
top_user_without_purchase_profile = top_user_without_purchase[['user_id','idbrowser','idos','idoperator','country_code','clicks']].drop_duplicates().sort_values('clicks',ascending=False)
top_user_without_purchase_profile.to_csv("top_users_without_purchase.txt",index=False)


print ('%.2f%% out of top users with most clicks have no purchase generate %.2f%% of total clicks' 
       %(perc_top_without_purchase*100,perc_clicks_without_purchase*100))

diff_clicks = list(set(top_users_id_click) - set(top_users_id_purchase))
diff_purchase = list(set(top_users_id_purchase) - set(top_users_id_click))
diff_click_purchase = list(set(top_users_id_click) - set(top_cvr_users))
print len(diff_clicks)
perc_top_click_top_purchase = float(len(diff_clicks))/float(top_6perc_users.shape[0])
print ('%.2f%% out of top users with most clicks are also with most purchase' 
       %(perc_top_click_top_purchase*100))

print len(diff_purchase)
perc_top_purchase_not_in_top_click = float(len(diff_purchase))/float(len(set(top_users_id_purchase)))
print ('%.2f%% out of top users with most purchase are not in the top click user set' 
       %(perc_top_purchase_not_in_top_click*100))

## More purchase means more cvr?
print len(set(top_users_id_purchase))




# In[ ]:

#top_user_cvr_profile = pd.merge(top_user_cvr,user_profile_index,on=['user_id'],how='left')
#print top_user_cvr_profile
'''
plt.ioff()
fig = plt.figure()
plt.plot(sort_users_cvr.index.values,sort_users_cvr['cvr_h'],'.')
plt.xlabel("user_id")
plt.ylabel("cvr")
plt.grid()
plt.savefig('user_cvr')
plt.close(fig)
'''

