
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
from statsmodels.distributions.empirical_distribution import ECDF
from datetime import datetime, timedelta
import seaborn.apionly as sns


# In[2]:

train = pd.read_csv("../Data/initial-parse/train_1519.txt", header=0,sep=',',index_col=False)
test = pd.read_csv("../Data/initial-parse/test_2021.txt", header=0, sep=',', index_col=False)

train['uniform_price'].fillna(0,inplace=True)
test['uniform_price'].fillna(0,inplace=True)

####Exclude aff_type = None and hour, weekday = None
train_df = train[["hour", "weekday", "country_code", "idoperator","idhardware", "idbrowser", "idos",
      "idcampaign", "idcat","idaffiliate", "aff_type","purchase", "idcampaign_diff_cvr_1", "user_id_diff_cvr_1","uniform_price","date_added_x","user_id"]].dropna()
test_df = test[["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
                "idcampaign", "idcat","idaffiliate", "aff_type","purchase", "idcampaign_diff_cvr_1", "user_id_diff_cvr_1","uniform_price","date_added_x","user_id"]].dropna()


# In[3]:

a1 = train_df[["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
                              "idcampaign", "idcat", "idaffiliate", "aff_type"]]
a2 = test_df[["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
                              "idcampaign", "idcat", "idaffiliate", "aff_type"]]
total = 0
a12 = pd.concat([train_df, test_df])

#iteritems iterates over columns, calculate the number of unique values per column (Total number of features)
for name, values in a12.iteritems():
    total += values.nunique()
print ("Total number of features: %d" %total)


# In[20]:

#print train.head()
#print train.loc[train['date_added_full'] == 2015071506].index.tolist()[0]


# In[5]:

wholeset = pd.concat([train_df, test_df])


# In[6]:

print ("The total number of clicks (including purchases) is %d" %wholeset.shape[0]) 


# In[8]:

def top_C_revenue(df):

    by_campaign = df.groupby('idcampaign')
    df['revenue'] = by_campaign['uniform_price'].transform('sum')
    print ("In the training set, there are %d campaigns in total" %df['idcampaign'].nunique())

    campaign_revenuelist = df.sort_values(['revenue'],ascending = False).drop_duplicates(['idcampaign','revenue'])

    campaign_index_list = map(float,campaign_revenuelist.reset_index().index.values)
    campaign_revenuelist['cum_sum'] = campaign_revenuelist.revenue.cumsum()
    campaign_revenuelist['cum_perc'] = campaign_revenuelist.cum_sum/campaign_revenuelist.revenue.sum()
    #print campaign_revenuelist.head()


    campaign_revenuelist['cum_perc_u'] = map(lambda x:x/float(len(campaign_index_list)), campaign_index_list)

    ##x: percentage of number of campaigns
    ##y: percentage of revenue
    x = campaign_revenuelist['cum_perc_u']
    y = campaign_revenuelist['cum_perc']

    plt.ioff()
    fig = plt.figure()
    line = plt.plot(x,y,'-', color='k')
    xvalues = line[0].get_xdata()
    yvalues = line[0].get_ydata()
    idx = np.where(yvalues >= 0.8)
    print ("%.2f%% campaigngs generated 80%% revenue" %(xvalues[idx][0] * 100))

    plt.xlabel("fraction of campaigns")
    plt.ylabel("fraction of revenue")
    plt.grid()
    plt.savefig('../Figures/0715-19/campaign_revenue_cdf.pdf')
    plt.close(fig)
    return campaign_revenuelist

campaign_revenuelist = top_C_revenue(train_df)


# In[9]:

def calculate_campaign_cvr(df):
    ######Add campaign cvr 
    df_camp_count = pd.DataFrame(df.groupby(['idcampaign','purchase']).size().reset_index(name='count'))
    for name, group in df_camp_count.groupby(['idcampaign']): 
        counts = group['count'].values
    
        if ((len(counts) == 1) & (group['purchase'].values[0] == 0)):
            cvr_h = 0
        elif ((len(counts) == 1) & (group['purchase'].values[0] == 1)):
            cvr_h = 1
        else:
            cvr_h = float(counts[1])/float(counts.sum())
    
        df.loc[(df.idcampaign==name), 'campaign_overall_cvr'] = cvr_h
        df.loc[(df.idcampaign==name), 'clicks_all'] = float(counts.sum())

    by_campaign = df.groupby('idcampaign')
    df['eCPM'] = df['revenue'] / df['clicks_all']
    campaign_ecpm_list = df.sort_values(['eCPM'],ascending = False).drop_duplicates(['idcampaign','eCPM'])
    #print campaign_ecpm_list[['idcampaign']].values[:50]
    return campaign_ecpm_list

campaign_rankby_eCPM = calculate_campaign_cvr(train_df)    


# In[ ]:

print campaign_rankby_eCPM.head()


# In[11]:

#####find top campaigns with highest ePCM and revenue
top_ecpm = campaign_rankby_eCPM[['idcampaign']].values[:50].flatten()
top_revenue = campaign_revenuelist[['idcampaign']].values[:50].flatten()
top_campaigns = list(set().union(top_ecpm,top_revenue))
print "top campaigns: ecpm + revenue"
print len(top_campaigns)

#print top_campaigns
########### Top campaigns countries 
df_big_camp = train_df.loc[train_df['idcampaign'].isin(top_revenue)]
aggregation = {
    'country_code': {
        'countries': pd.Series.nunique
    },
    'idoperator':{
        'operators': pd.Series.nunique
    },
    'idcampaign': ["count"]
}


# In[12]:

print top_revenue[:5]


# In[13]:

df_big_camp = df_big_camp.sort_values(by='revenue',ascending=False)

#####Not ranked
check_bigcampaign_global = df_big_camp[['idcampaign','country_code','idoperator','eCPM']].groupby(['idcampaign']).agg(aggregation)
check_bigcampaign_global.columns = check_bigcampaign_global.columns.droplevel()
print check_bigcampaign_global.head(100)
#print select.sort_values(by='count',ascending=False)


# In[15]:

print df_big_camp.tail(2)


# In[ ]:

def add_delta_time(group):
    stop = None
    group.sort_values(['fr_time'],ascending=True,inplace=True)
    #group['delta'] = np.zeros(group.shape[0])
    #The default delta time is 5 days (same as the training data)
    group['delta'] =  [1440*5]*group.shape[0]
    group['purchase_delta'] = [1440*5]*group.shape[0]
    #group_pur = group.loc(group['purchase']==1)
    start = None
    last_delta = 1440*5
    #print group.loc[group['purchase'] == 1]
    
    #delta: current time (click or purchase) - the previous purchase
    #purchase delta: the time between the previous two purchase
    
    for index, row in group.iterrows():
        if ((row['purchase'] == 1) & (start is not None)):
            stop = row['fr_time']
            delta = stop - start
            
            row['delta'] = int(delta.total_seconds() // 60)
            group.set_value(index,'delta',row['delta'])
            
            row['purchase_delta'] = int(delta.total_seconds() // 60)
            group.set_value(index,'purchase_delta',row['purchase_delta'])
            
            last_delta = row['purchase_delta']
            start = row['fr_time']
            #print row['delta']
        elif ((row['purchase'] == 1) & (start is None)):
            start = row['fr_time']
        else:
            if ((row['purchase'] == 0) & (start is None)):
                continue
            else:
                current = row['fr_time']
                delta = current - start
                row['delta'] = int(delta.total_seconds() // 60)
                group.set_value(index,'delta',row['delta'])
                group.set_value(index,'purchase_delta',last_delta)
      
    ###Save the last purchase delta time to add into the test file
    if group.idcampaign.values[0] not in campaign_last_delta:
        campaign_last_delta[group.idcampaign.values[0]] = last_delta
        
    if (group.idcampaign.values[0]+group.user_id.values[0]) not in campaign_user_last_delta:
        campaign_user_last_delta[group.idcampaign.values[0]+group.user_id.values[0]] = last_delta

    ###Save the last purchase time
    if (group.idcampaign.values[0]+group.user_id.values[0]) not in campaign_user_last_purchase:
        if stop is not None:
            campaign_user_last_purchase[group.idcampaign.values[0]+group.user_id.values[0]] = stop
        elif start is not None:
            campaign_user_last_purchase[group.idcampaign.values[0]+group.user_id.values[0]] = start


    ###Add average purchase time
    valid_purchase_time = group['purchase_delta'][group['purchase_delta']!=1440*5]
    if len(valid_purchase_time) == 0:
        average_purchase = 1440*5
        group['mean_purchase'] = [1440*5]*group.shape[0]
    else:
        average_purchase = np.mean(valid_purchase_time)
        group['mean_purchase'] = [np.mean(valid_purchase_time)]*group.shape[0]
        
        
    ###Save the average purchase time
    if (group.idcampaign.values[0]+group.user_id.values[0]) not in campaign_user_average_purchase:
        campaign_user_average_purchase[group.idcampaign.values[0]+group.user_id.values[0]] = average_purchase
    return group


campaign_last_delta = {}
campaign_user_last_delta = {}
campaign_user_last_purchase = {}
campaign_user_average_purchase = {}


def campaign_delta_time(df):
    ####### Find top campaign purchase delta time
    df['fr_time'] = df['date_added_x'].map(lambda x:x[:x.index('+')])
    df['fr_time'] = df['fr_time'].map(lambda x :datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    #print df.head()
    
   

    #delta: current time - previous purchase, 2880, if there's no previous purchase
    #purchase_delta: time between two purchase, 2880, for the first purchase
    
    df_add_delta_time = df.groupby(['idcampaign','user_id']).apply(add_delta_time)
    #######group by only campaign
    #df_groupby_campaign = df.groupby(['idcampaign']).apply(add_delta_time)
    
    return df_add_delta_time
   
###add delta time to only top campaigns
df_add_delta_time = campaign_delta_time(df_big_camp)


###add delta time to all the training data 
#df_add_delta_time = campaign_delta_time(train_df)


# In[38]:

print df_add_delta_time.head(1)


# In[39]:

import numpy as np
import seaborn as sns
def plot_density(delta_type, input_data, groupby_only_campaign):

    plt.ioff()
    fig = plt.figure()
    #sns.reset_orig()
    #sns.set_style("whitegrid")
    data = input_data[input_data[delta_type] != 1440*5][delta_type].values
    px = sns.kdeplot(data, bw=0.1, color="k")
    x,y = px.get_lines()[0].get_data()
    xysel = np.array([(x,y) for x,y in zip(x,y) if x > 0])
    imax = np.argmax(xysel[:,1])
    #print the (x,y) for the peaks
    print xysel[imax]

    plt.xlim((0,1440*5))
    plt.xlabel("Purchase time interval (minute)")
    plt.ylabel("Density")
    plt.grid(True)
    if groupby_only_campaign == 0:
        if delta_type == 'purchase_delta':
            plt.title = "density function purchase time interval (top campaigns)"
            plt.savefig("../Figures/0715-19/density function purchase time interval (top campaigns).pdf")
        else:
            plt.title = "density function time to last purchase (top campaigns)"
            plt.savefig("../Figures/0715-19/density function time to last purchase (top campaigns).pdf")
    else:
        if delta_type == 'purchase_delta':
            plt.title = "density function purchase time interval (top campaigns)"
            plt.savefig("../Figures/0715-19/density function purchase time interval groupby only campaign.pdf")
        else:
            plt.title = "density function time to last purchase (top campaigns)"
            plt.savefig("../Figures/0715-19/density function time to last purchase groupby only campaign.pdf")
    plt.close(fig)
    


# In[40]:

#plot only purchase interval
plot_density("purchase_delta", df_add_delta_time, 0)
#plot time to last purchase (click to purchase/purchase to purchase)
plot_density("delta", df_add_delta_time, 0)


# In[ ]:

'''
plot_density("purchase_delta", df_groupby_campaign, 1)
plot_density("delta",df_groupby_campaign,1)
'''


# In[14]:

# Plot ccdf 
import matplotlib.pyplot as plt
def plot_ccdf(delta_type):

    data = df_add_delta_time[df_add_delta_time[delta_type] != 1440*5][delta_type].values
    plt.ioff()
    fig = plt.figure()
    #sns.set_style("whitegrid")
    plt.plot(np.sort(data), 1-np.linspace(0, 1, len(data), endpoint=False))
        
    plt.ylabel("CCDF")
    plt.grid(True)
    if delta_type == 'purchase_delta':
        plt.xlabel("purchase time interval(minute)")
        plt.savefig('../Figures/0715-19/ccdf_purchase_interval_topcampaigns.pdf')

    else:
        plt.xlabel("time to last purchase(minute)")
        plt.savefig('../Figures/0715-19/ccdf_time_to_purchase_topcampaigns.pdf')
    plt.close(fig)
    
plot_ccdf("delta")
plot_ccdf("purchase_delta")


# In[ ]:

##################Processing test set


# In[41]:

print test_df.head()
test_df['fr_time'] = test_df['date_added_x'].map(lambda x:x[:x.index('+')])
test_df['fr_time'] = test_df['fr_time'].map(lambda x :datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))


# In[ ]:

##Add delta time feature as the time to the previous purchase
def add_time_to_purchase(group):
    group.sort_values(['fr_time'],ascending=True,inplace=True)
    group['delta'] =  [7200]*group.shape[0]
    group['mean_purchase'] = [7200]*group.shape[0]
    group['purchase_delta'] = [7200]*group.shape[0]
    key = group.idcampaign.values[0]+group.user_id.values[0]
    if key in campaign_user_last_purchase:
         for index, row in group.iterrows():
                row['delta'] = int((row['fr_time'] - campaign_user_last_purchase[key]).total_seconds()// 60)
                
    if key in campaign_user_last_delta:
        group['purchase_delta'] = [campaign_user_last_delta[key]]*group.shape[0]
               
    if key in campaign_user_average_purchase:
        group['mean_purchase'] =  [campaign_user_average_purchase[key]]*group.shape[0]
    return group

df_test_add_delta = test_df.groupby(['idcampaign','user_id']).apply(add_time_to_purchase)


# In[ ]:

######normalize the delta time column 
def add_decay_delta_time(df,x,y,z):
    #df_add_delta_time['norm_delta'] = (x - x.min(0)) / x.ptp(0)
    df['decay_delta'] = np.exp(-x)
    df['decay_purchase_delta'] = np.exp(-y)
    df['decay_mean'] = np.exp(-z)

    return df

df_add_delta_time = add_decay_delta_time(df_add_delta_time,df_add_delta_time['delta'],df_add_delta_time['purchase_delta'],
                 df_add_delta_time['mean_purchase'])


df_test_add_delta = add_decay_delta_time(df_test_add_delta,df_test_add_delta['delta'],df_test_add_delta['purchase_delta'],
                 df_test_add_delta['mean_purchase'])


# In[16]:

colors = cm.gist_earth(np.linspace(0, 1, len(top_revenue[:5])))
print colors


# In[51]:

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import seaborn as sns

#sns.reset_orig()
# Turn interactive plotting off
plt.ioff()

#colors = cm.cubehelix(np.linspace(0, 1, len(top_revenue[:5])+2))
fig = plt.figure()
styles = ['D','o','^','.','s']
for i, c in zip(top_revenue[:5],styles):
    day_dict = {}
    campaign = train_df.loc[train_df['idcampaign']==i]
    #campaign = train_df.loc[(train_df['idcampaign']==i)&(train_df['weekday'] == 1)]
    campaign_purchase = pd.DataFrame(campaign.groupby(['weekday','purchase']).size().reset_index(name = "count"))
    for name, group in campaign_purchase.groupby('weekday'): 
        counts = group['count'].values
        if name not in day_dict:
            day_dict[name] = []
        if ( (len(counts) == 1) & (group['purchase'].values[0] == 1)):
        
            day_dict[name].append(1)
        elif ((len(counts) == 1) & (group['purchase'].values[0] == 0)):
            #print group['purchase'].values[0]
            day_dict[name].append(0)
        else:
            day_dict[name].append(float(counts[1])/float(counts.sum()))
            
    plt.plot(day_dict.keys(),day_dict.values(),'o-',color='k',marker=c,label= i,linestyle='--',markersize=6)
        
    #pd.DataFrame(campaign_purchase.groupby(['weekday']).agg(cal_cvr).reset_index(name="cvr"))
     #g.apply(lambda x: x.order(ascending=False).head(3))
    #for d in campaign_purchase['weekday'].unique():
    #    campaign_purchase.loc[campaign_purchase['weekday']]
    #purchase = campaign_reset.loc[campaign_reset['purchase']==1].count()
    #total = campaign.count()

lgd = plt.legend(bbox_to_anchor=(0.8, 1), loc='upper right', ncol=2,prop={'size':4},borderpad=1.5, labelspacing=2,numpoints=1)
ltext  = lgd.get_texts() 
plt.setp(ltext, fontsize='medium')  
plt.grid(True)
plt.xlabel('day of week')
plt.ylabel('cvr')
figname = "../Figures/0715-19/top5_ecpmcampaign_cvr_oneweek.pdf" 
plt.savefig(figname,bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close(fig)



# In[123]:

print campaign_purchase.head(2)


# In[50]:

########### Top campaigns CVR per hour 
## Turn interactive plotting off
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
sns.reset_orig()

print top_ecpm[:5]

plt.ioff()
colors = cm.rainbow(np.linspace(0, 1, len(top_revenue[:5])))
styles = ['D','o','^','.','s']
fig = plt.figure()
for i, c in zip(top_revenue[:5],styles):
    day_dict = {}
    campaign = train_df.loc[(train_df['idcampaign']==i) & (train_df['weekday']== 3)]
    campaign_country = campaign.groupby(['idcampaign'])
    campaign_purchase = pd.DataFrame(campaign.groupby(['hour','purchase']).size().reset_index(name = "count"))
    for name, group in campaign_purchase.groupby('hour'): 
        counts = group['count'].values
        if name not in day_dict:
            day_dict[name] = []
        if ( (len(counts) == 1) & (group['purchase'].values[0] == 1)):
        
            day_dict[name].append(1)
        elif ((len(counts) == 1) & (group['purchase'].values[0] == 0)):
            print group['purchase'].values[0]
            day_dict[name].append(0)
        else:
            day_dict[name].append(float(counts[1])/float(counts.sum()))
    
    plt.plot(day_dict.keys(),day_dict.values(),'o-',color='k',marker=c,label= i,linestyle='--',markersize=6)
        
    #pd.DataFrame(campaign_purchase.groupby(['weekday']).agg(cal_cvr).reset_index(name="cvr"))
     #g.apply(lambda x: x.order(ascending=False).head(3))
    #for d in campaign_purchase['weekday'].unique():
    #    campaign_purchase.loc[campaign_purchase['weekday']]
    #purchase = campaign_reset.loc[campaign_reset['purchase']==1].count()

    #total = campaign.count()
    
plt.xlim((0,24))
plt.ylim((0,0.045))
plt.xlabel('hour of the day')
plt.ylabel('cvr')
#lgd = plt.legend(bbox_to_anchor=(1.3, 0.5), loc='upper right', ncol=3,prop={'size':5})
lgd = plt.legend(bbox_to_anchor=(0.8, 1), loc='upper right', ncol=2,prop={'size':4},borderpad=1.5, labelspacing=2,numpoints=1)
ltext  = lgd.get_texts() 
plt.setp(ltext, fontsize='medium')  
plt.grid(True)
figname = "../Figures/0715-19/top5_ecpmcampaign_cvr_oneday.pdf" 
plt.savefig(figname,bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close(fig)


# In[182]:

def top_U_revenue(df):

    by_user = df.groupby('user_id')
    df['revenue'] = by_user['uniform_price'].transform('sum')
    

    user_revenuelist = df[['user_id','revenue']].sort_values(['revenue'],ascending = False).drop_duplicates(['user_id','revenue'])
    print user_revenuelist.loc[user_revenuelist.user_id== 8414]
    user_revenuelist.drop(['user_id'], axis = 1 , inplace= True,errors= 'ignore')
    
    
    user_index_list = map(float,user_revenuelist.reset_index().index.values)
    user_revenuelist['cum_sum'] = user_revenuelist.revenue.cumsum()
    user_revenuelist['cum_perc'] = user_revenuelist.cum_sum/user_revenuelist.revenue.sum()
    #print campaign_revenuelist.head()


    user_revenuelist['cum_perc_u'] = map(lambda x:x/float(len(user_index_list)), user_index_list)

    ##x: percentage of number of campaigns
    ##y: percentage of revenue
    x = user_revenuelist['cum_perc_u']
    y = user_revenuelist['cum_perc']

    plt.ioff()
    fig = plt.figure()
    line = plt.plot(x,y,'.')
    xvalues = line[0].get_xdata()
    yvalues = line[0].get_ydata()
    idx = np.where(yvalues >= 0.8)
    print ("%.2f%% user profile generated 80%% revenue" %(xvalues[idx][0] * 100))

    plt.xlabel("fraction of user profile")
    plt.ylabel("fraction of revenue")
    plt.grid()
    plt.savefig('../Figures/0715-19/user_revenue_cdf.pdf')
    plt.close(fig)
    
    
    
    ###revenue by user_id + campaign
    by_usercampaign = df.groupby(['user_id','idcampaign'])
    df['revenue_per_campaign'] = by_usercampaign['uniform_price'].transform('sum')
    user_campaign_revenuelist = df[['user_id','idcampaign','revenue_per_campaign']].drop_duplicates(['user_id','idcampaign','revenue_per_campaign'])
    
    
    return (user_revenuelist.iloc[:idx[0][0]],user_campaign_revenuelist)

top_users_revenue,user_campaign_revenuelist = top_U_revenue(df_add_delta_time)
print user_campaign_revenuelist.head()


# In[ ]:

print top_users_revenue.shape


# In[ ]:

'''


training_data = df_add_delta_time[["hour", "weekday", "country_code", "idoperator","idhardware", "idbrowser", "idos",
                                   "idcampaign", "idcat", "idaffiliate","aff_type", "idcampaign_diff_cvr_1","user_id_diff_cvr_1","uniform_price",
                                   "purchase","decay_purchase_delta","decay_delta","decay_mean","delta","purchase_delta","mean_purchase"]]
test_data = df_test_add_delta[["hour", "weekday", "country_code", "idoperator","idhardware", "idbrowser", "idos",
                               "idcampaign", "idcat", "idaffiliate","aff_type", "idcampaign_diff_cvr_1","user_id_diff_cvr_1","uniform_price",
                               "purchase","decay_purchase_delta","decay_delta","decay_mean","delta","purchase_delta","mean_purchase"]]
                               
training_data.to_csv("../Data/add_time_delta/train_1519_6features.txt",index=False)
test_data.to_csv("../Data/add_time_delta/test_2021_6features.txt",index=False)
'''


# In[ ]:

'''
#### select top user to plot purchase time interval density
top_campaign_5 = df_big_camp.drop_duplicates(['idcampaign','eCPM']).sort(['eCPM'],ascending = False)['idcampaign'][:5]
print df_big_camp.drop_duplicates(['idcampaign','eCPM']).sort(['eCPM'],ascending = False).head(10)
i = 1
for campaignid in top_campaign_5:
    
    df_top_campaign_1 = df_groupby_campaign.loc[df_groupby_campaign.idcampaign == campaignid]
    #print df_top_campaign_1.tail(20)

    ### plot purchase time interval for top user by purchase

    import numpy as np
    import seaborn as sns
    plt.ioff()
    fig = plt.figure()
    data = df_top_campaign_1[df_top_campaign_1['purchase_delta'] != 2880]['purchase_delta'].values
    #print data
    #sns.set_style('whitegrid')
    px = sns.kdeplot(data, bw=0.1)
    x,y = px.get_lines()[0].get_data()
    xysel = np.array([(x,y) for x,y in zip(x,y) if x > 0])

    imax = np.argmax(xysel[:,1])
    print xysel[imax]

    lim_x = np.max(x) + 5
    plt.xlim((0, lim_x))
    plt.xlabel('Purchase time interval (minute)')
    plt.ylabel('Density')
    #plt.title('top user profile with most purchase rank 1')
    plt.grid(True)
    plt.savefig("density function purchase time interval top campaign " + str(i) + " " + str(campaignid))
    i += 1
    plt.close(fig)
'''


# In[160]:

####Call Top users
from TopUser import get_top_user_list 
from TopUser import plot_topuser 


# In[170]:

##### clicks include purchase 
uID_with_cvr_counts = get_top_user_list(df_add_delta_time)


#percentage_users, top_users_click = plot_topuser(uID_with_cvr_counts,"clicks",0)
#percentage_users_pur, top_users_purchase= plot_topuser(uID_with_cvr_counts,"purchase_total",0)


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
            ax1 = fig.add_subplot(211)
            line = ax1.plot(x,y,linestyle='-',color='k')
            xvalues = line[0].get_xdata()
            yvalues = line[0].get_ydata()
            idx = np.where(yvalues >= 0.8)
            ax1.set_ylabel("fraction of clicks")
            #ax1.set_xlabel("fraction of user profiles")
            sort_users_1 = sort_users
            i += 1
            ax1.grid(True)
        else:
            ax2 = fig.add_subplot(212)
            line = ax2.plot(x,y,'-',color='k')
            xvalues_2 = line[0].get_xdata()
            yvalues_2 = line[0].get_ydata()
            idx_2 = np.where(yvalues_2 >= 0.8)
            ax2.set_ylabel("fraction of purchase")
            ax2.set_xlabel("fraction of user profiles")
    
    #plt.xlabel("fraction of user profiles")
    plt.grid(True)
    plt.savefig('../Figures/0715-19/user_click_purchase_cdf.pdf')
    #if plot_type == "purchase_total":
    #    plt.ylabel("fraction of purchase")
    #else:
    #    plt.ylabel("fraction of " + plot_type)
    
    
    return (xvalues[idx][0],xvalues_2[idx_2][0], sort_users_1, sort_users)


percentage_users, percentage_users_pur, top_users_click,top_users_purchase = plot_topuser(uID_with_cvr_counts,["clicks","purchase_total"],0)


# In[165]:

print percentage_users, percentage_users_pur


# In[ ]:

######Top user cvr
uID_with_cvr_counts_sorted =  uID_with_cvr_counts.sort_values(['cvr_h'],ascending = False)
print uID_with_cvr_counts_sorted.loc[uID_with_cvr_counts_sorted.clicks > 10000].head(10)


# In[43]:

def output_purchasetime_for_topUtopC(group):
    
    group = group.sort_values(['fr_time'])
   
    purchase_time_interval = [float((t - s).total_seconds())/float(60) for s, t in zip(group['fr_time'], 
                                                                       group['fr_time'][1:])]
    purchase_time_interval.insert(0, 0)
    
    time_series = np.cumsum(purchase_time_interval)
    
    group['purchase_time_accum'] = time_series  
    
    uID = str(group['user_id'][0])
    cID = str(group['idcampaign'][0])
    
    #if (uID == "100") & (cID == "4755"):
        #print time_series
    #    print group.shape[0]
    
    
    if len(time_series) > 100:
    
        outfile = open('../Data/purchase-time-series/0715-19/all_purchase_morethan100_revenue_0729/uID' + uID + 'cID' + cID +'.txt', 'w')
        for item in time_series:
            outfile.write("%.3f\n" % item)
        
        outfile.close()
    
    return group


# In[44]:

#print top_campaigns
#topU_topC = df_add_delta_time.loc[(df_add_delta_time['user_id'].isin(top_users_purchase['user_id']))&
#                                  (df_add_delta_time['idcampaign'].isin(top_campaigns)) &
#                                  (df_add_delta_time.purchase ==1)]

#print top_users_purchase.head(1)
#topU_topC = topU_topC.groupby(['user_id','idcampaign']).apply(output_purchasetime_for_topUtopC)


topU_topC = df_add_delta_time[(df_add_delta_time.purchase ==1)].groupby(['user_id','idcampaign']).apply(output_purchasetime_for_topUtopC)


# In[45]:

print topU_topC.shape


# In[ ]:

print df_add_delta_time.head()


# In[ ]:

hawkes = [100,281,324,324,2500,8414]
hawkes_campaign = [4755, 4398, 4755, 4848, 4848, 4755]
print top_users_revenue.head(1)

top_U_revenue = top_users_revenue.drop(['user_id'], axis = 1 , errors= 'ignore')
top_U_revenue = top_U_revenue.reset_index()

#user_campaign_revenue = user_campaign_revenuelist.drop(['user_id','idcampaign'], axis = 1,errors= 'ignore')
#user_campaign_revenue = user_campaign_revenue.reset_index()

samples = zip(hawkes, hawkes_campaign)
samples = pd.DataFrame(samples, columns=['user_id', 'idcampaign'])

result = pd.merge(samples, user_campaign_revenue, how='inner')

print result


#top_users_revenue.reset_index()

#hawkes_better = df_add_delta_time.loc[(df_add_delta_time['user_id'].isin*(hawkes))&(df_add_delta_time.idcampaign==)]
#print top_users_revenue.index
print list(set(hawkes).intersection(set(top_U_revenue['user_id'].values)))
#top_users_click
#top_user_revenue
#print set(top_users_revenue.index.levels[1])



####how much revenue they generate? the user profile which fits 
#revenue_hawkes = result.loc[result['user_id'].isin(hawkes),'revenue_per_campaign'].sum()

revenue_hawkes = result.loc[result['user_id'].isin(hawkes),'revenue_per_campaign'].sum()
user_revenue_list = df_add_delta_time.drop_duplicates(['user_id','revenue'])

revenue_hawkes_user = user_revenue_list.loc[user_revenue_list['user_id'].isin(hawkes),'revenue'].sum()
totalrevenue = df_add_delta_time.drop_duplicates(['user_id','revenue'])['revenue'].sum()





print float(revenue_hawkes_user)/float(totalrevenue)

#print top_users_revenue.loc[top_users_revenue.index.levels[1].isin(hawkes)]
# df.loc[df['a'] == 1, 'b'].sum()
print revenue_hawkes


# In[ ]:

print uID_with_cvr_counts.loc[uID_with_cvr_counts.user_id == 22]


# In[ ]:

'''
for uID in top_users_purchase['user_id'].values:
    df_top_user_1 = df_add_delta_time.loc[(df_add_delta_time.user_id == 22)&(df_add_delta_time.idcampaign == 4755)]
    print df_top_user_1.head()
    print df_top_user_1[["country_code","idos","idbrowser","idoperator"]].head()

df_top_campaign_1 = df_add_delta_time.loc[df_add_delta_time.idcampaign == 4755]
#print df_top_campaign_1.tail(20)
'''


# In[ ]:

'''
### plot purchase time interval for top user by purchase

import numpy as np

plt.ioff()
fig = plt.figure()
data = df_top_user_1[df_top_user_1['purchase_delta'] != 2880]['purchase_delta'].values
print data
#sns.set_style('whitegrid')
px = sns.kdeplot(data, bw=0.1)
x,y = px.get_lines()[0].get_data()
xysel = np.array([(x,y) for x,y in zip(x,y) if x > 0])

imax = np.argmax(xysel[:,1])
print xysel[imax]

plt.xlim((0,400))
plt.xlabel('Purchase time interval (minute)')
plt.ylabel('Density')
#plt.title('top user profile with most purchase rank 1')
plt.grid(True)
plt.savefig("density function purchase time interval top user 22")
plt.close(fig)
'''


# In[ ]:

'''

##generate data for purchase time
#user_campaign_purchase = df_top_user_1.loc[df_top_user_1.purchase == 1].sort_values(['fr_time'])

##generate data for non-purchase time
user_campaign_nonpurchase = df_top_user_1.loc[df_top_user_1.purchase == 0].sort_values(['fr_time'])
print user_campaign_nonpurchase


#print user_campaign_purchase['fr_time'].head()
#user_campaign_purchase['fr_time'] = user_campaign_purchase['fr_time'].map(lambda x :datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

### count which campaign this user profile purchase the most
user_campaign_purchase_list = user_campaign_purchase[['user_id','idcampaign','purchase']].groupby(['user_id','idcampaign']).agg(['count'])
#print user_campaign_purchase_list.head(10)
'''


# In[ ]:

'''
campaign_purchase = df_top_campaign_1.loc[df_top_campaign_1.purchase == 1].sort_values(['fr_time'])
campaign_purchase_list = campaign_purchase[['idcampaign','purchase']].groupby(['idcampaign']).agg(['count'])
print campaign_purchase_list.head(10)

'''


# In[ ]:

'''


user_campaign1_purchase = user_campaign_purchase.loc[user_campaign_purchase.idcampaign == 4755].sort_values(['fr_time'])


#purchase_time_interval = [int((t - s).total_seconds()) for s, t in zip(campaign_purchase['fr_time'], 
#                                                                       campaign_purchase['fr_time'][1:])]

purchase_time_interval = [int((t - s).total_seconds()) for s, t in zip(user_campaign_nonpurchase['fr_time'], 
                                                                       user_campaign_nonpurchase['fr_time'][1:])]

print len(purchase_time_interval)
print purchase_time_interval
time_series = np.cumsum(purchase_time_interval)

print time_series[:10]
outfile = open('Data/purchase-time-series/uID22_cID4755.nonpurchase.time.txt', 'w')
for item in time_series:
    outfile.write("%d\n" % item)
outfile.close()


'''


# In[ ]:

'''



#### fit hawkes process, groupby user_id + campaign

def sort_by_time(group):
    group = group.sort_values(['fr_time'])
   
    purchase_time_interval = [int((t - s).total_seconds()) for s, t in zip(group['fr_time'], 
                                                                       group['fr_time'][1:])]
    purchase_time_interval.insert(0, 0)
    
    time_series = np.cumsum(purchase_time_interval)
    
    group['purchase_time_accum'] = time_series  
    return group
    

df_update = df_top_user_1.loc[df_top_user_1.purchase == 1].groupby(['user_id','idcampaign']).apply(sort_by_time)
update_accum_purchase_interval_bycampaign = np.sort(df_update[['purchase_time_accum']].values.flatten())

outfile = open('Data/purchase-time-series/uID43.purchase.groupbyCam.txt', 'w')
for item in update_accum_purchase_interval_bycampaign:
    outfile.write("%d\n" % item)
outfile.close()

'''


# In[184]:


#### get top cvr users
sub_cvr = uID_with_cvr_counts.loc[uID_with_cvr_counts.cvr_h != 0]
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
perc_clicks_without_purchase = float(top_6perc_users.loc[top_6perc_users.purchase_total == 0].clicks.values.sum())/float(uID_with_cvr_counts['clicks'].values.sum())


#####print top user without purchase profile
top_user_without_purchase = top_6perc_users.loc[top_6perc_users.purchase_total == 0]
#top_user_without_purchase_profile = df[df['user_id'].isin(top_user_without_purchase)][['user_id','idbrowser','idos','idoperator','country_code']].drop_duplicates()
top_user_without_purchase = pd.merge(top_user_without_purchase, train_df[['user_id','idbrowser','idos','idoperator','country_code']], on =['user_id'], how="left",left_index=True)
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


# In[235]:

#########Use Test set to generate 
print df_add_delta_time.shape
print train_df.shape
print float(df_add_delta_time.shape[0])/float(train_df.shape[0])


# In[49]:

####### Take df_add_delta's [user_id, campaign_id] where has purchase = 1 

trainingset_purchase_user = df_add_delta_time[(df_add_delta_time.purchase ==1)][['user_id','idcampaign']].drop_duplicates()


# In[50]:

test_select_purchase = pd.merge(trainingset_purchase_user,test_df,how="left",on=['user_id','idcampaign'])


# In[244]:

print test_select_purchase.head()


# In[51]:

def output_purchasetime_for_merge(group):
    group.reset_index(inplace=True)
    
    group = group.sort_values(['fr_time'])
   
    # purchase time measured by minutes
    purchase_time_interval = [float((t - s).total_seconds())/float(60) for s, t in zip(group['fr_time'], 
                                                                       group['fr_time'][1:])]
    purchase_time_interval.insert(0, 0)
    
    time_series = np.cumsum(purchase_time_interval)
    
    group['purchase_time_accum'] = time_series  
    
    uID = str(group['user_id'][0])
    cID = str(group['idcampaign'][0])
    
    #if (uID == "100") & (cID == "4755"):
        #print time_series
    #    print group.shape[0]
    
    
    #if len(time_series) > 100:
    if group.loc[group.setty==0].shape[0] > 100:
        outfile = open('../Data/purchase-time-series/0715-19//train-test-merge/uID' + uID + 'cID' + cID +'.txt', 'w')
        for index, row in group.iterrows():
            outfile.write("%.3f\t%d\n" % (row['purchase_time_accum'],row['setty']))
        
        outfile.close()
    
    return group


# In[52]:

df_add_delta_time[['fr_time','user_id','idcampaign','purchase']].index.names = ['A','B','C']
train_drop_index = df_add_delta_time.reset_index(drop=True)
sLength = train_drop_index.shape[0]
train_drop_index['setty'] = 0
test_select_purchase['setty'] = 1

print test_select_purchase[['fr_time','user_id','idcampaign','purchase','setty']].head()
train_test_merge = pd.concat([train_drop_index[['fr_time','user_id','idcampaign','purchase','setty']],
                              test_select_purchase[['fr_time','user_id','idcampaign','purchase','setty']]], axis=0)
merge_data = train_test_merge[(train_test_merge.purchase ==1)].groupby(['user_id','idcampaign']).apply(output_purchasetime_for_merge)


# In[53]:

print merge_data.shape


# In[58]:

######Read hawkes prob. files
import os
import re
files = os.listdir("../Data/purchase-time-series/0715-19/train-test-merge-add-intensity/")
path = "../Data/purchase-time-series/0715-19/train-test-merge-add-intensity/"

wholeset = pd.concat([train_drop_index[["hour", "weekday", "country_code", "idoperator",
                                                              "idhardware", "idbrowser", "idos","idcampaign", "idcat",
                                                              "idaffiliate", "aff_type","purchase", "idcampaign_diff_cvr_1", 
                                                              "user_id_diff_cvr_1","user_id", "fr_time","setty"]], test_select_purchase[["hour", "weekday", "country_code", "idoperator",
                                                              "idhardware", "idbrowser", "idos","idcampaign", "idcat",
                                                              "idaffiliate", "aff_type","purchase", "idcampaign_diff_cvr_1", 
                                                              "user_id_diff_cvr_1","user_id", "fr_time","setty"]]])


# In[62]:

hawkes_dict = {}
for filename in files[1:]:
    #filename = path + filename
    #print filename
    infile = pd.read_csv(path+filename, header=0, sep='\t', index_col=False)
    ID = re.findall('\d+', filename)
    user_id = ID[0]
    campaign = ID[1]
    infile['minutes'] = infile['minutes'].apply(lambda x: int(str(x).split('.')[0]))
    if str(user_id)+str(campaign) not in hawkes_dict:
        hawkes_dict[str(user_id)+ "c" + str(campaign)] = infile
        
        


# In[65]:

print wholeset.head(1)


# In[59]:

def add_time_to_first_purchase(group):
    print group.head(1)
    #group.reset_index(inplace = True)
    stop = None
    group.sort_values(['fr_time'],ascending=True,inplace=True)
    #group['delta'] = np.zeros(group.shape[0])
    group['delta'] =  [1440*7]*group.shape[0]
    
    #group_pur = group.loc(group['purchase']==1)
    start = None
    
    
    for index, row in group.iterrows():
        
        if ((row['purchase'] == 1) & (start is None)):
            start = row['fr_time']
            row['delta'] = 0 
       
        else:
            if ((row['purchase'] == 0) & (start is None)):
                row['delta'] = 0
                group.set_value(index,'delta',row['delta'])
                continue
            else:
                current = row['fr_time']
                delta = current - start
                if row["setty"] == 0:
                
                    row['delta'] = int(delta.total_seconds() // 60)
                    #group.set_value(index,'delta',row['delta'])
                else:
                    row['delta'] = int(delta.total_seconds() // 60) - 1
        group.set_value(index,'delta',row['delta'])
    #hawke_index = str(row["user_id"]) + "c" + str(row["idcampaign"])
    print group['user_id'].values
    new_group = pd.merge(group, hawkes_dict[row["hawkes_name"]], how="left", left_on = "delta", right_on = "minutes")
    
    return new_group


hawkes_init = pd.concat([group for _, group in wholeset.groupby(['idcampaign','user_id']) if (group['user_id']+"c"+group["idcampaign"]) in hawkes_dict.keys()])

output_file_for_LR = hawkes_init.groupby(['idcampaign','user_id']).apply(add_time_to_first_purchase)


# In[61]:

def add_time_to_first_purchase(group):
    #group.reset_index(inplace = True)
    stop = None
    group.sort_values(['fr_time'],ascending=True,inplace=True)
    #group['delta'] = np.zeros(group.shape[0])
    group['delta'] =  [1440*7]*group.shape[0]
    
    #group_pur = group.loc(group['purchase']==1)
    start = None
    
    print group.head(1)
    for index, row in group.iterrows():
        
        if ((row['purchase'] == 1) & (start is None)):
            start = row['fr_time']
            row['delta'] = 0 
       
        else:
            if ((row['purchase'] == 0) & (start is None)):
                row['delta'] = 0
                group.set_value(index,'delta',row['delta'])
                continue
            else:
                current = row['fr_time']
                delta = current - start
                if row["setty"] == 0:
                
                    row['delta'] = int(delta.total_seconds() // 60)
                    group.set_value(index,'delta',row['delta'])
                else:
                    row['delta'] = int(delta.total_seconds() // 60) - 1
                    group.set_value(index,'delta',row['delta'])
    #hawke_index = str(row["user_id"]) + "c" + str(row["idcampaign"])
    #print group['user_id'].values
    group = pd.merge(group, hawkes_dict[row["hawkes_name"]], how="left", left_on = "delta", right_on = "minutes")
    
    return group


hawkes_init = pd.concat([group for _, group in wholeset.groupby(['idcampaign','user_id']) if (group['user_id']+"c"+group["idcampaign"]) in hawkes_dict.keys()])

output_file_for_LR = hawkes_init.groupby(['user_id','idcampaign']).apply(lambda x: add_time_to_first_purchase(x))


# In[77]:

print hawkes_init['hawkes_name'].nunique()
groups = hawkes_init.groupby(['user_id','idcampaign'])
#print output_file_for_LR['hawkes_name'].values


# In[78]:

start_df="None"
for profile, group in groups: 
    print profile
    stop = None
    group.sort_values(['fr_time'],ascending=True,inplace=True)
    #group['delta'] = np.zeros(group.shape[0])
    group['delta'] =  [1440*7]*group.shape[0]
    
    #group_pur = group.loc(group['purchase']==1)
    start = None
    
    #print group.head(3)
    for index, row in group.iterrows():
        
        if ((row['purchase'] == 1) & (start is None)):
            start = row['fr_time']
            row['delta'] = 0
            row['delta_2'] = 0
            row['delta_3'] = 0
            first_purchase_ix = index    
       
        else:
            if ((row['purchase'] == 0) & (start is None)):
                row['delta'] = 0
                row['delta_2'] = 0
                row['delta_3'] = 0
                group.set_value(index,'delta',row['delta'])
                group.set_value(index,'delta_2',row['delta_2'])
                group.set_value(index,'delta_3',row['delta_3'])
                continue
            else:
                current = row['fr_time']
                delta = current - start
                if row["setty"] == 0:
                
                    row['delta'] = int(delta.total_seconds() // 60) - 1
                    row['delta_2'] = int(delta.total_seconds() // 60) - 2
                    row['delta_3'] = int(delta.total_seconds() // 60) - 3
                    #group.set_value(index,'delta',row['delta'])
                else:
                    row['delta'] = int(delta.total_seconds() // 60) - 1
                    row['delta_2'] = int(delta.total_seconds() // 60) - 2
                    row['delta_3'] = int(delta.total_seconds() // 60) - 3
        group.set_value(index,'delta',row['delta'])
        group.set_value(index,'delta_2',row['delta_2'])
        group.set_value(index,'delta_3',row['delta_3'])
    hawke_index = str(profile[0]) + "c" + str(profile[1])
    group = pd.merge(group, hawkes_dict[hawke_index], how="left", left_on = "delta", right_on = "minutes")
    group = pd.merge(group, hawkes_dict[hawke_index], how="left", left_on = "delta_2", right_on = "minutes",suffixes=('_1', '_2'))
    group = pd.merge(group, hawkes_dict[hawke_index], how="left", left_on = "delta_3", right_on = "minutes")

    #print hawkes_dict[hawke_index].head(1)
    
    if start_df == "None":
        start_df = "Start"
        last_df = group
        continue
    else:
        final_df = pd.concat([last_df,group])
        last_df = final_df


# In[81]:

final_df['intensity_2'].fillna(0,inplace=True)
final_df['intensity'].fillna(0,inplace=True)
final_df['intensity_1'].fillna(0,inplace=True)


hist,edges = np.histogram(final_df['intensity'], bins=100)
final_df['bin_3'] = np.digitize(final_df['intensity'], edges)
hist, edges = np.histogram(final_df['intensity_2'], bins=100)
final_df['bin_2'] = np.digitize(final_df['intensity_2'], edges)
hist, edges = np.histogram(final_df['intensity_1'], bins=100)
final_df['bin_1'] = np.digitize(final_df['intensity_1'], edges)


#hist, edges = np.histogram(final_df['minutes'], bins=100)
#final_df['min_bin'] = np.digitize(final_df['minutes'], edges)


# In[80]:

print final_df.head()


# In[602]:

print set(list(final_df['intensity'].values))


# In[472]:

print final_df['hawkes_name'].nunique()


# In[578]:

print final_df.loc[final_df.minutes_1.isnull()].tail()
#print final_df.loc[final_df.hawkes_name=="17354c4755"].tail()
#abc = hawkes_dict["17354c4755"].ix[hawkes_dict["17354c4755"].minutes==5039].index.values
#abc = hawkes_dict["17354c4755"].loc[[5040]]
#print abc


# In[83]:

####Write files 
import numpy as np
import pandas as pd
from lr_model import LR_predict,LR_fit_country
import scipy as sp
from scipy.sparse import coo_matrix, hstack
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score, mean_squared_error

output_LR_rep = final_df.replace([-1],[2])


#, "idcampaign_diff_cvr_1", "user_id_diff_cvr_1"
train_features = output_LR_rep.loc[final_df.setty == 0][["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
                              "idcampaign", "idcat", "idaffiliate", "aff_type"]].values

test_features = output_LR_rep.loc[final_df.setty == 1][["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
                              "idcampaign", "idcat", "idaffiliate", "aff_type"]].values




label_train = output_LR_rep.loc[output_LR_rep.setty == 0][['purchase']].values
label_test = output_LR_rep.loc[output_LR_rep.setty == 1][['purchase']].values

#train_df = train_df.astype(int)
#test_df = test_df.astype(int)
wholeset_hawke = np.concatenate((train_features,test_features),axis = 0)
#wholeset = np.asmatrix(wholeset)

#print type(wholeset)

print "started encoding"
#encoded_features = pd.get_dummies(train_df)
#print encoded_features.shape
enc = OneHotEncoder(dtype=int)
encoded_features = enc.fit_transform(wholeset_hawke)
mtx_train = encoded_features[:output_LR_rep.loc[final_df.setty == 0].shape[0],]
mtx_test = encoded_features[output_LR_rep.loc[final_df.setty == 0].shape[0]:,]





pCVR, predict_CVR, auc_score, lg_rmse = LR_predict(mtx_train,label_train,mtx_test,label_test)
print ("LR model with AUC: %.6f, RMSE: %.4f" %(auc_score, lg_rmse))
pCVR, predict_CVR, auc_score, nb_rmse = NB_predict(train_features,label_train,test_features,label_test)
#rate = float(predict_CVR)/cvr
#print ("LR model with AUC: %.4f, RMSE: %.4f, CVR_ratio: %.4f" %(auc_score, lg_rmse, rate))
print ("GaussianNB model with AUC: %.4f, RMSE: %.4f" %(auc_score, nb_rmse))






from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier(n_estimators=100, max_depth=50, min_samples_split=1, random_state=0)
label_train = np.ravel(label_train)
clf_lg = clf_RF.fit(mtx_train,label_train)
pCVR = clf_lg.predict_proba(mtx_test)
predict_CVR = np.mean(pCVR[:,1])
auc_score = roc_auc_score(label_test,pCVR[:,1])
lg_rmse = sqrt(mean_squared_error(label_test, pCVR[:,1]))
print auc_score, lg_rmse




'''
#########add binned features


print ("after add hawkes features")
train_features = output_LR_rep.loc[final_df.setty == 0][["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
                              "idcampaign", "idcat", "idaffiliate", "aff_type", "idcampaign_diff_cvr_1", "user_id_diff_cvr_1","bin_1","bin_2","bin_3"]].values

test_features = output_LR_rep.loc[final_df.setty == 1][["hour", "weekday", "country_code", "idoperator", "idhardware", "idbrowser", "idos",
                              "idcampaign", "idcat", "idaffiliate", "aff_type", "idcampaign_diff_cvr_1", "user_id_diff_cvr_1","bin_1","bin_2","bin_3"]].values




label_train = output_LR_rep.loc[output_LR_rep.setty == 0][['purchase']].values
label_test = output_LR_rep.loc[output_LR_rep.setty == 1][['purchase']].values

#train_df = train_df.astype(int)
#test_df = test_df.astype(int)
wholeset_hawke = np.concatenate((train_features,test_features),axis = 0)
#wholeset = np.asmatrix(wholeset)



enc = OneHotEncoder(dtype=int)
encoded_features = enc.fit_transform(wholeset_hawke)
mtx_train = encoded_features[:output_LR_rep.loc[final_df.setty == 0].shape[0],]
mtx_test = encoded_features[output_LR_rep.loc[final_df.setty == 0].shape[0]:,]



#prob_train = sp.sparse.csr_matrix(output_LR_rep.loc[output_LR_rep.setty == 0][["intensity"]].values)
#prob_test = sp.sparse.csr_matrix(output_LR_rep.loc[output_LR_rep.setty == 1][["intensity"]].values)

#mtx_train = hstack([mtx_train,prob_train])
#mtx_test = hstack([mtx_test,prob_test])

pCVR, predict_CVR, auc_score, lg_rmse = LR_predict(mtx_train,label_train,mtx_test,label_test)
print ("LR model with AUC: %.6f, RMSE: %.4f" %(auc_score, lg_rmse))
pCVR, predict_CVR, auc_score, nb_rmse = NB_predict(train_features,label_train,test_features,label_test)
#rate = float(predict_CVR)/cvr
#print ("LR model with AUC: %.4f, RMSE: %.4f, CVR_ratio: %.4f" %(auc_score, lg_rmse, rate))
print ("GaussianNB model with AUC: %.4f, RMSE: %.4f" %(auc_score, nb_rmse))
'''


# In[82]:

print final_df.shape


# In[611]:

from sklearn.linear_model import LogisticRegression
###print logistic regression coeffi

lg = LogisticRegression(random_state=44,penalty='l2')
label_train = np.ravel(label_train)

#start = timeit.default_timer()
clf_lg = lg.fit(mtx_train,label_train)
#stop = timeit.default_timer()
coef_lg = clf_lg.coef_
print len(coef_lg[0])


# In[606]:

from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier(n_estimators=100, max_depth=50, min_samples_split=1, random_state=0)
label_train = np.ravel(label_train)
clf_lg = clf_RF.fit(mtx_train,label_train)
pCVR = clf_lg.predict_proba(mtx_test)
predict_CVR = np.mean(pCVR[:,1])
auc_score = roc_auc_score(label_test,pCVR[:,1])
lg_rmse = sqrt(mean_squared_error(label_test, pCVR[:,1]))
print auc_score, lg_rmse


# In[555]:

from math import sqrt
prob_test_test = output_LR_rep.loc[output_LR_rep.setty == 1][["intensity"]].values
auc_score = roc_auc_score(label_test,prob_test_test)
print auc_score


# In[557]:

from NB_model import NB_predict
pCVR, predict_CVR, auc_score, nb_rmse = NB_predict(train_features,label_train,test_features,label_test)
#rate = float(predict_CVR)/cvr
#print ("LR model with AUC: %.4f, RMSE: %.4f, CVR_ratio: %.4f" %(auc_score, lg_rmse, rate))

print ("GaussianNB model with AUC: %.4f, RMSE: %.4f" %(auc_score, nb_rmse))


# In[550]:

print len(label_train)
print len(label_test)
print output_LR_rep.loc[output_LR_rep.setty == 0]['purchase'].value_counts()


# In[66]:

print len(hawkes_dict.keys())
if "281c4398" in hawkes_dict.keys():
    print "yes"
    
wholeset['hawkes_name'] = wholeset['user_id'].astype(str) + "c" + wholeset['idcampaign'].astype(str)


# In[67]:

##Select hawkes process
hawkes_init = wholeset.loc[wholeset['hawkes_name'].isin(hawkes_dict.keys())]
#hawkes_init = wholeset.groupby(['idcampaign','user_id'])


# In[68]:

print hawkes_init.shape
print wholeset.shape


# In[ ]:




# In[ ]:



