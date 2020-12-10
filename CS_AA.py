import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib.pylab as plt

pd.set_option('display.max_columns', 25)


# ------ Define functions ------
def run_kmeans(n_clusters_f, init_f, df_f):
    k_means_model_f = KMeans(n_clusters=n_clusters_f,init=init_f).fit(df_f)
    df_f['predict_cluster_kmeans'] = k_means_model_f.labels_
    df_f['predict_cluster_kmeans'] = df_f['predict_cluster_kmeans'].astype(int)

    # summarize cluster attributes
    k_means_model_f_summary = df_f.groupby('predict_cluster_kmeans').agg(attribute_summary_method_dict)
    return k_means_model_f, k_means_model_f_summary    

# ------ Import data ------

df = pd.read_csv('final data.csv').fillna(0)

df['tier'] = df['tier'].astype(str)
df['payment_period'] = df['payment_period'].astype(str)
categorical_columns= df[['payment_period','tier','preferred_genre','intended_use','male_TF','attribution_survey','attribution_technical','package_type','current_sub_TF']]
dum_df = pd.get_dummies(categorical_columns)
pd.set_option('display.max_columns', None)

new_df = df.join(dum_df)
clean_df = new_df.drop(['F1', 'country','op_sys','months_per_bill_period','language'], axis='columns')

# ------ RUN CLUSTERING -----
# --- set parameters
n_clusters = 3
init_point_selection_method = 'k-means++'

# --- select data
cols_for_clustering = ['payment_period_0','payment_period_1','payment_period_2','payment_period_3','num_weekly_services_utilized','weekly_consumption_hour','num_ideal_streaming_services','join_fee','age','monthly_price','preferred_genre_0','preferred_genre_comedy','preferred_genre_drama','preferred_genre_international','preferred_genre_other','preferred_genre_regional','intended_use_0','intended_use_access to exclusive content','intended_use_education','intended_use_expand international access','intended_use_expand regional access','intended_use_other','intended_use_replace OTT','intended_use_supplement OTT','male_TF_0','male_TF_伪','male_TF_真','attribution_survey_0','attribution_survey_affiliate','attribution_survey_amfm_radio','attribution_survey_direct_mail','attribution_survey_facebook','attribution_survey_facebook_organic','attribution_survey_other','attribution_survey_ott','attribution_survey_pinterest','attribution_survey_podcast','attribution_survey_public_radio','attribution_survey_referral','attribution_survey_search','attribution_survey_sirius xm','attribution_survey_social_organic','attribution_survey_tv','attribution_survey_youtube','package_type_0','package_type_base','package_type_economy','package_type_enhanced','current_sub_TF_伪','current_sub_TF_真','tier_0.0','tier_1.0','tier_2.0','tier_3.0','tier_4.0','tier_5.0','tier_6.0','tier_7.0','tier_8.0','attribution_technical_affiliate','attribution_technical_appstore','attribution_technical_bing','attribution_technical_bing_organic','attribution_technical_brand sem intent bing','attribution_technical_brand sem intent google','attribution_technical_content_greatist','attribution_technical_criteo','attribution_technical_direct_mail','attribution_technical_discovery','attribution_technical_display','attribution_technical_email','attribution_technical_email_blast','attribution_technical_facebook','attribution_technical_facebook_organic','attribution_technical_google_organic','attribution_technical_influencer','attribution_technical_internal','attribution_technical_organic','attribution_technical_other','attribution_technical_ott','attribution_technical_pinterest','attribution_technical_pinterest_organic','attribution_technical_playstore','attribution_technical_podcast','attribution_technical_quora','attribution_technical_referral','attribution_technical_samsung','attribution_technical_search','attribution_technical_tv','attribution_technical_twitter','attribution_technical_vod','attribution_technical_youtube']
df_cluster = clean_df.loc[:, cols_for_clustering]

# --- split to test and train
df_cluster_train, df_cluster_test, _, _, = train_test_split(df_cluster, [1]*df_cluster.shape[0], test_size=0.33)   # ignoring y values for unsupervised

# --- fit model
attribute_summary_method_dict = {'payment_period_0':sum,'payment_period_1':sum,'payment_period_2':sum,'payment_period_3':sum,'num_weekly_services_utilized':np.mean,'weekly_consumption_hour':np.mean,'join_fee':np.mean,'num_ideal_streaming_services':np.mean,'age':np.mean,'monthly_price':np.mean,'preferred_genre_0':sum,'preferred_genre_comedy':sum,'preferred_genre_drama':sum,'preferred_genre_international':sum,'preferred_genre_other':sum,'preferred_genre_regional':sum,'intended_use_0':sum,'intended_use_access to exclusive content':sum,'intended_use_education':sum,'intended_use_expand international access':sum,'intended_use_expand regional access':sum,'intended_use_other':sum,'intended_use_replace OTT':sum,'intended_use_supplement OTT':sum,'male_TF_0':sum,'male_TF_伪':sum,'male_TF_真':sum,'attribution_survey_0':sum,'attribution_survey_affiliate':sum,'attribution_survey_amfm_radio':sum,'attribution_survey_direct_mail':sum,'attribution_survey_facebook':sum,'attribution_survey_facebook_organic':sum,'attribution_survey_other':sum,'attribution_survey_ott':sum,'attribution_survey_pinterest':sum,'attribution_survey_podcast':sum,'attribution_survey_public_radio':sum,'attribution_survey_referral':sum,'attribution_survey_search':sum,'attribution_survey_sirius xm':sum,'attribution_survey_social_organic':sum,'attribution_survey_tv':sum,'attribution_survey_youtube':sum,'package_type_0':sum,'package_type_base':sum,'package_type_economy':sum,'package_type_enhanced':sum,'current_sub_TF_伪':sum,'current_sub_TF_真':sum,'tier_0.0':sum,'tier_1.0':sum,'tier_2.0':sum,'tier_3.0':sum,'tier_4.0':sum,'tier_5.0':sum,'tier_6.0':sum,'tier_7.0':sum,'tier_8.0':sum,'attribution_technical_affiliate':sum,'attribution_technical_appstore':sum,'attribution_technical_bing':sum,'attribution_technical_bing_organic':sum,'attribution_technical_brand sem intent bing':sum,'attribution_technical_brand sem intent google':sum,'attribution_technical_content_greatist':sum,'attribution_technical_criteo':sum,'attribution_technical_direct_mail':sum,'attribution_technical_discovery':sum,'attribution_technical_display':sum,'attribution_technical_email':sum,'attribution_technical_email_blast':sum,'attribution_technical_facebook':sum,'attribution_technical_facebook_organic':sum,'attribution_technical_google_organic':sum,'attribution_technical_influencer':sum,'attribution_technical_internal':sum,'attribution_technical_organic':sum,'attribution_technical_other':sum,'attribution_technical_ott':sum,'attribution_technical_pinterest':sum,'attribution_technical_pinterest_organic':sum,'attribution_technical_playstore':sum,'attribution_technical_podcast':sum,'attribution_technical_quora':sum,'attribution_technical_referral':sum,'attribution_technical_samsung':sum,'attribution_technical_search':sum,'attribution_technical_tv':sum,'attribution_technical_twitter':sum,'attribution_technical_vod':sum,'attribution_technical_youtube':sum}
col_output_order = ['payment_period_0','payment_period_1','payment_period_2','payment_period_3','num_weekly_services_utilized','weekly_consumption_hour','num_ideal_streaming_services','age','join_fee','monthly_price','preferred_genre_0','preferred_genre_comedy','preferred_genre_drama','preferred_genre_international','preferred_genre_other','preferred_genre_regional','intended_use_0','intended_use_access to exclusive content','intended_use_education','intended_use_expand international access','intended_use_expand regional access','intended_use_other','intended_use_replace OTT','intended_use_supplement OTT','male_TF_0','male_TF_伪','male_TF_真','attribution_survey_0','attribution_survey_affiliate','attribution_survey_amfm_radio','attribution_survey_direct_mail','attribution_survey_facebook','attribution_survey_facebook_organic','attribution_survey_other','attribution_survey_ott','attribution_survey_pinterest','attribution_survey_podcast','attribution_survey_public_radio','attribution_survey_referral','attribution_survey_search','attribution_survey_sirius xm','attribution_survey_social_organic','attribution_survey_tv','attribution_survey_youtube','package_type_0','package_type_base','package_type_economy','package_type_enhanced','current_sub_TF_伪','current_sub_TF_真','tier_0.0','tier_1.0','tier_2.0','tier_3.0','tier_4.0','tier_5.0','tier_6.0','tier_7.0','tier_8.0','attribution_technical_affiliate','attribution_technical_appstore','attribution_technical_bing','attribution_technical_bing_organic','attribution_technical_brand sem intent bing','attribution_technical_brand sem intent google','attribution_technical_content_greatist','attribution_technical_criteo','attribution_technical_direct_mail','attribution_technical_discovery','attribution_technical_display','attribution_technical_email','attribution_technical_email_blast','attribution_technical_facebook','attribution_technical_facebook_organic','attribution_technical_google_organic','attribution_technical_influencer','attribution_technical_internal','attribution_technical_organic','attribution_technical_other','attribution_technical_ott','attribution_technical_pinterest','attribution_technical_pinterest_organic','attribution_technical_playstore','attribution_technical_podcast','attribution_technical_quora','attribution_technical_referral','attribution_technical_samsung','attribution_technical_search','attribution_technical_tv','attribution_technical_twitter','attribution_technical_vod','attribution_technical_youtube']

# training data
train_model, train_model_summary = run_kmeans(n_clusters, init_point_selection_method, df_cluster_train.reindex())
# testing data
test_model, test_model_summary = run_kmeans(n_clusters, init_point_selection_method, df_cluster_test.reindex())
# all data
model, model_summary = run_kmeans(n_clusters, init_point_selection_method, df_cluster)

# --- run for various number of clusters
##### add the code to run the clustering algorithm for various numbers of clusters
wcss = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i, init=init_point_selection_method).fit(df_cluster)
    wcss.append(kmeans.inertia_)

# --- draw elbow plot
##### create an elbow plot for your numbers of clusters in previous step
plt.plot(range(1, 10), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

genre_col_names = ['preferred_genre_0','preferred_genre_comedy','preferred_genre_drama','preferred_genre_international','preferred_genre_other','preferred_genre_regional']
df_cluster['preferred_genre'] = None
for t_col in genre_col_names:
    df_cluster.loc[df_cluster[t_col] == 1, 'preferred_genre'] = t_col.split('_')[2]

payment_period_col_names = ['payment_period_0','payment_period_1','payment_period_2','payment_period_3']
df_cluster['payment_period'] = None
for t_col in payment_period_col_names:
    df_cluster.loc[df_cluster[t_col] == 1, 'payment_period'] = t_col.split('_')[2]

intended_use_col_names = ['intended_use_0','intended_use_access to exclusive content','intended_use_education','intended_use_expand international access','intended_use_expand regional access','intended_use_other','intended_use_replace OTT','intended_use_supplement OTT']
df_cluster['intended_use'] = None
for t_col in intended_use_col_names:
    df_cluster.loc[df_cluster[t_col] == 1, 'intended_use'] = t_col.split('_')[2]

gender_col_names = ['male_TF_0','male_TF_伪','male_TF_真']
df_cluster['gender'] = None
for t_col in gender_col_names:
    df_cluster.loc[df_cluster[t_col] == 1, 'gender'] = t_col.split('_')[2]

attribution_survey_col_names = ['attribution_survey_0','attribution_survey_affiliate','attribution_survey_amfm_radio','attribution_survey_direct_mail','attribution_survey_facebook','attribution_survey_facebook_organic','attribution_survey_other','attribution_survey_ott','attribution_survey_pinterest','attribution_survey_podcast','attribution_survey_public_radio','attribution_survey_referral','attribution_survey_search','attribution_survey_sirius xm','attribution_survey_social_organic','attribution_survey_tv','attribution_survey_youtube']
df_cluster['attribution_survey'] = None
for t_col in attribution_survey_col_names:
    df_cluster.loc[df_cluster[t_col] == 1, 'attribution_survey'] = t_col.split('_')[2]

package_type_col_names = ['package_type_0','package_type_base','package_type_economy','package_type_enhanced']
df_cluster['package_type'] = None
for t_col in package_type_col_names:
    df_cluster.loc[df_cluster[t_col] == 1, 'package_type'] = t_col.split('_')[2]

current_sub_col_names = ['current_sub_TF_伪','current_sub_TF_真']
df_cluster['current_sub'] = None
for t_col in current_sub_col_names:
    df_cluster.loc[df_cluster[t_col] == 1, 'current_sub'] = t_col.split('_')[3]

tier_col_names = ['tier_0.0','tier_1.0','tier_2.0','tier_3.0','tier_4.0','tier_5.0','tier_6.0','tier_7.0','tier_8.0']
df_cluster['tier'] = None
for t_col in tier_col_names:
    df_cluster.loc[df_cluster[t_col] == 1, 'tier'] = t_col.split('_')[1][0]

attribution_technical_col_names = ['attribution_technical_affiliate','attribution_technical_appstore','attribution_technical_bing','attribution_technical_bing_organic','attribution_technical_brand sem intent bing','attribution_technical_brand sem intent google','attribution_technical_content_greatist','attribution_technical_criteo','attribution_technical_direct_mail','attribution_technical_discovery','attribution_technical_display','attribution_technical_email','attribution_technical_email_blast','attribution_technical_facebook','attribution_technical_facebook_organic','attribution_technical_google_organic','attribution_technical_influencer','attribution_technical_internal','attribution_technical_organic','attribution_technical_other','attribution_technical_ott','attribution_technical_pinterest','attribution_technical_pinterest_organic','attribution_technical_playstore','attribution_technical_podcast','attribution_technical_quora','attribution_technical_referral','attribution_technical_samsung','attribution_technical_search','attribution_technical_tv','attribution_technical_twitter','attribution_technical_vod','attribution_technical_youtube']
df_cluster['attribution_technical'] = None
for t_col in attribution_technical_col_names:
    df_cluster.loc[df_cluster[t_col] == 1, 'attribution_technical'] = t_col.split('_')[2]


convert_TF = []
for i in range(0,191489):
    if df_cluster.iloc[i]['payment_period'] == '0' and df_cluster.iloc[i]['current_sub'] == '真':
        convert_TF.append('')
    elif df_cluster.iloc[i]['payment_period'] == '0' and df_cluster.iloc[i]['current_sub'] == '伪':
        convert_TF.append(False)
    else:
        convert_TF.append(True)

df_cluster['convert_TF'] = convert_TF

df_cluster.to_csv('final_project_clustering_output.csv')

import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib.pylab as plt

df = pd.read_csv('final_project_clustering_output.csv')

import json
df = df[['convert_TF','attribution_survey','attribution_technical','tier','join_fee','monthly_price','payment_period','predict_cluster_kmeans']]
df = df.loc[df['attribution_survey'].isin(['bing', 'display', 'facebook', 'search', 'youtube']) | df['attribution_technical'].isin(['bing', 'display', 'facebook', 'search', 'youtube'])]
channel_spend = pd.read_csv('channel_spend_undergraduate.csv')
json.loads(channel_spend['spend'][0].replace("'", '"'))

channel_spend_dict = {}
tier_list = ['tier1', 'tier2', 'tier3','tier4', 'tier5', 'tier6','tier7', 'tier8']
for i in range(8):
    channel_spend_dict[tier_list[i]] = json.loads(channel_spend['spend'][i].replace("'", '"'))
# channel_spend_dict
channel_spend = channel_spend_dict


# ----- Set parameters -----
touch_col_prepend = 'touch'
direct_label = 'direct'
first_weight = 0.4
last_weight = 0.4
cred_col_post_pend = '_credit'
select_model_types = ['attribution_survey','attribution_technical']
write_to_file = True

# total spending for all 8 tier experiments
channel_spend['total'] = dict()
for t_name, t in channel_spend.items():
    if t_name != 'total':
        for c in t.keys():
            try:
                channel_spend['total'][c] = channel_spend['total'][c] + t[c]
            except KeyError:
                channel_spend['total'].update({c: 0})

# ----- Format dataframe -----
# --- create credit columns
cred_col_names = ['bing_credit','display_credit','facebook_credit','search_credit','youtube_credit']
#df = pd.concat([df, pd.DataFrame(data=0, columns=cred_col_names, index=df.index)], axis=1, ignore_index=False)

df = df.reset_index()

from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

Counter(list(df['attribution_survey']))

Counter(list(df['attribution_technical']))

df[df.convert_TF==True].attribution_survey.value_counts()
df = df[df.tier != 0]
df['convert_TF'] = df['convert_TF'].astype(bool)
df.loc[df['convert_TF']==True]

def assign_credit(t_row, cred_col_names_f, cred_col_post_pend_f, model_type_f, first_weight_f=0.5, last_weight_f=0.5):
    # function assigns a credit to each relevant channel based on user specified model type, e.g. "last_touch_point", "first_touch_point", etc.
    t_dict = dict(zip(cred_col_names_f, [0]*len(cred_col_names_f)))

    if model_type_f == 'attribution_survey':
        if t_row['attribution_survey'] == 'bing' or t_row['attribution_survey'] == 'display'or t_row['attribution_survey'] ==  'facebook' or t_row['attribution_survey'] == 'search' or t_row['attribution_survey'] == 'youtube':
            t_dict.update({t_row['attribution_survey'] + cred_col_post_pend_f: 1})
        return t_dict
    if model_type_f == 'attribution_technical':
        if t_row['attribution_technical'] == 'bing' or t_row['attribution_survey'] == 'display'or t_row['attribution_survey'] ==  'facebook' or t_row['attribution_survey'] == 'search' or t_row['attribution_survey'] == 'youtube':
            t_dict.update({t_row['attribution_technical'] + cred_col_post_pend_f: 1})
        return t_dict
    
def get_attribution_by_channel(df_f, credit_col_postpend_f):
    allocated_conversions = df_f[cred_col_names].sum()
    print(allocated_conversions)
    n_allocated_conversions = df_f[cred_col_names].sum().sum()
    print(n_allocated_conversions)
    n_total_conversions = df_f['convert_TF'].sum()
    channel_allocation_f = pd.Series(dict(zip([x.split(credit_col_postpend_f)[0] for x in allocated_conversions.keys()], list(allocated_conversions.array))))
    return channel_allocation_f


def calc_avg_CAC(channel_allocation_f, channel_spend_f):
    t_df = pd.DataFrame(channel_allocation_f)
    t_df.columns = ['channel_allocation']
    for t_ind, _ in t_df.iterrows():
        t_df.loc[t_ind, 'channel_spend'] = channel_spend_f[t_ind]

    t_df['CAC'] = t_df['channel_spend'] / t_df['channel_allocation']
    t_df['CAC'].replace(np.inf, 0, inplace=True)
    return t_df

def calc_marginal_CAC(n_conversions_low_tier, spend_low_tier, n_conversions_high_tier, spend_high_tier):
    ##### fill in this code to create the three variables in output dictionary
    marginal_conversions = n_conversions_high_tier - n_conversions_low_tier
    marginal_spend = spend_high_tier - spend_low_tier
    marginal_CAC = marginal_spend/marginal_conversions
    return {'marginal_conversions': marginal_conversions, 'marginal_spend': marginal_spend,
            'marginal_CAC': marginal_CAC}

df.convert_TF = df.convert_TF.fillna(False)
# df = df.sample(n=5000)
# ----- RUN MODELS -----
CAC_dfs = dict()
for model_type in select_model_types:
    print('Processing model %s' % model_type)

    # ----- Run attribution model -----
    print('Running attribution model')
    df_convert = df.loc[df['convert_TF']==True] # only run calculation for conversion rows
    info_to_add = list()
    for t_ind, t_row in df_convert.iterrows():
        t_credit_dict = assign_credit(t_row, cred_col_names, cred_col_post_pend, model_type, first_weight, last_weight)
        info_to_add.append(t_credit_dict)
    df_convert = pd.concat([df_convert.reset_index(drop=True), pd.DataFrame(info_to_add).reset_index(drop=True)], axis=1)
    #df.loc[t_ind, list(t_credit_dict.keys())] = list(t_credit_dict.values())  # add credit to original dataframe

    # ----- Calculate CAC -----
    print('Calculating average and marginal CAC')
    # --- Average CAC ---
    channel_allocation = get_attribution_by_channel(df_convert, credit_col_postpend_f='_credit')
    df_CAC = calc_avg_CAC(channel_allocation, channel_spend['total'])
    
    # --- Marginal CAC ---
    credit_cols = [x for x in df_convert.columns if x.find('credit') > -1]
    df_CAC = pd.DataFrame(index=[x.split('_credit')[0] for x in credit_cols])
    base_col_names = ['marginal_conversions', 'marginal_spend', 'marginal_CAC']

    df_tier_sum = df_convert[['tier']+credit_cols].groupby(['tier']).sum()
    df_tier_sum.columns = [x.split('_credit')[0] for x in df_tier_sum.columns]
    for t_tier in df_tier_sum.index:
        for t_channel in df_CAC.index:
            if t_tier > 1:
                n_conversions_low_tier = df_tier_sum.loc[t_tier - 1, t_channel]
                spend_low_tier = channel_spend['tier' + str(t_tier - 1)][t_channel]
                n_conversions_high_tier = df_tier_sum.loc[t_tier, t_channel]
                spend_high_tier = channel_spend['tier' + str(t_tier)][t_channel]
            else:
                n_conversions_low_tier = 0
                spend_low_tier = 0
                n_conversions_high_tier = df_tier_sum.loc[t_tier, t_channel]
                spend_high_tier = channel_spend['tier' + str(t_tier)][t_channel]

            t_df_CAC_colnames = [x + '_t' + str(t_tier) for x in base_col_names]
            for i in t_df_CAC_colnames:
                if i not in list(df_CAC.columns):
                    df_CAC[i] = float('nan')
            
            t_marginal_dict = calc_marginal_CAC(n_conversions_low_tier, spend_low_tier, n_conversions_high_tier, spend_high_tier)
            df_CAC.loc[t_channel, t_df_CAC_colnames] = [t_marginal_dict[x] for x in base_col_names]
    CAC_dfs.update({model_type: df_CAC})


# print implied CAC
for m in CAC_dfs.keys():
    print('\n%s attribution model implied CAC:' % m)
    print(CAC_dfs[m][['marginal_CAC_t1', 'marginal_CAC_t2', 'marginal_CAC_t3','marginal_CAC_t4', 'marginal_CAC_t5', 'marginal_CAC_t6','marginal_CAC_t7', 'marginal_CAC_t8']])

# write marginal CAC output
if write_to_file:
    for key, value in CAC_dfs.items():
        with open(key + '_model_marginal_implied_CAC.csv', 'w') as f:
            value.to_csv(f)

df_technical = pd.read_csv('attribution_technical_model_marginal_implied_CAC.csv')
df_survey = pd.read_csv('attribution_survey_model_marginal_implied_CAC.csv')
df_convert[['join_fee', 'payment_period','monthly_price']] = df_convert[['join_fee', 'payment_period','monthly_price']].apply(pd.to_numeric)
df_convert['revenue'] = df_convert['join_fee']+df_convert['payment_period']*4*df_convert['monthly_price']

cac_survey = []
cac_technical =[]
for i in range(len(df_convert)):
    if df_convert.loc[i,'attribution_technical'] == 'bing':
        tier = df_convert.loc[i,'tier']
        cac_technical.append(CAC_dfs['attribution_technical'].loc['bing','marginal_CAC_t'+str(tier)])
    elif df_convert.loc[i,'attribution_technical'] == 'youtube':
        tier = df_convert.loc[i,'tier']
        cac_technical.append(CAC_dfs['attribution_technical'].loc['youtube','marginal_CAC_t'+str(tier)])
    elif df_convert.loc[i,'attribution_technical'] == 'facebook':
        tier = df_convert.loc[i,'tier']
        cac_technical.append(CAC_dfs['attribution_technical'].loc['facebook','marginal_CAC_t'+str(tier)])
    elif df_convert.loc[i,'attribution_technical'] == 'search':
        tier = df_convert.loc[i,'tier']
        cac_technical.append(CAC_dfs['attribution_technical'].loc['search','marginal_CAC_t'+str(tier)])
    elif df_convert.loc[i,'attribution_technical'] == 'display':
        tier = df_convert.loc[i,'tier']
        cac_technical.append(CAC_dfs['attribution_technical'].loc['display','marginal_CAC_t'+str(tier)])
    else:
        cac_technical.append(0)

for i in range(len(df_convert)):
    if df_convert.loc[i,'attribution_survey'] == 'bing':
        tier = df_convert.loc[i,'tier']
        cac_survey.append(CAC_dfs['attribution_survey'].loc['bing','marginal_CAC_t'+str(tier)])
    elif df_convert.loc[i,'attribution_survey'] == 'youtube':
        tier = df_convert.loc[i,'tier']
        cac_survey.append(CAC_dfs['attribution_survey'].loc['youtube','marginal_CAC_t'+str(tier)])
    elif df_convert.loc[i,'attribution_survey'] == 'facebook':
        tier = df_convert.loc[i,'tier']
        cac_survey.append(CAC_dfs['attribution_survey'].loc['facebook','marginal_CAC_t'+str(tier)])
    elif df_convert.loc[i,'attribution_survey'] == 'search':
        tier = df_convert.loc[i,'tier']
        cac_survey.append(CAC_dfs['attribution_survey'].loc['search','marginal_CAC_t'+str(tier)])
    elif df_convert.loc[i,'attribution_survey'] == 'display':
        tier = df_convert.loc[i,'tier']
        cac_survey.append(CAC_dfs['attribution_survey'].loc['display','marginal_CAC_t'+str(tier)])
    else:
        cac_survey.append(0)

df_convert['marginal_cac_technical'] = cac_technical
df_convert['marginal_cac_survey'] = cac_survey

df_convert['CLV_technical'] = df_convert['revenue']-df_convert['marginal_cac_technical']
df_convert['CLV_survey'] = df_convert['revenue']-df_convert['marginal_cac_survey']
df_convert['CLV/CAC_ratio_technical'] = df_convert['CLV_technical']/df_convert['marginal_cac_technical']
df_convert['CLV/CAC_ratio_survey'] = df_convert['CLV_survey']/df_convert['marginal_cac_survey']

df_convert.to_csv('clv_cac_analysis.csv')