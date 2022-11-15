# %%
import pandas as pd
import numpy as np
import matplotlib as mpl
from os import listdir
from datetime import datetime
import statistics
import tarfile
import gzip
from io import BytesIO
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:,.2f}'.format

# %%
features = pd.read_csv('data/train_labeld.csv')

# %%
features = features.astype({'actor_account_id':'str'})
features = features.reindex(columns=['actor_account_id','churn_yn','survival_time',
'event_num','enterworld_num','levelup_num','joinparty_num','spendmoney_num','duel_num',
'duel_kd','partybattle_num','completechallengetoday_num','completechallengeweek_num',
'itemupgrade_successrate','trade_num','buyitemnowmainauction_num','guildlevelup_num','level_min',
'level_max','class','longest_time_between_events','faction1','faction2','targetaccountid_num','sessions_num','has_smurf_yn'])

# %%
tar1 = tarfile.open('data/traindata_1.tar.gz')
for member in tar1.getmembers():
    f = tar1.extractfile(member).read()
    with gzip.GzipFile(fileobj=BytesIO(f)) as fp:
        df = pd.read_csv(fp, usecols=['actor_account_id','logid','log_detail_code','entity_code','actor_level','actor_job','time','actor_faction','actor_faction2','target_account_id','session','actor_id'])
        dict_merge = {}
        dict_merge['actor_account_id'] = str(df.actor_account_id.unique()[0])
        dict_merge['event_num'] = len(df)
        dict_merge['enterworld_num'] = len(df[df.logid==1003])
        dict_merge['levelup_num'] = len(df[df.logid==1013])
        dict_merge['joinparty_num'] = len(df[df.logid==1102]) 
        dict_merge['spendmoney_num'] = len(df[df.logid==1018]) 
        dict_merge['duel_num'] = len(df[(df.logid == 1404) | (df.logid == 1406)])
        try:
            dict_merge['duel_kd'] = (len(df[((df.logid==1404) | (df.logid==1406)) & (df.log_detail_code==1)])) / (len(df[((df.logid==1404) | (df.logid==1406)) & (df.log_detail_code==2)]))
        except ZeroDivisionError:
            dict_merge['duel_kd'] = 0
        dict_merge['partybattle_num'] = len(df[df.entity_code==80])
        dict_merge['completechallengetoday_num'] = len(df[df.logid==5011])
        dict_merge['completechallengeweek_num'] = len(df[df.logid==5015])
        try:
            dict_merge['itemupgrade_successrate'] = (len(df[((df.logid==2126) | (df.logid==2127)) & (df.log_detail_code==1)])) / (len(df[((df.logid==2126) | (df.logid==2127)) & (df.log_detail_code==2)]))
        except ZeroDivisionError:
            dict_merge['itemupgrade_successrate'] = 0
        dict_merge['trade_num'] = len(df[(df.logid==2201) | (df.logid==2202)])
        dict_merge['buyitemnowmainauction_num'] = len(df[df.logid==2307])
        dict_merge['guildlevelup_num'] = len(df[df.logid==6003])
        dict_merge['level_min'] = min(df.actor_level)
        dict_merge['level_max'] = max(df.actor_level)
        dict_merge['class'] = df.actor_job.value_counts().index[0]
        df['time'] = df.apply(lambda row: datetime.strptime(row.time, '%Y-%m-%d %H:%M:%S.%f'), axis=1)
        df['time_diff'] = df.time.diff(-1).dt.total_seconds()
        dict_merge['longest_time_between_events'] = max(-df.time_diff)
        dict_merge['faction1'] = df.actor_faction.value_counts().index[0]
        dict_merge['faction2'] = df.actor_faction2.value_counts().index[0]
        dict_merge['targetaccountid_num'] = len(df.target_account_id.unique())
        dict_merge['sessions_num'] = len(df.session.unique())
        if len(df.actor_id.unique()) > 1:
            dict_merge['has_smurf_yn'] = 1
        else:
            dict_merge['has_smurf_yn'] = 0

        for key in dict_merge:
            dict_merge[key] = [dict_merge[key]]
        df_merge = pd.DataFrame.from_dict(dict_merge)

        features = pd.concat((features,df_merge)).groupby('actor_account_id').first().reset_index()


# %%
tar2 = tarfile.open('data/traindata_2.tar.gz')
for member in tar2.getmembers():
    f = tar2.extractfile(member).read()
    with gzip.GzipFile(fileobj=BytesIO(f)) as fp:
        df = pd.read_csv(fp, usecols=['actor_account_id','logid','log_detail_code','entity_code','actor_level','actor_job','time','actor_faction','actor_faction2','target_account_id','session','actor_id'])
        dict_merge = {}
        dict_merge['actor_account_id'] = str(df.actor_account_id.unique()[0])
        dict_merge['event_num'] = len(df)
        dict_merge['enterworld_num'] = len(df[df.logid==1003])
        dict_merge['levelup_num'] = len(df[df.logid==1013])
        dict_merge['joinparty_num'] = len(df[df.logid==1102]) 
        dict_merge['spendmoney_num'] = len(df[df.logid==1018]) 
        dict_merge['duel_num'] = len(df[(df.logid == 1404) | (df.logid == 1406)])
        try:
            dict_merge['duel_kd'] = (len(df[((df.logid==1404) | (df.logid==1406)) & (df.log_detail_code==1)])) / (len(df[((df.logid==1404) | (df.logid==1406)) & (df.log_detail_code==2)]))
        except ZeroDivisionError:
            dict_merge['duel_kd'] = 0
        dict_merge['partybattle_num'] = len(df[df.entity_code==80])
        dict_merge['completechallengetoday_num'] = len(df[df.logid==5011])
        dict_merge['completechallengeweek_num'] = len(df[df.logid==5015])
        try:
            dict_merge['itemupgrade_successrate'] = (len(df[((df.logid==2126) | (df.logid==2127)) & (df.log_detail_code==1)])) / (len(df[((df.logid==2126) | (df.logid==2127)) & (df.log_detail_code==2)]))
        except ZeroDivisionError:
            dict_merge['itemupgrade_successrate'] = 0
        dict_merge['trade_num'] = len(df[(df.logid==2201) | (df.logid==2202)])
        dict_merge['buyitemnowmainauction_num'] = len(df[df.logid==2307])
        dict_merge['guildlevelup_num'] = len(df[df.logid==6003])
        dict_merge['level_min'] = min(df.actor_level)
        dict_merge['level_max'] = max(df.actor_level)
        dict_merge['class'] = df.actor_job.value_counts().index[0]
        df['time'] = df.apply(lambda row: datetime.strptime(row.time, '%Y-%m-%d %H:%M:%S.%f'), axis=1)
        df['time_diff'] = df.time.diff(-1).dt.total_seconds()
        dict_merge['longest_time_between_events'] = max(-df.time_diff)
        dict_merge['faction1'] = df.actor_faction.value_counts().index[0]
        dict_merge['faction2'] = df.actor_faction2.value_counts().index[0]
        dict_merge['targetaccountid_num'] = len(df.target_account_id.unique())
        dict_merge['sessions_num'] = len(df.session.unique())
        if len(df.actor_id.unique()) > 1:
            dict_merge['has_smurf_yn'] = 1
        else:
            dict_merge['has_smurf_yn'] = 0

        for key in dict_merge:
            dict_merge[key] = [dict_merge[key]]
        df_merge = pd.DataFrame.from_dict(dict_merge)

        features = pd.concat((features,df_merge)).groupby('actor_account_id').first().reset_index()


# %%
tar3 = tarfile.open('data/traindata_3.tar.gz')
for member in tar3.getmembers():
    f = tar3.extractfile(member).read()
    with gzip.GzipFile(fileobj=BytesIO(f)) as fp:
        df = pd.read_csv(fp, usecols=['actor_account_id','logid','log_detail_code','entity_code','actor_level','actor_job','time','actor_faction','actor_faction2','target_account_id','session','actor_id'])
        dict_merge = {}
        dict_merge['actor_account_id'] = str(df.actor_account_id.unique()[0])
        dict_merge['event_num'] = len(df)
        dict_merge['enterworld_num'] = len(df[df.logid==1003])
        dict_merge['levelup_num'] = len(df[df.logid==1013])
        dict_merge['joinparty_num'] = len(df[df.logid==1102]) 
        dict_merge['spendmoney_num'] = len(df[df.logid==1018]) 
        dict_merge['duel_num'] = len(df[(df.logid == 1404) | (df.logid == 1406)])
        try:
            dict_merge['duel_kd'] = (len(df[((df.logid==1404) | (df.logid==1406)) & (df.log_detail_code==1)])) / (len(df[((df.logid==1404) | (df.logid==1406)) & (df.log_detail_code==2)]))
        except ZeroDivisionError:
            dict_merge['duel_kd'] = 0
        dict_merge['partybattle_num'] = len(df[df.entity_code==80])
        dict_merge['completechallengetoday_num'] = len(df[df.logid==5011])
        dict_merge['completechallengeweek_num'] = len(df[df.logid==5015])
        try:
            dict_merge['itemupgrade_successrate'] = (len(df[((df.logid==2126) | (df.logid==2127)) & (df.log_detail_code==1)])) / (len(df[((df.logid==2126) | (df.logid==2127)) & (df.log_detail_code==2)]))
        except ZeroDivisionError:
            dict_merge['itemupgrade_successrate'] = 0
        dict_merge['trade_num'] = len(df[(df.logid==2201) | (df.logid==2202)])
        dict_merge['buyitemnowmainauction_num'] = len(df[df.logid==2307])
        dict_merge['guildlevelup_num'] = len(df[df.logid==6003])
        dict_merge['level_min'] = min(df.actor_level)
        dict_merge['level_max'] = max(df.actor_level)
        dict_merge['class'] = df.actor_job.value_counts().index[0]
        df['time'] = df.apply(lambda row: datetime.strptime(row.time, '%Y-%m-%d %H:%M:%S.%f'), axis=1)
        df['time_diff'] = df.time.diff(-1).dt.total_seconds()
        dict_merge['longest_time_between_events'] = max(-df.time_diff)
        dict_merge['faction1'] = df.actor_faction.value_counts().index[0]
        dict_merge['faction2'] = df.actor_faction2.value_counts().index[0]
        dict_merge['targetaccountid_num'] = len(df.target_account_id.unique())
        dict_merge['sessions_num'] = len(df.session.unique())
        if len(df.actor_id.unique()) > 1:
            dict_merge['has_smurf_yn'] = 1
        else:
            dict_merge['has_smurf_yn'] = 0

        for key in dict_merge:
            dict_merge[key] = [dict_merge[key]]
        df_merge = pd.DataFrame.from_dict(dict_merge)

        features = pd.concat((features,df_merge)).groupby('actor_account_id').first().reset_index()


# %%
tar4 = tarfile.open('data/traindata_4.tar.gz')
for member in tar4.getmembers():
    f = tar4.extractfile(member).read()
    with gzip.GzipFile(fileobj=BytesIO(f)) as fp:
        df = pd.read_csv(fp, usecols=['actor_account_id','logid','log_detail_code','entity_code','actor_level','actor_job','time','actor_faction','actor_faction2','target_account_id','session','actor_id'])
        dict_merge = {}
        dict_merge['actor_account_id'] = str(df.actor_account_id.unique()[0])
        dict_merge['event_num'] = len(df)
        dict_merge['enterworld_num'] = len(df[df.logid==1003])
        dict_merge['levelup_num'] = len(df[df.logid==1013])
        dict_merge['joinparty_num'] = len(df[df.logid==1102]) 
        dict_merge['spendmoney_num'] = len(df[df.logid==1018]) 
        dict_merge['duel_num'] = len(df[(df.logid == 1404) | (df.logid == 1406)])
        try:
            dict_merge['duel_kd'] = (len(df[((df.logid==1404) | (df.logid==1406)) & (df.log_detail_code==1)])) / (len(df[((df.logid==1404) | (df.logid==1406)) & (df.log_detail_code==2)]))
        except ZeroDivisionError:
            dict_merge['duel_kd'] = 0
        dict_merge['partybattle_num'] = len(df[df.entity_code==80])
        dict_merge['completechallengetoday_num'] = len(df[df.logid==5011])
        dict_merge['completechallengeweek_num'] = len(df[df.logid==5015])
        try:
            dict_merge['itemupgrade_successrate'] = (len(df[((df.logid==2126) | (df.logid==2127)) & (df.log_detail_code==1)])) / (len(df[((df.logid==2126) | (df.logid==2127)) & (df.log_detail_code==2)]))
        except ZeroDivisionError:
            dict_merge['itemupgrade_successrate'] = 0
        dict_merge['trade_num'] = len(df[(df.logid==2201) | (df.logid==2202)])
        dict_merge['buyitemnowmainauction_num'] = len(df[df.logid==2307])
        dict_merge['guildlevelup_num'] = len(df[df.logid==6003])
        dict_merge['level_min'] = min(df.actor_level)
        dict_merge['level_max'] = max(df.actor_level)
        dict_merge['class'] = df.actor_job.value_counts().index[0]
        df['time'] = df.apply(lambda row: datetime.strptime(row.time, '%Y-%m-%d %H:%M:%S.%f'), axis=1)
        df['time_diff'] = df.time.diff(-1).dt.total_seconds()
        dict_merge['longest_time_between_events'] = max(-df.time_diff)
        dict_merge['faction1'] = df.actor_faction.value_counts().index[0]
        dict_merge['faction2'] = df.actor_faction2.value_counts().index[0]
        dict_merge['targetaccountid_num'] = len(df.target_account_id.unique())
        dict_merge['sessions_num'] = len(df.session.unique())
        if len(df.actor_id.unique()) > 1:
            dict_merge['has_smurf_yn'] = 1
        else:
            dict_merge['has_smurf_yn'] = 0

        for key in dict_merge:
            dict_merge[key] = [dict_merge[key]]
        df_merge = pd.DataFrame.from_dict(dict_merge)

        features = pd.concat((features,df_merge)).groupby('actor_account_id').first().reset_index()


# %%
tar5 = tarfile.open('data/traindata_5.tar.gz')
for member in tar5.getmembers():
    f = tar5.extractfile(member).read()
    with gzip.GzipFile(fileobj=BytesIO(f)) as fp:
        df = pd.read_csv(fp, usecols=['actor_account_id','logid','log_detail_code','entity_code','actor_level','actor_job','time','actor_faction','actor_faction2','target_account_id','session','actor_id'])
        dict_merge = {}
        dict_merge['actor_account_id'] = str(df.actor_account_id.unique()[0])
        dict_merge['event_num'] = len(df)
        dict_merge['enterworld_num'] = len(df[df.logid==1003])
        dict_merge['levelup_num'] = len(df[df.logid==1013])
        dict_merge['joinparty_num'] = len(df[df.logid==1102]) 
        dict_merge['spendmoney_num'] = len(df[df.logid==1018]) 
        dict_merge['duel_num'] = len(df[(df.logid == 1404) | (df.logid == 1406)])
        try:
            dict_merge['duel_kd'] = (len(df[((df.logid==1404) | (df.logid==1406)) & (df.log_detail_code==1)])) / (len(df[((df.logid==1404) | (df.logid==1406)) & (df.log_detail_code==2)]))
        except ZeroDivisionError:
            dict_merge['duel_kd'] = 0
        dict_merge['partybattle_num'] = len(df[df.entity_code==80])
        dict_merge['completechallengetoday_num'] = len(df[df.logid==5011])
        dict_merge['completechallengeweek_num'] = len(df[df.logid==5015])
        try:
            dict_merge['itemupgrade_successrate'] = (len(df[((df.logid==2126) | (df.logid==2127)) & (df.log_detail_code==1)])) / (len(df[((df.logid==2126) | (df.logid==2127)) & (df.log_detail_code==2)]))
        except ZeroDivisionError:
            dict_merge['itemupgrade_successrate'] = 0
        dict_merge['trade_num'] = len(df[(df.logid==2201) | (df.logid==2202)])
        dict_merge['buyitemnowmainauction_num'] = len(df[df.logid==2307])
        dict_merge['guildlevelup_num'] = len(df[df.logid==6003])
        dict_merge['level_min'] = min(df.actor_level)
        dict_merge['level_max'] = max(df.actor_level)
        dict_merge['class'] = df.actor_job.value_counts().index[0]
        df['time'] = df.apply(lambda row: datetime.strptime(row.time, '%Y-%m-%d %H:%M:%S.%f'), axis=1)
        df['time_diff'] = df.time.diff(-1).dt.total_seconds()
        dict_merge['longest_time_between_events'] = max(-df.time_diff)
        dict_merge['faction1'] = df.actor_faction.value_counts().index[0]
        dict_merge['faction2'] = df.actor_faction2.value_counts().index[0]
        dict_merge['targetaccountid_num'] = len(df.target_account_id.unique())
        dict_merge['sessions_num'] = len(df.session.unique())
        if len(df.actor_id.unique()) > 1:
            dict_merge['has_smurf_yn'] = 1
        else:
            dict_merge['has_smurf_yn'] = 0

        for key in dict_merge:
            dict_merge[key] = [dict_merge[key]]
        df_merge = pd.DataFrame.from_dict(dict_merge)

        features = pd.concat((features,df_merge)).groupby('actor_account_id').first().reset_index()


# %%
features.to_csv('features.csv', sep=',', float_format='{:.2f}'.format)

# %%



