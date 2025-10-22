
！
# %% import package
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import os
import copy
from tqdm import tqdm

#%%
##################################
### Raw ENV data preprocessing ###
##################################

# %% load raw data
data_raw = pd.read_excel('./Data/env/isusm.xlsx')
pd.set_option('display.max_columns', None)
data_raw.describe()


# %% Replace the outliers with NaN
def fix_outlier(df: pd.DataFrame):
    df.loc[df.tmpf < -40, 'tmpf'] = np.nan
    df.loc[df.precip > 5, 'precip'] = np.nan
    df.loc[df.solar > 4e6, 'solar'] = np.nan
    df.loc[df.relh < 0, 'relh'] = 0
    df.loc[df.relh > 100, 'relh'] = np.nan
    df.loc[df.solar < 0, 'solar'] = 0
    df.loc[df.speed < 0, 'speed'] = 0
    df.loc[df.drct < 0, 'drct'] = 0
    df.loc[df.et < 0, 'et'] = 0

    df.loc[df.soil04t < 0, 'soil04t'] = np.nan
    df.loc[df.soil04t > 100, 'soil04t'] = np.nan
    df.loc[df.soil12t < 0, 'soil12t'] = np.nan
    df.loc[df.soil12t > 100, 'soil12t'] = np.nan
    df.loc[df.soil24t < 0, 'soil24t'] = np.nan
    df.loc[df.soil24t > 100, 'soil24t'] = np.nan
    df.loc[df.soil50t < 0, 'soil50t'] = np.nan
    df.loc[df.soil50t > 100, 'soil50t'] = np.nan

    df.loc[df.soil12vwc < 0, 'soil12vwc'] = np.nan
    df.loc[df.soil12vwc > 100, 'soil12vwc'] = np.nan
    df.loc[df.soil24vwc < 0, 'soil24vwc'] = np.nan
    df.loc[df.soil24vwc > 100, 'soil24vwc'] = np.nan
    df.loc[df.soil50vwc < 0, 'soil50vwc'] = np.nan
    df.loc[df.soil50vwc > 100, 'soil50vwc'] = np.nan

    return df


# %%
data = fix_outlier(data_raw)

ames = data[data.station == 'BOOI4']
green = data[data.station == 'GREI4']
curtis = data[data.station == 'CRFI4']

green.loc[(green.valid.astype('datetime64[ns]') <= pd.Timestamp('2015-05-26 23:00:00')) & (green.solar > 0), 'solar'] = np.nan
curtis.loc[(curtis.valid.astype('datetime64[ns]') <= pd.Timestamp('2015-05-22 21:00:00')) & (curtis.solar > 0), 'solar'] = np.nan

green.loc[14538:19166, 'soil50t'] = np.nan
green.loc[14538:19166, 'soil50vwc'] = np.nan

# %% preprocess
def fix_time(df):
    df = df.drop_duplicates(subset=['valid'])
    df.valid = df.valid.astype('datetime64[ns]')
    df = df[df.valid >= pd.Timestamp('2015-01-01')]
    df = df.reset_index(drop=True)

    df = df.sort_values(by=['valid'])
    ts = df.valid
    n_df = len(df)
    for i in range(n_df - 1):
        if ts.iloc[i + 1] - ts.iloc[i] != pd.Timedelta('0 days 01:00:00'):
            # print(ts.iloc[i-2:i+1])
            # print(ts.iloc[i+1:i+3])

            gap = ts.iloc[i + 1] - ts.iloc[i]
            gap = int(gap.value / 3600000000000 - 1)

            df_new = copy.copy(df.iloc[i:i + gap, :])
            df_new.iloc[:, 2:] = np.nan
            t0 = df_new.valid.iloc[0]
            for g in range(gap):
                df_new.iloc[g, 1] = t0 + pd.Timedelta(seconds=3600 * (g + 1))

            # df = df.append(df_new, ignore_index=True)
            df = pd.concat([df, df_new], ignore_index=True)

    # df = df.sort_values(by=['valid'])
    # df = df.reset_index(drop=True)
    return df.sort_values(by=['valid']).reset_index(drop=True)


def check_time(df):
    ts = df.valid.astype('datetime64[ns]')
    if ts.iloc[0] != pd.Timestamp('2015-01-01'):
        print('error')

    for i in range(len(ts) - 1):
        if ts.iloc[i + 1] - ts.iloc[i] != pd.Timedelta('0 days 01:00:00'):
            print(ts.iloc[i - 2:i + 1])
            print(ts.iloc[i + 1:i + 3])


# %%
ames = fix_time(ames)
green = fix_time(green)
curtis = fix_time(curtis)

check_time(ames)
check_time(green)
check_time(curtis)


# %% Fill nan with mean of same date in other years
def fix_nan(df: pd.DataFrame):
    n_row = len(df)
    n_col = len(df.columns)
    isna = df.isna()
    ts = copy.copy(df.valid)

    for row in tqdm(range(n_row)):
        ttt = []
        for col in range(n_col - 2):
            if isna.iloc[row, col + 2]:
                if ttt:
                    if pd.isna(df.iloc[ttt, col + 2].mean()):
                        df.iloc[row, col + 2] = df.iloc[row-2:row+3, col + 2].mean()
                    else:
                        df.iloc[row, col + 2] = df.iloc[ttt, col + 2].mean()
                else:
                    # start = time.time()
                    t = df.valid.iloc[row]
                    temp = abs(ts - t) % pd.Timedelta(days=365)
                    temp = temp < pd.Timedelta(days=10)
                    ts_filtered = ts[temp]
                    for i in range(len(ts_filtered)):
                        tt = ts_filtered.iloc[i]
                        if tt.month == t.month and tt.day == t.day and tt.hour == t.hour:
                            ttt.append(ts_filtered.index[i])
                    if pd.isna(df.iloc[ttt, col + 2].mean()):
                        df.iloc[row, col + 2] = df.iloc[row-2:row+3, col + 2].mean()
                    else:
                        df.iloc[row, col + 2] = df.iloc[ttt, col + 2].mean()

                    # print(time.time()-start)
                    # df.iloc[row, col+2] = df[ttt].iloc[:, col+2].mean()
    return df


# %%
ames = fix_nan(ames)  # Time used 57:11
green = fix_nan(green)  # Time used 4:00:00
curtis = fix_nan(curtis)


# %%
def change_column_names(df: pd.DataFrame):
    df_t = df.rename(columns={
        'station': 'Station',
        'valid': 'Time',
        'tmpf': 'AirTemp',
        'soil04t': 'SoilTemp1',
        'soil12t': 'SoilTemp2',
        'soil24t': 'SoilTemp3',
        'soil50t': 'SoilTemp4',
        'soil12vwc': 'SoilMoist2',
        'soil24vwc': 'SoilMoist3',
        'soil50vwc': 'SoilMoist4',
        'relh': 'Humidity',
        'solar': 'Light',
        'speed': 'Wind',
        'et': 'Evap',
        'precip': 'SoilMoist1'
    })
    return df_t

# %%
ames = change_column_names(ames)
green = change_column_names(green)
curtis = change_column_names(curtis)

env = {
    'ames': ames,
    'green': green,
    'curtis': curtis,
}


# %% Save data
ames.to_csv('../Data/env/ames.csv')
green.to_csv('../Data/env/green.csv')
curtis.to_csv('../Data/env/curtis.csv')

ames.to_excel('../Data/env/ames.xlsx', index=False)
green.to_excel('../Data/env/green.xlsx', index=False)
curtis.to_excel('../Data/env/curtis.xlsx', index=False)

import pickle
with open('../Data/env/env.pkl', 'wb') as f:
    pickle.dump(env, f)

with open('../Data/env/env.pkl', 'rb') as f:
    ddt = pickle.load(f)



# %% visualization of raw data
loc = ['ames', 'green', 'curtis']
for i in range(14):
    fig, axs = plt.subplots(3)
    fig.set_figheight(9)
    fig.set_figwidth(24)
    axs[0].plot(ames.iloc[:, 1], ames.iloc[:, i + 2], '-', label='ames')
    axs[1].plot(green.iloc[:, 1], green.iloc[:, i + 2], '-', label='green')
    axs[2].plot(curtis.iloc[:, 1], curtis.iloc[:, i + 2], '-', label='curtis')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.suptitle(ames.columns[i + 2])
    axs[0].get_xaxis().set_visible(False)
    axs[1].get_xaxis().set_visible(False)
    axs[2].get_xaxis().set_visible(False)
    plt.savefig('../Result/Data/' + data_raw.columns[i + 2] + '.png')


#%%
####################################
### Raw PHENO data preprocessing ###
####################################

pheno_raw = pd.read_csv('./Data/pheno/Yield_data_sorghm.csv')


pheno_large = pd.read_csv('./Data/pheno/pheno_large_clean.csv')
# ['Loc', 'REP', 'Type', 'Mother', 'Father', 'ID', 'StandCounts',
# 'PlotSize', 'Heights', 'Lodging', 'Yield', 'Drypercent', 'Range', 'Row',
# 'Density']
pheno_small = pd.read_csv('./Data/pheno/pheno_small_clean.csv')
# ['Year', 'Location', 'Replication', 'Female code', 'Male code', 'Type',
# 'Yield', 'Density', 'Lodging', 'Dayi', 'Hi']

#%%
n_large = len(pheno_large)
n_small = len(pheno_small)
n_all = n_large + n_small

info_large = pheno_large[['Loc', 'Type', 'Mother', 'Father']]
info_small = pheno_small[['Location', 'Type', 'Female code', 'Male code']]
info_small = info_small.rename(columns={'Location': 'Loc', 'Female code': 'Mother', 'Male code': 'Father'})

info = pd.concat([info_large, info_small])
# info.to_excel('../Data/info.xlsx', index=False)

#%%
Density = np.concatenate([pheno_large.Density.to_numpy(), pheno_small.Density.to_numpy()])
Yield = np.concatenate([pheno_large.Yield.to_numpy(), pheno_small.Yield.to_numpy()])
Lodging = np.concatenate([pheno_large.Lodging.to_numpy(), pheno_small.Lodging.to_numpy()])
Lodging = np.floor(Lodging)
Lodging[Lodging > 5] = 5
Lodging[Lodging < 0] = 0


Biomass_small = pheno_small.iloc[:, 9:9+180].to_numpy()
Biomass_large = np.empty((n_large, 180))
Biomass_large[:] = np.nan
Biomass = np.concatenate([Biomass_large, Biomass_small])

Height_small = pheno_small.iloc[:, 9+180:].to_numpy()
Height_large = np.empty((n_large, 180))
Height_large[:] = np.nan
Height = np.concatenate([Height_large, Height_small])

pheno = {
    'Yield': Yield,
    'Height': Height,
    'Biomass': Biomass,
    'Lodging': Lodging,
    'Density': Density,
}

#%%
import pickle
# with open('../Data/pheno.pkl', 'wb') as f:
#     pickle.dump(pheno, f)

#%%
#################################
### Raw MG data preprocessing ###
#################################
mg_large = pd.read_csv('../Data/mg/mg_large.csv')
mg_small = pd.read_csv('../Data/mg/mg_short.csv')
mg = pd.concat([mg_large, mg_small])
mg.to_excel('../Data/mg.xlsx', index=False)

#%%
ames = pd.read_excel('../Data/training/ames.xlsx')
curtis = pd.read_excel('../Data/training/curtis.xlsx')
green = pd.read_excel('../Data/training/green.xlsx')


env_ames = ames.iloc[:, [2, 9, 10, 11, 12, 5, 13, 14, 15, 3, 4, 6, 8]].to_numpy()
env_curtis = curtis.iloc[:, [2, 9, 10, 11, 12, 5, 13, 14, 15, 3, 4, 6, 8]].to_numpy()
env_green = green.iloc[:, [2, 9, 10, 11, 12, 5, 13, 14, 15, 3, 4, 6, 8]].to_numpy()

info

env = np.zeros((len(info), 4320, 13))
mg_new = np.zeros((len(info), 3))
mg_new[:, 1] = Density
mg_new[:, 2] = Lodging

for i in range(len(mg)):
    loc = mg.iloc[i, 0]
    P = mg.iloc[i, 2]
    H = mg.iloc[i, 1]

    id = ames.index[ames['Time'] == pd.Timestamp(P)].tolist()
    if not id:
        id = 3624
    else:
        id = id[0]

    if 'Ames' in loc or 'Craw' in loc:
        env[info.Loc == loc, :, :] = env_ames[id:id+4320]
    elif 'Curtis' in loc:
        env[info.Loc == loc, :, :] = env_curtis[id:id+4320]
    elif 'Green' in loc:
        env[info.Loc == loc, :, :] = env_green[id:id+4320]

    mg_new[info.Loc == loc, 0] = mg.iloc[i, 1]


for i in range(len(mg_large)):
    loc = mg.iloc[i, 0]
    P = mg.iloc[i, 2]
    H = mg.iloc[i, 1]

    id = ames.index[ames['Time'] == pd.Timestamp(P)].tolist()
    if not id:
        id = 3624
    else:
        id = id[0]
    pheno['Height'][info.Loc == loc, int(H)] = pheno_large['Heights'][info_large.Loc == loc]


#%%
#################################
### Saving ###
#################################


year = copy.copy(info.Loc.to_numpy())
for i in range(len(year)):
    year[i] = int(year[i][:2])

env_new = torch.from_numpy(env[year >= 15])
pheno_new = {
    'Biomass': torch.from_numpy(pheno['Biomass'][year >= 15]),
    'Height': torch.from_numpy(pheno['Height'][year >= 15]),
    'Yield': torch.from_numpy(pheno['Yield'][year >= 15]),
    'Lodging': torch.from_numpy(pheno['Lodging'][year >= 15])
}

info_new = info[year >= 15]
mg_new = torch.from_numpy(mg_new[year >= 15])



data = {
    'env': env_new,
    'mg': mg_new,
    'pheno': pheno_new,
    'info': info_new.reset_index(),
    'types': info_new.Type.to_numpy(),
}

with open('../Data/data_raw.pkl', 'wb') as f:
    pickle.dump(data, f)

# with open('../Data/data_raw.pkl', 'rb') as f:
#     data = pickle.load(f)
