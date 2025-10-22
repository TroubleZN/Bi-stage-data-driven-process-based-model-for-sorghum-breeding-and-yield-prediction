#
!

#%%
import os.path
import sys
import copy
import pickle
import random

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from GenoToTreno.GenoToTreno import MAP_GT
from TrenoToPheno.make_treno import make_treno
from TrenoToPheno.gSimulator import simu_sorghum
from Utilities.utilities import reproduce, RRMSE, expend_info, make_info, listdiff, get_loc
from Models.GenoToTreno import Geno_to_Treno_MAP
from Models.PhenotypicSelection import PhenotypicSelection
from Models.GET_models import GET, rank_geno, fix_sub_treno
from Simulated_GroundTrue_main import load_data, get_sub_data

# torch.autograd.set_detect_anomaly(True)
seed_num = 1314
torch.random.manual_seed(seed_num)
random.seed(seed_num)
np.random.seed(seed_num)

Treno_0, Ltreno, Utreno, dtreno = make_treno()
Treno_0 = torch.tensor(Treno_0).to(torch.float32)
Utreno = torch.tensor(Utreno).to(torch.float32)
Ltreno = torch.tensor(Ltreno).to(torch.float32)
dtreno = torch.tensor(dtreno).to(torch.float32)

#%%
with open('../Data/data_imputed.pkl', 'rb') as f:
    data = pickle.load(f)

pheno = data['pheno']
info = data['info']
Types = np.array(data['types'])
envi = data['env']
mg = data['mg']

info['Year'] = info.Loc
info['Position'] = info.Loc
for i in range(len(info)):
    info.iloc[i, 4] = int(info.iloc[i, 4][:2])
    if 'Ames' in info.iloc[i, 5]:
        info.iloc[i, 5] = 'Ames'
    elif 'Craw' in info.iloc[i, 5]:
        info.iloc[i, 5] = 'Craw'
    elif 'Green' in info.iloc[i, 5]:
        info.iloc[i, 5] = 'Green'
    elif 'Curtis' in info.iloc[i, 5]:
        info.iloc[i, 5] = 'Ames'

info[info.Type == 'Dp'] = 'DP'
row_uniq = info.drop_duplicates()
# row_uniq = info.iloc[:, -4:-2].drop_duplicates()
id = row_uniq.index.to_list()
# info = info.iloc[id, :]
# Types = Types[id]
id = torch.tensor(id)

pheno, info, Types, envi, mg = get_sub_data(pheno, info, Types, envi, mg, id)

#%%
pheno_best = pheno['Biomass'].detach()
F = pheno_best[Types == 'F']
PS = pheno_best[Types == 'PS']
G = pheno_best[Types == 'G']
DP = pheno_best[Types == 'DP']

Days = 160
fig, ax = plt.subplots()
ax.scatter(torch.linspace(0, Days, Days), F.nanmean(0)[:Days].to('cpu'), label='F', color='y', alpha=0.6)
ax.scatter(torch.linspace(0, Days, Days), PS.nanmean(0)[:Days].to('cpu'), label='PS', color='g', alpha=0.6)
ax.scatter(torch.linspace(0, Days, Days), G.nanmean(0)[:Days].to('cpu'), label='G', color='r', alpha=0.6)
ax.scatter(torch.linspace(0, Days, Days), DP.nanmean(0)[:Days].to('cpu'), label='DP', color='b', alpha=0.6)
# ax.scatter(torch.linspace(0, Days, Days), pheno.nanmean(0)[1:Days+1].to('cpu'))
ax.set(xlabel='Days since planting', ylabel='Dry Biomass (Leaf + Stem)/g')
plt.xlim(0, Days)
plt.ylim(0, 300)
plt.xticks(np.linspace(0, Days, int(Days/20)+1).astype(int))
plt.yticks(np.linspace(0, 300, int(300/20 + 1)))
ax.legend(loc='upper left')
ax.grid('on')
plt.show()

# fig.savefig('../Result/Pheno/Biomass_true_mean.png', dpi=300)

#%%
pheno_best = pheno['Height'].detach()
F = pheno_best[Types == 'F']
PS = pheno_best[Types == 'PS']
G = pheno_best[Types == 'G']
DP = pheno_best[Types == 'DP']

Days = 160
fig, ax = plt.subplots()
ax.scatter(torch.linspace(0, Days, Days), F.nanmean(0)[:Days].to('cpu'), label='F', color='y', alpha=0.6)
ax.scatter(torch.linspace(0, Days, Days), PS.nanmean(0)[:Days].to('cpu'), label='PS', color='g', alpha=0.6)
ax.scatter(torch.linspace(0, Days, Days), G.nanmean(0)[:Days].to('cpu'), label='G', color='r', alpha=0.6)
ax.scatter(torch.linspace(0, Days, Days), DP.nanmean(0)[:Days].to('cpu'), label='DP', color='b', alpha=0.6)
# ax.scatter(torch.linspace(0, Days, Days), pheno.nanmean(0)[1:Days+1].to('cpu'))
ax.set(xlabel='Days since planting', ylabel='Plant Height/m')
plt.xlim(0, Days)
plt.ylim(0, 5)
plt.xticks(np.linspace(0, Days, int(Days/20)+1).astype(int))
plt.yticks(np.linspace(0, 5, 11))
ax.legend(loc='upper left')
ax.grid('on')
plt.show()
# fig.savefig('../Result/Pheno/Height_true_mean.png', dpi=300)


#%% load simulated data


# with open('../Data/simulated_data_new.pkl', 'rb') as f:
with open('C:\\Users\\zni\\Downloads\\simulated_data (2).pkl', 'rb') as f:
    data = pickle.load(f)
    print('simulated data loaded successfully from file')

Pheno = data['pheno']
Treno = data['treno']
info = data['info']
types = data['types']
envi = data['envi']
mg = data['mg']
id = data['id']
model = data['GenoToTreno']
g_father = data['genotype_father']
g_mother = data['genotype_mother']
father_dict = data['father_dict']
mother_dict = data['mother_dict']

data = 0


Geno = reproduce(g_father, g_mother, father_dict, mother_dict, info)
Treno = model.get_Treno(Geno)
# Pheno = simu_sorghum(envi, mg, Treno)

#%%
pheno_best = Pheno['Biomass'].detach().cpu()
F = pheno_best[types == 'F']
PS = pheno_best[types == 'PS']
G = pheno_best[types == 'G']
DP = pheno_best[types == 'DP']

Days = 140
fig, ax = plt.subplots(dpi=300)
ax.plot(torch.linspace(0, Days, Days), F.nanmean(0)[:Days].to('cpu'), label='F', color='y', alpha=0.8)
ax.fill_between(torch.linspace(0, Days, Days), F.quantile(0.25, 0)[:Days], F.quantile(0.75, 0)[:Days],
                color='y', linestyle="--", alpha=0.3)

ax.plot(torch.linspace(0, Days, Days), PS.nanmean(0)[:Days].to('cpu'), label='PS', color='g', alpha=0.8)
ax.fill_between(torch.linspace(0, Days, Days), PS.quantile(0.25, 0)[:Days], PS.quantile(0.75, 0)[:Days],
                color='g', linestyle="--", alpha=0.3)

ax.plot(torch.linspace(0, Days, Days), G.nanmean(0)[:Days].to('cpu'), label='G', color='r', alpha=0.8)
ax.fill_between(torch.linspace(0, Days, Days), G.quantile(0.25, 0)[:Days], G.quantile(0.75, 0)[:Days],
                color='r', linestyle="--", alpha=0.3)

ax.plot(torch.linspace(0, Days, Days), DP.nanmean(0)[:Days].to('cpu'), label='DP', color='b', alpha=0.8)
ax.fill_between(torch.linspace(0, Days, Days), DP.quantile(0.25, 0)[:Days], DP.quantile(0.75, 0)[:Days],
                color='b', linestyle="--", alpha=0.3)

# ax.scatter(torch.linspace(0, Days, Days), pheno.nanmean(0)[1:Days+1].to('cpu'))
ax.set(xlabel='Days since planting', ylabel='Dry Biomass (Leaf + Stem)/g')
plt.xlim(0, Days)
plt.ylim(0, 300)
plt.xticks(np.linspace(0, Days, int(Days/20)+1).astype(int))
plt.yticks(np.linspace(0, 300, int(300/20 + 1)))
ax.grid('on')
ax.legend(loc='upper left')
plt.show()
# fig.savefig('../Result/Pheno/Biomass_simulated_mean.png', dpi=300)

#%%
pheno_best = Pheno['Height'].detach()
# pheno_best = Yield['Biomass'].detach()
F = pheno_best[types == 'F']
PS = pheno_best[types == 'PS']
G = pheno_best[types == 'G']
DP = pheno_best[types == 'DP']

Days = 140
fig, ax = plt.subplots(dpi=300)
ax.plot(torch.linspace(0, Days, Days), F.nanmean(0)[:Days].to('cpu'), label='F', color='y', alpha=1)
ax.fill_between(torch.linspace(0, Days, Days), F.quantile(0.25, 0)[:Days], F.quantile(0.75, 0)[:Days],
                color='y', linestyle="--", alpha=0.3)

ax.plot(torch.linspace(0, Days, Days), PS.nanmean(0)[:Days].to('cpu'), label='PS', color='g', alpha=0.6)
ax.fill_between(torch.linspace(0, Days, Days), PS.quantile(0.25, 0)[:Days], PS.quantile(0.75, 0)[:Days],
                color='g', linestyle="--", alpha=0.3)

ax.plot(torch.linspace(0, Days, Days), G.nanmean(0)[:Days].to('cpu'), label='GR', color='r', alpha=0.6)
ax.fill_between(torch.linspace(0, Days, Days), G.quantile(0.25, 0)[:Days], G.quantile(0.75, 0)[:Days],
                color='r', linestyle="--", alpha=0.3)

ax.plot(torch.linspace(0, Days, Days), DP.nanmean(0)[:Days].to('cpu'), label='DP', color='b', alpha=0.6)
ax.fill_between(torch.linspace(0, Days, Days), DP.quantile(0.25, 0)[:Days], DP.quantile(0.75, 0)[:Days],
                color='b', linestyle="--", alpha=0.3)

ax.legend(loc='upper left')
plt.xlim(0, Days)
plt.ylim(0, 5)
plt.xticks(np.linspace(0, Days, int(Days/20)+1).astype(int))
plt.yticks(np.linspace(0, 5, 11))
ax.set(xlabel='Days since planting', ylabel='Plant Height/m')
ax.grid('on')
ax.legend()
plt.show()
# fig.savefig('../Result/Pheno/Height_simulated_mean.png', dpi=300)

###############################################################################################
###############################################################################################
#%%
device = 'cuda'

Fathers = []
Mothers = []
for f in father_dict.keys():
    for m in mother_dict.keys():
        Fathers.append(f)
        Mothers.append(m)
info_all = pd.DataFrame({
    'Mother': Mothers,
    'Father': Fathers,
})

envi_dir = '../Result/Simulated_data/env/Ames/15/env.pt'
envi_new = torch.load(envi_dir).to(device)
mg_dir = '../Result/Simulated_data/env/Ames/15/mg.pt'
mg_new = torch.load(mg_dir).to(device)

Treno_father = model.get_Treno(g_father)
Treno_mother = model.get_Treno(g_mother)

Yield_father = simu_sorghum(
    envi_new.repeat(len(Treno_father), 1, 1).to(device),
    mg_new[[0]].repeat(len(Treno_father), 1).to(device),
    Treno_father.to(device),
    device=device
)['Yield']

Yield_mother = simu_sorghum(
    envi_new.repeat(len(Treno_mother), 1, 1).to(device),
    mg_new[[0]].repeat(len(Treno_mother), 1).to(device),
    Treno_mother.to(device),
    device=device
)['Yield']

N_all = len(info_all)
Geno_all = reproduce(g_father, g_mother, father_dict, mother_dict, info_all)
Treno_all = model.get_Treno(Geno_all)

tt = 8
Yield_all = torch.zeros(N_all).to(device)
for i in range(tt):
    Yield_all[i * int(N_all / tt):(i + 1) * int(N_all / tt)] = simu_sorghum(
        envi_new.repeat(int(N_all / tt), 1, 1).to(device),
        mg_new[[0]].repeat(int(N_all / tt), 1).to(device),
        Treno_all[i * int(N_all / tt):(i + 1) * int(N_all / tt)].to(device),
        device=device
    )['Yield']

Yield_all_norm = torch.zeros(N_all)
for i in range(N_all):
    Yield_all_norm[i] = (Yield_all[i] - (Yield_father[father_dict[info_all.Father.iloc[i]]] + Yield_mother[mother_dict[info_all.Mother.iloc[i]]])/2)/(Yield_father[father_dict[info_all.Father.iloc[i]]] - Yield_mother[mother_dict[info_all.Mother.iloc[i]]]).abs()


###############################################################################################
###############################################################################################
#%%
methods = ['G0E0T0', 'G0E1T0', 'G0E1T1', 'G1E0T0', 'G1E1T0', 'G1E1T1']
RRMSE_test = {}
RRMSE_train = {}
for method in methods:
    RRMSE_train[method] = []
    RRMSE_test[method] = []
    for N_alleles in range(20, 220, 20):
        model = '../Result/Reduced_models/Server/Reduced_models2/' + method + '/N_alleles_' + str(N_alleles) + '.pkl'
        if os.path.exists(model):
            with open(model, 'rb') as f:
                res = pickle.load(f)
            Geno_F = res['g_father']
            Geno_M = res['g_mother']
            GroundTrueModel = res['model']
            loss_train = res['loss_train']
            loss_test = res['loss_test']
            RRMSE_test[method].append(loss_test.tolist())
            RRMSE_train[method].append(loss_train.tolist())
    RRMSE_train[method] = np.array(RRMSE_train[method])
    RRMSE_test[method] = np.array(RRMSE_test[method])
    RRMSE_train[method][RRMSE_train[method] > 0.5] = 0
    RRMSE_test[method][RRMSE_test[method] > 0.5] = 0

fig, ax = plt.subplots(dpi=300, figsize=(10, 5))
X = range(20, 250, 20)
plt.plot(X, RRMSE_test['G0E0T0'], 's-', color=(1.0, 0.3, 0.3), label='G0E0T0', linewidth=2.5)
plt.plot(X, RRMSE_test['G1E0T0'], 'x-', color=(0.8, 0.3, 0.3), label='G1E0T0', linewidth=2.5)
plt.plot(X, RRMSE_test['G0E1T0'], 's-', color=(0.3, 0.9, 0.3), label='G0E1T0', linewidth=2.5)
plt.plot(X, RRMSE_test['G1E1T0'], 'x-', color=(0.3, 0.6, 0.3), label='G1E1T0', linewidth=2.5)
plt.plot(X, RRMSE_test['G0E1T1'], 's-', color=(0.3, 0.4, 1.0), label='G0E1T1', linewidth=2.5)
# plt.plot(X, RRMSE_test['G1E1T1'], 'x-', color=(0.3, 0.3, 0.6), label='G1E1T1', linewidth=2.5)

plt.xlabel("Number of alleles")
plt.ylabel("Test RRMSE")
plt.legend(bbox_to_anchor=(0.5, 1), loc='upper center', ncol=3)
plt.xticks(np.linspace(20, 300, 15).astype(int))
plt.yticks(np.linspace(0.2, 0.6, 9))
ax.grid('on')

# plt.savefig("../Result/Results_n_alleles/Test_RRMSE.png")
plt.show()

###############################################################################################
###############################################################################################
#%%
with open('../Result/GTM2.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
g_father = data['g_father']
g_mother = data['g_mother']
father_dict = data['father_dict']
mother_dict = data['mother_dict']
info = data['info']
envi = data['envi']
mg = data['mg']

data = 0

#%%

N_all = len(mother_dict) * len(father_dict)
Fathers = []
Mothers = []
for f in father_dict.keys():
    for m in mother_dict.keys():
        Fathers.append(f)
        Mothers.append(m)
info_all = pd.DataFrame({
    'Mother': Mothers,
    'Father': Fathers,
})
Geno_all = reproduce(g_father, g_mother, father_dict, mother_dict, info_all)
Treno_all = model.get_Treno(Geno_all)
N_all = len(Treno_all)
pairs_all = [[info_all.Father.iloc[i], info_all.Mother.iloc[i]] for i in range(len(info_all))]


#%% Temporal and spatial comparison
N_pairs = 10
Loc = 'Ames'
device = 'cuda'
N_alleles = 140

RCs_TS = {}
Yields_TS = {}

methods = ['Phenotypic selection', 'G0E0T0', 'G0E1T0', 'G0E1T1', 'G1E0T0', 'G1E1T0', 'G1E1T1']
RCs_TS['Year'] = []

for dir in methods:
    RCs_TS[dir] = []

info_train = info.loc[:, ['Mother', 'Father']]

RES_all = {}

#%%
for method in methods:
    RES_all[method] = []
    #%%
    for Year in range(15, 22, 1):
        #%% load next year environment data
        envi_dir = '../Result/Simulated_data/env/' + Loc + '/' + str(Year) + '/env.pt'
        envi_new = torch.load(envi_dir).to(device)
        mg_dir = '../Result/Simulated_data/env/' + Loc + '/' + str(Year) + '/mg.pt'
        mg_new = torch.load(mg_dir).to(device)
        mg_new[1:, :] = mg_new[0, :]

        # Get all Yields in this year
        tt = 8
        Yield_all = torch.zeros(N_all).to(device)
        for i in range(tt):
            Yield_all[i * int(N_all / tt):(i + 1) * int(N_all / tt)] = simu_sorghum(
                envi_new.repeat(int(N_all / tt), 1, 1).to(device),
                mg_new[i * int(N_all / tt):(i + 1) * int(N_all / tt)].to(device),
                Treno_all[i * int(N_all / tt):(i + 1) * int(N_all / tt)].to(device),
                device=device
            )['Yield']

        if method == 'Phenotypic selection':
            pairs_train = [[info_train.Father.iloc[i], info_train.Mother.iloc[i]] for i in range(len(info_train))]
            i_train = get_loc(pairs_train, pairs_all)
            Yield_train = Yield_all[i_train]
            pairs_new = PhenotypicSelection(N_pairs, Yield_train, info_train, father_dict, mother_dict)
        else:
            if method == 'G0E0T0':
                (G_true, E_reduce, T_reduce) = (False, False, False)
            elif method == 'G0E1T0':
                (G_true, E_reduce, T_reduce) = (False, True, False)
            elif method == 'G0E1T1':
                (G_true, E_reduce, T_reduce) = (False, True, True)
            elif method == 'G1E0T0':
                (G_true, E_reduce, T_reduce) = (True, False, False)
            elif method == 'G1E1T0':
                (G_true, E_reduce, T_reduce) = (True, True, False)
            elif method == 'G1E1T1':
                (G_true, E_reduce, T_reduce) = (True, True, True)
            model = '../Result/Reduced_models/Server/Reduced_models/' + method + '/N_alleles_' + str(
                N_alleles) + '.pkl'
            if os.path.exists(model):
                with open(model, 'rb') as f:
                    res = pickle.load(f)
                g_father_train = res['g_father']
                g_mother_train = res['g_mother']
                GroundTrueModel = res['model']
                loss_train = res['loss_train']
                loss_test = res['loss_test']

            model_train = res['model']
            Geno_train = reproduce(g_father_train, g_mother_train, father_dict, mother_dict, info_all)
            Treno_train = model_train.get_Treno(Geno_train)

            if T_reduce:
                Treno_train = fix_sub_treno(Treno_train, Treno_0.to(device))

            tt = 8
            Yield_pred = torch.zeros(N_all).to(device)
            for i in range(tt):
                Yield_pred[i * int(N_all / tt):(i + 1) * int(N_all / tt)] = simu_sorghum(
                    envi_new.repeat(int(N_all / tt), 1, 1).to(device),
                    mg_new[i * int(N_all / tt):(i + 1) * int(N_all / tt)].to(device),
                    Treno_train[i * int(N_all / tt):(i + 1) * int(N_all / tt)].to(device),
                    device=device
                )['Yield']

            i_sort = Yield_pred.sort(descending=True)[1][:N_pairs]
            pairs_new = np.array(pairs_all)[i_sort.cpu()]


        i_new = get_loc(pairs_new.tolist(), pairs_all)
        Yield_new = Yield_all[i_new]
        RC = Yield_new.sort(descending=True)[0][:10].mean() / Yield_all.sort(descending=True)[0][:10].mean()
        print('\nRC(' + method + ' ' + str(N_alleles) +')='+str(RC))
        RES_all[method].append(RC.tolist())

#%%
methods = ['PhenotypicSelection', 'G0E0T0', 'G0E1T0', 'G0E1T1', 'G1E0T0', 'G1E1T0', 'G1E1T1']
X = range(15, 22)
RC = {}
Yield_mean = []
res_dir = '../Result/Active_training/Server/Active_training2/'
for method in methods:
    with open(res_dir + method + '_res.pkl', 'rb') as f:
        res = pickle.load(f)
    RC[method] = res['RC']
    for year in range(15, 22):
        Yield_mean.append(res['Data_new_15']['Yield'].mean())

fig, ax = plt.subplots(dpi=300, figsize=(10, 5))
plt.plot(X, RC['PhenotypicSelection'], 'o-', color=(0, 0, 0), label='PS', linewidth=2.5)
plt.plot(X, RC['G0E0T0'], 's-', color=(1.0, 0.3, 0.3), label='G0E0T0', linewidth=2.5)
plt.plot(X, RC['G1E0T0'], 'x-', color=(0.8, 0.3, 0.3), label='G1E0T0', linewidth=2.5)
plt.plot(X, RC['G0E1T0'], 's-', color=(0.3, 0.9, 0.3), label='G0E1T0', linewidth=2.5)
plt.plot(X, RC['G1E1T0'], 'x-', color=(0.3, 0.6, 0.3), label='G1E1T0', linewidth=2.5)
plt.plot(X, RC['G0E1T1'], 's-', color=(0.3, 0.4, 1.0), label='G0E1T1', linewidth=2.5)
plt.plot(X, RC['G1E1T1'], 'x-', color=(0.3, 0.4, 0.8), label='G1E1T1', linewidth=2.5)

plt.xlabel("Year")
plt.ylabel("Recovery rate")

plt.legend(bbox_to_anchor=(0.5, 1), loc='upper center', ncol=4)
plt.xticks(np.linspace(15, 21, 7).astype(int))
plt.yticks(np.linspace(0.54, 0.9, 19))
ax.grid('on')
plt.savefig(res_dir + "RC.png")
plt.show()