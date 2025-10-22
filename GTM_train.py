# !
#%% package importation
import random

import torch
import copy
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression

from GenoToTreno.GenoToTreno import MAP_GT
from TrenoToPheno.make_treno import make_treno
from TrenoToPheno.gSimulator import simu_sorghum
from Models.GenoToTreno import Geno_to_Treno_MAP
from Utilities.utilities import reproduce

# torch.manual_seed(0)
# torch.autograd.set_detect_anomaly(True)


def get_sub_data(pheno, info, Types, envi, mg, id, device='cpu'):
    envi_new = envi[id].to(device)
    mg_new = mg[id].to(device)
    pheno_new = {}
    pheno_new['Height'] = pheno['Height'][id].to(device)
    pheno_new['Yield'] = pheno['Yield'][id].to(device)
    pheno_new['Biomass'] = pheno['Biomass'][id].to(device)
    pheno_new['Lodging'] = pheno['Lodging'][id].to(device)
    info_new = info.iloc[id]
    info_new = info_new.reset_index(drop=True)
    Types_new = Types[id]
    return pheno_new, info_new, Types_new, envi_new, mg_new


#%%
with open('../Data/data_raw.pkl', 'rb') as f:
    data = pickle.load(f)

pheno = data['pheno']
pheno['Biomass'] = pheno['Biomass'].to(torch.float32)
pheno['Height'] = pheno['Height'].to(torch.float32)
pheno['Yield'] = pheno['Yield'].to(torch.float32)
pheno['Lodging'] = pheno['Lodging'].to(torch.float32)
info = data['info']
Types = np.array(data['types'])
envi = data['env'].to(torch.float32)
mg = data['mg'].to(torch.float32)

#%%
# Yield = pheno['Yield']
# i_bad = ((~pheno['Biomass'][:, :130].isnan()).sum(1) == 0)
# i_good = ((~pheno['Biomass'][:, :130].isnan()).sum(1) != 0)
# Biomass_good = pheno['Biomass'][i_good][:, :130] / pheno['Yield'][i_good].reshape(i_good.sum(), 1).repeat(1, 130)
# Biomass_good = Biomass_good.nanmean(0)
# pheno['Biomass'][i_bad, :130] = torch.matmul(torch.diag(pheno['Yield'][i_bad]), Biomass_good.repeat(i_bad.sum(), 1))

# i_bad = ((~pheno['Height'][:, :130].isnan()).sum(1) == 0)
# i_good = ((~pheno['Height'][:, :130].isnan()).sum(1) != 0)
# Height_good = pheno['Height'][i_good][:, :130] / pheno['Yield'][i_good].reshape(i_good.sum(), 1).repeat(1, 130)
# Height_good = Height_good.nanmean(0)
# pheno['Height'][i_bad, :130] = torch.matmul(torch.diag(pheno['Yield'][i_bad]), Height_good.repeat(i_bad.sum(), 1))

Y_PS = pheno['Yield'][Types == 'PS'].nanmean()
Y_F = pheno['Yield'][Types == 'F'].nanmean()
Y_DP = pheno['Yield'][Types == 'DP'].nanmean()
Y_G = pheno['Yield'][Types == 'G'].nanmean()

B_PS = pheno['Biomass'][Types == 'PS', :50].nanmean(0)
B_F = pheno['Biomass'][Types == 'F', :50].nanmean(0)

B_DP = (B_PS/Y_PS + B_F/Y_F) / 2 * Y_DP
pheno['Biomass'][Types == 'DP', :50] = B_DP
pheno['Biomass'][Types == 'DP', 120] = 120

B_G = (B_PS/Y_PS + B_F/Y_F) / 2 * Y_G
pheno['Biomass'][Types == 'G', :50] = B_G
pheno['Biomass'][Types == 'G', 120] = 80

H_PS = pheno['Height'][Types == 'PS', :50].nanmean(0)
H_F = pheno['Height'][Types == 'F', :50].nanmean(0)

H_DP = (H_PS/Y_PS + H_F/Y_F) / 2 * Y_DP
pheno['Height'][Types == 'DP', :50] = H_DP
H_G = (H_PS/Y_PS + H_F/Y_F) / 2 * Y_G
pheno['Height'][Types == 'G', :50] = H_G

pheno['Yield'][pheno['Yield'] == 0] = torch.nan



#%%
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

# %% Generate Mother-Father dictionary
Mothers = info.Mother.unique()
Mothers.sort()
Fathers = info.Father.unique()
Fathers.sort()

Mother_dict = {}
for i in range(len(Mothers)):
    Mother_dict[Mothers[i]] = i

Father_dict = {}
for i in range(len(Fathers)):
    Father_dict[Fathers[i]] = i

#%%
device = 'cuda'
Treno, Lgeno, Ugeno, dgeno = make_treno()
Treno = torch.from_numpy(Treno).to(device)
Lgeno = torch.from_numpy(Lgeno).to(device)
Ugeno = torch.from_numpy(Ugeno).to(device)
dgeno = torch.from_numpy(dgeno).to(device)

#%%
N_sample = info.__len__()
GroundTrueModel = MAP_GT(N_alleles=2000)
GroundTrueModel.start()

#%%
Geno_F = torch.randint(2, (len(Fathers), GroundTrueModel.N_alleles))
Geno_M = torch.randint(2, (len(Mothers), GroundTrueModel.N_alleles))
Geno = reproduce(Geno_F, Geno_M, Father_dict, Mother_dict, info)

#%%
_, Ltreno, Utreno, dtreno = make_treno()
Utreno = torch.tensor(Utreno).to(torch.float32)
Ltreno = torch.tensor(Ltreno).to(torch.float32)

#%% Step 2: Treno prediction
# prepare data
Yield = pheno['Yield']
N_father = len(Father_dict)
N_mother = len(Mother_dict)
Fathers = list(Father_dict.keys())
Mothers = list(Mother_dict.keys())

S_father = np.zeros(N_father)
S_mother = np.zeros(N_mother)
for id in Father_dict.keys():
    if Yield[info.Father == id].numel() != 0:
        info_P = info[info.Father == id].reset_index(drop=True)
        Y_child = Yield[info.Father == id]
        Y_child[info_P.Type == 'PS'] = Y_child[info_P.Type == 'PS'] / 1.5
        S_father[Father_dict[id]] = Y_child.nanmean()
    else:
        S_father[Father_dict[id]] = 0
for id in Mother_dict.keys():
    if Yield[info.Mother == id].numel() != 0:
        info_P = info[info.Mother == id].reset_index(drop=True)
        Y_child = Yield[info.Mother == id]
        Y_child[info_P.Type == 'PS'] = Y_child[info_P.Type == 'PS'] / 1.5
        S_mother[Mother_dict[id]] = Y_child.nanmean()
    else:
        S_mother[Mother_dict[id]] = 0

S_mother = torch.tensor(S_mother).to(torch.float32)
S_father = torch.tensor(S_father).to(torch.float32)

#%%
pheno_F = {
    'Height': torch.zeros(len(S_father), 180),
    'Biomass': torch.zeros(len(S_father), 180),
    'Yield': S_father,
    'Lodging': torch.zeros(len(S_father)),
}
pheno_M = {
    'Height': torch.zeros(len(S_mother), 180),
    'Biomass': torch.zeros(len(S_mother), 180),
    'Yield': S_mother,
    'Lodging': torch.zeros(len(S_mother)),
}

pheno_F['Biomass'][:] = torch.nan
pheno_F['Height'][:] = torch.nan
pheno_M['Biomass'][:] = torch.nan
pheno_M['Height'][:] = torch.nan

# B_father = torch.matmul(torch.diag(S_father), Biomass_good.repeat(len(S_father), 1))
# pheno_F['Biomass'][:, :130] = B_father
# B_mother = torch.matmul(torch.diag(S_mother), Biomass_good.repeat(len(S_mother), 1))
# pheno_M['Biomass'][:, :130] = B_mother
# H_father = torch.matmul(torch.diag(S_father), Height_good.repeat(len(S_father), 1))
# pheno_F['Height'][:, :130] = H_father
# H_mother = torch.matmul(torch.diag(S_mother), Height_good.repeat(len(S_mother), 1))
# pheno_M['Height'][:, :130] = H_mother

B_father = torch.matmul(torch.diag(S_father), ((B_PS/Y_PS + B_F/Y_F) / 2).repeat(len(S_father), 1))
pheno_F['Biomass'][:, :50] = B_father
B_mother = torch.matmul(torch.diag(S_mother), ((B_PS/Y_PS + B_F/Y_F) / 2).repeat(len(S_mother), 1))
pheno_M['Biomass'][:, :50] = B_mother
H_father = torch.matmul(torch.diag(S_father), ((H_PS/Y_PS + H_F/Y_F) / 2).repeat(len(S_father), 1))
pheno_F['Height'][:, :50] = H_father
H_mother = torch.matmul(torch.diag(S_mother), ((H_PS/Y_PS + H_F/Y_F) / 2).repeat(len(S_mother), 1))
pheno_M['Height'][:, :50] = H_mother

#%%
#
# N_all = len(Yield)
# Yield_all_norm = torch.zeros(N_all)
# for i in range(N_all):
#     Yield_all_norm[i] = (Yield[i] - (S_father[Father_dict[info.Father.iloc[i]]] + S_mother[Mother_dict[info.Mother.iloc[i]]])/2)/(S_father[Father_dict[info.Father.iloc[i]]] - S_mother[Mother_dict[info.Mother.iloc[i]]]).abs()
#
# aa = Yield_all_norm[~Yield_all_norm.isnan()]
# aa = aa[~aa.isinf()]
#
# fig, ax = plt.subplots(dpi=300, figsize=(20, 5))
# plt.hist(aa, bins=10000)
# plt.xlim(-25, 25)
# plt.xticks(range(-25, 26))
# # plt.savefig('../Result/Yield_norm_dist.png')
# plt.show()
#
# for i in range(N_all):
#     Yield_all_norm[i] = (Yield[i] - (S_father[Father_dict[info.Father.iloc[i]]] + S_mother[Mother_dict[info.Mother.iloc[i]]])/2)/(S_father[Father_dict[info.Father.iloc[i]]] + S_mother[Mother_dict[info.Mother.iloc[i]]])*2
#
# info_aa = info[~Yield_all_norm.isnan().numpy()].reset_index(drop=True)
# aa = Yield_all_norm[~Yield_all_norm.isnan()]
# aa = aa[~aa.isinf()]
#
# fig, ax = plt.subplots(dpi=300, figsize=(20, 5))
# plt.hist(aa[info_aa.Type=='PS'], bins=200, alpha=0.5, label='PS')
# plt.hist(aa[info_aa.Type=='DP'], bins=200, alpha=0.5, label='DP')
# plt.hist(aa[info_aa.Type=='F'], bins=200, alpha=0.5, label='F')
# plt.hist(aa[info_aa.Type=='G'], bins=200, alpha=0.5, label='G')
# plt.legend()
# plt.xlim(-2, 2)
# plt.xticks(np.linspace(-2, 2, 21))
# # plt.savefig('../Result/Yield_norm_dist2.png')
# plt.show()



#%%
# training
model_whole = Geno_to_Treno_MAP(GroundTrueModel.W_eQTL.shape[0] - 1, 153).to(device)

def forward(Geno, envi, mg, device):
    N_sample = len(Geno)
    temp = model_whole(GroundTrueModel.Geno_to_eQTL(Geno).to(device)).reshape(N_sample, 3, 51)
    Treno_pred = temp * (Utreno.to(device) - Ltreno.to(device)) + Ltreno.to(device)
    Treno_pred = torch.maximum(Treno_pred, Ltreno.to(device))
    Treno_pred = torch.minimum(Treno_pred, Utreno.to(device))
    Pheno_pred = simu_sorghum(envi, mg, Treno_pred, device=device)
    return Pheno_pred, Treno_pred

model_old = copy.deepcopy(model_whole.state_dict())
model_start = copy.deepcopy(model_old)

# %%
finished = 0
loss_test_old = 1e17
loss_train_old = 1e17
loss_test_best = 1e17
loss_test_wrong = 0
i_fail = 0
epochs = 10
lr_start = 1e-5
batch_size = 1500
seed_num = 1213

optimizer = torch.optim.Adam(model_whole.parameters(), lr=lr_start)

def RRMSE(x, x_pred):
    x_flat = x.flatten()[~x.flatten().isnan()]
    x_pred_flat = x_pred.flatten()[~x.flatten().isnan()]
    return (x_flat-x_pred_flat).square().mean().sqrt()/x_pred_flat.mean()

def loss_fn(pheno_true, pheno_pred):
    RRMSE_Yield = RRMSE(pheno_true['Yield'], pheno_pred['Yield'])
    RRMSE_Biomass = RRMSE(pheno_true['Biomass'], pheno_pred['Biomass'])
    RRMSE_Height = RRMSE(pheno_true['Height'], pheno_pred['Height'])
    RRMSEs = torch.zeros(3)
    RRMSEs[0] = RRMSE_Yield
    RRMSEs[1] = RRMSE_Biomass
    RRMSEs[2] = RRMSE_Height
    # return (RRMSE_Yield + RRMSE_Biomass + RRMSE_Height)/3
    return RRMSEs[~RRMSEs.isnan()].mean()

def get_yield_std(Yield_pred, Yield_M, Yield_F, info, Mother_dict, Father_dict):
    Yield_pred_std = torch.zeros(len(Yield_pred['Yield']))
    for i in range(len(info)):
        PM_mean = (Yield_F['Yield'][Father_dict[info.Father.iloc[i]]] + Yield_M['Yield'][Mother_dict[info.Mother.iloc[i]]])/2
        Yield_pred_std[i] = (Yield_pred['Yield'][i]-PM_mean)/PM_mean
    return Yield_pred_std


#%%
def sub_pheno(pheno, index):
    pheno_new = copy.deepcopy(pheno)
    pheno_new['Yield'] = pheno['Yield'][index]
    pheno_new['Biomass'] = pheno['Biomass'][index]
    pheno_new['Height'] = pheno['Height'][index]
    return pheno_new

#%%
Geno_all = torch.concat((Geno.sum(2), Geno_F*2, Geno_M*2), 0)
pheno_all = copy.deepcopy(pheno)
pheno_all['Yield'] = torch.concat((pheno['Yield'], pheno_F['Yield'], pheno_M['Yield']), 0)
pheno_all['Biomass'] = torch.concat((pheno['Biomass'], pheno_F['Biomass'], pheno_M['Biomass']), 0)
pheno_all['Height'] = torch.concat((pheno['Height'], pheno_F['Height'], pheno_M['Height']), 0)

envi_all = torch.concat((envi, envi[228].repeat(len(S_father)+len(S_mother), 1, 1)), 0)
mg_all = torch.concat((mg, mg[228].repeat(len(S_father)+len(S_mother), 1)), 0)
#
# Geno_all = Geno
# pheno_all = copy.deepcopy(pheno)
# envi_all = envi
# mg_all = mg

#%%
Geno_all = Geno_all.to(device)
Yield_all = copy.deepcopy(pheno_all)
Yield_all['Yield'] = pheno_all['Yield'].to(device)
Yield_all['Biomass'] = pheno_all['Biomass'].to(device)
Yield_all['Height'] = pheno_all['Height'].to(device)
envi_all = envi_all.to(device)
mg_all = mg_all.to(device)

#%%
GTM_file = '../Result/Server/GTM2.pkl'
# GTM_file = '../Result/GTM2.pkl'
with open(GTM_file, 'rb') as f:
    GTM = pickle.load(f)

GroundTrueModel = GTM['model']
Geno_F = GTM['g_father']
Geno_M = GTM['g_mother']

# with open('../Data/simulated_data.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print('simulated data loaded successfully from file')
# GroundTrueModel = data['GenoToTreno']
# Geno_F = data['genotype_father']
# Geno_M = data['genotype_mother']

Geno = reproduce(Geno_F, Geno_M, Father_dict, Mother_dict, info)
Geno_all = torch.concat((Geno.sum(2), Geno_F*2, Geno_M*2), 0)
# Geno_all = Geno.sum(2)

model_whole.linear_one.weight.data = GroundTrueModel.W_eQTL[1:].t().to(device)
model_whole.linear_one.bias.data = GroundTrueModel.W_eQTL[0].to(device)
model_whole.linear_two.weight.data = GroundTrueModel.W_poly[1:].t().to(device)
model_whole.linear_two.bias.data = GroundTrueModel.W_poly[0].to(device)
model_whole.linear_three.weight.data = GroundTrueModel.W_fold[1:].t().to(device)
model_whole.linear_three.bias.data = GroundTrueModel.W_fold[0].to(device)

# %%

while finished == 0:
    # %% splitting data
    train_dataset, test_dataset = torch.utils.data.random_split(range(len(Geno_all)), [0.3, 0.7])

    Geno_train = Geno_all[train_dataset]
    Yield_train = sub_pheno(Yield_all, train_dataset)
    envi_train = envi_all[train_dataset]
    mg_train = mg_all[train_dataset]
    Geno_test = Geno_all[test_dataset]
    Yield_test = sub_pheno(Yield_all, test_dataset)
    envi_test = envi_all[test_dataset]
    mg_test = mg_all[test_dataset]
    info_train = info.iloc[train_dataset.indices]
    info_test = info.iloc[test_dataset.indices]

    with torch.no_grad():
        Yield_pred, Treno_pred = forward(Geno_train, envi_train, mg_train, device)
        Yield_pred['Biomass'] = Yield_pred['StemDry'] + Yield_pred['LeafDry']

        Yield_M, Treno_M = forward(Geno_M*2, envi[228].repeat(len(Geno_M), 1, 1), mg[228].repeat(len(Geno_M), 1), device)
        Yield_F, Treno_F = forward(Geno_F*2, envi[228].repeat(len(Geno_F), 1, 1), mg[228].repeat(len(Geno_F), 1), device)
        Yield_pred_std = get_yield_std(Yield_pred, Yield_M, Yield_F, info_train, Mother_dict, Father_dict)

        # calculating the loss between original and predicted data points
        loss_1 = loss_fn(Yield_train, Yield_pred)
        loss_2 = (((Yield_pred_std[info_train.Type.to_numpy() == 'PS'] - 0.5).square().sum() + (
                    Yield_pred_std[info_train.Type.to_numpy() != 'PS'] - 0).square().sum()) / len(info_train)).sqrt()
        loss_train_start = loss_1 + loss_2 * 0.3


    loss_increase = 0
    bar = tqdm(range(epochs))

    # for (Yield_train_t, Geno_train_t, envi_train_t, mg_train_t) in bar:
    for i in bar:
        total = 0
        # making predictions with forward pass
        Yield_pred, Treno_pred = forward(Geno_train, envi_train, mg_train, device)
        Yield_pred['Biomass'] = Yield_pred['StemDry'] + Yield_pred['LeafDry']

        Yield_M, Treno_M = forward(Geno_M*2, envi[228].repeat(len(Geno_M), 1, 1), mg[228].repeat(len(Geno_M), 1), device)
        Yield_F, Treno_F = forward(Geno_F*2, envi[228].repeat(len(Geno_F), 1, 1), mg[228].repeat(len(Geno_F), 1), device)
        Yield_pred_std = get_yield_std(Yield_pred, Yield_M, Yield_F, info_train, Mother_dict, Father_dict)

        # calculating the loss between original and predicted data points
        loss_1 = loss_fn(Yield_train, Yield_pred)
        loss_2 = (((Yield_pred_std[info_train.Type.to_numpy() == 'PS']-0.5).square().sum() + (Yield_pred_std[info_train.Type.to_numpy() != 'PS']-0).square().sum())/len(info_train)).sqrt()

        loss_train = loss_1 + loss_2 * 0.3
        bar.set_postfix(loss1=float(loss_1), loss2=float(loss_2), lr=optimizer.param_groups[0]['lr'], fail=i_fail)
        # bar.set_postfix(loss_train=float(loss_train), lr=optimizer.param_groups[0]['lr'], fail=i_fail)
        torch.cuda.empty_cache()

        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        optimizer.zero_grad()
        loss_train.backward()

        model_whole.linear_one.weight.grad.data[model_whole.linear_one.weight.grad.data.isnan()] = 0
        model_whole.linear_one.bias.grad.data[model_whole.linear_one.bias.grad.data.isnan()] = 0
        model_whole.linear_two.weight.grad.data[model_whole.linear_two.weight.grad.data.isnan()] = 0
        model_whole.linear_two.bias.grad.data[model_whole.linear_two.bias.grad.data.isnan()] = 0
        model_whole.linear_three.weight.grad.data[model_whole.linear_three.weight.grad.data.isnan()] = 0
        model_whole.linear_three.bias.grad.data[model_whole.linear_three.bias.grad.data.isnan()] = 0

        optimizer.step()

        if loss_train.isnan():
            lr_start = lr_start / 10
            for g in optimizer.param_groups:
                g['lr'] = lr_start
            i_fail = 0
            optimizer = torch.optim.Adam(model_whole.parameters(), lr=lr_start)
            model_whole.zero_grad()
            break

    if loss_train.isnan():
        model_whole.load_state_dict(model_start)
        loss_test_old = 100
        continue

    print('Train Loss = {},\tTrain Loss Old = {}'.format(loss_train.item(), loss_train_start))
    with torch.no_grad():
        Yield_pred_all, Treno_pred_all = forward(Geno_all, envi_all, mg_all, device)
        Yield_pred_all['Biomass'] = Yield_pred_all['StemDry'] + Yield_pred_all['LeafDry']

        Yield_all_std = get_yield_std(Yield_pred_all, Yield_M, Yield_F, info, Mother_dict, Father_dict)

        # calculating the loss between original and predicted data points
        loss_1 = loss_fn(Yield_all, Yield_pred_all)
        loss_2 = (((Yield_all_std[info.Type.to_numpy() == 'PS'] - 0.5).square().sum() + (
                    Yield_all_std[info.Type.to_numpy() != 'PS'] - 0).square().sum()) / len(info)).sqrt()

        loss_all = loss_1 + loss_2 * 0.3

        print('Overall Loss = {}'.format(loss_all.item()))

    # loss_test_wrong += 1
    if (loss_train - loss_train_start) / loss_train_start >= 0:
        # model_whole = model_old
        model_whole.load_state_dict(model_old)
        i_fail += 1
        # halfing the learning rate when loss rising back
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] / 2

        # if i_fail >= 100 or loss_train < 0.05:
        #     finished = 1
        #     model_whole.load_state_dict(model_best)
    else:
        if loss_train_start > 0.4:
            if 0 < (loss_train_start - loss_train) / loss_train_start < 0.001:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] * 10
        else:
            if 0 < (loss_train_start - loss_train) / loss_train_start < 0.001:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] * 10

        model_old = copy.deepcopy(model_whole.state_dict())
        i_fail = 0
        model_best = copy.deepcopy(model_whole.state_dict())

        # Mapping from small model_whole to my GTM
        paras = list(model_whole.parameters())
        GroundTrueModel.W_eQTL[1:] = paras[0].t().detach()
        GroundTrueModel.W_eQTL[0] = paras[1].detach()
        GroundTrueModel.W_poly[1:] = paras[2].t().detach()
        GroundTrueModel.W_poly[0] = paras[3].detach()
        GroundTrueModel.W_fold[1:] = paras[4].t().detach()
        GroundTrueModel.W_fold[0] = paras[5].detach()

        res = {
            'model': GroundTrueModel,
            'g_father': Geno_F,
            'g_mother': Geno_M,
            'loss': loss_all.cpu(),
            'mother_dict': Mother_dict,
            'father_dict': Father_dict,
            # 'torch_model': model_whole.state_dict(),
            'info': info.reset_index().iloc[:, 1:],
            'types': Types,
            'envi': envi_all.data.to('cpu').to(torch.float32),
            'mg': mg_all.data.to('cpu').to(torch.float32),
        }

        with open('../Result/GTM.pkl', 'wb') as f:
            pickle.dump(res, f)

        # plt.close('all')

        pheno_best = Yield_pred_all['Biomass'].detach().cpu()[:len(info)]
        F = pheno_best[Types == 'F']
        PS = pheno_best[Types == 'PS']
        G = pheno_best[Types == 'G']
        DP = pheno_best[Types == 'DP']

        Days = 140
        fig1, ax1 = plt.subplots(dpi=100)
        ax1.plot(torch.linspace(0, Days, Days), F.nanmean(0)[:Days].to('cpu'), label='F', color='y', alpha=0.8)
        ax1.fill_between(torch.linspace(0, Days, Days), F.quantile(0.25, 0)[:Days], F.quantile(0.75, 0)[:Days],
                        color='y', linestyle="--", alpha=0.3)

        ax1.plot(torch.linspace(0, Days, Days), PS.nanmean(0)[:Days].to('cpu'), label='PS', color='g', alpha=0.8)
        ax1.fill_between(torch.linspace(0, Days, Days), PS.quantile(0.25, 0)[:Days], PS.quantile(0.75, 0)[:Days],
                        color='g', linestyle="--", alpha=0.3)

        ax1.plot(torch.linspace(0, Days, Days), G.nanmean(0)[:Days].to('cpu'), label='G', color='r', alpha=0.8)
        ax1.fill_between(torch.linspace(0, Days, Days), G.quantile(0.25, 0)[:Days], G.quantile(0.75, 0)[:Days],
                        color='r', linestyle="--", alpha=0.3)

        ax1.plot(torch.linspace(0, Days, Days), DP.nanmean(0)[:Days].to('cpu'), label='DP', color='b', alpha=0.8)
        ax1.fill_between(torch.linspace(0, Days, Days), DP.quantile(0.25, 0)[:Days], DP.quantile(0.75, 0)[:Days],
                        color='b', linestyle="--", alpha=0.3)

        # ax.scatter(torch.linspace(0, Days, Days), pheno.nanmean(0)[1:Days+1].to('cpu'))
        ax1.set(xlabel='Days since planting', ylabel='Dry Biomass (Leaf + Stem)/g')
        plt.xlim(0, Days)
        plt.ylim(0, 300)
        plt.xticks(np.linspace(0, Days, int(Days / 20) + 1).astype(int))
        plt.yticks(np.linspace(0, 300, int(300 / 20 + 1)))
        ax1.grid('on')
        ax1.legend(loc='upper left')
        fig1.savefig('../Result/Biomass_simulated3.png')

        ##########################
        pheno_best = Yield_pred_all['Height'].detach().cpu()[:len(info)]
        F = pheno_best[Types == 'F']
        PS = pheno_best[Types == 'PS']
        G = pheno_best[Types == 'G']
        DP = pheno_best[Types == 'DP']

        Days = 140
        fig2, ax2 = plt.subplots(dpi=100)
        ax2.plot(torch.linspace(0, Days, Days), F.nanmean(0)[:Days].to('cpu'), label='F', color='y', alpha=1)
        ax2.fill_between(torch.linspace(0, Days, Days), F.quantile(0.25, 0)[:Days], F.quantile(0.75, 0)[:Days],
                        color='y', linestyle="--", alpha=0.3)

        ax2.plot(torch.linspace(0, Days, Days), PS.nanmean(0)[:Days].to('cpu'), label='PS', color='g', alpha=0.6)
        ax2.fill_between(torch.linspace(0, Days, Days), PS.quantile(0.25, 0)[:Days], PS.quantile(0.75, 0)[:Days],
                        color='g', linestyle="--", alpha=0.3)

        ax2.plot(torch.linspace(0, Days, Days), G.nanmean(0)[:Days].to('cpu'), label='G', color='r', alpha=0.6)
        ax2.fill_between(torch.linspace(0, Days, Days), G.quantile(0.25, 0)[:Days], G.quantile(0.75, 0)[:Days],
                        color='r', linestyle="--", alpha=0.3)

        ax2.plot(torch.linspace(0, Days, Days), DP.nanmean(0)[:Days].to('cpu'), label='DP', color='b', alpha=0.6)
        ax2.fill_between(torch.linspace(0, Days, Days), DP.quantile(0.25, 0)[:Days], DP.quantile(0.75, 0)[:Days],
                        color='b', linestyle="--", alpha=0.3)

        ax2.legend(loc='upper left')
        plt.xlim(0, Days)
        plt.ylim(0, 5)
        plt.xticks(np.linspace(0, Days, int(Days / 20) + 1).astype(int))
        plt.yticks(np.linspace(0, 5, 11))
        ax2.set(xlabel='Days since planting', ylabel='Plant Height/m')
        ax2.grid('on')
        ax2.legend()
        fig2.savefig('../Result/Height_simulated3.png')

        fig, ax = plt.subplots(dpi=300, figsize=(20, 5))
        plt.hist(Yield_all_std[info.Type == 'PS'], bins=100, alpha=0.5, label='PS')
        plt.hist(Yield_all_std[info.Type == 'DP'], bins=100, alpha=0.5, label='DP')
        plt.hist(Yield_all_std[info.Type == 'F'], bins=100, alpha=0.5, label='F')
        plt.hist(Yield_all_std[info.Type == 'G'], bins=100, alpha=0.5, label='G')
        plt.legend()
        plt.xlim(-2, 2)
        plt.xticks(np.linspace(-2, 2, 21))
        plt.savefig('../Result/Yield_norm_dist3.png')
        # plt.show()

#%%
GTM_file = '../Result/Server/GTM2.pkl'
GTM_file = '../Result/GTM2.pkl'
with open(GTM_file, 'rb') as f:
    GTM = pickle.load(f)

Geno_F = GTM['g_father']
Geno_M = GTM['g_mother']
GroundTrueModel = GTM['model']
envi_saved = GTM['envi']
mg_saved = GTM['mg']
info = GTM['info']

Geno = reproduce(Geno_F, Geno_M, Father_dict, Mother_dict, info)
Treno_pred = GroundTrueModel.get_Treno(Geno)
Pheno_pred = simu_sorghum(envi_saved[:len(info)], mg_saved[:len(info)], Treno_pred)
Pheno_pred['Biomass'] = Pheno_pred['StemDry'] + Pheno_pred['LeafDry']


Treno_M_pred = GroundTrueModel.get_Treno(Geno_M * 2)
Pheno_M_pred = simu_sorghum(envi_saved[-len(Geno_M):], mg_saved[-len(Geno_M):], Treno_M_pred)
Pheno_M_pred['Biomass'] = Pheno_M_pred['StemDry'] + Pheno_M_pred['LeafDry']

Treno_F_pred = GroundTrueModel.get_Treno(Geno_F * 2)
Pheno_F_pred = simu_sorghum(envi_saved[-(len(Geno_M)+len(Geno_F)):-len(Geno_M)], mg_saved[-(len(Geno_M)+len(Geno_F)):-len(Geno_M)], Treno_F_pred)
Pheno_F_pred['Biomass'] = Pheno_F_pred['StemDry'] + Pheno_F_pred['LeafDry']

Yield_pred_std = get_yield_std(Pheno_pred, Pheno_M_pred, Pheno_F_pred, info, Mother_dict, Father_dict)

aa = Yield_pred_std[~Yield_pred_std.isnan()]
aa = aa[~aa.isinf()]

fig, ax = plt.subplots(dpi=300, figsize=(20, 5))
plt.hist(aa[info.Type=='PS'], bins=100, alpha=0.5, label='PS')
plt.hist(aa[info.Type=='DP'], bins=100, alpha=0.5, label='DP')
plt.hist(aa[info.Type=='F'], bins=100, alpha=0.5, label='F')
plt.hist(aa[info.Type=='G'], bins=100, alpha=0.5, label='G')
plt.legend()
plt.xlim(-2, 2)
plt.xticks(np.linspace(-2, 2, 21))
# plt.savefig('../Result/Yield_norm_dist2.png')
plt.show()

