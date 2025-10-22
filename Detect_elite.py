!
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
import matplotlib.patches as patches

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

pair_counts = info.groupby(['Mother', 'Father', 'Type']).size().reset_index(name='count')
main_type = pair_counts.groupby(['Mother', 'Father'])['count'].idxmax()
main_pairs = pair_counts.loc[main_type, ['Mother', 'Father', 'Type']]
info = info.merge(main_pairs, on=['Mother', 'Father'], suffixes=('', '_correct'), how='left')
info['Type'] = info['Type_correct']
info = info.drop(columns=['Type_correct'])

info = info[~((info['Mother'] == 131) & (info['Father'] == 677))]
id = info.index
info = info.reset_index(drop=True)
Types = Types[id]
envi = envi[id]
mg = mg[id]

pheno['Biomass'] = pheno['Biomass'][id]
pheno['Height'] = pheno['Height'][id]
pheno['Yield'] = pheno['Yield'][id]
pheno['Lodging'] = pheno['Lodging'][id]
pheno['Yield'][pheno['Yield'] == 0] = torch.nan


#%%
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
row_uniq = info.iloc[:, [0, 2, 3, 4, 5]].drop_duplicates()
id = row_uniq.index.to_list()
# info = info.iloc[id, :]
# Types = Types[id]
id = torch.tensor(id)

pheno, info, Types, envi, mg = get_sub_data(pheno, info, Types, envi, mg, id)
types = Types



#%%
GTM_file = '../GTM.pkl'
# GTM_file = '../Result/GTM.pkl'
with open(GTM_file, 'rb') as f:
    GTM = pickle.load(f)


#%%
model = GTM['model']
g_father = GTM['g_father']
g_mother = GTM['g_mother']
loss = GTM['loss']
mother_dict = GTM['mother_dict']
father_dict = GTM['father_dict']
# info = GTM['info']
# types = GTM['types']
# envi = GTM['envi']
# mg = GTM['mg']
train_id = GTM['train_id']
test_id = GTM['test_id']



#%%
geno_pred = reproduce(g_father, g_mother, father_dict, mother_dict, info)
treno_pred = model.get_Treno(geno_pred)
# pheno_pred = simu_sorghum(envi[[0]].repeat(5200, 1, 1), mg[[0]].repeat(5200, 1), treno_pred)
pheno_pred = simu_sorghum(envi, mg, treno_pred)


#%%
def scatter_plot(data, pheno, Days, ax, type=None):
    pheno_best = data[pheno].detach()
    F = pheno_best[Types == 'F']
    PS = pheno_best[Types == 'PS']
    G = pheno_best[Types == 'G']
    DP = pheno_best[Types == 'DP']

    ax.scatter(torch.linspace(0, Days, Days), F.nanmean(0)[:Days].to('cpu'), label='F', color='y', alpha=0.6)
    ax.scatter(torch.linspace(0, Days, Days), PS.nanmean(0)[:Days].to('cpu'), label='PS', color='g', alpha=0.6)
    ax.scatter(torch.linspace(0, Days, Days), G.nanmean(0)[:Days].to('cpu'), label='G', color='r', alpha=0.6)
    ax.scatter(torch.linspace(0, Days, Days), DP.nanmean(0)[:Days].to('cpu'), label='DP', color='b', alpha=0.6)

    # ax.set_xlim(0, Days)
    # if pheno == 'Biomass':
    #     ax.set_ylim([0, 300])
    #     ax.set_yticks(np.linspace(0, 300, int(300 / 20 + 1)))
    # else:
    #     ax.set_ylim([0, 5])
    #     ax.set_yticks(np.linspace(0, 5, 11))
    # ax.set_xticks(np.linspace(0, Days, int(Days / 20) + 1).astype(int))
    if type == 'F':
        ax.set_title('F')
    elif type == 'G':
        ax.set_title('G')
    else:
        ax.set_title(type)
    ax.legend(loc='upper left')
    ax.grid('on')


def box_plot(data, pheno, Days, ax, type=None):
    pheno_best = data[pheno].detach()
    F = pheno_best[Types == 'F']
    PS = pheno_best[Types == 'PS']
    G = pheno_best[Types == 'G']
    DP = pheno_best[Types == 'DP']

    if type == 'PS':
        data = PS
        box_color = 'g'
    if type == 'F':
        data = F
        box_color = 'y'
    if type == 'G':
        data = G
        box_color = 'r'
    if type == 'DP':
        data = DP
        box_color = 'b'

    valid_columns = [col for col in range(180) if not np.isnan(data[:, col]).all()]
    valid_data = [data[:, col][~torch.isnan(data[:, col])] for col in valid_columns]

    # 设定 x 坐标
    x_positions = valid_columns

    # 绘制箱线图
    boxplots = ax.boxplot(valid_data, positions=x_positions, widths=2, patch_artist=True, showfliers=False)
    for patch in boxplots['boxes']:
        patch.set_facecolor(box_color)
        patch.set_alpha(0.8)  # 设定透明度

    import matplotlib.patches as mpatches
    legend_patch = mpatches.Patch(color=box_color, alpha=0.8, label='True data')
    ax.legend(handles=[legend_patch])

    ax.set_xlim(0, Days)
    if pheno == 'Biomass':
        ax.set_ylim([0, 360])
        ax.set_yticks(np.linspace(0, 360, int(360 / 40 + 1)))
    else:
        ax.set_ylim([0, 5])
        ax.set_yticks(np.linspace(0, 5, 11))
    ax.set_xticks(np.linspace(0, Days, int(Days / 20) + 1).astype(int), np.linspace(0, Days, int(Days / 20) + 1).astype(int))

    if type == 'F':
        ax.set_title('F')
    elif type == 'G':
        ax.set_title('G')
    else:
        ax.set_title(type)
    # ax.legend(loc='upper left')
    ax.grid('on')



def region_plot(data, pheno, Days, ax, type=None):
    pheno_best = data[pheno].detach().cpu()
    F = pheno_best[Types == 'F']
    PS = pheno_best[Types == 'PS']
    G = pheno_best[Types == 'G']
    DP = pheno_best[Types == 'DP']

    if type == 'F' or type is None:
        ax.plot(torch.linspace(0, Days, Days), F.nanmean(0)[:Days].to('cpu'), label='F', color='y', alpha=0.8)
        ax.fill_between(torch.linspace(0, Days, Days), F.quantile(0.25, 0)[:Days], F.quantile(0.75, 0)[:Days],
                        color='y', linestyle="--", alpha=0.3)
    if type == 'PS' or type is None:
        ax.plot(torch.linspace(0, Days, Days), PS.nanmean(0)[:Days].to('cpu'), label='PS', color='g', alpha=0.8)
        ax.fill_between(torch.linspace(0, Days, Days), PS.quantile(0.25, 0)[:Days], PS.quantile(0.75, 0)[:Days],
                        color='g', linestyle="--", alpha=0.3)
    if type == 'G' or type is None:
        ax.plot(torch.linspace(0, Days, Days), G.nanmean(0)[:Days].to('cpu'), label='G', color='r', alpha=0.8)
        ax.fill_between(torch.linspace(0, Days, Days), G.quantile(0.25, 0)[:Days], G.quantile(0.75, 0)[:Days],
                        color='r', linestyle="--", alpha=0.3)
    if type == 'DP' or type is None:
        ax.plot(torch.linspace(0, Days, Days), DP.nanmean(0)[:Days].to('cpu'), label='DP', color='b', alpha=0.8)
        ax.fill_between(torch.linspace(0, Days, Days), DP.quantile(0.25, 0)[:Days], DP.quantile(0.75, 0)[:Days],
                        color='b', linestyle="--", alpha=0.3)

    # ax.scatter(torch.linspace(0, Days, Days), pheno.nanmean(0)[1:Days+1].to('cpu'))
    # if pheno == 'Biomass':
    #     ax.set(xlabel='Days since planting', ylabel='Dry Biomass (Leaf + Stem)/g')
    # else:
    #     ax.set(xlabel='Days since planting', ylabel='Plant Height/m')
    ax.set_xlim(0, Days)
    if pheno == 'Biomass':
        ax.set_ylim([0, 360])
        ax.set_yticks(np.linspace(0, 360, int(360 / 40 + 1)))
    else:
        ax.set_ylim([0, 5])
        ax.set_yticks(np.linspace(0, 5, 11))
    ax.set_xticks(np.linspace(0, Days, int(Days / 20) + 1).astype(int))
    if type == 'F':
        ax.set_title('F')
    elif type == 'G':
        ax.set_title('G')
    else:
        ax.set_title(type)
    ax.grid('on')
    # ax.legend(loc='upper left')


#%%
fig, axs = plt.subplots(2, 4, figsize=(15, 8), dpi=400)

scatter_plot(pheno, 'Biomass', 160, axs[0, 0], type='PS')
region_plot(pheno_pred, 'Biomass',160, axs[0, 0], type='PS')

scatter_plot(pheno, 'Biomass', 160, axs[0, 1], type='F')
region_plot(pheno_pred, 'Biomass',160, axs[0, 1], type='F')

scatter_plot(pheno, 'Biomass', 160, axs[0, 2], type='DP')
region_plot(pheno_pred, 'Biomass',160, axs[0, 2], type='DP')

scatter_plot(pheno, 'Biomass', 160, axs[0, 3], type='G')
region_plot(pheno_pred, 'Biomass',160, axs[0, 3], type='G')

scatter_plot(pheno, 'Height', 160, axs[1, 0], type='PS')
region_plot(pheno_pred, 'Height',160, axs[1, 0], type='PS')

scatter_plot(pheno, 'Height', 160, axs[1, 1], type='F')
region_plot(pheno_pred, 'Height',160, axs[1, 1], type='F')

scatter_plot(pheno, 'Height', 160, axs[1, 2], type='DP')
region_plot(pheno_pred, 'Height',160, axs[1, 2], type='DP')

scatter_plot(pheno, 'Height', 160, axs[1, 3], type='G')
region_plot(pheno_pred, 'Height',160, axs[1, 3], type='G')

fig.text(0.5, 0.05, 'Days since planting', ha='center', va='center', fontsize=12)
fig.text(0.09, 0.7, 'Dry Biomass (Leaf + Stem)/g', ha='center', va='center', rotation='vertical', fontsize=12)
fig.text(0.09, 0.3, 'Plant Height/m', ha='center', va='center', rotation='vertical', fontsize=12)

# fig.savefig('../Predictive_Capability.png', bbox_inches='tight', pad_inches=0.1)
plt.show()


#%%
pheno_best = pheno_pred['Biomass'].detach().cpu()
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


#%%
def RRMSE(x, x_pred):
    x_flat = x.flatten()[~x.flatten().isnan()]
    x_pred_flat = x_pred.flatten()[~x.flatten().isnan()]
    return (x_flat-x_pred_flat).square().mean().sqrt()/x_pred_flat.mean()

pheno_pred = simu_sorghum(envi, mg, treno_pred)

RRMSE_Yield = RRMSE(pheno['Yield'], pheno_pred['Yield'])
RRMSE_Biomass = RRMSE(pheno['Biomass'], pheno_pred['Biomass'])
RRMSE_Height = RRMSE(pheno['Height'], pheno_pred['Height'])

RRMSE_Yield_train = RRMSE(pheno['Yield'][train_id], pheno_pred['Yield'][train_id])
RRMSE_Biomass_train = RRMSE(pheno['Biomass'][train_id], pheno_pred['Biomass'][train_id])
RRMSE_Height_train = RRMSE(pheno['Height'][train_id], pheno_pred['Height'][train_id])

RRMSE_Yield_test = RRMSE(pheno['Yield'][test_id], pheno_pred['Yield'][test_id])
RRMSE_Biomass_test = RRMSE(pheno['Biomass'][test_id], pheno_pred['Biomass'][test_id])
RRMSE_Height_test = RRMSE(pheno['Height'][test_id], pheno_pred['Height'][test_id])

train_pos = np.array([True if i in train_id else False for i in range(5200)])

RRMSE_Yield_PS = RRMSE(pheno['Yield'][(types=='PS') & ~train_pos], pheno_pred['Yield'][(types=='PS') & ~train_pos])
RRMSE_Yield_F = RRMSE(pheno['Yield'][(types=='F') & ~train_pos], pheno_pred['Yield'][(types=='F') & ~train_pos])
RRMSE_Yield_DP = RRMSE(pheno['Yield'][(types=='DP') & ~train_pos], pheno_pred['Yield'][(types=='DP') & ~train_pos])
RRMSE_Yield_G = RRMSE(pheno['Yield'][(types=='G') & ~train_pos], pheno_pred['Yield'][(types=='G') & ~train_pos])

RRMSE_Biomass_PS = RRMSE(pheno['Biomass'][(types=='PS') & ~train_pos], pheno_pred['Biomass'][(types=='PS') & ~train_pos])
RRMSE_Biomass_F = RRMSE(pheno['Biomass'][(types=='F') & ~train_pos], pheno_pred['Biomass'][(types=='F') & ~train_pos])
RRMSE_Biomass_DP = RRMSE(pheno['Biomass'][(types=='DP') & ~train_pos], pheno_pred['Biomass'][(types=='DP') & ~train_pos])
RRMSE_Biomass_G = RRMSE(pheno['Biomass'][(types=='G') & ~train_pos], pheno_pred['Biomass'][(types=='G') & ~train_pos])

RRMSE_Height_PS = RRMSE(pheno['Height'][(types=='PS') & ~train_pos], pheno_pred['Height'][(types=='PS') & ~train_pos])
RRMSE_Height_F = RRMSE(pheno['Height'][(types=='F') & ~train_pos], pheno_pred['Height'][(types=='F') & ~train_pos])
RRMSE_Height_DP = RRMSE(pheno['Height'][(types=='DP') & ~train_pos], pheno_pred['Height'][(types=='DP') & ~train_pos])
RRMSE_Height_G = RRMSE(pheno['Height'][(types=='G') & ~train_pos], pheno_pred['Height'][(types=='G') & ~train_pos])


#%%
Biomass_pred = pheno_pred['Biomass']
Height_pred = pheno_pred['Height']
Grain_pred = pheno_pred['GrainDry']
Yield_pred = pheno_pred['Yield']

PS = Biomass_pred[types == 'PS', 140]
PS = Yield_pred[types == 'PS']
_, id_PS = PS.sort(descending=True)
F = Biomass_pred[types == 'F', 140]
F = Yield_pred[types == 'F']
_, id_F = F.sort(descending=True)
DP = Biomass_pred[types == 'DP', 140] + Grain_pred[types=='DP', 140]
DP = Yield_pred[types == 'DP'] + Grain_pred[types=='DP', 140]
_, id_DP = DP.sort(descending=True)
G = Grain_pred[types == 'G', 140]
_, id_G = G.sort(descending=True)


#%%
figs, ax = plt.subplots(3, 1, figsize=(7, 8), dpi=400)

yield_all = pheno['Yield'].flatten().cpu().detach().numpy()

ax[0].hist(yield_all[info.Type == 'PS'], bins=20, alpha=0.5, label='PS', color='blue', histtype='barstacked', rwidth=0.9)
ax[0].set_xlim([0, 50000])
ax[0].set_ylim([0, 500])
ax[0].axvline((pheno['Yield'][types=='PS'])[id_PS[:5]].max(), color='red', linestyle='--', label='Selected')

ax[1].hist(yield_all[info.Type == 'F'], bins=20, alpha=0.5, label='F', color='orange', histtype='barstacked', rwidth=0.9)
ax[1].set_xlim([0, 50000])
ax[1].set_ylim([0, 500])
ax[1].axvline((pheno['Yield'][types=='F'])[id_F[:5]].max(), color='red', linestyle='--', label='Selected')

ax[2].hist(yield_all[info.Type == 'DP'], bins=20, alpha=0.5, label='DP', color='green', histtype='barstacked', rwidth=0.9)
ax[2].set_xlim([0, 50000])
ax[2].set_ylim([0, 500])
ax[2].axvline((pheno['Yield'][types=='DP'])[id_DP[:5]].max(), color='red', linestyle='--', label='Selected')
#
# ax[3].hist(yield_all[info.Type == 'G'], bins=20, alpha=0.6, label='GR', color='red', histtype='barstacked', rwidth=0.9)
# ax[3].set_xlim([0, 50000])
# ax[3].set_ylim([0, 500])
# ax[3].axvline((pheno['Yield'][types=='G'])[id_G[:5]].max(), color='red', linestyle='--', label='Selected')

ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
ax[2].legend(loc='upper right')
# ax[3].legend(loc='upper right')

# ax[0].set_xlabel('Final yield (dry stem+leaf kg/hectare)')
# ax[1].set_xlabel('Final yield (dry stem+leaf kg/hectare)')
# ax[2].set_xlabel('Final yield (dry stem+leaf kg/hectare)')
# ax[3].set_xlabel('Final yield (dry stem+leaf kg/hectare)')
# ax[0].set_ylabel('Frequency')
# ax[1].set_ylabel('Frequency')
# ax[2].set_ylabel('Frequency')
# ax[3].set_ylabel('Frequency')
# figs.supxlabel('Dry yield (kg/hectare)')
figs.text(0.5, 0.05, 'Final yield (kg/hectare)', ha='center', va='center', fontsize=12)
figs.text(0.05, 0.5, 'Frequency', ha='center', va='center', rotation='vertical', fontsize=12)


figs.savefig('../Yield_distribution.png', bbox_inches='tight', pad_inches=0.2)
plt.show()


#%%
import matplotlib.pyplot as plt
import numpy as np

species = ("PS", "F", "DP", "G")
Phenotypes = {
    'Biomass': ((Biomass_pred[types=='PS', 140])[id_PS[0]].round(decimals=1),
                (Biomass_pred[types=='F', 140])[id_F[0]].round(decimals=1),
                (Biomass_pred[types=='DP', 140])[id_DP[0]].round(decimals=1),
                (Biomass_pred[types=='G', 140])[id_G[0]].round(decimals=1)),
    'Grain':  ((Grain_pred[types=='PS', 140])[id_PS[0]].round(decimals=1),
               (Grain_pred[types=='F', 140])[id_F[0]].round(decimals=1),
               (Grain_pred[types=='DP', 140])[id_DP[0]].round(decimals=1),
               (Grain_pred[types=='G', 140])[id_G[0]].round(decimals=1)),
    # 'Height': (5.2, 1.3, 4.3, 2.8),
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained', dpi=400)

for attribute, measurement in Phenotypes.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Weight (g)')
ax.set_title('Elite sorghum phenotypes by plant types')
ax.set_xticks(x + width, species)
ax.set_ylim(0, 400)

Height=((Height_pred[types=='PS', 140])[id_PS[0]].round(decimals=1),
        (Height_pred[types=='F', 140])[id_F[0]].round(decimals=1),
        (Height_pred[types=='DP', 140])[id_DP[0]].round(decimals=1),
        (Height_pred[types=='G', 140])[id_G[0]].round(decimals=1))
ax2 = ax.twinx()
offset = width * multiplier
rects = ax2.bar(x + offset, Height, width, label='Height', color='green')
ax2.bar_label(rects, padding=3)

ax2.set_ylabel('Height (m)')
ax2.set_ylim(0, 10)

ax.legend(loc='upper left', ncols=3)
ax2.legend(loc='upper right', ncols=3)
ax.set_xlabel('Sorghum types')

# fig.savefig('../elite_phenotype.png', bbox_inches='tight', pad_inches=0.2)

plt.show()



#%%
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


#%%
Grain_all = {}
Biomass_all = {}
Height_all = {}
Yield_all = {}

for year in [15, 16, 17, 18, 19, 21]:
# year = 21
    print(year)
    location = 'Ames'
    id = info.loc[(info.Year == year) & (info.Position == location)].index.to_list()[1]

    envi_yl = envi[[id]]
    mg_yl = mg[[id]]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_all = len(Treno_all)
    tt = 8

    Grain_all[year] = torch.zeros(N_all).to(device)
    Biomass_all[year] = torch.zeros(N_all).to(device)
    Height_all[year] = torch.zeros(N_all).to(device)
    Yield_all[year] = torch.zeros(N_all).to(device)
    for i in range(tt):
        pp = simu_sorghum(
            envi_yl.repeat(int(N_all / tt), 1, 1).to(device),
            mg_yl.repeat(int(N_all / tt), 1).to(device),
            Treno_all[i * int(N_all / tt):(i + 1) * int(N_all / tt)].to(device),
            device=device
        )
        Grain_all[year][i * int(N_all / tt):(i + 1) * int(N_all / tt)] = pp['GrainDry'][:, 120]
        Biomass_all[year][i * int(N_all / tt):(i + 1) * int(N_all / tt)] = pp['Biomass'][:, 120]
        Height_all[year][i * int(N_all / tt):(i + 1) * int(N_all / tt)] = pp['Height'][:, 120]
        Yield_all[year][i * int(N_all / tt):(i + 1) * int(N_all / tt)] = pp['Yield']

    Yield_all[year] = Yield_all[year].cpu()
    Grain_all[year] = Grain_all[year].cpu()
    Biomass_all[year] = Biomass_all[year].cpu()
    Height_all[year] = Height_all[year].cpu()

#%%
year = 15
Pheno = Yield_all[year]
pheno_matrix = Pheno.reshape(len(mother_dict), len(father_dict))
pheno_matrix_max, id_m = pheno_matrix.max(0, keepdim=True)
_, id_f = pheno_matrix_max.sort(descending=True)

id_f = id_f[0, :20]
id_m = id_m[0, id_f]

Pheno = Grain_all[year]
pheno_matrix2 = Pheno.reshape(len(mother_dict), len(father_dict))
pheno_matrix_max2, id_m2 = pheno_matrix2.max(0, keepdim=True)
_, id_f2 = pheno_matrix_max2.sort(descending=True)

id_f2 = id_f2[0, :20]
id_m2 = id_m2[0, id_f]

id_f = torch.cat((id_f, id_f2), dim=0).unique()
id_m = torch.cat((id_m, id_m2), dim=0).unique()



#%%
fig, ax = plt.subplots(figsize=(10, 6), dpi=400)

# Plot the heatmap
pheno_matrix = Yield_all[year].reshape(len(mother_dict), len(father_dict))[id_m, :]
pheno_matrix = pheno_matrix[:, id_f]
im = ax.imshow(pheno_matrix)

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax, fraction=0.1, pad=0.02, cmap='crest')
cbar.ax.set_ylabel('Final yield (dry stem+leaf kg/hectare)', rotation=90, fontdict={'fontsize': 12})
im.set_clim(0, 45000)

# Show all ticks and label them with the respective list entries.
ax.set_xticks(np.arange(pheno_matrix.shape[1]), labels=np.array(list(father_dict.keys()))[id_f], fontsize=8)
ax.set_yticks(np.arange(pheno_matrix.shape[0]), labels=np.array(list(mother_dict.keys()))[id_m], fontsize=8)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=90, ha="center",
         rotation_mode="default")

# Turn spines off and create white grid.
ax.spines[:].set_visible(False)

ax.set_xticks(np.arange(pheno_matrix.shape[1]+1)-.5, minor=True)
ax.set_yticks(np.arange(pheno_matrix.shape[0]+1)-.5, minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)
ax.set_axisbelow(True)

plt.xlabel('Male', fontsize=12)
plt.ylabel('Female', fontsize=12)

import matplotlib.patches as patches

pp = Yield_all[year].reshape(len(mother_dict), len(father_dict))
pp = pp[id_m, :]
pp = pp[:, id_f]
pp_max, y = pp.max(0, keepdim=True)
_, x = pp_max.sort(descending=True)
y = y[0, x[0, 0]]
x = x[0, 0]

ax.add_patch(
    patches.Rectangle(
        (-0.5+x, -0.5+y),   # (x,y)
        1,          # width
        1,          # height
        fill=False,
        edgecolor='r',
        linewidth=2,
        label='PS'
    )
)

ax.add_patch(
    patches.Rectangle(
        (-0.5+28, -0.5+15),   # (x,y)
        1,          # width
        1,          # height
        fill=False,
        edgecolor='g',
        linewidth=2,
        label='F'
    )
)

pp = (Yield_all[year].reshape(len(mother_dict), len(father_dict))/Yield_all[year].max() +
      Grain_all[year].reshape(len(mother_dict), len(father_dict))/Grain_all[year].max())
pp = pp[id_m, :]
pp = pp[:, id_f]
pp_max, y = pp.max(0, keepdim=True)
_, x = pp_max.sort(descending=True)
y = y[0, x[0, 0]]
x = x[0, 0]

ax.add_patch(
    patches.Rectangle(
        (-0.5+x, -0.5+y),   # (x,y)
        1,          # width
        1,          # height
        fill=False,
        edgecolor='b',
        linewidth=2,
        label='DP'
    )
)

pp = Grain_all[year].reshape(len(mother_dict), len(father_dict))
pp = pp[id_m, :]
pp = pp[:, id_f]
pp_max, y = pp.max(0, keepdim=True)
_, x = pp_max.sort(descending=True)
y = y[0, x[0, 0]]
x = x[0, 0]
#
# ax.add_patch(
#     patches.Rectangle(
#         (-0.5+x, -0.5+y),   # (x,y)
#         1,          # width
#         1,          # height
#         fill=False,
#         linewidth=2,
#         edgecolor='k',
#         label='G'
#     )
# )

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncols=4, title='Selected elite')

# fig.savefig('../Yield_elite.png', bbox_inches='tight', pad_inches=0.2)
plt.show()

#%%
fig, ax = plt.subplots(figsize=(10, 6), dpi=400)

# Plot the heatmap
pheno_matrix = Grain_all[year].reshape(len(mother_dict), len(father_dict))[id_m, :]
pheno_matrix = pheno_matrix[:, id_f]
im = ax.imshow(pheno_matrix)

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax, fraction=0.1, pad=0.02, cmap='crest')
cbar.ax.set_ylabel('Dry Grain on day 140 (g)', rotation=90, fontdict={'fontsize': 12})
im.set_clim(0, 130)

# Show all ticks and label them with the respective list entries.
ax.set_xticks(np.arange(pheno_matrix.shape[1]), labels=np.array(list(father_dict.keys()))[id_f], fontsize=8)
ax.set_yticks(np.arange(pheno_matrix.shape[0]), labels=np.array(list(mother_dict.keys()))[id_m], fontsize=8)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=90, ha="center",
         rotation_mode="default")

# Turn spines off and create white grid.
ax.spines[:].set_visible(False)

ax.set_xticks(np.arange(pheno_matrix.shape[1]+1)-.5, minor=True)
ax.set_yticks(np.arange(pheno_matrix.shape[0]+1)-.5, minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)
ax.set_axisbelow(True)

plt.xlabel('Male', fontsize=12)
plt.ylabel('Female', fontsize=12)


pp = Yield_all[year].reshape(len(mother_dict), len(father_dict))
pp = pp[id_m, :]
pp = pp[:, id_f]
pp_max, y = pp.max(0, keepdim=True)
_, x = pp_max.sort(descending=True)
y = y[0, x[0, 0]]
x = x[0, 0]

ax.add_patch(
    patches.Rectangle(
        (-0.5+x, -0.5+y),   # (x,y)
        1,          # width
        1,          # height
        fill=False,
        edgecolor='r',
        linewidth=2,
        label='PS'
    )
)

ax.add_patch(
    patches.Rectangle(
        (-0.5+28, -0.5+15),   # (x,y)
        1,          # width
        1,          # height
        fill=False,
        edgecolor='g',
        linewidth=2,
        label='F'
    )
)

pp = (Yield_all[year].reshape(len(mother_dict), len(father_dict))/Yield_all[year].max()+
      Grain_all[year].reshape(len(mother_dict), len(father_dict))/Grain_all[year].max())
pp = pp[id_m, :]
pp = pp[:, id_f]
pp_max, y = pp.max(0, keepdim=True)
_, x = pp_max.sort(descending=True)
y = y[0, x[0, 0]]
x = x[0, 0]

ax.add_patch(
    patches.Rectangle(
        (-0.5+x, -0.5+y),   # (x,y)
        1,          # width
        1,          # height
        fill=False,
        edgecolor='b',
        linewidth=2,
        label='DP'
    )
)

pp = Grain_all[year].reshape(len(mother_dict), len(father_dict))
pp = pp[id_m, :]
pp = pp[:, id_f]
pp_max, y = pp.max(0, keepdim=True)
_, x = pp_max.sort(descending=True)
y = y[0, x[0, 0]]
x = x[0, 0]

ax.add_patch(
    patches.Rectangle(
        (-0.5+x, -0.5+y),   # (x,y)
        1,          # width
        1,          # height
        fill=False,
        linewidth=2,
        edgecolor='k',
        label='G'
    )
)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncols=4, title='Selected elite')

# fig.savefig('../Grain_elite.png', bbox_inches='tight', pad_inches=0.2)
plt.show()


#%%
info_test = info.iloc[np.array(test_id)]

id_f
yield_test = pheno['Yield'][test_id]

info_PS = info_test[info_test.Type == 'PS']
PS_ID = yield_test[np.array(info_test.Type == 'PS')].sort()[1][-24:-14]
info_PS = info_PS.iloc[PS_ID]

info_DP = info_test[info_test.Type == 'DP']
DP_ID = yield_test[np.array(info_test.Type == 'DP')].sort()[1][-10:]
info_DP = info_DP.iloc[DP_ID]

info_G = info_test[info_test.Type == 'G']
G_ID = yield_test[np.array(info_test.Type == 'G')].sort()[1][-11:-1]
info_G = info_G.iloc[G_ID]

info_F = info_test[info_test.Type == 'F']
F_ID = yield_test[np.array(info_test.Type == 'F')].sort()[1][-12:-2]
info_F = info_F.iloc[F_ID]

id_f = torch.cat((torch.tensor(info_PS.Father.tolist()), torch.tensor(info_DP.Father.tolist())), dim=0).unique()
id_f = torch.cat((id_f, torch.tensor(info_G.Father.tolist())), dim=0).unique()
id_f = torch.cat((id_f, torch.tensor(info_F.Father.tolist())), dim=0).unique()

id_m = torch.cat((torch.tensor(info_PS.Mother.tolist()), torch.tensor(info_DP.Mother.tolist())), dim=0).unique()
id_m = torch.cat((id_m, torch.tensor(info_G.Mother.tolist())), dim=0).unique()
id_m = torch.cat((id_m, torch.tensor(info_F.Mother.tolist())), dim=0).unique()

id_f = torch.tensor([father_dict[i] for i in id_f.tolist()])
id_m = torch.tensor([mother_dict[i] for i in id_m.tolist()])


#%%
fig, ax = plt.subplots(figsize=(10, 6), dpi=400)

# Plot the heatmap
pheno_matrix = Yield_all[year].reshape(len(mother_dict), len(father_dict))[id_m, :]
pheno_matrix = pheno_matrix[:, id_f]

f_sort = pheno_matrix.quantile(0.5, 0).sort(descending=True)
m_sort = pheno_matrix.quantile(0.5, 1).sort(descending=True)

id_f = id_f[f_sort[1]]
id_m = id_m[m_sort[1]]
pheno_matrix = Yield_all[year].reshape(len(mother_dict), len(father_dict))[id_m, :]
pheno_matrix = pheno_matrix[:, id_f]

im = ax.imshow(pheno_matrix)

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.02, cmap='crest')
cbar.ax.set_ylabel('Final yield (dry stem+leaf kg/hectare)', rotation=90, fontdict={'fontsize': 12})
im.set_clim(0, 45000)

# Show all ticks and label them with the respective list entries.
ax.set_xticks(np.arange(pheno_matrix.shape[1]), labels=np.array(list(father_dict.keys()))[id_f], fontsize=8)
ax.set_yticks(np.arange(pheno_matrix.shape[0]), labels=np.array(list(mother_dict.keys()))[id_m], fontsize=8)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=90, ha="center",
         rotation_mode="default")

# Turn spines off and create white grid.
ax.spines[:].set_visible(False)

ax.set_xticks(np.arange(pheno_matrix.shape[1]+1)-.5, minor=True)
ax.set_yticks(np.arange(pheno_matrix.shape[0]+1)-.5, minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)
ax.set_axisbelow(True)

plt.xlabel('Father', fontsize=12)
plt.ylabel('Mother', fontsize=12)


i_m_dict = {}
for i in range(len(id_m)):
    i_m_dict[id_m[i].tolist()] = i
i_f_dict = {}
for i in range(len(id_f)):
    i_f_dict[id_f[i].tolist()] = i


id_MM = [mother_dict[i] for i in info_PS.Mother]
id_FF = [father_dict[i] for i in info_PS.Father]
for i in range(len(info_PS)):
    x = i_f_dict[id_FF[i]]
    y = i_m_dict[id_MM[i]]
    ax.add_patch(
        patches.Rectangle(
            (-0.5 + x, -0.5 + y),  # (x,y)
            1,  # width
            1,  # height
            fill=False,
            edgecolor='r',
            linewidth=2,
            label='PS'
        )
    )

id_MM = [mother_dict[i] for i in info_F.Mother]
id_FF = [father_dict[i] for i in info_F.Father]
for i in range(len(info_PS)):
    x = i_f_dict[id_FF[i]]
    y = i_m_dict[id_MM[i]]
    ax.add_patch(
        patches.Rectangle(
            (-0.5 + x, -0.5 + y),  # (x,y)
            1,  # width
            1,  # height
            fill=False,
            edgecolor='g',
            linewidth=2,
            label='F'
        )
    )

id_MM = [mother_dict[i] for i in info_DP.Mother]
id_FF = [father_dict[i] for i in info_DP.Father]
for i in range(len(info_PS)):
    x = i_f_dict[id_FF[i]]
    y = i_m_dict[id_MM[i]]
    ax.add_patch(
        patches.Rectangle(
            (-0.5 + x, -0.5 + y),  # (x,y)
            1,  # width
            1,  # height
            fill=False,
            edgecolor='b',
            linewidth=2,
            label='DP'
        )
    )

id_MM = [mother_dict[i] for i in info_G.Mother]
id_FF = [father_dict[i] for i in info_G.Father]
for i in range(len(info_PS)):
    x = i_f_dict[id_FF[i]]
    y = i_m_dict[id_MM[i]]
    ax.add_patch(
        patches.Rectangle(
            (-0.5 + x, -0.5 + y),  # (x,y)
            1,  # width
            1,  # height
            fill=False,
            edgecolor='k',
            linewidth=2,
            label='G'
        )
    )

plt.show()


#%%

import seaborn as sns

fig, ax = plt.subplots(dpi=100)


# Plot the heatmap
im = ax.imshow(pheno_matrix)

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.005, cmap='crest')
cbar.ax.set_ylabel('Biomass (Dry g)', rotation=90, fontdict={'fontsize': 20})
im.set_clim(0, 150)

# Show all ticks and label them with the respective list entries.
ax.set_xticks(np.arange(pheno_matrix.shape[1]), labels=torch.tensor(list(father_dict.keys()))[id_f])
ax.set_yticks(np.arange(pheno_matrix.shape[0]), labels=torch.tensor(list(mother_dict.keys()))[id_m])

# Let the horizontal axes labeling appear on top.
# ax.tick_params(top=False, bottom=False,
#                labeltop=True, labelbottom=False)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")

# Turn spines off and create white grid.
ax.spines[:].set_visible(False)

ax.set_xticks(np.arange(pheno_matrix.shape[1]+1)-.5, minor=True)
ax.set_yticks(np.arange(pheno_matrix.shape[0]+1)-.5, minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)

plt.xlabel('Father', fontsize=25)
plt.ylabel('Mother', fontsize=25)

# fig.savefig('../Biomass_'+str(year)+'.png', bbox_inches='tight', pad_inches=0.2)
# fig.savefig('../test.png', bbox_inches='tight', pad_inches=0.2)
plt.show()

#
# sns.heatmap(Yield_matrix, linewidth=0.05,
#             yticklabels=mother_dict.keys(),
#             xticklabels=father_dict.keys(),
#             cmap="crest",
#             cbar_kws={'pad': 0.005},
#             )
#
# plt.xlabel('Father', fontsize=25)
# plt.ylabel('Mother', fontsize=25)

# fig.xlabel('asd')

# inset Axes....
# x1, x2, y1, y2 = 10, 11, 10, 11  # subregion of the original image
# axins = ax.inset_axes(
#     [0.2, 0.2, 0.2, 0.2],
#     xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
# # axins.imshow(Z2, extent=extent, origin="lower")
# #
# # axins.sns.heatmap(Yield_matrix[8:12, 8:12], linewidth=0.05,
# #             cmap="crest",
# #             cbar_kws={'pad': 0.005},
# #             )
#
# ax.indicate_inset_zoom(axins, edgecolor="black")

# im = ax.imshow(Yield_matrix)

# !



#%%
geno_pred = reproduce(g_father, g_mother, father_dict, mother_dict, info)
treno_pred = model.get_Treno(geno_pred)
# pheno_15Ames = simu_sorghum(envi[[0]].repeat(5200, 1, 1), mg[[0]].repeat(5200, 1), treno_pred)

#%%
year = 15
loc = 'Ames'

env_y = torch.load("../Result/Simulated_data/env/" + loc + '/' + str(year) + '/env.pt')
if loc == 'Ames':
    mg_y = torch.load("../Result/Simulated_data/env/" + loc + '/' + str(year) + '/mg.pt')[[0]]
else:
    mg_y = torch.load("../Result/Simulated_data/env/" + loc + '/' + str(year) + '/mg.pt').reshape(1, 3)

pheno_pred = simu_sorghum(env_y.repeat(len(treno_pred), 1, 1), mg_y.repeat(len(treno_pred), 1), treno_pred)

Biomass_pred = pheno_pred['Biomass']
Height_pred = pheno_pred['Height']
Grain_pred = pheno_pred['GrainDry']
Yield_pred = pheno_pred['Yield']

PS = Biomass_pred[types == 'PS', 140]
_, id_PS = PS.sort(descending=True)
F = Biomass_pred[types == 'F', 140]
_, id_F = F.sort(descending=True)
DP = Biomass_pred[types == 'DP', 140] + Grain_pred[types=='DP', 140]
_, id_DP = DP.sort(descending=True)
G = Grain_pred[types == 'G', 140]
_, id_G = G.sort(descending=True)

id_PS = info.loc[info.Type=='PS'].index[id_PS[0].tolist()]
id_F = info.loc[info.Type=='F'].index[id_F[0].tolist()]
id_DP = info.loc[info.Type=='DP'].index[id_DP[0].tolist()]
id_G = info.loc[info.Type=='G'].index[id_G[0].tolist()]

#%%
loc = 'Ames'
# fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
fig, ax = plt.subplots(2, 3, figsize=(10, 9), dpi=300)

for i, loc in enumerate(['Ames', 'Craw', 'Green']):
    for year in range(15, 22):
        env_y = torch.load("../Result/Simulated_data/env/" + loc + '/' + str(year) + '/env.pt')
        if loc == 'Ames':
            mg_y = torch.load("../Result/Simulated_data/env/" + loc + '/' + str(year) + '/mg.pt')[[0]]
        else:
            mg_y = torch.load("../Result/Simulated_data/env/" + loc + '/' + str(year) + '/mg.pt').reshape(1,3)
        pheno_pred = simu_sorghum(env_y, mg_y, treno_pred[[id_PS]])
        ax[0, i].plot(pheno_pred['Biomass'][0, :140], label='20'+str(year))
        ax[0, i].set_ylim((0, 400))
        ax[0, i].set_xlim((0, 140))
        ax[0, i].legend()
        ax[0, i].set_title(loc)


#%%
year = 15
loc = 'Ames'

env_y = torch.load("../Result/Simulated_data/env/" + loc + '/' + str(year) + '/env.pt')
if loc == 'Ames':
    mg_y = torch.load("../Result/Simulated_data/env/" + loc + '/' + str(year) + '/mg.pt')[[0]]
else:
    mg_y = torch.load("../Result/Simulated_data/env/" + loc + '/' + str(year) + '/mg.pt').reshape(1, 3)

pheno_pred = simu_sorghum(env_y.repeat(len(treno_pred), 1, 1), mg_y.repeat(len(treno_pred), 1), treno_pred)

Biomass_pred = pheno_pred['Biomass']
Height_pred = pheno_pred['Height']
Grain_pred = pheno_pred['GrainDry']
Yield_pred = pheno_pred['Yield']

PS = Biomass_pred[types == 'PS', 140]
_, id_PS = PS.sort(descending=True)
F = Biomass_pred[types == 'F', 140]
_, id_F = F.sort(descending=True)
DP = Biomass_pred[types == 'DP', 140] + Grain_pred[types=='DP', 140]
_, id_DP = DP.sort(descending=True)
G = Grain_pred[types == 'G', 140]
_, id_G = G.sort(descending=True)

id_PS = info.loc[info.Type=='PS'].index[id_PS[0].tolist()]
id_F = info.loc[info.Type=='F'].index[id_F[0].tolist()]
id_DP = info.loc[info.Type=='DP'].index[id_DP[0].tolist()]
id_G = info.loc[info.Type=='G'].index[id_G[0].tolist()]

#%%
for i, loc in enumerate(['Ames', 'Craw', 'Green']):
    for year in range(15, 22):
        env_y = torch.load("../Result/Simulated_data/env/" + loc + '/' + str(year) + '/env.pt')
        if loc == 'Ames':
            mg_y = torch.load("../Result/Simulated_data/env/" + loc + '/' + str(year) + '/mg.pt')[[0]]
        else:
            mg_y = torch.load("../Result/Simulated_data/env/" + loc + '/' + str(year) + '/mg.pt').reshape(1,3)
        pheno_pred = simu_sorghum(env_y, mg_y, treno_pred[[id_G]])
        ax[1, i].plot(pheno_pred['GrainDry'][0, :140], label='20'+str(year))
        ax[1, i].set_ylim((0, 130))
        ax[1, i].set_xlim((0, 140))
        ax[1, i].legend()
        ax[1, i].set_title(loc)
# fig.savefig('../Grain_diff_' + loc + '.png', bbox_inches='tight', pad_inches=0.2)
plt.show()


#%%
year = 15
loc = 'Ames'

types = info.Type.to_numpy()

env_y = torch.load("../Result/Simulated_data/env/" + loc + '/' + str(year) + '/env.pt')
if loc == 'Ames':
    mg_y = torch.load("../Result/Simulated_data/env/" + loc + '/' + str(year) + '/mg.pt')[[0]]
else:
    mg_y = torch.load("../Result/Simulated_data/env/" + loc + '/' + str(year) + '/mg.pt').reshape(1,3)

pheno_pred = simu_sorghum(env_y.repeat(len(treno_pred), 1, 1), mg_y.repeat(len(treno_pred), 1), treno_pred)

Biomass_pred = pheno_pred['Biomass']
Height_pred = pheno_pred['Height']
Grain_pred = pheno_pred['GrainDry']
Yield_pred = pheno_pred['Yield']

PS = Biomass_pred[types == 'PS', 140]
# PS = Yield_pred[types == 'PS']
_, id_PS = PS.sort(descending=True)
F = Biomass_pred[types == 'F', 140]
_, id_F = F.sort(descending=True)
DP = Biomass_pred[types == 'DP', 140] + Grain_pred[types=='DP', 140]
_, id_DP = DP.sort(descending=True)
G = Grain_pred[types == 'G', 140]
_, id_G = G.sort(descending=True)


id_PS = info.loc[info.Type=='PS'].index[id_PS[:50].tolist()]
id_PS = info.iloc[id_PS].drop_duplicates(('Mother', 'Father')).index


id_F = info.loc[info.Type=='F'].index[id_F[:50].tolist()]
id_F = info.iloc[id_F].drop_duplicates(('Mother', 'Father')).index

id_G = info.loc[info.Type=='G'].index[id_G[:50].tolist()]
id_G = info.iloc[id_G].drop_duplicates(('Mother', 'Father')).index

id_DP = info.loc[info.Type=='DP'].index[id_DP[:50].tolist()]
id_DP = info.iloc[id_DP].drop_duplicates(('Mother', 'Father')).index

#%%
i_want = 'G'

if i_want == 'PS':
    id = id_PS[:10]
if i_want == 'F':
    id = id_F[:10]
if i_want == 'DP':
    id = id_DP[:10]
if i_want == 'G':
    id = id_G[:10]

res = torch.zeros((10, 3*7))
i = 0
for loc in ['Ames', 'Green', 'Craw']:
    for year in range(15, 22):
        print(loc, year)
        env_y = torch.load("../Result/Simulated_data/env/" + loc + '/' + str(year) + '/env.pt')
        if loc == 'Ames':
            mg_y = torch.load("../Result/Simulated_data/env/" + loc + '/' + str(year) + '/mg.pt')[[0]]
        else:
            mg_y = torch.load("../Result/Simulated_data/env/" + loc + '/' + str(year) + '/mg.pt').reshape(1, 3)

        mg_y = mg_y.repeat(len(id), 1)
        if (year == 18 and loc == 'Green'):
            mg_y[:, 2] = 5
        if year == 19:
            mg_y[:, 2] = 4
        if (year == 19 and loc == 'Ames'):
            mg_y[:, 2] = 5
        if year == 20:
            mg_y[:, 2] = 5
        # mg_y[:, 2] = 0
        # mg_y[:, 2] = mg[id, 2]

        pheno_pred = simu_sorghum(env_y.repeat(len(id), 1, 1), mg_y, treno_pred[[id]])
        if i_want == 'PS' or 'F':
            res[:, i] = pheno_pred['Yield']
        if i_want == 'DP':
            res[:, i] = pheno_pred['GrainDry'][:10, 120] + pheno_pred['Biomass'][:10, 120]
        if i_want == 'G':
            res[:, i] = pheno_pred['GrainDry'][:10, 120]
        i = i + 1

#%%
fig, ax = plt.subplots(figsize=(8, 5), dpi=400)

# Plot the heatmap
im = ax.imshow(res)

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax, fraction=0.023, pad=0.02, cmap='crest')
if i_want == 'PS' or 'F':
    cbar.ax.set_ylabel('Predicted Final Yield (Dry Leaf + Stem kg/hectare)', rotation=90, fontdict={'fontsize': 9})
    im.set_clim(0, 45000)
if i_want == 'DP':
    cbar.ax.set_ylabel('Predicted dry whole weight on Day 140 (g)', rotation=90, fontdict={'fontsize': 12})
    im.set_clim(0, 250)
if i_want == 'G':
    cbar.ax.set_ylabel('Predicted dry grain weight on Day 140 (g)', rotation=90, fontdict={'fontsize': 12})
    im.set_clim(0, 120)

# Show all ticks and label them with the respective list entries.
ax.set_xticks(np.arange(res.shape[1]), labels=[str(a)+b for b in ['Ames', 'Green', 'Craw'] for a in range(15, 22)], fontsize=8)
ax.set_yticks(np.arange(res.shape[0]), labels=[str(info.Mother.iloc[i]) + 'x' + str(info.Father.iloc[i]) for i in id[:10]], fontsize=8)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=90, ha="center",
         rotation_mode="default")

# Turn spines off and create white grid.
ax.spines[:].set_visible(False)

ax.set_xticks(np.arange(res.shape[1]+1)-.5, minor=True)
ax.set_yticks(np.arange(res.shape[0]+1)-.5, minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)
ax.set_axisbelow(True)

plt.xlabel('Environment', fontsize=12)
plt.ylabel('Genotype', fontsize=12)


# ax.add_patch(
#     patches.Rectangle(
#         (-0.5+0, -0.5+0),   # (x,y)
#         1,          # width
#         1,          # height
#         fill=False,
#         linewidth=2,
#         edgecolor='r',
#         label = ''
#     )
# )

for i in range(21):
    i_max = res[:, i].sort(descending=True)[1][0]
    print(i_max)
    ax.add_patch(
        patches.Rectangle(
            (-0.5 + i, -0.5 + i_max.tolist()),  # (x,y)
            1,  # width
            1,  # height
            fill=False,
            linewidth=1.5,
            edgecolor='r',
        )
    )

fig.savefig('../GxE_' + i_want + '.png', bbox_inches='tight', pad_inches=0.2)
plt.show()


#%%
fig, ax = plt.subplots(figsize=(20, 6), dpi=400)

plt.plot(env_15[0, :, 12], alpha=0.5)
plt.plot(env_17[0, :, 12], alpha=0.5)
plt.plot(env_20[0, :, 12], alpha=0.5)

plt.show()