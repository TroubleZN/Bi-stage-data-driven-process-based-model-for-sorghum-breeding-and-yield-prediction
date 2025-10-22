# %%
import copy
import pickle
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from GenoToTreno.GenoToTreno import MAP_GT
from TrenoToPheno.make_treno import make_treno
from TrenoToPheno.gSimulator import simu_sorghum
from Utilities.utilities import reproduce, RRMSE
from Models.GenoToTreno import Geno_to_Treno_MAP
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# torch.autograd.set_detect_anomaly(True)
seed_num = 127
torch.random.manual_seed(seed_num)
random.seed(seed_num)
np.random.seed(seed_num)

# %%
def rank_geno(Yield, Geno):
    y = Yield.flatten()
    X_all = Geno.sum(2)
    Scores = torch.zeros(X_all.shape[1])
    for i in range(X_all.shape[1]):
        X = X_all[:, i].reshape(-1, 1)
        X = torch.concat((X, X == 1), 1).to(torch.float)
        reg = LinearRegression().fit(X, y)
        Scores[i] = reg.score(X, y)
    v, id_g_sort = Scores.sort(descending=True)

    # y = Yield.flatten().numpy()
    # X_all = Geno.sum(2).numpy()
    # mod = sm.OLS(y, X_all)
    # fii = mod.fit()
    # p_values = fii.summary2().tables[1]['P>|t|'].to_numpy()
    # p_values = torch.tensor(p_values.tolist())
    # v, id_g_sort = p_values.sort()
    return id_g_sort


def fix_sub_treno(treno, treno_true):
    N_treno = treno_true.shape[1] * treno_true.shape[2]
    treno_fixed = treno.clone()
    treno_fixed.flatten(1)[:, 0:N_treno:5] = treno_true.flatten(1)[:, 0:N_treno:5]
    return treno_fixed


def RRMSE(x, x_pred):
    x_flat = x.flatten()[~x.flatten().isnan()]
    x_pred_flat = x_pred.flatten()[~x.flatten().isnan()]
    return (x_flat - x_pred_flat).square().mean().sqrt() / x_pred_flat.mean()


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
        PM_mean = (Yield_F['Yield'][Father_dict[info.Father.iloc[i]]] + Yield_M['Yield'][
            Mother_dict[info.Mother.iloc[i]]]) / 2
        Yield_pred_std[i] = (Yield_pred['Yield'][i] - PM_mean) / PM_mean
    return Yield_pred_std

def sub_pheno(pheno, index):
    pheno_new = copy.deepcopy(pheno)
    pheno_new['Yield'] = pheno['Yield'][index]
    pheno_new['Biomass'] = pheno['Biomass'][index]
    pheno_new['Height'] = pheno['Height'][index]
    return pheno_new



def GET(pheno, envi, mg, info, father_dict, mother_dict,
        N_alleles=200,
        G_true=False, G_mother_true=0, G_father_true=0, G_rank=0,
        E_reduce=False,
        T_reduce=False,
        device='cpu',
        pretrained=False,
        lr_start=0.0001,
        plot=False):

    envi = envi.to(device)
    mg = mg.to(device)

    if E_reduce:
        env_simple = envi.clone()
        env_simple[:, :, 2:5] = env_simple[:, :, 2:5].mean(dim=1).reshape(len(env_simple), 1, 3)
        env_simple[:, :, 6:9] = env_simple[:, :, 6:9].mean(dim=1).reshape(len(env_simple), 1, 3)
        envi = env_simple

    # %%
    Treno_0, Ltreno, Utreno, dtreno = make_treno()
    Treno_0 = torch.tensor(Treno_0).to(torch.float32)
    Utreno = torch.tensor(Utreno).to(torch.float32)
    Ltreno = torch.tensor(Ltreno).to(torch.float32)
    dtreno = torch.tensor(dtreno).to(torch.float32)

    # %% initialize the Layer1
    GroundTrueModel = MAP_GT(N_alleles=N_alleles)
    GroundTrueModel.start(N_eQTL=N_alleles, N_dominant=N_alleles)

    model_whole = Geno_to_Treno_MAP(GroundTrueModel.W_eQTL.shape[0] - 1, 153, GroundTrueModel).to(device)

    if G_true:
        Geno_F = G_father_true[:, G_rank[:N_alleles]]
        Geno_M = G_mother_true[:, G_rank[:N_alleles]]
    else:
        Geno_F = torch.randint(2, (len(father_dict), GroundTrueModel.N_alleles))
        Geno_M = torch.randint(2, (len(mother_dict), GroundTrueModel.N_alleles))

    def forward(Geno, envi, mg, device):
        N_sample = len(Geno)
        temp = model_whole(GroundTrueModel.Geno_to_eQTL(Geno).to(device)).reshape(N_sample, 3, 51)
        Treno_pred = temp * (Utreno.to(device) - Ltreno.to(device)) + Ltreno.to(device)
        Treno_pred = torch.maximum(Treno_pred, Ltreno.to(device))
        Treno_pred = torch.minimum(Treno_pred, Utreno.to(device))

        if T_reduce:
            Treno_pred = fix_sub_treno(Treno_pred, Treno_0.to(device))

        Pheno_pred = simu_sorghum(envi, mg, Treno_pred, device=device)
        return Pheno_pred, Treno_pred

    if pretrained is not False:
        with open(pretrained, 'rb') as f:
            res = pickle.load(f)

        GroundTrueModel_saved = res['model']
        Geno_F_saved = res['g_father']
        Geno_M_saved = res['g_mother']
        N_alleles_saved = Geno_F_saved.shape[1]

        torch.nn.init.zeros_(model_whole.linear_one.weight)
        torch.nn.init.zeros_(model_whole.linear_two.weight)
        torch.nn.init.zeros_(model_whole.linear_three.weight)
        torch.nn.init.zeros_(model_whole.linear_one.bias)
        torch.nn.init.zeros_(model_whole.linear_two.bias)
        torch.nn.init.zeros_(model_whole.linear_three.bias)

        if N_alleles_saved == N_alleles:
            Geno_F = Geno_F_saved
            Geno_M = Geno_M_saved
            GroundTrueModel = GroundTrueModel_saved
            model_whole.linear_one.weight.data += GroundTrueModel.W_eQTL[1:].t().to(device)
            model_whole.linear_one.bias.data += GroundTrueModel.W_eQTL[0].to(device)
            model_whole.linear_two.weight.data += GroundTrueModel.W_poly[1:].t().to(device)
            model_whole.linear_two.bias.data += GroundTrueModel.W_poly[0].to(device)
            model_whole.linear_three.weight.data += GroundTrueModel.W_fold[1:].t().to(device)
            model_whole.linear_three.bias.data += GroundTrueModel.W_fold[0].to(device)
        else:
            Geno_F[:, :N_alleles] = Geno_F_saved
            Geno_M[:, :N_alleles] = Geno_M_saved
            model_whole.linear_one.weight.data += GroundTrueModel.W_eQTL[1:].t().to(device)
            model_whole.linear_one.bias.data += GroundTrueModel.W_eQTL[0].to(device)
            model_whole.linear_two.weight.data += GroundTrueModel.W_poly[1:].t().to(device)
            model_whole.linear_two.bias.data += GroundTrueModel.W_poly[0].to(device)
            model_whole.linear_three.weight.data += GroundTrueModel.W_fold[1:].t().to(device)
            model_whole.linear_three.bias.data += GroundTrueModel.W_fold[0].to(device)

    Geno = reproduce(Geno_F, Geno_M, father_dict, mother_dict, info)
    Geno_all = Geno.sum(2)

    Geno_all = Geno_all.to(device)
    Yield_all = copy.deepcopy(pheno)
    Yield_all['Yield'] = pheno['Yield'].to(device)
    Yield_all['Biomass'] = pheno['Biomass'].to(device)
    Yield_all['Height'] = pheno['Height'].to(device)
    envi_all = envi.to(device)
    mg_all = mg.to(device)

    model_old = copy.deepcopy(model_whole.state_dict())
    model_start = copy.deepcopy(model_old)

    # %%
    finished = 0
    i_fail = 0
    epochs = 5
    lr_start = 1e-6
    loss_all_best = 100

    # config optimizer
    optimizer = torch.optim.Adam(model_whole.parameters(), lr=lr_start)

    # %%
    while i_fail <= 50:
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
            loss_1 = loss_fn(Yield_train, Yield_pred)
            loss_train_start = loss_1

        loss_increase = 0
        bar = tqdm(range(epochs))

        # for (Yield_train_t, Geno_train_t, envi_train_t, mg_train_t) in bar:
        for i in bar:
            total = 0
            # making predictions with forward pass
            Yield_pred, Treno_pred = forward(Geno_train, envi_train, mg_train, device)
            Yield_pred['Biomass'] = Yield_pred['StemDry'] + Yield_pred['LeafDry']

            # calculating the loss between original and predicted data points
            loss_1 = loss_fn(Yield_train, Yield_pred)

            loss_train = loss_1
            bar.set_postfix(loss1=float(loss_1),  lr=optimizer.param_groups[0]['lr'], fail=i_fail)
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

            # calculating the loss between original and predicted data points
            loss_1 = loss_fn(Yield_all, Yield_pred_all)
            loss_all = loss_1

            print('Overall Loss = {}'.format(loss_all.item()))

        # loss_test_wrong += 1
        if (loss_train - loss_train_start) / loss_train_start >= 0:
            # model_whole = model_old
            model_whole.load_state_dict(model_old)
            i_fail += 1
            # halfing the learning rate when loss rising back
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / 2
        elif (loss_all - loss_all_best) / loss_all_best >= 0:
            # model_whole = model_old
            model_whole.load_state_dict(model_old)
            i_fail += 1
        else:
            if 0 < (loss_train_start - loss_train) / loss_train_start < 0.001:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] * 10
            loss_all_best = loss_all
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
                'mother_dict': mother_dict,
                'father_dict': father_dict,
                # 'torch_model': model_whole.state_dict(),
                'info': info.reset_index().iloc[:, 1:],
            }

            with open('../Result/G'+ str(G_true*1) +'E' + str(E_reduce * 1) + 'T' + str(T_reduce * 1) + '.pkl', 'wb') as f:
                pickle.dump(res, f)
    return res
