#%% package importation
import os
import random

import torch
import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm


from GenoToTreno.GenoToTreno import MAP_GT
from TrenoToPheno.make_treno import make_treno
from TrenoToPheno.main import forward, get_loss
from Simulated_GroundTrue_main import Geno_to_Treno_MAP, get_sub_data
from TrenoToPheno.gSimulator import simu_sorghum


#%%
class myModel(object):
    _, Ltreno, Utreno, dtreno = make_treno()
    Ltreno = torch.tensor(Ltreno)
    Utreno = torch.tensor(Utreno)
    dtreno = torch.tensor(dtreno)

    def __init__(self, n_samples=200, n_alleles=20):
        self.n_alleles = n_alleles
        self.n_samples = n_samples
        # self.GT_map = torch.rand((2*n_alleles+1, 153))
        # self.GT_map[n_alleles:2*n_alleles, :] = 0
        self.GT_map = MAP_GT(N_alleles=n_alleles)
        self.GT_map.start(N_eQTL=n_alleles, N_dominant=n_alleles)

    def geno_to_treno(self, geno, device='cpu'):
        Treno_t = self.GT_map.get_Treno(geno).to(device)
        return Treno_t

    def treno_to_pheno(self, treno, env, mg, device='cpu'):
        env = env.to(device).detach()
        mg = mg.to(device).detach()
        treno = treno.to(device).detach()
        out = simu_sorghum(env, mg, treno, device=device)
        out['Yield'] = out['Yield'].reshape(len(out['Yield']), 1)
        out['Biomass'] = out['StemDry'] + out['LeafDry']
        return out

    def forward(self, geno, env, mg, device='cpu'):
        geno = geno.detach()
        env = env.to(device).detach()
        mg = mg.to(device).detach()
        treno = self.geno_to_treno(geno, device).detach()
        out = simu_sorghum(env, mg, treno, device=device)
        out['Yield'] = out['Yield'].reshape(len(out['Yield']), 1)
        out['Biomass'] = out['StemDry'] + out['LeafDry']
        return out


#%%
def reproduce(Mothers, G_Mother, Mother_Dict, Fathers, G_Father, Father_Dict):
    N = len(Mothers)
    G_child = torch.zeros((N, G_Father.shape[1]))
    for i in range(N):
        G_child[i] = G_Mother[Mother_Dict[Mothers[i]]] + G_Father[Father_Dict[Fathers[i]]]
    return G_child


def load_simulated_data():
    with open('../Data/simulated_data.pkl', 'rb') as f:
        data = pickle.load(f)
        print('simulated data loaded successfully from file')

    Y = data['yield']
    Treno = data['treno']
    info = data['info']
    types = data['types']
    envi = data['envi']
    mg = data['mg']
    id = data['id']
    model = data['GenoToTreno']
    genotype_father = data['genotype_father']
    genotype_mother = data['genotype_mother']
    father_dict = data['father_dict']
    mother_dict = data['mother_dict']

    return Y, Treno, info, types, envi, mg, id, model, mother_dict, father_dict, genotype_father, genotype_mother


def get_loss(y_true, y_pred, loss_type='rrmse'):
    Yield_diff = y_true['Yield'].cpu() - y_pred['Yield'].flatten().cpu()
    Biomass_diff = y_true['Biomass'].cpu() - y_pred['Biomass'].cpu()
    Height_diff = y_true['Height'].cpu() - y_pred['Height'].cpu()

    # diff = torch.cat([Yield_diff.flatten(), Biomass_diff.flatten(), Height_diff.flatten()])
    Yield_rmse = Yield_diff.square().nanmean().sqrt()
    Biomass_rmse = Biomass_diff.square().nanmean().sqrt()
    Height_rmse = Height_diff.square().nanmean().sqrt()

    if loss_type == 'rrmse':
        Yield_rrmse = Yield_rmse / y_true['Yield'].nanmean()
        Biomass_rrmse = Biomass_rmse / y_true['Biomass'].nanmean()
        Height_rrmse = Height_rmse / y_true['Height'].nanmean()
        # return (Yield_rrmse + Biomass_rrmse + Height_rrmse) / 3
        return Biomass_rrmse


def save_temp(myModel, g_mother, g_father, pheno, rrmse):
    res = {
        'model_whole': myModel.GT_map,
        'g_mother': g_mother,
        'g_father': g_father,
        'pheno': pheno,
        'rrmse': rrmse,
    }
    with open('../Result/res.pkl', 'wb') as f:
        pickle.dump(res, f)
        print(' Temporary results saved successfully to file !!')


def train(Pheno, myModel,
          Mothers, g_mother, Mother_dict, Fathers, g_father, Father_dict,
          envi, mg, types, device):

    g_child = reproduce(Mothers, g_mother, Mother_dict, Fathers, g_father, Father_dict)
    treno_t = myModel_simple.geno_to_treno(g_child, device=device)
    out = myModel_simple.treno_to_pheno(treno_t, envi, mg, device=device)

    current_loss = get_loss(out, Pheno)
    pheno_best = out

    if os.path.isfile('../Result/start.pkl'):
        with open('../Result/start.pkl', 'rb') as f:
            res = pickle.load(f)
        if res['rrmse'] < current_loss:
            current_loss = res['rrmse']
            myModel.GT_map = res['model_whole']
            g_mother = res['g_mother']
            g_father = res['g_father']
            pheno_best = res['pheno']
            print('The loaded loss:' + str(current_loss))

    n_mothers = len(g_mother)
    n_fathers = len(g_father)
    i_try = 0
    i_fail = 0
    d_step = 1
    while current_loss > 0.1:
        # try geno
        print('Starting to try geno!')
        flag_break = 0

        loop = tqdm(range(20))
        loop.set_description(f"Try Geno! Epoch")
        for epoch in loop:
            i_try += 1
            ForM = random.randint(0, 1)
            g_mother_new = copy.deepcopy(g_mother)
            g_father_new = copy.deepcopy(g_father)
            if ForM == 0:
                i = random.randint(0, n_mothers-1)
                j = random.randint(0, n_alleles-1)
                g_mother_new[i, j] = 1 - g_mother_new[i, j]
            else:
                i = random.randint(0, n_fathers-1)
                j = random.randint(0, n_alleles-1)
                g_father_new[i, j] = 1 - g_father_new[i, j]

            g_child = reproduce(Mothers, g_mother_new, Mother_dict, Fathers, g_father_new, Father_dict)
            treno_t = myModel_simple.geno_to_treno(g_child, device=device)
            out = myModel_simple.treno_to_pheno(treno_t, envi, mg, device=device)
            new_loss = get_loss(out, Pheno)

            if new_loss < current_loss:
                g_mother = g_mother_new
                g_father = g_father_new
                current_loss = new_loss
                pheno_best = out
                flag_break += 1
                i_try = 0
            loop.set_postfix(loss=current_loss.tolist(), N_try=i_try, step=d_step)
            if flag_break == 10:
                break
        save_temp(myModel, g_mother, g_father, pheno_best, current_loss)

        # try GT_map
        print('Starting to try GT_map!')

        loop = tqdm(range(100))
        loop.set_description(f"Try GT_map! Epoch")
        flag_break = 0
        d_step = 1
        i_fail = 0
        for epoch in loop:
            # if i_fail % 40 == 1:
            #     d_step = d_step / 2

            i_change = random.randint(0, 2)
            if i_change == 0:
                W = myModel.GT_map.W_eQTL
                dim_W = W.shape
            elif i_change == 1:
                W = myModel.GT_map.W_poly
                dim_W = W.shape
            elif i_change == 2:
                W = myModel.GT_map.W_fold
                dim_W = W.shape

            d_try = torch.rand(dim_W)/100
            d_try[d_try < 0.8/100] = 0

            d_t = torch.rand(dim_W)
            d_try = copy.deepcopy(W).detach()
            d_try[d_t < 0.98] = 0
            d_try = d_try * 0.001

            i_try += 1
            flag_break = 0
            GT_map_old = copy.deepcopy(myModel.GT_map)
            for rate in [-1, -2/3, -1/3, 1/3, 2/3, 1]:
                W = W + d_step * d_try * rate

                g_child = reproduce(Mothers, g_mother, Mother_dict, Fathers, g_father, Father_dict)
                treno_t = myModel_simple.geno_to_treno(g_child, device=device)
                out = myModel_simple.treno_to_pheno(treno_t, envi, mg, device=device)
                new_loss = get_loss(out, Pheno)

                if new_loss < current_loss:
                    current_loss = new_loss
                    pheno_best = out
                    flag_break += 1
                    i_fail = 0
                    d_step = 1
                else:
                    myModel.GT_map = GT_map_old
                    i_fail += 1
                    if i_fail % 10 == 1:
                        d_step = d_step / 2
                loop.set_postfix(loss=current_loss.tolist(), N_fail=i_fail, step=rate)
                if flag_break:
                    break
        save_temp(myModel, g_mother, g_father, pheno_best, current_loss)


#%%
if __name__ == '__main__':
    #%%
    device = 'cuda'

    Y, Treno, info, types, envi, mg, id, model_true, Mother_dict, Father_dict, _, _ = load_simulated_data()

    Mothers = info.Mother.to_numpy()
    Fathers = info.Father.to_numpy()

    n_samples = len(info)
    n_fathers = Father_dict.__len__()
    n_mothers = Mother_dict.__len__()
    n_alleles = 200

    g_mother = torch.randint(2, (n_mothers, n_alleles)).to(torch.float)
    g_father = torch.randint(2, (n_fathers, n_alleles)).to(torch.float)

    g_child = reproduce(Mothers, g_mother, Mother_dict, Fathers, g_father, Father_dict).to(device)

    myModel_simple = myModel(n_samples, n_alleles)
    treno = myModel_simple.geno_to_treno(g_child.cpu(), device=device)

    envi = envi.to(device)
    mg = mg.to(device)
    Y['Biomass'] = Y['Biomass'].detach().to(device)
    Y['Height'] = Y['Height'].detach().to(device)
    Y['Yield'] = Y['Yield'].detach().to(device)

    g_child = reproduce(Mothers, g_mother, Mother_dict, Fathers, g_father, Father_dict)
    treno_t = myModel_simple.geno_to_treno(g_child, device=device)
    out = myModel_simple.treno_to_pheno(treno_t, envi, mg, device=device)

    i_train = torch.randint(0, len(Mothers), (100,))
    i_train = torch.linspace(0, len(Mothers)-1, len(Mothers)).to(int)

    Y_train = copy.deepcopy(Y)
    Y_train['Biomass'] = Y_train['Biomass'][i_train]
    Y_train['Height'] = Y_train['Height'][i_train]
    Y_train['Yield'] = Y_train['Yield'][i_train]

    #%%
    train(Y_train, myModel_simple,
          Mothers[i_train], g_mother, Mother_dict, Fathers[i_train], g_father, Father_dict,
          envi[i_train], mg[i_train], types[i_train], device)

    #%%
    F = Y[types=='F']
    PS = Y[types=='PS']
    G = Y[types=='G']
    DP = Y[types=='DP']

    Days = 120
    fig, ax = plt.subplots()
    ax.plot(torch.linspace(0, Days, Days), F.median(0).values[1:Days+1].to('cpu'), label='F')
    ax.plot(torch.linspace(0, Days, Days), PS.median(0).values[1:Days+1].to('cpu'), label='PS')
    ax.plot(torch.linspace(0, Days, Days), G.median(0).values[1:Days+1].to('cpu'), label='G')
    ax.plot(torch.linspace(0, Days, Days), DP.median(0).values[1:Days+1].to('cpu'), label='DP')
    ax.set(xlabel='Day(s)', ylabel='Biomass')
    ax.legend()
    plt.show()
    # fig.savefig('Median_of_biomass.png', dpi=300)


