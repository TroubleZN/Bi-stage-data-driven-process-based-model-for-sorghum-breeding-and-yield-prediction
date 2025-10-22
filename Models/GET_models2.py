#%%
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

# torch.autograd.set_detect_anomaly(True)
seed_num = 1314
torch.random.manual_seed(seed_num)
random.seed(seed_num)
np.random.seed(seed_num)


#%%
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
    return id_g_sort

def fix_sub_treno(treno, treno_true):
    N_treno = treno_true.shape[1] * treno_true.shape[2]
    treno_fixed = treno.clone()
    treno_fixed.flatten(1)[:, 0:N_treno:5] = treno_true.flatten(1)[:, 0:N_treno:5]
    return treno_fixed


def G0ET(Yield_true, envi, mg, info, father_dict, mother_dict,
        N_alleles=200,
        E_reduce=False,
        T_reduce=False,
        device='cpu',
        pretrained=False,
        lr_start=0.0001,
        plot=True):

    N_sample = len(info)

    Yield_true = Yield_true.to(device)
    envi = envi.to(device)
    mg = mg.to(device)

    if E_reduce:
        env_simple = envi.clone()
        env_simple[:, :, 2:5] = env_simple[:, :, 2:5].mean(dim=1).reshape(len(env_simple), 1, 3)
        env_simple[:, :, 6:9] = env_simple[:, :, 6:9].mean(dim=1).reshape(len(env_simple), 1, 3)
        envi = env_simple

    #%%
    Treno_0, Ltreno, Utreno, dtreno = make_treno()
    Treno_0 = torch.tensor(Treno_0).to(torch.float32)
    Utreno = torch.tensor(Utreno).to(torch.float32)
    Ltreno = torch.tensor(Ltreno).to(torch.float32)
    dtreno = torch.tensor(dtreno).to(torch.float32)

    #%% initialize the Layer1
    GroundTrueModel = MAP_GT(N_alleles=N_alleles)
    GroundTrueModel.start(N_eQTL=N_alleles, N_dominant=N_alleles)

    model_whole = Geno_to_Treno_MAP(GroundTrueModel.W_eQTL.shape[0] - 1, 153, GroundTrueModel).to(device)

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
        Yield_pred = Pheno_pred['Yield']
        return Yield_pred, Treno_pred

    if pretrained is not False:
        with open(pretrained, 'rb') as f:
        # with open('../Result/Active_training/starting_' + str(N_alleles) + '.pkl', 'rb') as f:
            res = pickle.load(f)
        Geno_F = res['g_father']
        Geno_M = res['g_mother']
        GroundTrueModel = res['model']

        torch.nn.init.zeros_(model_whole.linear_one.weight)
        torch.nn.init.zeros_(model_whole.linear_two.weight)
        torch.nn.init.zeros_(model_whole.linear_three.weight)
        torch.nn.init.zeros_(model_whole.linear_one.bias)
        torch.nn.init.zeros_(model_whole.linear_two.bias)
        torch.nn.init.zeros_(model_whole.linear_three.bias)

        model_whole.linear_one.weight.data += GroundTrueModel.W_eQTL[1:].t().to(device)
        model_whole.linear_one.bias.data += GroundTrueModel.W_eQTL[0].to(device)
        model_whole.linear_two.weight.data += GroundTrueModel.W_poly[1:].t().to(device)
        model_whole.linear_two.bias.data += GroundTrueModel.W_poly[0].to(device)
        model_whole.linear_three.weight.data += GroundTrueModel.W_fold[1:].t().to(device)
        model_whole.linear_three.bias.data += GroundTrueModel.W_fold[0].to(device)

    model_old = copy.deepcopy(model_whole.state_dict())
    model_start = copy.deepcopy(model_old)


    # %%
    finished = 0
    if pretrained:
        loss_test_old = 1
    else:
        loss_test_old = 1e17
    loss_train_old = 1e17
    loss_train = 1e17
    loss_test = 1e17
    i_fail = 0
    epochs = 10
    lr_start = lr_start

    #%%
    # config optimizer
    # optimizer = torch.optim.Adadelta(model_whole.parameters(), lr=0.1)
    optimizer = torch.optim.Adam(model_whole.parameters(), lr=lr_start)
    loss_fn = torch.nn.MSELoss()

    #%% splitting data
    generator0 = torch.Generator().manual_seed(seed_num)
    train_dataset, test_dataset = torch.utils.data.random_split(range(N_sample), [0.7, 0.3], generator=generator0)

    #%%
    while finished == 0:
        i_rand = random.randint(0, len(Geno_M) + len(Geno_F) - 1)
        i_pos = random.randint(N_alleles * 0.2, N_alleles - 1)
        Geno_M_t = copy.deepcopy(Geno_M)
        Geno_F_t = copy.deepcopy(Geno_F)
        if i_rand >= len(Geno_M):
            Geno_F_t[i_rand - len(Geno_M), i_pos] = 1 - Geno_F[i_rand - len(Geno_M), i_pos]
        else:
            Geno_M_t[i_rand, i_pos] = 1 - Geno_M[i_rand, i_pos]
        Geno_t = reproduce(Geno_F_t, Geno_M_t, father_dict, mother_dict, info)

        Geno_train = Geno_t[train_dataset]
        Yield_train = Yield_true[train_dataset]
        envi_train = envi[train_dataset]
        mg_train = mg[train_dataset]
        Geno_test = Geno_t[test_dataset]
        Yield_test = Yield_true[test_dataset]
        envi_test = envi[test_dataset]
        mg_test = mg[test_dataset]

        loss_increase = 0
        loss_train_last = 1e17
        bar = tqdm(range(epochs))
        bar.set_description('G1E{}T{}'.format(E_reduce*1, T_reduce*1))
        for epoch in bar:
            total = 0
            # making predictions with forward pass
            Yield_pred, Treno_pred = forward(Geno_train, envi_train, mg_train, device)

            # calculating the loss between original and predicted data points
            loss_train = RRMSE(Yield_train, Yield_pred)
            bar.set_postfix(loss=float(loss_train), lr=optimizer.param_groups[0]['lr'], fail=i_fail)

            if loss_train.data > loss_train_last:
                loss_increase += 1
            loss_train_last = loss_train.data

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
            torch.cuda.empty_cache()

            if loss_train.isnan():
                lr_start = lr_start / 10
                for g in optimizer.param_groups:
                    g['lr'] = lr_start
                i_fail = 0
                optimizer = torch.optim.Adam(model_whole.parameters(), lr=lr_start)
                model_whole.zero_grad()
                break


        with torch.no_grad():
            Yield_pred, Treno_pred = forward(Geno_train, envi_train, mg_train, device)
            loss_train = RRMSE(Yield_train, Yield_pred)

        if loss_train.isnan():
            model_whole.load_state_dict(model_start)
            loss_test_old = 100
            continue

        print('Train Loss = {},\tTrain Loss Old = {}'.format(loss_train.item(), loss_train_old))
        with torch.no_grad():
            Yield_pred_test, Treno_pred_test = forward(Geno_test, envi_test, mg_test, device)
            loss_test = RRMSE(Yield_test, Yield_pred_test)
            print('Test Loss = {},\tTest Loss Old = {}'.format(loss_test.item(), loss_test_old))

        # if (loss_test_old - loss_test) / loss_test_old <= 0.05 or ((loss_test - loss_train) / loss_train > 0.5 and loss_train < 0.3):
        if (loss_train_old - loss_train) / loss_train_old <= 0 or ((loss_test - loss_train) / loss_train > 1 and loss_train < 0.3):
            # model_whole = model_old
            model_whole.load_state_dict(model_old)
            i_fail += 1
            if i_fail >= 5 or loss_train < 0.05:
                finished = 1
            elif loss_increase >= 2 and loss_train_old <= 0.5:
                # halfing the learning rate when loss rising back
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / 5
            elif loss_increase < 2:
                epochs = epochs + 5
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] * 2
        else:
            loss_test_old = loss_test
            loss_train_old = loss_train
            # model_old = copy.copy(model_whole)
            model_old = copy.deepcopy(model_whole.state_dict())
            i_fail = 0

            # for g in optimizer.param_groups:
            #     g['lr'] = lr_start

            # Mapping from small model_whole to my GTM
            paras = list(model_whole.parameters())
            GroundTrueModel.W_eQTL[1:] = paras[0].t().detach()
            GroundTrueModel.W_eQTL[0] = paras[1].detach()
            GroundTrueModel.W_poly[1:] = paras[2].t().detach()
            GroundTrueModel.W_poly[0] = paras[3].detach()
            GroundTrueModel.W_fold[1:] = paras[4].t().detach()
            GroundTrueModel.W_fold[0] = paras[5].detach()

            with torch.no_grad():
                Yield_pred_all, Treno_pred_all = forward(Geno_t, envi, mg, device)

            res = {
                'model': GroundTrueModel,
                'g_father': Geno_F,
                'g_mother': Geno_M,
                'loss': loss_test_old.cpu(),
                # 'Yield_true': Yield_true.cpu(),
                # 'Yield_pred': Yield_pred_all.cpu(),
                # 'envi': envi.cpu(),
                # 'mg': mg.cpu(),
                # 'info': info,
            }

            with open('../Result/Active_training/G0E' + str(E_reduce*1) + 'T' + str(T_reduce*1) + '.pkl', 'wb') as f:
                pickle.dump(res, f)

            if plot:
                fig, ax = plt.subplots()
                ax.scatter(Yield_pred.detach().cpu(), Yield_train.detach().cpu(), marker='*', color='g', alpha=0.5, label='Train')
                ax.scatter(Yield_pred_test.detach().cpu(), Yield_test.detach().cpu(), marker='+', color='b', alpha=0.5, label='Test')
                y_range = np.linspace(5000, 45000, 100)
                ax.plot(y_range, y_range, color='r', lw=2.5, ls='--')
                plt.legend()
                plt.show()

            if optimizer.param_groups[0]['lr'] < 1e-7:
                break
    return res



def G1ET(Yield_true, envi, mg, info, father_dict, mother_dict,
        N_alleles=200,
        G_mother_true=0, G_father_true=0, G_rank=0,
        E_reduce=False,
        T_reduce=False,
        device='cpu',
        pretrained=False,
        lr_start=0.0001,
        plot=True):

    N_sample = len(info)

    Yield_true = Yield_true.to(device)
    envi = envi.to(device)
    mg = mg.to(device)

    if E_reduce:
        env_simple = envi.clone()
        env_simple[:, :, 2:5] = env_simple[:, :, 2:5].mean(dim=1).reshape(len(env_simple), 1, 3)
        env_simple[:, :, 6:9] = env_simple[:, :, 6:9].mean(dim=1).reshape(len(env_simple), 1, 3)
        envi = env_simple

    #%%
    Treno_0, Ltreno, Utreno, dtreno = make_treno()
    Treno_0 = torch.tensor(Treno_0).to(torch.float32)
    Utreno = torch.tensor(Utreno).to(torch.float32)
    Ltreno = torch.tensor(Ltreno).to(torch.float32)
    dtreno = torch.tensor(dtreno).to(torch.float32)

    #%% initialize the Layer1
    GroundTrueModel = MAP_GT(N_alleles=N_alleles)
    GroundTrueModel.start(N_eQTL=N_alleles, N_dominant=N_alleles)

    model_whole = Geno_to_Treno_MAP(GroundTrueModel.W_eQTL.shape[0] - 1, 153, GroundTrueModel).to(device)

    Geno_F = G_father_true[:, G_rank[:N_alleles]]
    Geno_M = G_mother_true[:, G_rank[:N_alleles]]

    def forward(Geno, envi, mg, device):
        N_sample = len(Geno)
        temp = model_whole(GroundTrueModel.Geno_to_eQTL(Geno).to(device)).reshape(N_sample, 3, 51)
        Treno_pred = temp * (Utreno.to(device) - Ltreno.to(device)) + Ltreno.to(device)
        Treno_pred = torch.maximum(Treno_pred, Ltreno.to(device))
        Treno_pred = torch.minimum(Treno_pred, Utreno.to(device))

        if T_reduce:
            Treno_pred = fix_sub_treno(Treno_pred, Treno_0.to(device))

        Pheno_pred = simu_sorghum(envi, mg, Treno_pred, device=device)
        Yield_pred = Pheno_pred['Yield']
        return Yield_pred, Treno_pred

    if pretrained is not False:
        with open(pretrained, 'rb') as f:
        # with open('../Result/Active_training/starting_' + str(N_alleles) + '.pkl', 'rb') as f:
            res = pickle.load(f)
        GroundTrueModel = res['model']

        torch.nn.init.zeros_(model_whole.linear_one.weight)
        torch.nn.init.zeros_(model_whole.linear_two.weight)
        torch.nn.init.zeros_(model_whole.linear_three.weight)
        torch.nn.init.zeros_(model_whole.linear_one.bias)
        torch.nn.init.zeros_(model_whole.linear_two.bias)
        torch.nn.init.zeros_(model_whole.linear_three.bias)

        model_whole.linear_one.weight.data += GroundTrueModel.W_eQTL[1:].t().to(device)
        model_whole.linear_one.bias.data += GroundTrueModel.W_eQTL[0].to(device)
        model_whole.linear_two.weight.data += GroundTrueModel.W_poly[1:].t().to(device)
        model_whole.linear_two.bias.data += GroundTrueModel.W_poly[0].to(device)
        model_whole.linear_three.weight.data += GroundTrueModel.W_fold[1:].t().to(device)
        model_whole.linear_three.bias.data += GroundTrueModel.W_fold[0].to(device)

    model_old = copy.deepcopy(model_whole.state_dict())
    model_start = copy.deepcopy(model_old)

    # %%
    finished = 0
    if pretrained:
        loss_test_old = 1
    else:
        loss_test_old = 1
    loss_train_old = 1e17
    i_fail = 0
    epochs = 10
    lr_start = lr_start

    #%%
    # config optimizer
    # optimizer = torch.optim.Adadelta(model_whole.parameters(), lr=0.1)
    optimizer = torch.optim.Adam(model_whole.parameters(), lr=lr_start)
    loss_fn = torch.nn.MSELoss()

    #%% splitting data
    generator0 = torch.Generator().manual_seed(seed_num)
    train_dataset, test_dataset = torch.utils.data.random_split(range(N_sample), [0.6, 0.4], generator=generator0)

    Geno_t = reproduce(Geno_F, Geno_M, father_dict, mother_dict, info)

    Geno_train = Geno_t[train_dataset]
    Yield_train = Yield_true[train_dataset]
    envi_train = envi[train_dataset]
    mg_train = mg[train_dataset]
    Geno_test = Geno_t[test_dataset]
    Yield_test = Yield_true[test_dataset]
    envi_test = envi[test_dataset]
    mg_test = mg[test_dataset]

    #%%
    while finished == 0:
        loss_train_last = torch.inf
        loss_increase = 0
        bar = tqdm(range(epochs))
        bar.set_description('G1E{}T{}'.format(E_reduce*1, T_reduce*1))
        for epoch in bar:
        # for epoch in range(epochs):
            total = 0
            # making predictions with forward pass
            Yield_pred, Treno_pred = forward(Geno_train, envi_train, mg_train, device)

            # calculating the loss between original and predicted data points
            loss_train = RRMSE(Yield_train, Yield_pred)
            bar.set_postfix(loss=float(loss_train), lr=optimizer.param_groups[0]['lr'], fail=i_fail)

            if loss_train.data > loss_train_last:
                loss_increase += 1
            loss_train_last = loss_train.data

            # backward pass for computing the gradients of the loss w.r.t to learnable parameters
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            torch.cuda.empty_cache()

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

        print('Train Loss = {},\tTrain Loss Old = {}'.format(loss_train.item(), loss_train_old))
        with torch.no_grad():
            Yield_pred_test, Treno_pred_test = forward(Geno_test, envi_test, mg_test, device)
            loss_test = RRMSE(Yield_test, Yield_pred_test)
            print('Test Loss = {},\tTest Loss Old = {}'.format(loss_test.item(), loss_test_old))

        # if (loss_test_old - loss_test) / loss_test_old <= 0.05 or ((loss_test - loss_train) / loss_train > 0.5 and loss_train < 0.3):
        if (loss_train_old - loss_train) / loss_train_old <= 0 or ((loss_test - loss_train) / loss_train > 1 and loss_train < 0.3):
            # model_whole = model_old
            model_whole.load_state_dict(model_old)
            i_fail += 1
            if i_fail >= 5 or loss_train < 0.05:
                finished = 1
            elif loss_increase >= 2 and loss_train_old <= 0.5:
                # halfing the learning rate when loss rising back
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / 5
            elif loss_increase < 2:
                epochs = epochs + 5
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] * 2
        else:
            if 0 < (loss_train_old - loss_train) / loss_train_old < 0.05:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] * 2
            loss_test_old = loss_test
            # model_old = copy.copy(model_whole)
            model_old = copy.deepcopy(model_whole.state_dict())
            i_fail = 0

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
                'loss_train': loss_train.cpu(),
                'loss_test': loss_test.cpu(),
            }

            with open('../Result/Active_training/G1E' + str(E_reduce*1) + 'T' + str(T_reduce*1) + '.pkl', 'wb') as f:
                pickle.dump(res, f)

            if plot:
                fig, ax = plt.subplots()
                ax.scatter(Yield_pred.detach().cpu(), Yield_train.detach().cpu(), marker='*', color='g', alpha=0.5, label='Train')
                ax.scatter(Yield_pred_test.detach().cpu(), Yield_test.detach().cpu(), marker='+', color='b', alpha=0.5, label='Test')
                y_range = np.linspace(5000, 45000, 100)
                ax.plot(y_range, y_range, color='r', lw=2.5, ls='--')
                plt.legend()
                plt.show()
        if optimizer.param_groups[0]['lr'] < 1e-10:
            break
    return res


def GET(Yield_true, envi, mg, info, father_dict, mother_dict,
        N_alleles=200,
        G_true=False, G_mother_true=0, G_father_true=0, G_rank=0,
        E_reduce=False,
        T_reduce=False,
        device='cpu',
        pretrained=False,
        lr_start=0.0001,
        plot=False):
    if G_true:
        return G1ET(Yield_true, envi, mg, info, father_dict, mother_dict,
                    N_alleles=N_alleles,
                    G_mother_true=G_mother_true, G_father_true=G_father_true, G_rank=G_rank,
                    E_reduce=E_reduce,
                    T_reduce=T_reduce,
                    device=device,
                    pretrained=pretrained,
                    lr_start=lr_start,
                    plot=plot)
    else:
        return G0ET(Yield_true, envi, mg, info, father_dict, mother_dict,
                    N_alleles=N_alleles,
                    E_reduce=E_reduce,
                    T_reduce=T_reduce,
                    device=device,
                    pretrained=pretrained,
                    lr_start=lr_start,
                    plot=plot)


#%%
if __name__ == '__main__':
    # %% load simulated data
    with open('../../Data/simulated_data.pkl', 'rb') as f:
        data = pickle.load(f)
        print('simulated data loaded successfully from file')

    Pheno = data['yield']
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

    # %% starting point
    info = info.drop_duplicates(['Mother', 'Father'], ignore_index=True).loc[:, ['Type', 'Mother', 'Father']];
    N_sample = info.__len__()
    Geno = reproduce(g_father, g_mother, father_dict, mother_dict, info)
    Treno = model.get_Treno(Geno)

    # %% initialize the model
    N_alleles = 200
    Year = 15
    Loc = 'Ames'
    device = 'cpu'
    E_reduce = False
    T_reduce = False
    G_true = True

    # load environment data
    envi_dir = '../Result/Simulated_data/env/' + Loc + '/' + str(Year) + '/env.pt'
    envi = torch.load(envi_dir).repeat(N_sample, 1, 1).to(device)

    # simulate random management data
    mg = torch.tensor([[144 + random.randint(-5, 5),
                        np.random.randn() / 10 + 1.1,
                        random.choices([1, 2, 3, 4, 5], [0.3, 0.3, 0.2, 0.1, 0.1])[0]] for i in range(N_sample)]).to(
        device)

    Treno = Treno.to(device)
    Pheno_true = simu_sorghum(envi, mg, Treno, device=device)
    Yield_true = Pheno_true['Yield']

    G_rank = rank_geno(Yield_true.cpu(), Geno)

    #%%
    res = GET(Yield_true, envi, mg, info, father_dict, mother_dict,
              N_alleles=N_alleles,
              G_true=G_true, G_mother_true=g_mother, G_father_true=g_father, G_rank=G_rank,
              E_reduce=E_reduce,
              T_reduce=T_reduce,
              device=device)

