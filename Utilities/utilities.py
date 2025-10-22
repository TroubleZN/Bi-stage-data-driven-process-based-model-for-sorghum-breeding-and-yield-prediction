import numpy as np
import pandas as pd
import torch

#%%
def make_info(Fathers, Mothers):
    return pd.DataFrame({
        'Mother': Mothers,
        'Father': Fathers,
    })

def expend_info(old_info, Fathers=False, Mothers=False, new_info=False):
    Fathers_old = old_info.Father.to_numpy()
    Mothers_old = old_info.Mother.to_numpy()
    if new_info is not False:
        Fathers = new_info.Father.to_numpy()
        Mothers = new_info.Mother.to_numpy()
    Fathers_new = np.append(Fathers_old, np.array(Fathers))
    Mothers_new = np.append(Mothers_old, np.array(Mothers))
    return make_info(Fathers_new, Mothers_new)

#%%
def reproduce(G_F, G_M, Father_dict, Mother_dict, info):
    N_sample = len(info)
    Geno = torch.randint(2, (N_sample, G_F.shape[-1], 2))
    for i in range(N_sample):
        Geno[i, :, 1] = G_M[Mother_dict[info['Mother'].iloc[i]], :]
        Geno[i, :, 0] = G_F[Father_dict[info['Father'].iloc[i]], :]
    return Geno

def RRMSE(Yield_true, Yield_pred):
    # return torch.matmul((Yield_true - Yield_pred).square(), Yield_true/Yield_true.sum()).sqrt() / Yield_true.mean()
    # return torch.matmul((Yield_true - Yield_pred).square(), Yield_true/Yield_true.sum()).sqrt() / Yield_pred.mean()
    return (Yield_true - Yield_pred).square().mean().sqrt() / Yield_pred.mean()
    # return (Yield_true - Yield_pred).square().mean().sqrt()


def listdiff(li1, li2):
    temp3 = []
    for element in li1:
        if element not in li2:
            temp3.append(element)
    return temp3

def get_loc(list0, list_all):
    set0 = set(tuple(i) for i in list0)
    index = []
    for i in range(len(list_all)):
        if tuple(list_all[i]) in set0:
            index.append(i)
    rank = []
    list_new = np.array(list_all)[index].tolist()
    for i in range(len(list0)):
        for j in range(len(list_new)):
            if list0[i] == list_new[j]:
                rank.append(j)
    return np.array(index)[rank].tolist()


def get_candidate(N_pairs, Yield, info, father_dict, mother_dict):
    N_father = len(father_dict)
    N_mother = len(mother_dict)
    Fathers = np.array(list(father_dict.keys()))
    Mothers = np.array(list(mother_dict.keys()))

    S_father = np.zeros(N_father)
    S_mother = np.zeros(N_mother)
    for id in father_dict.keys():
        S_father[father_dict[id]] = Yield[info.Father == id].mean()
    for id in mother_dict.keys():
        S_mother[mother_dict[id]] = Yield[info.Mother == id].mean()

    S_father = torch.tensor(S_father)
    _, id_f = S_father.sort(descending=True)
    S_mother = torch.tensor(S_mother)
    _, id_m = S_mother.sort(descending=True)

    Mothers_new = Mothers[id_m[:N_pairs].numpy()].repeat(N_father)
    Fathers_new = np.tile(Fathers, N_pairs)

    Mothers_new = np.concatenate((Mothers_new, np.tile(Mothers, N_pairs)))
    Fathers_new = np.concatenate((Fathers_new, Fathers[id_f[:N_pairs].numpy()].repeat(N_mother)))

    info_new = make_info(Fathers_new, Mothers_new)
    return info_new