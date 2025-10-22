import time

import torch
import numpy as np

def PhenotypicSelection(N_pairs, Yield, info, father_dict, mother_dict, Yield_true=None, info_true=None):
    N_father = len(father_dict)
    N_mother = len(mother_dict)
    Fathers = list(father_dict.keys())
    Mothers = list(mother_dict.keys())

    Yield_t = Yield.cpu().numpy()
    # ts = time.time()
    S_father = np.zeros(N_father)
    S_mother = np.zeros(N_mother)
    for id in father_dict.keys():
        if len(Yield_t[info.Father == id]) != 0:
            # if len(Yield_t[info.Father == id]) >= 10:
            #     S_father[father_dict[id]] = np.nanmean(Yield_t[info.Father == id][:10])
            # else:
            #     S_father[father_dict[id]] = np.nanmean(Yield_t[info.Father == id])
            S_father[father_dict[id]] = np.nanmean(Yield_t[info.Father == id])
        # else:
        #     S_father[father_dict[id]] = 0
    # print(time.time()-ts)
    #
    # ts = time.time()
    # S_father2 = [np.nanmean(Yield_t[info.Father == id]) if len(Yield_t[info.Father == id]) != 0 else 0 for id in father_dict.keys()]
    # print(time.time()-ts)

    for id in mother_dict.keys():
        if len(Yield_t[info.Mother == id]) != 0:
            # if len(Yield_t[info.Mother == id]) >= 10:
            #     S_mother[mother_dict[id]] = np.nanmean(Yield_t[info.Mother == id][:10])
            # else:
            #     S_mother[mother_dict[id]] = np.nanmean(Yield_t[info.Mother == id])
            S_mother[mother_dict[id]] = np.nanmean(Yield_t[info.Mother == id])
    # else:
        #     S_mother[mother_dict[id]] = 0

    S_child = np.zeros(N_father * N_mother)
    pair_child = np.zeros((N_father * N_mother, 2))
    id = 0
    for i in range(N_father):
        for j in range(N_mother):
            S_child[id] = (S_father[i] + S_mother[j])/2
            pair_child[id, :] = [Fathers[i], Mothers[j]]
            id += 1

    if Yield_true is not None:
        info_true_np = info_true.to_numpy()
        Yield_true_t = Yield_true.cpu().numpy()
        for i in range(len(info_true_np)):
            S_child[(pair_child == info_true_np[i]).sum(1) == 2] = (S_child[(pair_child == info_true_np[i]).sum(1) == 2] + Yield_true_t[i])/2

    S_child = torch.tensor(S_child)
    _, id = S_child.sort(descending=True)

    return pair_child[id[:N_pairs]]


#%%
def PhenotypicPrediction(Yield, info_test, info, father_dict, mother_dict):
    N_father = len(father_dict)
    N_mother = len(mother_dict)
    Fathers = list(father_dict.keys())
    Mothers = list(mother_dict.keys())

    Yield_t = Yield.cpu().numpy()

    S_father = np.zeros(N_father)
    S_mother = np.zeros(N_mother)

    for id in father_dict.keys():
        if len(Yield_t[info.Father == id]) != 0:
            S_father[father_dict[id]] = np.nanmean(Yield_t[info.Father == id])

    for id in mother_dict.keys():
        if len(Yield_t[info.Mother == id]) != 0:
            S_mother[mother_dict[id]] = np.nanmean(Yield_t[info.Mother == id])

    S_child_test = np.zeros(len(info_test))
    for i in range(len(info_test)):
        S_child_test[i] = (S_father[father_dict[info_test.iloc[i].Father]] + S_mother[mother_dict[info_test.iloc[i].Mother]])/2

    S_child = np.zeros(N_father * N_mother)
    pair_child = np.zeros((N_father * N_mother, 2))
    id = 0
    for i in range(N_father):
        for j in range(N_mother):
            S_child[id] = (S_father[i] + S_mother[j])/2
            pair_child[id, :] = [Fathers[i], Mothers[j]]
            id += 1

    S_child = torch.tensor(S_child)
    _, id = S_child.sort(descending=True)

    return S_child_test