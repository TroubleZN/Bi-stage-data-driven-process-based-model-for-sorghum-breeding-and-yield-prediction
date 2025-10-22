import random

import pandas as pd
import numpy as np

from copy import copy


#%%
def reproduce(Mothers, G_Mother, Mother_Dict, Fathers, G_Father, Father_Dict):
    N = len(Mothers)
    G_child = np.zeros((N, 2))
    for i in range(N):
        G_child[i] = G_Mother[Mother_Dict == Mothers[i]] + G_Father[Father_Dict == Fathers[i]]
    return G_child


def loss(G_child, is_PS):
    is_both_dom = (G_child > 0).sum(axis=1) == 2
    return (is_both_dom == is_PS).mean()


#%%
if __name__ == '__main__':
    #%%
    info = pd.read_csv("../../Data/clean/python/info.csv")

    info = info.iloc[:, 3:6]
    info = info.drop_duplicates()

    N = len(info)

    Mothers = info.Mother
    Mother_Dict = Mothers.unique()
    Mothers = Mothers.to_numpy()
    n_Mother = len(Mother_Dict)  # n_Mother = 141

    Fathers = info.Father
    Father_Dict = Fathers.unique()
    Fathers = Fathers.to_numpy()
    n_Father = len(Father_Dict)  # n_Father = 665

    Types = info.Type.to_numpy()

    is_PS = Types == 'PS'

    G_Father = np.random.randint(0, 2, (n_Father, 2))
    G_Mother = np.random.randint(0, 2, (n_Mother, 2))

    G_child = reproduce(Mothers, G_Mother, Mother_Dict, Fathers, G_Father, Father_Dict)

    nF, nM = 0, 0
    while nF < 645 or nM < 135:
        i_all = np.arange(0, N)
        np.random.shuffle(i_all)

        N_train = np.int64(np.floor(0.9*N))
        i_train = i_all[:N_train]
        i_test = i_all[N_train:]

        nF = np.unique(Fathers[i_train]).__len__()
        nM = np.unique(Mothers[i_train]).__len__()

    train_loss = loss(G_child[i_train], is_PS[i_train])
    test_loss = loss(G_child[i_test], is_PS[i_test])

    i_fail = 0
    while i_fail < 1000:
        G_Father_new = copy(G_Father)
        G_Mother_new = copy(G_Mother)

        i_change = np.random.randint(0, n_Father + n_Mother)
        i_loc = np.random.randint(0, 2)
        if i_change <= n_Father - 1:
            G_Father_new[i_change][i_loc] = 1 - G_Father_new[i_change][i_loc]
        else:
            G_Mother_new[i_change - n_Father][i_loc] = 1 - G_Mother_new[i_change - n_Father][i_loc]

        G_child = reproduce(Mothers, G_Mother_new, Mother_Dict, Fathers, G_Father_new, Father_Dict)

        new_train_loss = loss(G_child[i_train], is_PS[i_train])
        new_test_loss = loss(G_child[i_test], is_PS[i_test])

        if new_train_loss > train_loss:
            G_Father = G_Father_new
            G_Mother = G_Mother_new
            train_loss = new_train_loss
            print("train_accuracy: ", new_train_loss, " test_accuracy: ", new_test_loss, "i_fail: ", i_fail)
            i_fail = 0
        else:
            i_fail += 1
