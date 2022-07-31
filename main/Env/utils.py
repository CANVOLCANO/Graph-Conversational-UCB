import torch
import numpy as np
import random
from conf import seeds_set
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = seeds_set[-1]
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def edge_probability(n):
    return 3 * np.log(n) / n


def is_power2(n):
    return n > 0 and ((n & (n - 1)) == 0)


def factT(T):
    return np.sqrt((1 + np.log(1 + T)) / (1 + T))


def evolve(T, t, N, delta):
    return min(int(np.log(1 + t / T * delta) / np.log(2) * N) + 1, N)

def fixSeed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def weaklyDominatingSet(adj_mat):
    '''
    adj_mat:

    return: a weakly dominating set of the graph using approximating algo (0-indexed)
    '''
    universe=list(range(len(adj_mat[0])))
    setClass=[] # list of sets
    for node in universe:
        setClass.append(set(np.nonzero(adj_mat[node])[0].tolist()))
    return setClass        

def GreedySetCover(universe,setClass):
    '''
    args: 

    universe: list

    setClass: list of sets

    plz refer to 

    https://www.cs.umd.edu/class/fall2017/cmsc451-0101/Lects/lect09-set-cover.pdf

    https://www.cs.dartmouth.edu/~ac/Teach/CS105-Winter05/Notes/wan-ba-notes.pdf

    return: indexes of the set in the minimal set cover
    '''
    res = []
    while len(universe)>0:
        tem = -1
        idx = -1
        for i in range(len(setClass)):
            se = setClass[i][1]
            num = len(set(universe).intersection(se))
            if num > tem:
                tem = num
                idx = i
        se = setClass[idx][1]
        universe = list(set(universe)-set(se))
        res.append(setClass[idx][0])
        del setClass[idx]
    return sorted(res)
