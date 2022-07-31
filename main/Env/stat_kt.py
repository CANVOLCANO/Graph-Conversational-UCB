import gzip
import numpy as np
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm 
import math
import json

'''
on movieLen, Gopt-Sum-Pro_gfN_2-hop-tanimoto+cos_SumOuter is the best
'''

reward_list = ['Averaged Reward', 'Cumulative Regret']
dirr = '/data/czzhao_hdd/GraphCRS/datasets/Amazon/amazon-book/'
dataset = 'amazon'

kt_num = 3757
horizon = 1000
seed_num = 10
user_num = 999
tot = user_num * horizon
checkpoint = [int(5e4), int(1e5), int(2e5), int(2.5e5), int(3e5), int(4e5), int(5e5), int(1e6)]

color_list = ['aqua', 'blue',  'black', 'pink', 'red',  'saddlebrown']
algorithm_list = [
    '[ConUCB]', 
    '[Gopt-Pro_1-hop-tanimoto+cos_SumOuter_0.15C]'
]

def sta(algorithm_name):
    '''
    res_X[0]: avg_reward
    res_X[1]: cum_regret
    '''
    res = np.zeros(kt_num)
    cnt_faled = 0
    for seed in range(seed_num):
        for i in tqdm(range(user_num)):
            filename = "Kt_U" + str(i).zfill(4) + '.gz'
            df = pd.read_csv(dirr + str(seed) + "/" + algorithm_name + "/" + filename, compression='gzip', nrows=horizon,
                             sep=',', quotechar='"', error_bad_lines=False)
            if df.shape[0]<30 or np.any(np.isnan(df['kt'])):
                cnt_faled += 1
                print('in prepare failed: seed={} user={}'.format(seed, i))
                continue
            res[df['kt'].to_numpy().tolist()] += 1

    with open('Kt_stat_{}.json'.format(algorithm_name), 'w') as f:
        json.dump(dict(zip(np.where(res>0)[0].tolist(), res[np.where(res>0)].tolist())), fp=f)

    print('in prepare {} final failed: {}'.format(algorithm_name, cnt_faled))

if __name__ == '__main__':
    for i in range(len(algorithm_list)):
        sta(algorithm_list[i])
        
