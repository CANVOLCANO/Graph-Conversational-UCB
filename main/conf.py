import math

linucb_para = {'lambda': 1.0, 'sigma': 0.05, 'alpha': 0.5}
conucb_para = {'lambda': 0.5, 'sigma': 0.05, 'tilde_lambda': 1.0, 'alpha': 0.25, 'tilde_alpha': 0.25}
lse_soft_para = {'sigma':0.5, 'delta':0.05, 'alpha':1.0, 'step_size_beta':0.5, 
                 'step_size_gamma':0.01, 'weight1':1.0,' beta':1.2, 'gamma':0}
epsilonGreedy_para = {'C':0.5, 'D':0.1}

epsilon = 1e-6
train_iter = 0
test_iter = 1000
armNoiseScale = 0.1
suparmNoiseScale = 0.1
ts_nu = 1.0
batch_size = 50
bt = lambda t: 5 * int(math.log(t + 1))
minRecommend, maxRecommend = 1, 5
seeds_set = [2756048, 675510, 807110, 2165051, 9492253, 927, 218, 495, 515, 452]
