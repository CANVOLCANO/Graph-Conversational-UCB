import numpy as np
import torch
from torch import mm, mv, ger
import conf
from conf import seeds_set
import time
import gzip
import random,os
from Env import utils
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConUCB:
    def __init__(self, nu, d, T, args):
        # fix random seed
        self.args = args
        self.seed = seeds_set[args.seed]
        utils.fixSeed(self.seed)
        # Initial ConUCB Hyper-parameters
        self.lamb = conf.conucb_para['lambda']
        self.tilde_lamb = conf.conucb_para['tilde_lambda']
        self.sigma = conf.conucb_para['sigma']
        self.alpha = conf.conucb_para['alpha']
        if 'tilde_alpha' in conf.conucb_para:
            self.tilde_alpha = conf.conucb_para['tilde_alpha']
            self.cal_alpha = False
        else:
            self.cal_alpha = True
        self.staU,self.endU = args.staU,args.endU
        # Set setting parameters
        self.nu = nu
        self.d = d
        self.T = T
        self.all_T = self.T * (self.endU - self.staU) # self.T is the horizon per user here
        # Initial item-level parameters
        self.S = torch.stack([torch.eye(d, device=device) * (1 - self.lamb) for _ in range(nu)])
        self.b = torch.stack([torch.zeros(d, device=device) for _ in range(nu)])
        self.Sinv = torch.stack([torch.eye(d, device=device) / (1 - self.lamb) for _ in range(nu)])
        self.theta = torch.stack([torch.zeros(d, device=device) for _ in range(nu)])
        self.N = np.zeros(nu)
        # Initial attr-level parameters
        self.S_tilde = torch.stack([torch.eye(d, device=device) * self.tilde_lamb for _ in range(nu)])
        self.b_tilde = torch.stack([torch.zeros(d, device=device) for _ in range(nu)])
        self.Sinv_tilde = torch.stack([torch.eye(d, device=device) / self.tilde_lamb for _ in range(nu)])
        self.theta_tilde = torch.stack([torch.zeros(d, device=device) for _ in range(nu)])
        self.N_tilde = np.zeros(nu)
        self.attrPolicy = args.attrPolicy

        self.AlgType = args.AlgType
        self.RMRho = args.RMRho  
        self.RSetting = args.RSetting
        self.RSampling = args.RSampling

        self.TV = args.TV
        

    def _genBL(self, filename):
        self.blackList = filename
        if self.RMRho > 0:
            if self.RSetting == 1:
                with open(self.blackList, 'r') as f:
                    self.blackList = dict(json.load(f))
                if self.RSampling == 'positive':
                    self.removeRate = np.array(list(self.blackList.values()))
                    self.removeRate /= np.sum(self.removeRate)
                self.blackList = np.random.choice(a=list(self.blackList.keys()), size=(int(len(self.blackList)*self.RMRho)), replace=False, p=self.removeRate).astype(int).tolist()
        else:
            self.blackList = []

    def _update_inverse(self, i, x, y, tilde):
        if tilde:
            self.N_tilde[i] += 1
            self.Sinv_x = mv(self.Sinv_tilde[i], x)
            self.S_tilde[i] += ger(x, x)
            self.b_tilde[i] += y * x
            self.Sinv_tilde[i] -= ger(self.Sinv_x, self.Sinv_x) / (1 + (x * self.Sinv_x).sum())
            self.theta_tilde[i] = mv(self.Sinv_tilde[i], self.b_tilde[i])
        else:
            self.N[i] += 1
            self.Sinv_x = mv(self.Sinv[i], x) 
            self.S[i] += ger(x, x) * self.lamb               
            self.Sinv[i] -= ger(self.Sinv_x, self.Sinv_x) / (1 / self.lamb + (x * self.Sinv_x).sum())
            self.b[i] += y * x * self.lamb
        self.theta[i] = mv(self.Sinv[i], self.b[i] + (1 - self.lamb) * self.theta_tilde[i])

    def getProb(self, N_tilde, Sinv_tilde, Sinv, theta):
        fv = theta
        if self.cal_alpha:
            self.tilde_alpha = np.sqrt(2 * (self.d * np.log(6) + np.log(2 * N_tilde / self.sigma + 1)))
            self.tilde_alpha += 2 * np.sqrt(self.tilde_lamb) * fv.norm()
        self.FM_Sinv = mm(self.X_t, Sinv)
        var1 = (self.FM_Sinv * self.X_t).sum(dim=1).sqrt()
        var2 = (mm(self.FM_Sinv, Sinv_tilde) * self.FM_Sinv).sum(dim=1).sqrt()
        pta_1 = mv(self.X_t, fv)
        pta_2 = var1 * self.lamb * self.alpha
        pta_3 = var2 * (1 - self.lamb) * self.tilde_alpha
        return torch.argmax(pta_1 + pta_2 + pta_3)


    def getCredit(self, Minv, tilde_Minv):
        tilde_Minv_FM = mm(tilde_Minv, self.tildeX_t)
        norm_M = torch.chain_matmul(self.X_t, Minv, tilde_Minv_FM).norm(dim=0)
        result_b = 1 + (self.tildeX_t * tilde_Minv_FM).sum(dim=0)
        res = norm_M * norm_M / result_b
        val, idx = torch.sort(res, descending=True)
        idx = idx.cpu().detach().numpy().tolist()
        res = -1
        if self.RMRho > 0:
            for i in idx:
                if i not in self.blackList:
                    res = i
                    break
            if res == -1:
                res = idx[-1]
        else:
            res = idx[0]
        return res

    def recommend_sumper_arm(self, i):
        res = self.getCredit(self.Sinv[i], self.Sinv_tilde[i])
        res = torch.tensor(res)
        return res

    def recommend(self, i):
        return self.getProb(self.N_tilde[i], self.Sinv_tilde[i], self.Sinv[i], self.theta[i])

    def store_info(self, i, x, y):
        self._update_inverse(i, x, y, tilde=False)

    def store_info_suparm(self, i, x, y):
        self._update_inverse(i, x, y, tilde=True)

    def update(self, t):
        return self.nu

    def getAddiBudget(self, cur_bt, iter_):
        left_budget = cur_bt(iter_) - (0 if iter_ == 0 else cur_bt(iter_ - 1))
        return int(left_budget) if left_budget > 0 else -1

    def getAlgorithmFileName(self, prefix, rating_matrix_name = ''):
        inner = self.__class__.__name__
        if self.RMRho > 0:
            inner += '_{}_RSetting{}_{}'.format(str(self.RMRho)+'R', self.RSetting, self.RSampling)
        # fn = prefix + '[' + self.__class__.__name__ + ']' # e.g., ../datasets/con_data_50_0.01_0.5/0/[Baseline]+[ArmCon]
        inner += '_TV{}'.format(str(self.TV))
        fn = prefix + '[' + inner + '_{}'.format(rating_matrix_name) + ']'
        return fn

    def run(self, envir):
        self.X_t, self.tildeX_t = envir.X_t, envir.tildeX_t 
        self._genBL(envir.BLFilename)

        for user in range(self.staU,self.endU):
            # set file name
            fn = self.getAlgorithmFileName(envir.out_file_name, envir.args.reviewMatrix)
            if not os.path.exists(fn):
                os.mkdir(fn)
            fn = self.getAlgorithmFileName(envir.out_file_name, envir.args.reviewMatrix) + '/U%04d' % user
            fn_kt = self.getAlgorithmFileName(envir.out_file_name, envir.args.reviewMatrix) + '/Kt_U%04d' % user
            f = gzip.open(fn + '.gz', 'wt')
            fn_kt_w = gzip.open(fn_kt + '.gz', 'wt')
            total_reward, t = 0, 0
            f.write(','.join(['iteration', 'average_reward', 'user_reward', 't']) + '\n')
            fn_kt_w.write(','.join(['iteration', 'kt', 'multiHop_kt', 'detS_tilde', 'detSinv_tilde']) + '\n')
            start = time.time()
            while t < self.T:
                i = user
                rounds = envir.get_rounds()
                for _ in range(rounds):
                    Addi_budget = self.getAddiBudget(conf.bt, self.N[i])
                    for _ in range(Addi_budget):
                        k_tilde = self.recommend_sumper_arm(i=i)
                        x_tilde, r_tilde, _ = envir.feedback(i=i, k=k_tilde, kind='attr')
                        self.store_info_suparm(i=i, x=x_tilde, y=r_tilde)

                        fn_kt_w.write(','.join([
                                str(t), str(k_tilde.cpu().detach().item()), '-1', str(torch.det(self.S_tilde[i]).cpu().detach().item()), str(torch.det(self.Sinv_tilde[i]).cpu().detach().item())
                            ]) + '\n')

                    k = self.recommend(i=i)
                    x, r, reg = envir.feedback(i=i, k=k, kind='item')
                    self.store_info(i=i, x=x, y=r)

                    total_reward += r
                    average_reward = total_reward / (t + 1)
                    f.write(','.join([
                        str(t), str(average_reward.tolist()), str(r.tolist()), str(time.time())
                    ]) + '\n')

                    if t % 1000 == 0:
                        f.close()
                        print('======== Iter:%d,  Attr-Num:%d, Reward:%f, time:%f ========' % (
                            t, self.tildeX_t.shape[1], total_reward / (t + 1), time.time() - start))
                        print('Iteration: %d, User: %d, user Reward: %f' % (t, i, r))
                        f = gzip.open(fn + '.gz', 'at')
                        start = time.time()
                    t += 1

