import numpy as np
import torch,os,gzip,time
from Agent.BASE import ConUCB
import conf
import random
from conf import seeds_set
from Env import utils
import torch
from torch import mm, mv, ger
import math
from tqdm import tqdm 
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraConUCB(ConUCB):
    def __init__(self, nu, d, T, args):
        super(GraConUCB, self).__init__(nu, d, T, args)
        '''
        for epsilon greedy
        '''
        self.args = args
        self.eGreedy_C = conf.epsilonGreedy_para['C']
        self.eGreedy_D = conf.epsilonGreedy_para['D']
        self.recNumSupArm = '' # the num of each sup arm in the weakly dominating set being recommended, for coumputing the empirical mean
        self.empiricalMeanSupArm = '' # the empirical mean of supArm in the weakly dominating set

        self.adj_mat = '' # adj_mat[i,j] denotes there is an arc from node_i to node_j with weight adj_mat[i,j] and 0 means there is no arc from node_i to node_j
        self.weaklyDS = '' # ordered
        self.weaklyDS_map = '' # map: index of node i in the graph -> index of node i in the weakly dominating
        self.setClass = '' # setClass[i] is a set containing the out-neighbours of node i and node i itself
        self.similarity = args.similarity
        self.attrPolicy = args.attrPolicy
        self.itemPolicy = args.itemPolicy
        self.DS = args.DS
        self.GF = args.GF
        self.wkMask = '' # size: (# key-terms), 1 if the key-term is in the weakly ds or 0 otherwise
        self.goptMask = '' # size:(# key-terms), 1 if the key-term is in the G-optimal ds or 0 otherwise
        self.dsMask = ''
        self.samples = ''
        self.samples_prob = ''
        self.U_set = []
        self.L_set = []
        self.gamma = ''

        self.C = args.C

        self.RSetting = args.RSetting
        self.RSampling = args.RSampling
        self.RMRhoEnv = args.RMRhoEnv

        self.TV = args.TV


    def _update_inverse(self, i, x, y, tilde):
        if tilde:
            self.Sinv_x = mv(self.Sinv_tilde[i], x)
            self.S_tilde[i] += ger(x, x)
            self.b_tilde[i] += y * x  
            self.Sinv_tilde[i] -= ger(self.Sinv_x, self.Sinv_x) / (1 + (x * self.Sinv_x).sum())
            self.theta_tilde[i] = mv(self.Sinv_tilde[i], self.b_tilde[i])
        else:
            self.Sinv_x = mv(self.Sinv[i], x)            
            self.b[i] += y * x * self.lamb
            self.S[i] += ger(x, x) * self.lamb
            self.Sinv[i] -= ger(self.Sinv_x, self.Sinv_x) / (1 / self.lamb + (x * self.Sinv_x).sum())
        self.theta[i] = mv(self.Sinv[i], self.b[i] + (1 - self.lamb) * self.theta_tilde[i])
               

    def epsilonGreedy(self, t, K):
        '''
        return: 1 for exploration, 0 for exploitation
        '''
        epsilon_t = min(1,self.eGreedy_C*K/((self.eGreedy_D**2)*t))
        return np.random.binomial(1,epsilon_t)
    

    def update_gamma(self):        
        cl = len(np.unique(self.L_set))
        delta=0.6
        if cl==0:
            cl=1
        else:
            pass
        a=np.log((delta*cl)/(1-delta))
        max_s=torch.max(self.S_list)
        self.gamma=torch.tensor(a)/max_s 
   

    def recommend_sumper_arm(self, i):
        res = -1
        if self.attrPolicy == 'Gopt-Pro':
            idx = list(np.random.choice(self.samples, size=len(self.samples), replace=False, p=self.samples_prob))
            for i in idx:
                if i not in self.blackList:
                    res = i
                    break
            if res == -1:
                res = idx[-1]
        elif self.attrPolicy == 'ConUCB':
            res = self.getCredit(self.Sinv[i], self.Sinv_tilde[i])
        else:
            raise AssertionError
        return torch.tensor(res)

    # New store_info_suparm
    def store_info_suparm(self, i, y, k_tilde):
        '''
        i: user

        we make multi-updates here for all the attributes in the out-neighbours of node k_tilde
        '''
        self.N_tilde[i] += 1
        updateSet = [k_tilde] if self.GF != 'Y' else self.setClass[k_tilde]
        if len(updateSet) < self.tildeX_t.size(0)/20:
            for k in updateSet:
                x = torch.index_select(self.tildeX_t, 1, torch.Tensor([k]).long().to(device)).squeeze()
                self._update_inverse(i, x, y * self.adj_mat[k_tilde,k], tilde=True)
        else:
            updateSet = list(updateSet)
            x = self.tildeX_t[:, updateSet] 
            x1, x2 = x.T.unsqueeze(2), x.T.unsqueeze(1)
            self.S_tilde[i] += torch.bmm(x1, x2).sum(0)
            self.b_tilde[i] += torch.mv(x, y * torch.from_numpy( self.adj_mat[k_tilde, updateSet]).to(device).float())
            self.Sinv_tilde[i] = torch.inverse(self.S_tilde[i])
            self.theta_tilde[i] = mv(self.Sinv_tilde[i], self.b_tilde[i])
        return updateSet

    def getAlgorithmFileName(self, prefix, rating_matrix_name = ''):
        suffix = ''
        suffix += '{}'.format(self.attrPolicy)
        if self.DS == 'WK':
            suffix += '_{}'.format('ds')
        if self.DS == 'GOPT':
            suffix += '_{}'.format('gopt')
        if self.DS == 'GOPT+WK':
            suffix += '_{}'.format('gopt+ds')        
        if self.GF == 'Y':
            suffix += '_{}'.format('gfN')
        if self.itemPolicy == 'SoftUCB':
            suffix += '_{}'.format('SoftUCB')
        if self.args.useDynamicSimilarity == 'Y':
            suffix += '_{}'.format('Dynamic')
        if self.similarity == 'CosSim' :
            suffix += '_{}'.format('CosSim')
        if self.similarity == 'OneSub' :
            suffix += '_{}'.format('OneSub')
        if self.similarity == 'hop-CosSim' :
            suffix += '_{}'.format(self.args.k+'-hop-CosSim')

        if self.similarity == 'hop-tanimoto' :
            suffix += '_{}'.format(self.args.k+'-hop-tanimoto')
        if self.similarity == 'hop-tanimoto+cos' :
            suffix += '_{}'.format(self.args.k+'-hop-tanimoto+cos')

        if self.args.keyreward=='Binomial':
            suffix += '_{}'.format('Binomial')
        if self.args.useSumOuter == 'Y':
            suffix += '_{}'.format('SumOuter')
        # fn = prefix + '[' + self.__class__.__name__ + suffix + ']' # e.g., ../datasets/con_data_50_0.01_0.5/0/[Baseline]+[ArmCon]

        if self.attrPolicy == 'Gopt-Sum-Pro' or self.attrPolicy == 'Gopt-Pro':
            suffix += '_{}C'.format(self.C)

        # ------------------------------------------------- for removing key-term ----------------------------------------------------------------------
        if self.args.RMRho > 0:
            suffix += '_{}_RSetting{}_{}'.format(str(self.RMRho)+'R', self.RSetting, self.RSampling)
        # ------------------------------------------------- for removing key-term ----------------------------------------------------------------------

        if self.args.reviewMatrix == 'user_item_validation':
            suffix += '_{}'.format('validate')

        suffix += '_{}CosGamma'.format(self.args.cos_lambda)
        suffix += '_TV{}'.format(str(self.TV))

        # fn = prefix + '[' + suffix + ']' # e.g., ../datasets/con_data_50_0.01_0.5/0/[Baseline]+[ArmCon]
        fn = prefix + '[' + suffix + '_{}'.format(rating_matrix_name) + ']' # e.g., ../datasets/con_data_50_0.01_0.5/0/[Baseline]+[ArmCon]
        return fn


    def store_info(self, i, x, y):
        self.N[i] += 1
        self._update_inverse(i, x, y, tilde=False)


    def _gopt_old(self, A, C = 2):
        """A: (n, d)"""
        def gram(A):
            return sum([np.outer(x,x) for x in A])
        A = np.array(A)
        d = len(A[0])
        remaining_idx = list(range(len(A)))
        result = set()
        cnt = 0
        sample_cnt_threshold = 50
        while len(remaining_idx) > C * d * np.log(d):
            print('in G Opt Old cnt={} len(remaining_index)={}'.format(cnt, len(remaining_idx)))
            cnt += 1
            Vinv = np.linalg.pinv(gram(A[remaining_idx]))
            p = [1/(2* len(A)) + 1/(2*d) * np.dot(A[idx], np.dot(Vinv, A[idx])) for idx in remaining_idx]
            for ix in range(len(remaining_idx)):
                if p[ix] < 0:
                    idx = remaining_idx[ix]
                    print('in G Opt <0 idx={} norm={} p[ix]={}'.format(idx, np.dot(A[idx], np.dot(Vinv, A[idx])), p[ix]))
            p = np.maximum(np.array(p), 0).tolist()
            p = np.divide(p, sum(p))
            s = set()
            sampling = True
            sample_cnt = 0
            sample_fail_fg = 0
            while sampling:
                sample_cnt += 1
                samples = set(np.random.choice(remaining_idx, size=int(C*d*np.log(d)), p=p))
                s = np.array([A[i] for i in samples])
                Vinv = np.linalg.pinv(gram(s))
                index = [np.dot(x, np.dot(Vinv, x)) <= 1 for x in A[remaining_idx]]
                if sum(index) >= len(remaining_idx) / 2:
                    result = result | samples
                    sampling = False
                    remaining_idx = [remaining_idx[i] for i in range(len(remaining_idx)) if index[i] == 0]
                if sample_cnt>sample_cnt_threshold:
                    sampling = False
                    sample_fail_fg = 1
            if sample_fail_fg:
                break
        if not sample_fail_fg:
            result = result | set(remaining_idx)
        if len(result) == 0:
            result |= samples
        result = list(result)
        res_final = torch.zeros(A.shape[0])
        res_final[result] = 1/(len(result))
        print('in G Opt Old end cnt={} len(result)={} result={}'.format(cnt, len(result), result))
        return result, res_final.numpy()

      
    def G_optimal(self, envir):
        if self.args.useSumOuter == 'Y':
            SVDtildeX_t = envir.getSVDXTilde(self.tildeX_t)
            tildeX_t = [tuple(x) for x in SVDtildeX_t.T]
        else: 
            tildeX_t = [tuple(x) for x in self.tildeX_t.cpu().numpy().T] # (n, d)
        A = tildeX_t
        samples, p = self._gopt_old(A, C=self.C)
        samples = sorted(samples)
        self.samples = samples
        self.samples_prob = p[self.samples]
        with open('GOpt.json', 'w') as f:
            json.dump(dict(zip([int(i)  for i in self.samples], [float(i) for i in self.samples_prob])), fp=f)
        self.samples_prob /= sum(self.samples_prob)
        self.samples_sum_prob = np.dot(self.adj_mat, p)[self.samples]
        self.samples_sum_prob /= sum(self.samples_sum_prob)


    def run(self, envir):
        self.X_t, self.tildeX_t = envir.X_t, envir.tildeX_t 
        self._genBL(envir.BLFilename)
        self.adj_mat = envir.adj_mat
        self.setClass = utils.weaklyDominatingSet(self.adj_mat)
        self.G_optimal(envir)      
        for user in tqdm(range(self.staU,self.endU)):
            # set file name
            fn = self.getAlgorithmFileName(envir.out_file_name, envir.args.reviewMatrix)
            if not os.path.exists(fn):
                os.mkdir(fn)
            fn = self.getAlgorithmFileName(envir.out_file_name, envir.args.reviewMatrix) + '/U%04d' % user
            # record the updated items and the key-terms 
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
                        updateSet = self.store_info_suparm(i=i, y=r_tilde, k_tilde=k_tilde.cpu().detach().item())
                        fn_kt_w.write(','.join([
                                str(t), str(k_tilde.cpu().detach().item()), '_'.join([str(ti) for ti in updateSet]), str(torch.det(self.S_tilde[i]).cpu().detach().item()), str(torch.det(self.Sinv_tilde[i]).cpu().detach().item())
                            ]) + '\n')
                        if self.args.useDynamicSimilarity == 'Y' and (self.theta[i]!= 0).all():
                            A = self.tildeX_t.T @ self.theta[i]
                            A = A.repeat((self.tildeX_t.shape[1], 1))
                            self.adj_mat =  A / A.T                                 
                    if self.itemPolicy == 'ConUCB':                        
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
            f.close()
            fn_kt_w.close()

