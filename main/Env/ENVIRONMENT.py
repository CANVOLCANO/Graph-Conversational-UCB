# -*- coding: UTF-8 -*-
import numpy as np
import json
import torch
from Env import Arm
from conf import armNoiseScale, suparmNoiseScale, minRecommend, maxRecommend, seeds_set
import random
from Env.utils import fixSeed, KMeans, evolve
import os
from numpy import linalg as la
from tqdm import tqdm
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Environment:
    # p: frequency vector of users
    def __init__(self, args, file_name, seed):
        self.file_name = file_name
        self.args = args
        self.ssuffix = ''
        self.out_file_name = self.args.fn + '/'
        # fix random seed
        self.seed = seeds_set[seed]
        fixSeed(self.seed)
        print("Seed = ", self.seed)
        # interaction frequency
        self.min_rounds = minRecommend
        self.max_rounds = maxRecommend
        print('# Interaction frequency:', minRecommend, '-', maxRecommend)
        # load User
        self.theta, self.nu, self.d = self.load_users()
        print('# Users loaded:', self.theta.shape[0])
        # load Arm
        self.AM = Arm.ArmManager(file_name)
        self.AM.loadArms(filepath = '/' + self.args.itemEmbedding+'.txt')
        self.X_t = torch.stack([v.fv for _, v in self.AM.arms.items()])
        self.arm_id = list(self.AM.arms.keys())               
        print('# Arms loaded:', self.X_t.shape[0])
        # load Review
        self.user_item, self.user_attr = self.load_reviews(file_name), None
        # load Attribute
        # for key-term removing setting 4
        self.RMRhoEnv = self.args.RMRhoEnv
        self.supArmMap = dict() # key: supArm_id, value: a list containing all the arms associated with this sup arm, for computing adj_mat
        self.tildeX_t = self.getXTilde()
        print('# Superarm loaded:', self.tildeX_t.shape[1]) # (d, #key-terms)
        # load User Distribution
        #------------------------------------
        self.suparm_id = []
        self.init_u = self.load_init()
        #-----------------------------------
        self.p = self._get_half_frequency_vector()[0]
        self.adj_mat = self.buildKG(self.tildeX_t, self.args.similarity) # adj_mat(i, j): the similarity between item i and item j
        # set file path
        self.one_zero_adj_mat = np.where(self.adj_mat>0, 1, 0)
        self.BLFilename = self.file_name + '/' + args.blackList


    def IOMap(self, fn):
        return fn+self.ssuffix

    def getXTilde(self, L_set=[]):
        user_attr, suparms = self._read_attributes(L_set)
        self.user_attr = torch.stack(user_attr).T
        tildeX_t = torch.stack(suparms).T # (d,attribute_num)     
        return tildeX_t
    
    def getSVDXTilde(self, tildeX_t):      
        SVDtildeX_t_file = self.IOMap(self.file_name +'/'+ 'SVDtildeX_t_{}'.format(self.args.k)) + '.npz'
        if  os.path.exists(SVDtildeX_t_file):
            SVDtildeX_t = np.load(SVDtildeX_t_file, allow_pickle=True)['SVDtildeX_t']  
        else:
            adj_mat = self.one_zero_adj_mat
            tildeX_t_list = [tuple(x) for x in tildeX_t.T]
            outer_X = np.array([np.outer(x,x) for x in tildeX_t.cpu().numpy().T])
            outer_X_list = []
            print('in evn for loop1')
            for i in tqdm(range(tildeX_t.shape[1])):
                outer_X_list.append(np.sum(outer_X[np.where(adj_mat[i]==1)],axis=0))
                
            print('in evn for loop2')
            svd_x = np.zeros((tildeX_t.shape[1],tildeX_t.shape[0]))
            for i in tqdm(range(tildeX_t.shape[1])):
                u,sigma,vt = la.svd(outer_X_list[i])
                svd_x[i] = np.sqrt(sigma[0])*vt[0]
            SVDtildeX_t = svd_x.T
            np.savez(self.IOMap(self.file_name +'/'+ 'SVDtildeX_t_{}'.format(self.args.k)), SVDtildeX_t=SVDtildeX_t)
        return SVDtildeX_t 

    def load_reviews(self, file_name):
        fn = file_name + '/{}.npy'.format(self.args.reviewMatrix)
        if self.args.trueReward == 'Y':
            # load user_item matrix
            user_item = np.load(fn)
            user_item = torch.from_numpy(user_item).to(device).float() # (#user, #item)
            print('load user_item matrix as rewards')
        else:
            user_item = torch.matmul(self.theta,self.X_t.T)
            print('generate rewards')
        self.noise = True if self.args.rewardNoise == 'Y' else False
        return user_item

    def load_users(self):
        theta = []
        with open(self.file_name + '/{}.txt'.format(self.args.userEmbedding), 'r') as fr:
            for line in fr:
                j_s = json.loads(line)
                theta_u = j_s['preference_v']
                theta.append(theta_u)
        theta = torch.tensor(theta, device=device).squeeze()
        nu, d = theta.shape[0], theta.shape[1]
        return theta, nu, d

    def load_init(self):
        theta = torch.zeros((self.d))
        num = 0
        with open(self.file_name + '/{}.txt'.format(self.args.userTrainEmbedding), 'r') as fr:
            for line in fr:
                j_s = json.loads(line)
                theta_u = torch.from_numpy(np.array(j_s['preference_v'])).squeeze()
                theta += theta_u
                num += 1
        return (theta/num)

    def get_rounds(self):
        return np.random.randint(self.min_rounds, self.max_rounds)

    def feedback(self, i, k, kind='item'):
        if kind == 'item':
            x = torch.index_select(self.X_t, 0, k).squeeze()
            r = torch.index_select(self.user_item, 1, k)[i]
            reg = None
        else:
            x = torch.index_select(self.tildeX_t, 1, k.to(device)).squeeze()
            if self.args.keyreward=='Binomial':          
                r = torch.bernoulli(torch.index_select(self.user_attr, 1, k.to(device))[i])
            elif self.args.keyreward=='Continuous': 
                r = torch.index_select(self.user_attr, 1, k.to(device))[i]
            reg = None
        if self.noise:
            r += torch.randn(1, device=device) * (armNoiseScale if kind == 'item' else suparmNoiseScale)
        return x, r.squeeze(), reg

    def generate_users(self):
        X = np.random.multinomial(1, self.p)
        return np.nonzero(X)[0]

    def _get_half_frequency_vector(self, m=10):
        p0 = list(np.random.dirichlet(np.ones(m)))
        p = np.ones(self.nu)
        k = int(self.nu / m)
        for j in range(m):
            for i in range(k * j, k * (j + 1)):
                p[i] = p0[j] / k
        ps = [list(np.ones(self.nu) / self.nu), list(p), list(np.random.dirichlet(np.ones(self.nu)))]
        return ps

    def get_top_reward_keyterm(self, ratio):
        '''
        return the idx of key-terms according to their rewards (descending order)
        '''
        res = []
        if ratio>0:
            suparms, user_attr, relation = [], [], dict()
            file_name = self.file_name +'/'+ self.args.armRelation + '.txt'
            with open(file_name, 'r') as fr: # record the item key-term bipartite graph
                for index, line in enumerate(fr):
                    e = line.strip().split('\t')
                    a_id, s_ids = int(e[0]), e[1].strip(', ').split(',')
                    w = 1.0 / len(s_ids)
                    for key in s_ids:
                        key = int(key)
                        if key in relation:
                            relation[key][a_id] = w
                        else:
                            relation[key] = {a_id: w}

            s_ids = []
            for index, (s_id, arms) in enumerate(relation.items()): # generate the rewards and the feature vec of key-term 
                rw = torch.zeros((self.user_item.shape[0],), device=device)
                fv = torch.zeros((self.d,), device=device)
                sum_w = 0
                for a_id, w in arms.items():
                    rw += self.user_item[:, a_id] * w
                    fv += self.AM.arms[a_id].fv * w
                    sum_w += w
                if sum_w > 0:
                    user_attr.append(rw / sum_w) # user_attr = (# of key-terms, # of users)
                    suparms.append(fv / sum_w)
                s_ids.append(s_id)
            res = np.array(s_ids)[np.argsort(torch.stack(user_attr).mean(dim=1).cpu().numpy())][-int(len(s_ids)*ratio):].tolist()
        return res

    def _read_attributes(self, L_set):
        if self.args.useGCN != 'Y':
            suparms, user_attr, relation = [], [], dict()
            key_term_BLEnv = self.get_top_reward_keyterm(self.RMRhoEnv)
            if self.RMRhoEnv>0:
                print('in Env delted key-terms:{}'.format(key_term_BLEnv))
            file_name = self.file_name +'/'+ self.args.armRelation + '.txt'
            with open(file_name, 'r') as fr:
                for line in fr:
                    e = line.strip().split('\t')
                    a_id, s_ids = int(e[0]), e[1].strip(', ').split(',')
                    w = 1.0 / len(s_ids)
                    for key in s_ids:
                        key = int(key)
                        if key in relation:
                            relation[key][a_id] = w
                        else:
                            relation[key] = {a_id: w}
            supArmCnt = 0
            for s_id, arms in relation.items():
                if int(s_id) in key_term_BLEnv:
                    continue
                rw = torch.zeros((self.user_item.shape[0],), device=device)
                fv = torch.zeros((self.d,), device=device)
                sum_w = 0
                for a_id, w in arms.items():                   
                    if a_id in L_set:
                        continue
                    rw += self.user_item[:, a_id] * w
                    fv += self.AM.arms[a_id].fv * w
                    sum_w += w
                user_attr.append(rw / sum_w if sum_w != 0 else torch.zeros_like(rw))
                """
                set 0 or delete decided by 'if'
                """
                if not torch.equal(fv, torch.zeros((self.d,), device=device)):
                    suparms.append(fv / sum_w if sum_w != 0 else torch.zeros_like(fv))
                self.supArmMap[supArmCnt] = list(arms.keys())
                supArmCnt += 1
                
            if self.args.RMRho == 0 and self.RMRhoEnv == 0:
                with open(self.out_file_name[:-2]+'kt_mapping.json', 'w') as f:
                    print(self.out_file_name[:-2]+'kt_mapping.json')
                    json.dump(dict(zip(list(range(supArmCnt)), relation.keys())), fp=f)
        return user_attr, suparms

    def _computeTanimoto(self):
        '''
        compute the h hop Tanimoto similarity
        '''
        adj_mat = ""
        adj_mat_file = self.IOMap(self.file_name +'/'+ '{}_hop_tanimoto_EnvRmv{}'.format(self.args.k, self.RMRhoEnv)) + '.npz'
        if os.path.exists(adj_mat_file):
            adj_mat = np.load(adj_mat_file, allow_pickle=True)['adjMat'] 
            print('load precomputing tanimoto matrix')
        else:
            com_adj_mat_file = self.IOMap(self.file_name +'/'+ 'attr_adj_mat_commItem_EnvRmv{}'.format(self.RMRhoEnv)) + '.npz'
            if  os.path.exists(com_adj_mat_file):
                com_adj_mat = np.load(com_adj_mat_file, allow_pickle=True)['adjMat'] 
                print("load adj_mat_commItem finish!")
            else:               
                com_adj_mat = np.zeros((self.tildeX_t.shape[1], self.tildeX_t.shape[1]))
                print("calculate adj_mat_commItem")
                supArmNum = self.tildeX_t.shape[1]
                for k1 in range(supArmNum):
                    for k2 in range(k1,supArmNum):
                        commonItemNum = len(set(self.supArmMap[k1]).intersection(set(self.supArmMap[k2])))
                        com_adj_mat[k1,k2], com_adj_mat[k2,k1] = commonItemNum, commonItemNum
                com_adj_mat /= np.max(com_adj_mat, axis=1)
                np.savez(self.IOMap(self.file_name +'/'+ 'attr_adj_mat_commItem_EnvRmv{}'.format(self.RMRhoEnv)), adjMat=com_adj_mat)
                
            adj_mat_tem = torch.tensor(np.where(com_adj_mat>0, 1, 0), device=device, dtype=torch.float)
            adj_mat_tem2 = adj_mat_tem
            adj_mat_h = adj_mat_tem
            h_hop_list = []
            for h in range(1, int(self.args.k)):
                adj_mat_tem2 @= adj_mat_tem
                adj_mat_h += adj_mat_tem2

            edge_num = np.sum(adj_mat_h.cpu().numpy(), axis=1).squeeze()
            edge_num_r = np.column_stack([edge_num] * edge_num.shape[0])
            edge_num_c = np.row_stack([edge_num] * edge_num.shape[0])
            adj_mat_h = adj_mat_h.cpu().detach().numpy()
            adj_mat = adj_mat_h/(edge_num_c+edge_num_r-adj_mat_h)
            adj_mat[np.array(range(adj_mat.shape[0])), np.array(range(adj_mat.shape[0]))] = 1
            np.savez(self.IOMap(self.file_name +'/'+ '{}_hop_tanimoto_EnvRmv{}'.format(self.args.k, self.RMRhoEnv)), adjMat=adj_mat)   
            
        return adj_mat

    def _computeCos_sim(self):
        '''
        compute the h hop cos similarity
        '''
        adj_mat_file = self.IOMap(self.file_name +'/'+ '{}_hop_adj_mat_cosSim_EnvRmv{}'.format(self.args.k, self.RMRhoEnv)) + '.npz'
        if os.path.exists(adj_mat_file):
            adj_mat = np.load(adj_mat_file, allow_pickle=True)['adjMat'] 
        else:
            com_adj_mat_file = self.IOMap(self.file_name +'/'+ 'attr_adj_mat_commItem_EnvRmv{}'.format(self.RMRhoEnv)) + '.npz'
            if  os.path.exists(com_adj_mat_file) and self.args.itemPolicy != 'SoftUCB':
                com_adj_mat = np.load(com_adj_mat_file, allow_pickle=True)['adjMat'] 
                print("load adj_mat_commItem finish!")
            else:               
                com_adj_mat = np.zeros((self.tildeX_t.shape[1], self.tildeX_t.shape[1])) 
                print("calculate adj_mat_commItem")
                supArmNum = self.tildeX_t.shape[1]
                for k1 in tqdm(range(supArmNum)):
                    for k2 in range(k1,supArmNum):
                        commonItemNum = len(set(self.supArmMap[k1]).intersection(set(self.supArmMap[k2])))
                        com_adj_mat[k1,k2], com_adj_mat[k2,k1] = commonItemNum, commonItemNum
                com_adj_mat /= np.max(com_adj_mat, axis=1)
                np.savez(self.IOMap(self.file_name +'/'+ 'attr_adj_mat_commItem_EnvRmv{}'.format(self.RMRhoEnv)), adjMat=com_adj_mat)
                
            cos_adj_mat_file = self.IOMap(self.file_name +'/'+ 'attribute_cosSim_EnvRmv{}'.format(self.RMRhoEnv)) + '.npz'
            if  os.path.exists(cos_adj_mat_file) and self.args.itemPolicy != 'SoftUCB':
                cos_adj_mat = np.load(cos_adj_mat_file, allow_pickle=True)['adjMat']
                print("load adj_mat_cosSim finish!")
            else: 
                print("calculate adj_mat_cosSim")
                supArmNum = self.tildeX_t.shape[1]      
                cos_adj_mat = torch.mm(self.tildeX_t.T, self.tildeX_t)
                norm2 = torch.norm(self.tildeX_t, p=2, dim=0)
                norm2 = torch.ger(norm2, norm2)   
                norm2[norm2 == 0] = 1e10
                cos_adj_mat = (cos_adj_mat/norm2).cpu().numpy()
                cos_adj_mat[np.diag_indices_from(cos_adj_mat)] = 1.0  
                cos_adj_mat = np.where(cos_adj_mat <1e-3, 0, cos_adj_mat )
                np.savez(self.IOMap(self.file_name +'/'+ 'attribute_cosSim_EnvRmv{}'.format(self.RMRhoEnv)), adjMat=cos_adj_mat)    
            sub_mat = torch.tensor(np.where(com_adj_mat>0, 1, 0))
            hop_mat = sub_mat
            final_mat = sub_mat
            for i in range(1,int(self.args.k)):
                hop_mat = hop_mat @ sub_mat  #calculate i-hop adj
                print("{}_hop_mat".format(i+1))
                final_mat += hop_mat         #calculate sum of 1~k-hop adj
            adj_mat = cos_adj_mat * np.where(final_mat.numpy()>0, 1, 0)
            np.savez(self.IOMap(self.file_name +'/'+ '{}_hop_adj_mat_cosSim_EnvRmv{}'.format(self.args.k, self.RMRhoEnv)), adjMat=adj_mat) 
        return adj_mat

    def buildKG(self, tildeX_t, similarity):
        if similarity == 'ComItem' :
            adj_mat_file = self.IOMap(self.file_name +'/'+ 'attr_adj_mat_commItem') + '.npz'
            if  os.path.exists(adj_mat_file) and self.args.itemPolicy != 'SoftUCB':
                print('using precomputed adj_mat: ComItem')
                adj_mat = np.load(adj_mat_file, allow_pickle=True)['adjMat']  
            else:               
                adj_mat = np.zeros((tildeX_t.shape[1], tildeX_t.shape[1]))
                supArmNum = tildeX_t.shape[1]
                for k1 in tqdm(range(supArmNum)):
                    for k2 in range(k1,supArmNum):
                        commonItemNum = len(set(self.supArmMap[k1]).intersection(set(self.supArmMap[k2])))
                        adj_mat[k1,k2], adj_mat[k2,k1] = commonItemNum, commonItemNum
                adj_mat /= np.max(adj_mat, axis=1)
                np.savez(self.IOMap(self.file_name +'/'+ 'attr_adj_mat_commItem'), adjMat=adj_mat)
                
        elif similarity == 'CosSim':
            adj_mat_file = self.IOMap(self.file_name +'/'+ 'attribute_cosSim') + '.npz'
            if  os.path.exists(adj_mat_file) and self.args.itemPolicy != 'SoftUCB':
                print('using precomputed adj_mat: CosSim')
                adj_mat = np.load(adj_mat_file, allow_pickle=True)['adjMat']
            else: 
                supArmNum = tildeX_t.shape[1]      
                adj_mat = torch.mm(tildeX_t.T, tildeX_t)
                norm2 = torch.norm(tildeX_t, p=2, dim=0)
                norm2 = torch.ger(norm2, norm2)   
                norm2[norm2 == 0] = 1e10
                adj_mat = (adj_mat/norm2).cpu().numpy()
                adj_mat[np.diag_indices_from(adj_mat)] = 1.0  
                np.savez(self.IOMap(self.file_name +'/'+ 'attribute_cosSim'), adjMat=adj_mat)
        elif similarity == 'hop-CosSim':
            adj_mat = self._computeCos_sim()
        elif similarity == 'hop-tanimoto':
            adj_mat = self._computeTanimoto()
        elif similarity == 'hop-tanimoto+cos':
            adj_mat = self._computeTanimoto() * (1-self.args.cos_lambda) + self.args.cos_lambda * self._computeCos_sim()
        return adj_mat

    def getSimilarity(self, tensor1, tensor2):
        '''
        get the similarity \in [-1,1] between 2 tensors
        '''
        return torch.nn.functional.cosine_similarity(tensor1.squeeze(), tensor2.squeeze(), dim=0)