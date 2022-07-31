import numpy as np
import torch
from Agent.BASE import ConUCB
import conf
import random
from conf import seeds_set

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LinUCB(ConUCB):
    def __init__(self, nu, d, T, args):
        super(LinUCB, self).__init__(nu, d, T, args)
        # fix random seed
        self.seed = seeds_set[args.seed]
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        # Initial LinUCB Hyper-parameters
        self.alpha = conf.linucb_para['alpha']

    def _update_inverse(self, i, x, y, tilde):
        self.N[i] += 1
        self.Sinv_x = torch.mv(self.Sinv[i], x)
        self.S[i] += torch.ger(x, x)
        self.b[i] += y * x
        self.Sinv[i] -= torch.ger(self.Sinv_x, self.Sinv_x) / (1 + (self.Sinv_x * x).sum())
        self.theta[i] = torch.mv(self.Sinv[i], self.b[i])

    def getProb(self, N_tilde, Sinv_tilde, Sinv, theta):
        if Sinv_tilde is None:
            X = self.X_t
            self.FM_Sinv = torch.mm(X, Sinv)
        else:
            X = self.tildeX_t.T
            self.FM_Sinv = torch.mm(X, Sinv_tilde)
        var1 = (self.FM_Sinv * X).sum(dim=1).sqrt()
        pta_1 = torch.mv(X, theta)
        pta_2 = var1 * self.alpha
        return torch.argmax(pta_1 + pta_2)

    def recommend(self, i):
        return self.getProb(self.N[i], None, self.Sinv[i], self.theta[i])

    def recommend_sumper_arm(self, i):
        return self.getProb(self.N[i], self.Sinv[i], None, self.theta[i])

    def store_info(self, i, x, y):
        self._update_inverse(i, x, y, tilde=False)

    def store_info_suparm(self, i, x, y):
        pass
