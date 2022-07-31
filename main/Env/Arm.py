import json
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Arm:
    def __init__(self, aid, fv=None, related_suparms={}):
        self.id = aid
        self.fv = fv
        self.suparms = related_suparms


class ArmManager:
    def __init__(self, in_folder):
        self.in_folder = in_folder
        self.arms = {}
        self.n_arms = 0
        self.dim = 0

    def loadArms(self, filepath='/arm_info.txt'):
        file_name = self.in_folder + filepath
        with open(file_name, 'r') as fr:
            for line in fr:
                j_s = json.loads(line)
                aid = j_s['a_id']
                fv = j_s['fv']
                self.dim = len(fv)
                self.arms[aid] = Arm(aid, torch.tensor(fv, device=device).squeeze())
        self.n_arms = len(self.arms)
