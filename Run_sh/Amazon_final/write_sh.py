import os
import json
import numpy as np

ds = 'Amazon_final'
gpu=[0,1,2,3]
# seeds = [-1]
seeds = list(range(10))
suffix = [
# '[ConUCB_0.5R]',
# '[ConUCB_0.9R]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_0.5R]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_0.9R]'

# '[ConUCB_0.2R]',
# '[ConUCB_0.4R]',
# '[ConUCB_0.6R]',
# '[ConUCB_0.8R]',
# '[ConUCB_1.0R]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_0.2R]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_0.4R]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_0.6R]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_0.8R]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_1.0R]'

# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_val_0.1Gamma]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_val_0.3Gamma]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_val_0.5Gamma]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_val_0.7Gamma]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_val_0.9Gamma]'

# '[ConUCB_0.1R_TV1.7]',
# '[ConUCB_0.2R_TV1.7]',
# '[ConUCB_0.3R_TV1.7]',
# '[ConUCB_0.4R_TV1.7]',
# '[ConUCB_0.5R_TV1.7]',
# '[ConUCB_0.6R_TV1.7]',
# '[ConUCB_0.7R_TV1.7]',
# '[ConUCB_0.8R_TV1.7]',
# '[ConUCB_0.9R_TV1.7]',
# '[ConUCB_1.0R_TV1.7]',

# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_0.1R_TV1.7_0.7Gam]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_0.2R_TV1.7_0.7Gam]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_0.3R_TV1.7_0.7Gam]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_0.4R_TV1.7_0.7Gam]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_0.5R_TV1.7_0.7Gam]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_0.6R_TV1.7_0.7Gam]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_0.7R_TV1.7_0.7Gam]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_0.8R_TV1.7_0.7Gam]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_0.9R_TV1.7_0.7Gam]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_1.0R_TV1.7_0.7Gam]',

# '[ConUCB_TV1.7]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_TV1.7_0.7Gam]',

# '[ConUCB_TV1.5]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_TV1.5_0.7Gam]',

# '[ConUCB_TV1.7_RSetting1_RMRho0.1]',
# '[ConUCB_TV1.7_RSetting1_RMRho0.2]',
# '[ConUCB_TV1.7_RSetting1_RMRho0.3]',
# '[ConUCB_TV1.7_RSetting1_RMRho0.4]',
# '[ConUCB_TV1.7_RSetting1_RMRho0.5]',
# '[ConUCB_TV1.7_RSetting1_RMRho0.6]',
# '[ConUCB_TV1.7_RSetting1_RMRho0.7]',
# '[ConUCB_TV1.7_RSetting1_RMRho0.8]',
# '[ConUCB_TV1.7_RSetting1_RMRho0.9]',
# '[ConUCB_TV1.7_RSetting1_RMRho1.0]',

# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_TV1.7_0.7Gam_RSetting1_RMRho0.1]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_TV1.7_0.7Gam_RSetting1_RMRho0.2]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_TV1.7_0.7Gam_RSetting1_RMRho0.3]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_TV1.7_0.7Gam_RSetting1_RMRho0.4]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_TV1.7_0.7Gam_RSetting1_RMRho0.5]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_TV1.7_0.7Gam_RSetting1_RMRho0.6]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_TV1.7_0.7Gam_RSetting1_RMRho0.7]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_TV1.7_0.7Gam_RSetting1_RMRho0.8]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_TV1.7_0.7Gam_RSetting1_RMRho0.9]',
# '[Gopt-Pro_gf_1-hop-tanimoto+CosSim_SumOuter_TV1.7_0.7Gam_RSetting1_RMRho1.0]',

# '[LinUCB_TV1.7]',
# '[Gopt-Pro_1-hop-tanimoto+CosSim_SumOuter_TV1.7_0.7Gam]',
# '[ConUCB_gf_1-hop-tanimoto+CosSim_SumOuter_TV1.7_0.7Gam]',

# '[CtoF-ConUCB+_TV1.7]'
'[CtoF-ConUCB+_TV1.7_U5_I0.05_Switch0.002]'
]

cnt = 0
for sufix in suffix:
    # fg = "/data/czzhao_hdd/GraphCRS/Run/{}/{}.json".format(ds, sufix)
    fg = "/data/czzhao_hdd/Gitee/GraphCRS/Run/{}/{}.json".format(ds, sufix)
    config = ""
    with open(fg,'r',encoding='utf8') as fp:
        config = json.load(fp)
    
    
    print(config)

    with open('../{}/on_{}_{}.sh'.format(ds, ds, sufix),'w') as f:
        for i in seeds:
            s = "CUDA_VISIBLE_DEVICES={} nohup python ../../main/__main__.py --seed={}".format(gpu[cnt%len(gpu)], i)
            # s = "CUDA_VISIBLE_DEVICES={} nohup python ../../main/__main__.py --seed={}".format(gpu[(cnt//5)%len(gpu)], i)
            for k in config:
                s += " --{}={}".format(k, config[k])
            print(s, '&\n', file=f)
            cnt+=1