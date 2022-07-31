import os
import time
import argparse

from Env.ENVIRONMENT import Environment
from Env.utils import edge_probability


def main(args, filename):
    envir = Environment(args=args, file_name=filename, seed=args.seed)
    d, num_users = envir.d, envir.nu
    if args.alg == 'ConUCB':
        from Agent.BASE import ConUCB
        agent = ConUCB(nu=num_users, d=d, T=2 ** args.N - 1, args=args)
    elif args.alg == 'LinUCB':
        from Agent.LinUCB import LinUCB
        agent = LinUCB(nu=num_users, d=d, T=2 ** args.N - 1, args=args)
    elif args.alg == 'GraConUCB':
        from Agent.GraConUCB import GraConUCB
        agent = GraConUCB(nu=num_users, d=d, T=2 ** args.N - 1, args=args)
    else:
        raise AssertionError
    agent.run(envir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', dest='seed', default=9, type=int)
    parser.add_argument('--N', dest='N', default=10, type=int) # N
    parser.add_argument('--fn', dest='fn', default='lastfm', type=str)
    parser.add_argument('--para', dest='para', default='Fix+700', type=str)
    parser.add_argument('--alg', dest='alg', default='ConUCB', type=str)
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float) # N
    parser.add_argument('--alpha_p', dest='alpha_p', default=None, type=float) # N
    parser.add_argument('--delta', dest='delta', default=0.1, type=float) # N

    parser.add_argument('--staU', dest='staU', default=0, type=int) # left closed
    parser.add_argument('--endU', dest='endU', default=100, type=int) # right open

    parser.add_argument('--armRelation', dest='armRelation', default='arm_suparm_relation', type=str) # N # the fine name of the relation between arm and supArm 
    parser.add_argument('--rewardNoise', dest='rewardNoise', default='N', type=str) # N # whether use noise when generate the rewards, Y: yes N: no
    parser.add_argument('--trueReward', dest='trueReward', default='Y', type=str) # N
    # whether use the primitive reward matrix when generate the rewards, 
    # Y: use the primitive user-item review matrix, N: use the inner product of user preference vector and item feature vector
    parser.add_argument('--useGCN', dest='useGCN', default='N', type=str) # N # whether use GCN to get the adj_mat between attributes, Y: yes N: no
    parser.add_argument('--itemEmbedding', dest='itemEmbedding', default='arm_info', type=str) # N # the file path of item embedding
    parser.add_argument('--attrEmbedding', dest='attrEmbedding', default='attribute_embedding_0_layer', type=str) # N # the file path of key-term embedding
    parser.add_argument('--userEmbedding', dest='userEmbedding', default='user_preference_test', type=str) # N # the file path of user item interaction review matrix
    parser.add_argument('--userTrainEmbedding', dest='userTrainEmbedding', default='user_preference_train', type=str) # N

    parser.add_argument('--reviewMatrix', dest='reviewMatrix', default='user_item_test', type=str) # N # the file path of user item interaction review matrix

    parser.add_argument('--DS', dest='DS', default='N', type=str) # N # whether to use the dominating set as the candidate set of key-terms, Dyn: use dynamic dominating set, Sta: use static dominating set, N: no
    parser.add_argument('--attrPolicy', dest='attrPolicy', default='ConUCB_MSN', type=str) # the policy used to select key-terms in GraConUCB: ConUCB, ConUCB_MSN (ConUCB Maximized Summed Neighbors)
    parser.add_argument('--itemPolicy', dest='itemPolicy', default='ConUCB', type=str) # N # the policy used to select items in ConUCB, SoftUCB
    parser.add_argument('--GF', dest='GF', default='N', type=str) # whether to use the graph feedback on key-term-level
    parser.add_argument('--similarity', dest='similarity', default='CosSim', type=str) # N # use cosine similarity or common Item as adj_mat elements
    parser.add_argument('--k', dest='k', default='1', type=str) # combine with 'similarity==hop-CosSim' to use k-hop-CosSim
    parser.add_argument('--cos_lambda', dest='cos_lambda', default='0.5', type=float) # balance the 2 similarity
    parser.add_argument('--keyreward', dest='keyreward', default='Continuous', type=str) # key-term reward value Binomial or Continuous
    parser.add_argument('--useSumOuter', dest='useSumOuter', default='N', type=str) # use sum neighbours outer as key-term outer.
    parser.add_argument('--useDynamicSimilarity', dest='useDynamicSimilarity', default='N', type=str)  # N # use Dynamic Similarity update key-term adjmat

    parser.add_argument('--C', dest='C', default='0.15', type=float) # used in G Opt
    parser.add_argument('--RMRho', dest='RMRho', default='0', type=float) # item key-term linking removing rho [0, 1]
    parser.add_argument('--AlgType', dest='AlgType', default='ConUCB', type=str) # work when RMRho>0, ConUCB | GraphConUCB 

    # for link removing setting 3
    parser.add_argument('--blackList', dest='blackList', default='Kt_stat_[ConUCB].json', type=str)
    parser.add_argument('--RSetting', dest='RSetting', default='1', type=int) # remove setting, 0 for newly generated data, 1 for constrained blackList, 2 for full list, 4 for Env top reward remove
    parser.add_argument('--RSampling', dest='RSampling', default='positive', type=str) # uniform, positive, negative

    # for key-terms removing setting 4
    parser.add_argument('--RMRhoEnv', dest='RMRhoEnv', default='0', type=float) # key-term removing rho on env side [0, 1]

    # torch version sign
    parser.add_argument('--TV', dest='TV', default='1.7', type=float) # key-term removing rho on env side [0, 1]

    args = parser.parse_args()
    prefix = args.fn 

    start_time = time.time()
    if args.seed < 0:
        for i in range(abs(args.seed) - 1, 10):
            args.seed = i
            args.fn = prefix + '/' + str(i) # e.g., ../datasets/con_data_50_0.01_0.5/0
            if not os.path.exists(args.fn):
                os.mkdir(args.fn)
            main(args=args, filename=prefix)
    else:
        args.fn = prefix + '/' + str(args.seed)
        if not os.path.exists(args.fn):
            os.mkdir(args.fn)
        main(args=args, filename=prefix)
    run_time = time.time() - start_time
    print('total time:', run_time)
