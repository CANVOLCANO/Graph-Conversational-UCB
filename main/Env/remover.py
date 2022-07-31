import json
import numpy as np

alg_name_list = ['[ConUCB]', '[Gopt-Pro_1-hop-tanimoto+cos_SumOuter_0.15C]']
alg_map = {
    '[ConUCB]':'ConUCB',
    '[Gopt-Pro_1-hop-tanimoto+cos_SumOuter_0.15C]':'GraphConUCB'
}
ratio = 0.5
pseudo_label = 987654321

if __name__ == '__main__':
    with open('kt_mapping.json', 'r') as f:
        kt_map = json.load(f)
        kt_map = dict(kt_map)

    for alg in alg_name_list:
        with open('Kt_stat_{}.json'.format(alg), 'r') as f:
            remove_list = dict(json.load(f))
            floatize_v = [float(i) for i in list(remove_list.values())]
            tot = sum(floatize_v)
            print(tot)
            remove_select = np.random.choice(a=list(remove_list.keys()), size=(int(len(remove_list)*ratio)), replace=False, p=np.array(floatize_v)/tot)

        mapped_remove_select = set([str(kt_map[str(r)]) for r in remove_select])

        file_name = 'arm_suparm_relation'
        res = []
        with open(file_name+'.txt', 'r') as fr:
            for line in fr:
                e = line.strip().split('\t')
                a_id, s_ids = int(e[0]), e[1].strip(', ').split(',')
                app = list(set(s_ids)-mapped_remove_select)
                if len(app) == 0:
                    app.append(str(pseudo_label))
                res.append(app)
        with open(file_name+'_{}_{}.txt'.format(alg_map[alg],str(ratio)), 'w') as f:
            for i in range(len(res)):
                print('{}\t{}'.format(i, ','.join(res[i])), file=f)