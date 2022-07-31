from pathlib import Path

t_alg = [
    '[ConUCB_0.1R_RSetting1_negative]',
    '[ConUCB_0.2R_RSetting1_negative]'
]

seeds = list(range(10))
tem = []
for s in seeds:
    ite = [a.name for a in Path(Path.cwd(),str(s)).iterdir()]
    for alg in t_alg:
        if alg not in ite:
            tem.append((s, alg))
        else:
            l = len(list(Path(Path.cwd(),str(s),alg).iterdir()))
            if l<1998:
                tem.append((s, alg))

tem = sorted(tem, key=lambda x:x[1])
for i in tem:
    print('fail in seed={} alg={}'.format(i[0], i[1]))