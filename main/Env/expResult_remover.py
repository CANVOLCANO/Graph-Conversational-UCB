import shutil
import os
from pathlib import Path

t_alg = [
    '[ConUCB_0.1R]',
    '[ConUCB_0.2R]',
    '[ConUCB_0.3R]',
    '[ConUCB_0.4R]',
    '[ConUCB_0.5R]',
    '[ConUCB_0.6R]',
    '[ConUCB_0.7R]',
    '[ConUCB_0.8R]',
    '[ConUCB_0.9R]',
    '[ConUCB_1.0R]',
]

seeds = list(range(10))
for s in seeds:
    for a in t_alg:
        p = str(Path(str(s), a))
        try:
            shutil.rmtree(p)
        except Exception as e:
            print(e)