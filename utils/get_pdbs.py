import numpy as np
import wget
from tqdm import tqdm
import time

pdbs=[]

for line in [l.strip() for l in open("/project/rohs_102/raktimmi/interactions/data.csv","r").readlines()[1:]]:
    split = line.split(",")[0]
    pdbs.append(split.split("_")[0])

#pdbl = PDBList(server="https://files.rcsb.org/download/1hh3.pdb1")
for i in tqdm(pdbs):
    try:
        wget.download("https://files.rcsb.org/download/" + i.lower() + ".pdb1", out='/project/rohs_102/raktimmi/interactions/data/raw_pdbs')
    except:
        print("could not find ", i)




