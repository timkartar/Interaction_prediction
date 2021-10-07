from Bio.PDB import PDBParser
from tqdm import tqdm
import sys
import numpy as np

pdbs=[]
chains_to_keep=[]
for line in [l.strip() for l in open("/project/rohs_102/raktimmi/interactions/data.csv","r").readlines()[1:]]:
    split = line.split(",")[0]
    pdbs.append(split.split("_")[0])
    chains_to_keep.append(split.split("_")[1].replace(":",""))


parser = PDBParser()



processed_path = "/project/rohs_102/raktimmi/interactions/data/processed/"
out_path = "/project/rohs_102/raktimmi/interactions/data/npz/"
raw_atom_types = "/project/rohs_102/raktimmi/interactions/utils/atom_types.txt"

def encode_features(model, out_path):
    pass


atom_keys = dict()

for line in [l.strip() for l in open(raw_atom_types,"r").readlines()]:
    atom, res  = tuple(line.split("_"))
    if(res in ['DA','DC','DG','DT']):
        if("'" in line):
            atom_keys[line] = "suger" + atom[0]
        elif("P" in atom):
            atom_keys[line] = "phosphate" + atom[0]
        else:
            atom_keys[line] = line
    elif(line == "N_PRO"):
        atom_keys[line] = line
    elif(atom in ['N', 'C', 'O']):
        atom_keys[line] = "aabackbone" + atom
    else:
        atom_keys[line] = line

atom_keys["OXT"] = "OXT"
print(atom_keys.values())

sys.exit()

for i in range(len(pdbs)):
    structure = parser.get_structure(pdbs[i], processed_path + pdbs[i].lower() + "_" + chains_to_keep[i] + ".pdb")
    model = structure[0]
    encode_features(model, out_path + pdbs[i].lower() + "_" + chains_to_keep[i] + ".npz")





