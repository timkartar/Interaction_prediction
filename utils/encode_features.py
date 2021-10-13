from Bio.PDB import PDBParser
from tqdm import tqdm
import sys
import numpy as np
from scipy.spatial import cKDTree
from Bio.PDB import is_aa
pdbs=[]
chains_to_keep=[]
Ys = []
for line in [l.strip() for l in open("/project/rohs_102/raktimmi/interactions/data.csv","r").readlines()[1:]]:
    split = line.split(",")[0]
    pdbs.append(split.split("_")[0])
    chains_to_keep.append(split.split("_")[1].replace(":",""))
    Ys.append(float(line.split(",")[1]))

parser = PDBParser()



processed_path = "/project/rohs_102/raktimmi/interactions/data/processed/"
out_path = "/project/rohs_102/raktimmi/interactions/data/npz/"
raw_atom_types = "/project/rohs_102/raktimmi/interactions/utils/atom_types.txt"

def encode_features(model, Y, out_path, atom_keys, feature_coord_map):
    atom_list = []
    V = []
    X = []
    E = []
    E_mask = []
    atom_feature_length = max(feature_coord_map.values()) + 2
    for atom in model.get_atoms():
        if(atom.element == 'H'):
            continue
        key = atom.get_name()+ "_" + atom.get_parent().get_resname()
        atom_feature = np.zeros(atom_feature_length)
        try:
            atom_feature[feature_coord_map[atom_keys[key]]] = 1
        except:
            #print(key)
            continue
        atom_list.append(atom)
        V.append(atom.coord)
        atom_feature[-1] = int(is_aa(atom.get_parent())) ## last coordinate is isprotein ()
        X.append(atom_feature)
    Kn = cKDTree(V)

    for i in range(len(V)):
        ret = Kn.query_ball_point(V[i], 4)[1:]
        for j_idx in ret:
            E.append([i,j_idx])
            E_mask.append((is_aa(atom_list[i].get_parent()) 
                != (is_aa(atom_list[j_idx].get_parent()))))
        
    
    npz_data = dict()
    npz_data['V'] = np.array(V)
    npz_data['X'] = np.array(X)
    npz_data['E'] = np.array(E)
    npz_data['E_mask'] = np.array(E_mask)
    npz_data['Y'] = Y
    return npz_data

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
feature_coord_map  = dict()
for i, k in enumerate(atom_keys.values()):
    feature_coord_map[k] = i
    

for i in tqdm(range(len(pdbs))):
    structure = parser.get_structure(pdbs[i], processed_path + pdbs[i].lower() + "_" + chains_to_keep[i] + ".pdb")
    model = structure[0]
    npz_data = encode_features(model, Ys[i], out_path + pdbs[i].lower() + "_" + chains_to_keep[i] + ".npz", atom_keys, feature_coord_map)
    np.savez(out_path + pdbs[i].lower() + "_" + chains_to_keep[i] + ".npz", **npz_data)




