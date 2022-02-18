from Bio.PDB import PDBParser
from tqdm import tqdm
import sys
import numpy as np
from scipy.spatial import cKDTree
from Bio.PDB import is_aa
from pyntcloud import PyntCloud
import pandas as pd
import numpy as np

pdbs=[]
chains_to_keep=[]
Ys = []
for line in [l.strip() for l in open("/project/rohs_102/raktimmi/interactions/pdbbind_index.txt","r").readlines()[1:]]:
    if(line[0] == "#"):
        continue
    split = line.split(" ")
    while "" in split:
        split.remove("")
    
    if(split[3][:2] != "Kd"):
        continue
    
    unit = (split[3][::-1][:2])[::-1]
    try:
        try:
            val = float(split[3].replace("Kd=","").replace(unit,""))
        except:
            val = float(split[3].replace("Kd~","").replace(unit,""))
    except:
        continue

    if(unit == "nM"):
        y = np.log(val/10**3)
    elif(unit == "pM"):
        y = np.log(val/10**6)
    else:
        y = np.log(val)
    
    #print(split[3])
    
    pdbs.append(split[0])
    #chains_to_keep.append(split.split("_")[1].replace(":",""))
    Ys.append(y)

Ys = np.array(Ys)

Ys = ((Ys - np.mean(Ys))/(np.std(Ys))).tolist()




parser = PDBParser()



processed_path = "/project/rohs_102/raktimmi/interactions/PN/"
out_path = "/project/rohs_102/raktimmi/interactions/data/npz/"
raw_atom_types = "/project/rohs_102/raktimmi/interactions/utils/atom_types.txt"

def encode_features(model, Y, out_path, atom_keys, feature_coord_map, idx = None):
    atom_list = []
    V = []
    X = []
    Xs = []
    E = []
    E_mask = []
    atom_feature_length = max(feature_coord_map.values()) + 2

    simple_map = {'C' : 0,
            'O': 1,
            'N': 2,
            'P': 3,
            'S': 4
            }
    for atom in model.get_atoms():
        if(atom.element == 'H'):
            continue
        key = atom.get_name()+ "_" + atom.get_parent().get_resname()
        atom_feature = np.zeros(atom_feature_length)
        simple_atom_feature = np.zeros(6)
        '''
        try:
            atom_feature[feature_coord_map[atom_keys[key]]] = 1
        except:
            print(key)
            continue
        '''
        try:
            simple_atom_feature[simple_map[atom.element]] = 1
        except Exception as e:
            #print(e, atom.element)
            continue
        
        atom_list.append(atom)
        V.append(atom.coord)
        
        #atom_feature[-1] = int(is_aa(atom.get_parent())) ## last coordinate is isprotein ()
        simple_atom_feature[-1] = int(is_aa(atom.get_parent())) ## last coordinate is isprotein ()
        X.append(atom_feature)
        Xs.append(simple_atom_feature)
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
    npz_data['Xs'] = np.array(Xs)
    npz_data['E'] = np.array(E)
    npz_data['E_mask'] = np.array(E_mask)
    npz_data['Y'] = Y
    #print((npz_data['Xs'][:,-1] == 1).sum(), npz_data['X'].shape)
    
    #print(X, Xs, Y)
    protein_cloud = PyntCloud(pd.DataFrame(
        data = npz_data['V'][npz_data['Xs'][:,-1] == 1, :],
        columns=["x", "y", "z"]
            ))
    k_neighbors = protein_cloud.get_neighbors(k=10)
    protein_cloud.add_scalar_field("normals", k_neighbors=k_neighbors)

    dna_cloud = PyntCloud(pd.DataFrame(
        data = npz_data['V'][npz_data['Xs'][:,-1] == 0, :],
        columns=["x", "y", "z"]
            ))
    k_neighbors = dna_cloud.get_neighbors(k=6)
    try:
        dna_cloud.add_scalar_field("normals", k_neighbors=k_neighbors)
    except:
        print(dna_cloud, dna_cloud.points, k_neighbors, idx)
    npz_data['N'] = np.zeros_like(V)
    npz_data['N'][np.where(npz_data['Xs'][:,-1] == 1), :] = protein_cloud.points.to_numpy()[:,3:]
    npz_data['N'][np.where(npz_data['Xs'][:,-1] == 0), :] = dna_cloud.points.to_numpy()[:,3:]
    return npz_data

atom_keys = dict()

for line in [l.strip() for l in open(raw_atom_types,"r").readlines()]:
    atom, res  = tuple(line.split("_"))
    if(res in ['DA','DC','DG','DT']):
        if("'" in line):
            atom_keys[line] = "sugar" + atom[0]
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
    try:
        #if(pdbs[i] != "1qfq"):
        #    continue
        structure = parser.get_structure(pdbs[i], processed_path + pdbs[i].lower() + ".ent.pdb")
        model = structure[0]
        npz_data = encode_features(model, Ys[i], out_path + pdbs[i].lower() + ".npz", atom_keys, feature_coord_map, idx = pdbs[i].lower())
        np.savez(out_path + pdbs[i].lower() + "_" + ".npz", **npz_data)
        print(pdbs[i], True)
    except Exception as e:
        print(e, pdbs[i])



