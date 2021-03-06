from Bio.PDB import PDBParser
from tqdm import tqdm
import sys
import numpy as np
from scipy.spatial import cKDTree
from Bio.PDB import is_aa
import pandas as pd

pdbs=[]
chains_to_keep=[]
Ys = []
for line in [l.strip() for l in open("/home/raktim/interactions/data.csv","r").readlines()[1:]]:
    split = line.split(",")[0]
    pdbs.append(split.split("_")[0])
    chains_to_keep.append(split.split("_")[1].replace(":",""))
    Ys.append(float(line.split(",")[1]))

Ys = np.array(Ys)

Ys = ((Ys - np.mean(Ys))/(np.std(Ys))).tolist()


parser = PDBParser()



processed_path = "/home/raktim/interactions/data/processed/"
out_path = "/home/raktim/interactions/data/npz/"
raw_atom_types = "/home/raktim/interactions/utils/atom_types.txt"

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
        try:
            atom_feature[feature_coord_map[atom_keys[key]]] = 1
        except:
            #print(key)
            continue
        try:
            simple_atom_feature[simple_map[atom.element]] = 1
        except Exception as e:
            print(e, atom.element)
            continue
        
        atom_list.append(atom)
        V.append(atom.coord)
        atom_feature[-1] = int(is_aa(atom.get_parent())) ## last coordinate is isprotein ()
        simple_atom_feature[-1] = int(is_aa(atom.get_parent())) ## last coordinate is isprotein ()
        X.append(atom_feature)
        Xs.append(simple_atom_feature)
    Kn = cKDTree(V)
    B = np.zeros_like(np.array(V)[:,0]).astype(int)

    for i in range(len(V)):
        ret = Kn.query_ball_point(V[i], 4.5)[1:]
        for j_idx in ret:
            E.append([i,j_idx])
            mask_val = (is_aa(atom_list[i].get_parent()) 
                != (is_aa(atom_list[j_idx].get_parent())))
            E_mask.append(mask_val)
            if mask_val:
                B[i] = 1
                B[j_idx] = 1
                    

        
    
    npz_data = dict()
    npz_data['V'] = np.array(V)
    npz_data['X'] = np.array(X)
    npz_data['Xs'] = np.array(Xs)
    npz_data['E'] = np.array(E)
    npz_data['E_mask'] = np.array(E_mask)
    npz_data['Y'] = Y
    npz_data['B'] = B
    
    where = npz_data['X'][:,-1] == 1
    npz_data['prot_mask'] = where
    npz_data['Xb'] = np.array(Xs)[where,:]
    npz_data['Vb'] = np.array(V)[where,:]
    npz_data['Yb'] = np.array(B)[where]

    Eb = []
    for i in range(len(npz_data['Vb'])):
        ret = Kn.query_ball_point(npz_data['Vb'][i], 4.5)[1:]
        for j_idx in ret:
            Eb.append([i,j_idx])
    npz_data['Eb'] = np.array(Eb)


    '''
    protein_cloud = PyntCloud(pd.DataFrame(
        data = npz_data['V'][npz_data['X'][:,250] == 1, :],
        columns=["x", "y", "z"]
            ))
    k_neighbors = protein_cloud.get_neighbors(k=10)
    protein_cloud.add_scalar_field("normals", k_neighbors=k_neighbors)

    dna_cloud = PyntCloud(pd.DataFrame(
        data = npz_data['V'][npz_data['X'][:,250] == 0, :],
        columns=["x", "y", "z"]
            ))
    k_neighbors = dna_cloud.get_neighbors(k=6)
    try:
        dna_cloud.add_scalar_field("normals", k_neighbors=k_neighbors)
    except:
        print(dna_cloud, dna_cloud.points, k_neighbors, idx)
    npz_data['N'] = np.zeros_like(V)
    npz_data['N'][np.where(npz_data['X'][:,250] == 1), :] = protein_cloud.points.to_numpy()[:,3:]
    npz_data['N'][np.where(npz_data['X'][:,250] == 0), :] = dna_cloud.points.to_numpy()[:,3:]
    '''
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
    npz_data = encode_features(model, Ys[i], out_path + pdbs[i].lower() + "_" + chains_to_keep[i] + ".npz", atom_keys, feature_coord_map, idx = pdbs[i].lower() + "_" + chains_to_keep[i])
    np.savez(out_path + pdbs[i].lower() + "_" + chains_to_keep[i] + ".npz", **npz_data)




