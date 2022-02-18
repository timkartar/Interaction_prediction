import torch
import os.path as osp
from pickle import dump, load

import numpy as np
from torch_geometric.data import Data

def _processData(data_files, x_key='X', y_key='Y'):
    data_list = []
    
    # read and process datafiles

    for f in data_files:
        data_arrays = np.load(f)
        
        ##########################################33
        data_arrays = dict(data_arrays)

        ###########################################
        
        if(x_key not in ['Xb']):
            data = Data(
                x=torch.tensor(data_arrays[x_key], dtype=torch.float32),
                pos=torch.tensor(data_arrays['V'], dtype=torch.float32),
                edge_attr=None,
                het_edge_index=torch.tensor(data_arrays['E'][data_arrays['E_mask']]),
                protein_mask = torch.tensor(data_arrays['prot_mask']),
                f = f
            )
        else:
            data = Data(
                x=torch.tensor(data_arrays[x_key], dtype=torch.float32),
                pos=torch.tensor(data_arrays['Vb'], dtype=torch.float32),
                edge_attr=None,
                f = f
            )
        if(y_key in ['Y']):
            data.y=torch.tensor(data_arrays[y_key], dtype=torch.float32).unsqueeze(0)
            data.mask = torch.tensor(np.ones_like(data.y[:,0]), dtype=torch.bool)
        elif(y_key in ['B','Yb']):
            data.y = torch.tensor(data_arrays[y_key], dtype=torch.int32)
            data.mask = torch.tensor(np.ones_like(data.y), dtype=torch.bool)
        elif("," in y_key):
            split = y_key.split(",")
            data.y = torch.tensor(data_arrays[split[0]], dtype=torch.float32).unsqueeze(0)
            data.y_cls = torch.tensor(data_arrays[split[1]], dtype=torch.int32)
            data.mask = torch.tensor(np.ones_like(data.y), dtype=torch.bool)

        data_list.append(data)
    
    
    return data_list

def loadDataset(data_files, x_key, labels_key, data_dir, **kwargs):
    if isinstance(data_files, str):
        with open(data_files) as FH:
            data_files = [_.strip() for _ in FH.readlines()]
    
    data_files = [osp.join(data_dir, f) for f in data_files]
        
    dataset = _processData(data_files, x_key = x_key, y_key=labels_key)
    info = {
        "num_features": int(getattr(dataset[0],x_key.lower()[0]).shape[1]),
        "num_instances": len(dataset)
    }
    
    return dataset, info
