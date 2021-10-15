import torch
import os.path as osp
from pickle import dump, load

import numpy as np
from torch_geometric.data import Data

def _processData(data_files):
    data_list = []
    
    # read and process datafiles

    for f in data_files:
        data_arrays = np.load(f)
        
        ##########################################33
        data_arrays = dict(data_arrays)

        ###########################################

        data = Data(
            x=torch.tensor(data_arrays['X'], dtype=torch.float32),
            y=torch.tensor(data_arrays['Y'], dtype=torch.float32).unsqueeze(0),
            pos=torch.tensor(data_arrays['V'], dtype=torch.float32),
            norm=torch.tensor(data_arrays['N'], dtype=torch.float32),
            edge_attr=None,
            het_edge_index=torch.tensor(data_arrays['E'][data_arrays['E_mask']])
        )
        #data.mask = torch.tensor(idxb, dtype=torch.bool)
        data_list.append(data)
    
    
    return data_list

def loadDataset(data_files, labels_key, data_dir, **kwargs):
    if isinstance(data_files, str):
        with open(data_files) as FH:
            data_files = [_.strip() for _ in FH.readlines()]
    
    data_files = [osp.join(data_dir, f) for f in data_files]
        
    dataset = _processData(data_files)
    info = {
        "num_features": int(dataset[0].x.shape[1]),
        "num_instances": len(dataset)
    }
    
    return dataset, info
