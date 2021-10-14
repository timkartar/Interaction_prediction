# builtin modules
import os.path as osp
import hashlib
from pickle import dump, load
import sys

# third party modules
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import StandardScaler
from dna2vec.multi_k_model import MultiKModel

# geobind modules
from geobind.nn.utils import balancedClassIndices
from nn.utils.encodings import one_hot_encode, one_hot_to_seq, rc_seq

class NodeScaler(object):
    def __init__(self):
        self._data_arrays = []
        self.scaler = StandardScaler()
    
    def update(self, array):
        self._data_arrays.append(array)
    
    def fit(self):
        self.scaler.fit(np.concatenate(self._data_arrays, axis=0))
    
    def scale(self, array):
        return self.scaler.transform(array)

class ClassificationDatasetMemory(InMemoryDataset):
    def __init__(self, data_files, nc, labels_key, data_dir,
            save_dir=None,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            max_motif_len=30,
            balance='balanced',
            percentage=1.0,
            remove_mask=False,
            unmasked_class=0,
            scale=True,
            scaler=None,
            feature_mask=None,
            kmer_encoding = "dna2vec"
        ):
        if(save_dir is None):
            save_dir = data_dir
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.data_files = data_files
        self.labels_key = labels_key
        self.nc = nc
        self.max_motif_len = max_motif_len
        self.balance = balance
        self.percentage = percentage
        self.remove_mask = remove_mask
        self.unmasked_class = unmasked_class
        self.scale = scale
        self.scaler = scaler
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.feature_mask = feature_mask
        self.kmer_encoding = kmer_encoding

        super(ClassificationDatasetMemory, self).__init__(save_dir, transform, pre_transform, pre_filter)
        # load data
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        # load scaler
        if self.scale and self.scaler is None:
            self.scaler = load(open(self.processed_paths[1], 'rb'))
        
    @property
    def raw_file_names(self):
        return self.data_files
    
    @property
    def processed_file_names(self):
        m = hashlib.md5()
        args = [
            self.nc,
            self.labels_key,
            self.balance,
            self.percentage,
            self.remove_mask,
            self.unmasked_class,
            self.scale
        ]
        args = "".join([str(_) for _ in args] + list(sorted(self.data_files)))
        m.update(args.encode('utf-8'))
        self.hash_name = m.hexdigest()
        return ['{}.pt'.format(self.hash_name), '{}_scaler.pkl'.format(self.hash_name)]
    
    @property
    def raw_dir(self):
        return self.data_dir

    @property
    def processed_dir(self):
        return self.save_dir
    
    def process(self):
        # get datalist and scaler
        data_list, transforms = _processData(self.raw_paths, self.nc, self.labels_key, max_motif_len=self.max_motif_len,
            balance=self.balance, 
            remove_mask=self.remove_mask,
            unmasked_class=self.unmasked_class,
            scaler=self.scaler,
            scale=self.scale,
            pre_filter=self.pre_filter,
            pre_transform=self.pre_transform,
            transform=self.transform,
            feature_mask=self.feature_mask,
            encoding = self.kmer_encoding
        )
        
        # save data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        # save scaler
        scaler = transforms['scaler']
        if scaler is not None:
            self.scaler = scaler
            dump(scaler, open(self.processed_paths[1], "wb"))

def _processData(data_files, nc, labels_key, max_motif_len=30,
        balance="unmasked",
        remove_mask=False,
        unmasked_class=0,
        scaler=None,
        scale=True,
        transform=None,
        pre_filter=None,
        pre_transform=None,
        feature_mask=None,
        encoding = "dna2vec"
    ):
    data_list = []
    
    # read and process datafiles
    #encoding = "dna2vec" # or "one_hot"
    padding = False #"center"
    one_hot_padding = False
    if(encoding == "dna2vec"):
        filepath = '/project/rohs_102/raktimmi/dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
        mk_model = MultiKModel(filepath)

    for f in data_files:
        data_arrays = np.load(f)
        
        if remove_mask:
            # remove any previous masking
            data_arrays[labels_key][(data_arrays[labels_key] == -1)] = unmasked_class
        
        if balance == 'balanced':
            idxb = balancedClassIndices(data_arrays[labels_key], range(nc), max_percentage=self.percentage)
        elif balance == 'unmasked':
            idxb = (data_arrays[labels_key] >= 0)
        elif balance == 'all':
            idxb = (data_arrays[labels_key] == data_arrays[labels_key])
        else:
            raise ValueError("Unrecognized value for `balance` keyword: {}".format(balance))
        
        if feature_mask is not None:
            X = data_arrays['X'][:, feature_mask]
        else:
            X = data_arrays['X']
        
        ##########################################33
        ## uniform dostribution padding
        data_arrays = dict(data_arrays)

        if (encoding == "one_hot" and padding != False):
            if (one_hot_padding):
                pad_unit = [[0,0,0,0,1]]
                #print(data_arrays[labels_key], np.zeros_like(data_arrays[labels_key][:,3].reshape(-1,1)))
                data_arrays[labels_key] = np.hstack( (data_arrays[labels_key], np.zeros_like(data_arrays[labels_key][:,3].reshape(-1,1))) )
            else:
                pad_unit = [[0.25, 0.25, 0.25, 0.25]]
            pad_length = max_motif_len - data_arrays[labels_key].shape[0]
            
            if (pad_length <= 1 or padding == "end"):
                if (data_arrays[labels_key].shape[0] < max_motif_len):
                    pad = pad_unit*(pad_length)
                    Y = np.vstack((data_arrays[labels_key],pad))
                    mask = np.vstack(( np.ones_like(data_arrays[labels_key], dtype=bool) , np.zeros_like(pad, dtype=bool) ))
                else:
                    Y = data_arrays[labels_key]
                    mask = np.ones_like(data_arrays[labels_key], dtype=bool)
            elif (padding == "center"):
                pad1 = pad_unit*(pad_length//2)
                pad2 = pad_unit*(pad_length - pad_length//2)
                Y = np.vstack((pad1, data_arrays[labels_key], pad2))
                mask = np.vstack(( np.zeros_like(pad1, dtype=bool), np.ones_like(data_arrays[labels_key], dtype=bool) , np.zeros_like(pad2, dtype=bool) ))
        elif(encoding == "one_hot"):
            Y = data_arrays[labels_key]
            mask = np.ones_like(data_arrays[labels_key], dtype=bool)
        elif(encoding == "dna2vec"):
            Y = one_hot_to_seq(data_arrays[labels_key])
            Y_rc = rc_seq(Y)
            Y = mk_model.vector(Y)
            Y_rc = mk_model.vector(Y_rc)
            mask = np.ones_like(Y, dtype=bool)
            #print(mk_model.model(3).similar_by_vector(v))
        #Y = np.vstack((Y, np.flip(Y,(0,1)) ))
        #mask = np.vstack((mask, np.flip(mask,(0,1)) ))
        ###########################################
        data = Data(
            x=torch.tensor(X, dtype=torch.float32),
            y=torch.tensor(Y, dtype=torch.float32),
            y_rc=torch.tensor(Y_rc, dtype=torch.float32),
            pos=torch.tensor(data_arrays['V'], dtype=torch.float32),
            norm=torch.tensor(data_arrays['N'], dtype=torch.float32),
            face=torch.tensor(data_arrays['F'].T, dtype=torch.int64),
            mask=torch.tensor(mask, dtype=torch.bool),
            edge_attr=None,
            edge_index=None
        )
        #data.mask = torch.tensor(idxb, dtype=torch.bool)
        data_list.append(data)
    
    # filter data
    if pre_filter is not None:
        data_list = [data for data in data_list if pre_filter(data)]
    
    # transform data
    if pre_transform is not None:
        data_list = [pre_transform(data) for data in data_list]
    
    # scale data
    if scale:
        # build a scaler
        if scaler is None:
            scaler = NodeScaler()
            for data in data_list:
                #scaler.update(data.x[data.mask])
                scaler.update(data.x)
            scaler.fit()
        
        # scale node features in each data object
        for data in data_list:
            data.x = torch.tensor(scaler.scale(data.x), dtype=torch.float32)
    
    transforms = {
        "scaler": scaler,
        "transform": transform,
        "pre_transform": pre_transform,
        "pre_filter": pre_filter
    }
    return data_list, transforms

def loadDataset(data_files, nc, labels_key, data_dir, cache_dataset=False, kmer_encoding="dna2vec", **kwargs):
    if isinstance(data_files, str):
        with open(data_files) as FH:
            data_files = [_.strip() for _ in FH.readlines()]
    
    if cache_dataset:
        dataset = ClassificationDatasetMemory(data_files, nc, labels_key, data_dir, **kwargs)
        transforms = {
            "scaler": dataset.scaler,
            "transform": dataset.transform,
            "pre_transform": dataset.pre_transform,
            "pre_filter": dataset.pre_filter
        }
        info = {
            "num_features": dataset.num_node_features,
            "num_classes": nc,
            "num_instances": len(dataset)
        }
    else:
        data_files = [osp.join(data_dir, f) for f in data_files]
        
        dataset, transforms = _processData(data_files, nc, labels_key, encoding=kmer_encoding, **kwargs)
        info = {
            "num_features": int(dataset[0].x.shape[1]),
            "num_classes": nc,
            "num_instances": len(dataset)
        }
    
    return dataset, transforms, info
