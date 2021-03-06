# third party modules
import torch
import sys

def processBatch(device, batch, xtras=None):
    batch_data = {}
    if isinstance(batch, list):
        batch_data['y'] = torch.cat([data.y for data in batch]).to(device)
        batch_data['mask'] = torch.cat([data.mask for data in batch]).to(device)
        if xtras is not None:
            for item in xtras:
                if(item == "f"):
                    batch_data[item] = [getattr(data, item) for data in batch]
                else:
                    batch_data[item] = torch.cat([getattr(data, item) for data in batch]).to(device)
        batch_data['batch'] = batch
    else:
        batch_data['batch'] = batch.to(device)
        batch_data['y'] = batch.y
        batch_data['mask'] = batch.mask
        if xtras is not None:
            for item in xtras:
                batch_data[item] = getattr(batch, item)
    
    return batch_data
