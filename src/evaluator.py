# third party modules
import torch
import numpy as np

# geobind modules
from process_batch import processBatch
from metrics import auroc, auprc, balanced_accuracy_score, recall_score, brier_score_loss
from metrics import precision_score, jaccard_score, f1_score, accuracy_score, matthews_corrcoef
from report_metrics import reportMetrics
from metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score

METRICS_FN = {
    'accuracy': accuracy_score,
    'balanced_accuracy': balanced_accuracy_score,
    'mean_iou': jaccard_score,
    'auroc': auroc,
    'auprc': auprc,
    'recall': recall_score,
    'precision': precision_score,
    'f1_score': f1_score,
    'brier_score': brier_score_loss,
    'matthews_corrcoef': matthews_corrcoef,
    'mean_squared_error': mean_squared_error,
    'mean_absolute_error': mean_absolute_error,
    'r2_score': r2_score
}

def registerMetric(name, fn):
    METRICS_FN[name] = fn

class Evaluator(object):
    def __init__(self, model, nc, device="cpu", metrics=None, post_process=None, negative_class=0, labels=None):
        self.model = model # must implement the 'forward' method
        self.device = device
        self.negative_class = negative_class
        
        if post_process is None:
            # identity function
            post_process = lambda x: x
        self.post = post_process
        
        # decide what metrics to use
        self.nc = nc
        if metrics == 'none':
            self.metrics = None
        elif metrics is None:
            if nc == 2:
                # binary classifier
                metrics={
                    'auroc': {'average': 'binary'},
                    'auprc': {'average': 'binary'},
                    'balanced_accuracy': {},
                    'mean_iou': {'average': 'weighted'},
                    'precision': {'average': 'binary', 'zero_division': 0},
                    'recall': {'average': 'binary', 'zero_division': 0},
                    'accuracy': {}
                }
            elif nc > 2:
                # three or more classes 
                if labels is None:
                    labels = list(range(nc))
                    labels.remove(negative_class)
                metrics={
                    'balanced_accuracy': {},
                    'mean_iou': {'average': 'weighted', 'labels': labels, 'zero_division': 0},
                    'precision': {'average': 'weighted', 'zero_division': 0, 'labels': labels},
                    'recall': {'average': 'weighted', 'zero_division': 0, 'labels': labels},
                    'accuracy': {},
                    'matthews_corrcoef': {},
                    'auroc': {'average': 'macro'},
                }
            else:
                metrics = {"mean_squared_error": {},
                        "mean_absolute_error": {},
                        "r2_score": {}
                        }
            self.metrics = metrics
        else:
            if not isinstance(metrics, dict):
                raise ValueError("The argument 'metrics' must be a dictionary of kwargs and metric names or 'none'!")
            self.metrics = metrics
    
    @torch.no_grad()
    def eval(self, dataset, eval_mode=True, batchwise=False, use_masks=False, return_masks=False, return_predicted=False, return_batches=True, xtras=None, split_batches=False, **kwargs):
        """Returns numpy arrays!!!"""        
        
        def _loop(batch, data_items, y_gts, y_prs, outps, masks, batches):
            batch_data = processBatch(self.device, batch, xtras=xtras)
            #batch, y, mask = batch_data['batch'], batch_data['y'], batch_data['mask']
            batch, y = batch_data['batch'], batch_data['y']
            output = self.model(batch)
            if (type(output) == tuple):
                output = output[0][0]
            if use_masks:
                y = y[mask].cpu().numpy()
                out = self.post(output[mask]).cpu().numpy()
            else:
                y = y.cpu().numpy()
                out = self.post(output).cpu().numpy()
            
            y_gts.append(y)
            outps.append(out)
            if return_masks:
                masks.append(mask.cpu().numpy())
            
            if return_predicted:
                y_prs.append(self.predictClass(out, y, **kwargs))
            
            if xtras is not None:
                # these items will not be masked even if `use_mask == True`
                for item in xtras:
                    data_items[item].append(batch_data[item].cpu().numpy())
            
            if return_batches:
                if isinstance(batch, list):
                    if batchwise:
                        batches.append(batch)
                    else:    
                        batches += batch
                else:
                    if batchwise:
                        batches.append([batch])
                    else:
                        batches.append(batch.to('cpu'))
        
        # eval or training
        if eval_mode:
            self.model.eval()
        
        # evaluate model on given dataset
        data_items = {}
        y_gts = []
        y_prs = []
        outps = []
        masks = []
        batches = []
        if xtras is not None:
            # these items will not be masked even if `use_mask == True`
            for item in xtras:
                data_items[item] = []
        
        # loop over dataset
        for batch in dataset:
            if split_batches:
                dl = batch.to_data_list()
                for d in dl:
                    _loop(d, data_items, y_gts, y_prs, outps, masks, batches)
            else:
                _loop(batch, data_items, y_gts, y_prs, outps, masks, batches)
        
        # decide what to do with each data item
        data_items['y'] = y_gts
        data_items['output'] = outps
        if return_masks:
            data_items['masks'] = masks
        if return_predicted:
            data_items['predicted_y'] = y_prs
        
        # concat batches if not batchwise
        if batchwise:
            data_items['num_batches'] = len(y_gts)
        else:
            for item in data_items:
                data_items[item] = np.concatenate(data_items[item], axis=0)
            data_items['num'] = len(data_items['y'])
        
        # add batches if requested
        if return_batches:
            data_items['batches'] = batches
        return data_items
    
    def getMetrics(self, *args, 
            eval_mode=True, 
            metric_values=None, 
            threshold=None, 
            threshold_metric='balanced_accuracy', 
            report_threshold=False,
            metrics_calculation="total",
            split_batches=False,
            **kwargs
        ):
        if self.metrics is None:
            return {}
        if metric_values is None:
            metric_values = {key: [] for key in self.metrics}
        if 'threshold' in metric_values:
            threshold = metric_values['threshold']
        
        if metrics_calculation == "total":
            batchwise = False
        elif metrics_calculation == "average_batches":
            batchwise = True
        else:
            raise ValueError("Invalid option for `metrics_calculation`: {}".format(metrics_calculation))
        
        # Determine what we were given (a dataset or labels/predictions)
        if len(args) == 1:
            evald = self.eval(args[0], eval_mode=eval_mode, use_masks=False, return_masks=False, return_batches=True, batchwise=batchwise, split_batches=split_batches, **kwargs)
            if batchwise:
                y_gt = evald['y']
                outs = evald['output']
                batches = evald['batches']
                #masks = evald['masks']
            else:
                y_gt = [evald['y']]
                outs = [evald['output']]
                batches = [evald['batches']]
                #masks = [evald['masks']]
        else:
            if len(args) == 3:
                y_gt, outs, masks = args
                batches = None
            else:
                y_gt, outs, masks, batches = args
                batches = [batches]
            y_gt = [y_gt]
            outs = [outs]
            #masks = [masks]
        
        # Get predicted class labels
        #for i in range(len(y_gt)):
        for metric, kw in self.metrics.items():
            metric_values[metric].append(METRICS_FN[metric](np.array(y_gt).reshape(-1,1), np.array(outs).reshape(-1,1), **kw))
        for key in metric_values:
            metric_values[key] = np.mean(metric_values[key])
        
        return metric_values
    
