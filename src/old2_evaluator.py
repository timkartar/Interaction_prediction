# third party modules
import torch
import numpy as np
import sys

# geobind modules
from process_batch import processBatch
from metrics import auroc, auprc, balanced_accuracy_score, recall_score, brier_score_loss
from metrics import precision_score, jaccard_score, f1_score, accuracy_score, matthews_corrcoef
from report_metrics import reportMetrics
from metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
from geobind.nn.metrics import chooseBinaryThreshold, meshLabelSmoothness

from scipy.spatial.distance import jensenshannon
from scipy.special import softmax

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
    'smoothness': meshLabelSmoothness,
    'mean_squared_error': mean_squared_error,
    'mean_absolute_error': mean_absolute_error,
    'r2_score': r2_score,
    'auroc_ovo': auroc,
    'jsd': jensenshannon
}

def registerMetric(name, fn):
    METRICS_FN[name] = fn

class Evaluator(object):
    def __init__(self, model, nc, device="cpu", metrics=None, post_process=None, negative_class=0, labels=None, soft= True, remove_zero_class=False):
        self.model = model # must implement the 'forward' method
        self.device = device
        self.negative_class = negative_class
        self.soft = soft
        self.remove_zero_class = remove_zero_class

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
                    'auroc': {'average': 'macro', 'multi_class':'ovr'},
                    'auroc_ovo': {'average': 'macro', 'multi_class':'ovo'}
                }
                if self.soft:
                    metrics = {"mean_squared_error": {},
                        "mean_absolute_error": {},
                        "auroc": {'average': 'macro', 'multi_class':'ovr'},
                        "auroc_ovo": {'average': 'macro', 'multi_class':'ovo'},
                        "jsd": {'base': 2, 'axis': 1}
                        }
            self.metrics = metrics
        else:
            if not isinstance(metrics, dict):
                raise ValueError("The argument 'metrics' must be a dictionary of kwargs and metric names or 'none'!")
            self.metrics = metrics
    
    @torch.no_grad()
    def eval(self, dataset, eval_mode=True, batchwise=False, use_masks=True, return_masks=False, return_predicted=False, return_batches=True, xtras=None, split_batches=False, **kwargs):
        """Returns numpy arrays!!!"""        
        
        def _loop(batch, data_items, y_gts, y_prs, outps, masks, batches):
            batch_data = processBatch(self.device, batch, xtras=xtras)
            batch, y, mask = batch_data['batch'], batch_data['y'], batch_data['mask']
            output = self.model(batch)
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
            evald = self.eval(args[0], eval_mode=eval_mode, use_masks=False, return_masks=True, return_batches=True, batchwise=batchwise, split_batches=split_batches, **kwargs)
            if batchwise:
                y_gt = evald['y']
                outs = evald['output']
                batches = evald['batches']
                masks = evald['masks']
            else:
                y_gt = [evald['y']]
                outs = [evald['output']]
                batches = [evald['batches']]
                masks = [evald['masks']]
        else:
            if len(args) == 3:
                y_gt, outs, masks = args
                batches = None
            else:
                y_gt, outs, masks, batches = args
                batches = [batches]
            y_gt = [y_gt]
            outs = [outs]
            masks = [masks]
        
        # Get predicted class labels
        for i in range(len(y_gt)):
            try:
                if(y_gt[i].shape[1] > 1 and not self.soft):
                    y_gt[i] = np.argmax(y_gt[i], axis = 1)
            except:
                pass
            y_pr = self.predictClass(outs[i], y_gt[i], metric_values, threshold=threshold, threshold_metric=threshold_metric, report_threshold=report_threshold)
            #print(y_pr, y_pr.shape, np.unique(y_pr))
            #if batches is not None:
                #y_gt[~masks] = self.negative_class
                #self.getGraphMetrics(batches[i], y_gt[i], y_pr, metric_values)
            if masks is not None:    
                if(self.remove_zero_class):
                    y_gt[i] = y_gt[i][masks[i]][:,1:]
                    outs[i] = softmax(outs[i][masks[i]][:,1:], axis = 1)
                else:
                    #print(y_gt[i].shape,outs[i].shape)
                    y_gt[i] = y_gt[i][masks[i]]
                    outs[i] = softmax(outs[i][masks[i]], axis = 1)

                y_pr = y_pr[masks[i]]
            #print(y_gt[i], outs[i])
                #print(outs[i])
            #print(y_pr, y_pr.shape, np.unique(y_pr))
            #print(y_gt[i], y_gt[i].shape, np.unique(y_gt[i]))
            # Compute metrics
            #print(y_pr,outs[i])
            for metric, kw in self.metrics.items():
                if (not self.soft) and metric in ['auprc', 'auroc', 'auroc_ovo']:
                    # AUC metrics
                    value = METRICS_FN[metric](y_gt[i], outs[i], **kw)
                    if(value is not None):
                        metric_values[metric].append(value)
                elif metric == 'smoothness':
                    # use `getGraphMetrics` for this
                    continue
                elif self.soft:
                    if(metric in ['auroc','auroc_ovo']):
                        #print(np.unique(np.sum(y_gt[i],axis=1)), np.unique(np.argmax(y_gt[i], axis=1)))
                        metric_values[metric].append(METRICS_FN[metric](np.argmax(y_gt[i], axis=1).flatten(), outs[i], **kw))        
                    elif(metric in ['jsd']):
                        metric_values[metric].append(np.mean(METRICS_FN[metric](y_gt[i], outs[i], **kw)))
                    else:
                        try:
                            metric_values[metric].append(METRICS_FN[metric](y_gt[i].reshape(-1,1), outs[i].reshape(-1,1), **kw))
                        except Exception as e:
                            print(e, y_gt[i].reshape(-1,1), outs[i].reshape(-1,1))
                            sys.exit()
                else:
                    metric_values[metric].append(METRICS_FN[metric](y_gt[i], y_pr, **kw))
        for key in metric_values:
            #metric_values[key] = np.mean(metric_values[key])
            metric_values[key] = np.nanmean(metric_values[key])
        
        return metric_values
    
    def getGraphMetrics(self, batches, y_gt, y_pr, metric_values=None):
        if self.metrics is None:
            return {}
        if metric_values is None:
            metric_values = {}
        
        smooth_gt = []
        smooth_pr = []
        ptr = 0
        if not isinstance(batches, list):
            batches = [batches]
        for batch in batches:
            edge_index = batch.edge_index.cpu().numpy()
            slc = slice(ptr, ptr + batch.num_nodes)
            if "smoothness" in self.metrics:
                smooth_gt.append(METRICS_FN["smoothness"](y_gt[slc], edge_index, **self.metrics['smoothness']))
                smooth_pr.append(METRICS_FN["smoothness"](y_pr[slc], edge_index, **self.metrics['smoothness']))
            ptr += batch.num_nodes
        
        if "smoothness" in self.metrics:
            smooth_pr = np.mean(smooth_pr)
            smooth_gt = np.mean(smooth_gt)
            metric_values['smoothness'].append(smooth_pr)
            metric_values['smoothness_relative'].append((smooth_pr - smooth_gt)/smooth_gt)
        
        return metric_values
    
    def predictClass(self, outs, y_gt=None, metrics_dict=None, threshold=None, threshold_metric='balanced_accuracy', report_threshold=False):
        # Decide how to determine `y_pr` from `outs`
        if self.nc == 2:
            if (threshold is None) and (y_gt is not None):
                # sample n_samples threshold values
                threshold, _  = chooseBinaryThreshold(y_gt, outs[:,1], metric_fn=METRICS_FN[threshold_metric], **self.metrics[threshold_metric])
            elif (threshold is None) and (y_gt is None):
                threshold = 0.5
            else:
                threshold = 0.5
            y_pr = (outs[:,1] >= threshold)
            if report_threshold and (metrics_dict is not None):
                metrics_dict['threshold'] = threshold
        elif self.nc > 2:
            y_pr = np.argmax(outs, axis=1).flatten()
        
        return y_pr
