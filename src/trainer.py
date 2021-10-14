# builtin modules
import logging
import json
from os.path import join as ospj
from collections import OrderedDict
import copy
import sys

# third party modules
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, ExponentialLR
from torch_geometric.nn import DataParallel
from tqdm import tqdm
# geobind modules
#from utils.class_weights import classWeights
from process_batch import processBatch
from report_metrics import reportMetrics

class Scheduler(object):
    def __init__(self, scheduler):
        self.epoch = 0
        self.scheduler = scheduler
        self.history = {
            "loss": 0,
            "batch_count": 0
        }
    
    def step(self, epoch, loss, **kwargs):
        # per-epoch schedulers
        new_epoch = (epoch > self.epoch)
        if new_epoch:
            # we are in a new epoch, update per-epoch schedulers
            self.epoch = epoch
            if isinstance(self.scheduler, ReduceLROnPlateau):
                mean_loss = self.history['loss']/self.history["batch_count"]
                self.scheduler.step(mean_loss)
                self.history["loss"] = 0
                self.history["batch_count"] = 0
            elif isinstance(self.scheduler, ExponentialLR):
                self.scheduler.step()
        
        # per-batch schedulers
        if isinstance(self.scheduler, OneCycleLR):
            self.scheduler.step()
        elif isinstance(self.scheduler, ReduceLROnPlateau):
            self.history["loss"] += loss
            self.history["batch_count"] += 1

class Trainer(object):
    def __init__(self, model, nc, optimizer, criterion, 
            device='cpu', scheduler=None, evaluator=None, 
            writer=None, checkpoint_path='.', quiet=True
            ):
        # parameters
        self.model = model
        self.nc = nc
        self.optimizer = optimizer
        self.criterion = criterion
        self.evaluator = evaluator
        self.writer = writer
        self.device = device
        self.quiet = quiet
        self.checkpoint_path = checkpoint_path
        self.loss_log = open("loss.txt","w")
        # variables to track training progress
        self.best_state = None
        self.best_state_metric = None
        self.best_epoch = None
        # set up scheduler
        if scheduler is not None:
            scheduler = Scheduler(scheduler)
        self.scheduler = scheduler
        
        # get model name
        if isinstance(self.model, DataParallel):
            self.model_name = self.model.module.name
        else:
            self.model_name = self.model.name
        
        # history
        self.metrics_history = {'epochs': []}
    
   
    def train(self, nepochs, dataset,
        validation_dataset=None, batch_loss_every=1, eval_every=2, debug=False,
        checkpoint_every=None, optimizer_kwargs={}, scheduler_kwargs={},
        best_state_metric=None, best_state_metric_threshold=None, 
        best_state_metric_dataset='validation', best_state_metric_goal='max', params_to_write=None
    ):
        # begin training
        if not self.quiet:
            logging.info("Beginning Training ({} epochs)".format(nepochs))
        
        if debug:
            mem_stats = {
                "current": [],
                "peak": [],
                "epoch_start": []
            }
        
        if best_state_metric_goal == 'max':
            self.best_state_metric = -999999
        else:
            self.best_state_metric = 999999
        
        batch_count = 0
        first_epoch = True
        for epoch in range(nepochs):
            # set model to training mode
            self.model.train()
            
            # forward + backward + update
            epoch_loss = 0
            n = 0
            for batch in dataset:
                oom = False
                # update the model weights
                batch_data = processBatch(self.device, batch)
                batch, y = batch_data['batch'], batch_data['y']
                #batch, y = batch_data['batch'], batch_data['y']
                
                # check for OOM errors
                try:
                    loss = self.optimizer_step(batch, y, **optimizer_kwargs)
                    #loss = self.optimizer_step(batch, y, **optimizer_kwargs)
                except RuntimeError as e: # out of memory
                    logging.info("Runtime error -- skipping batch.")
                    logging.debug("Error at loss computation.", exc_info=e)
                    oom = True
                    print(e)
                if oom:
                    continue
                    
                # update scheduler
                if self.scheduler is not None:
                    self.scheduler.step(epoch, loss, **scheduler_kwargs)
                
                # write batch-level stats
                if batch_count % batch_loss_every  == 0:
                    if self.writer:
                        self.writer.add_scalar("train/batch_loss", loss, batch_count)
                
                ######### adding parameters of model to tensorboard ######
                if (params_to_write is not None) and self.writer:
                    for name, param in self.model.named_parameters():
                        if (name in params_to_write) and param.requires_grad:
                            if(param.data.cpu().numpy().flatten().shape[0] == 1):
                                self.writer.add_scalar(name, param.data.cpu().numpy()[0], batch_count)
                                self.writer.add_scalar(name + "_grad", param.grad.cpu().numpy()[0], batch_count)
                            elif(param.data.cpu().numpy().flatten().shape[0] <= 4):
                                for i in range(1,param.data.cpu().numpy().flatten().shape[0] + 1):
                                    self.writer.add_scalar(name + "_" + str(i), param.data.cpu().numpy().flatten()[i-1], batch_count)
                                    self.writer.add_scalar(name + "_grad_" + str(i), param.grad.cpu().numpy().flatten()[i-1], batch_count)
                            else:
                                self.writer.add_histogram(name, param.data.cpu().numpy().flatten(), batch_count)
                                self.writer.add_histogram(name + "grad", param.grad.cpu().numpy().flatten(), batch_count)
                                self.writer.flush()  ## remove this line, if results in slowdown, flush only at end

                # update batch count
                batch_count += 1
                epoch_loss += loss
                n += 1
            self.loss_log.write("epoch loss_" + str(epoch) + " : " + str(epoch_loss) +"\n")
            
            epoch = epoch+1
            # compute metrics
            if (epoch % eval_every == 0) and (self.evaluator is not None):
                metrics = {}
                metrics['train'] = self.evaluator.getMetrics(dataset, eval_mode=True, report_threshold=True, threshold=0.5, metrics_calculation="total")
                metrics['train']['loss'] = epoch_loss/(n + 1e-5)
                
                if validation_dataset is not None:
                    metrics['validation'] = self.evaluator.getMetrics(validation_dataset, eval_mode=True, threshold=0.5, metrics_calculation="average_batches")
                    #print(metrics.keys(), metrics)
                # report performance
                if not self.quiet:
                    reportMetrics(metrics,
                        label=('Epoch', epoch),
                        header=first_epoch
                    )
                self.updateHistory(metrics, epoch)
                first_epoch = False
                
                if best_state_metric:
                    state_metric = metrics[best_state_metric_dataset][best_state_metric]
                    if best_state_metric_goal == 'max' and state_metric > best_state_metric_threshold:
                        if state_metric > self.best_state_metric:
                            self.best_state_metric = state_metric
                            self.best_state = copy.deepcopy(self.model.state_dict())
                            self.best_epoch = epoch
                            self.metrics_history['best_epoch'] = epoch
                    elif best_state_metric_goal == 'min' and state_metric < best_state_metric_threshold:
                        if state_metric < self.best_state_metric:
                            self.best_state_metric = state_metric
                            self.best_state = copy.deepcopy(self.model.state_dict())
                            self.best_epoch = epoch
                            self.metrics_history['best_epoch'] = epoch
                
            # checkpoint
            if checkpoint_every and (epoch % checkpoint_every == 0):
                fname = self.saveState(epoch, "{}.{}.tar".format(self.model_name, epoch))
                logging.info("Writing checkpoint to file {} at epoch {}".format(fname, epoch))
        self.endTraining()
        
    def optimizer_step(self, batch, y,  use_weight=False, weight=None):
        self.optimizer.zero_grad()
        output = self.model(batch)
        
        loss = torch.square(output - y)
        # compute gradients
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    

    def updateHistory(self, metrics, epoch):
        # update epoch
        self.metrics_history['epochs'].append(epoch)
        
        # update tags/metrics
        for tag in metrics:
            if tag not in self.metrics_history:
                self.metrics_history[tag] = {}
            
            for metric in metrics[tag]:
                # add metric to Tensorboard writer
                if(self.writer):
                    self.writer.add_scalar("{}/{}".format(tag, metric), metrics[tag][metric], epoch)
                
                # add metric to history
                if metric not in self.metrics_history[tag]:
                    self.metrics_history[tag][metric] = []
                self.metrics_history[tag][metric].append(metrics[tag][metric])
    
    def getHistory(self, tag, metric, epoch):
        if epoch == -1:
            ind = -1
        else:
            ind = self.metrics_history['epochs'].index(epoch)
        
        return self.metrics_history[tag][metric][ind]
    
    def saveState(self, epoch, suffix, metrics=True, optimizer=True, state=None):
        fname = ospj(self.checkpoint_path, suffix)
        
        # remove 'module' prefix from state_dict entries
        new_state_dict = OrderedDict()
        if state is None:
            state = self.model.state_dict()
        
        for k, v in state.items():
            name = k.replace("module.", "") # remove 'module.' prefix
            new_state_dict[name] = v
        
        data = {
            'model_state_dict': new_state_dict,
            'epoch': epoch
        }
        
        if metrics:
            data['history'] = self.metrics_history
            
        if optimizer:
            data['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(data, fname)
        
        return fname
    
    def endTraining(self, message="Training Successfully Ended."):
        """Stuff we want to do at the end of training"""
        logging.info(message)
        
        # Save best state to file if we kept it
        if self.best_state is not None:
            fname = self.saveState(
                self.best_epoch, "{}.{}.tar".format(self.model_name, "best"),
                metrics=False,
                optimizer=False,
                state=self.best_state
            )
            logging.info("Writing best state to file {} (epoch: {})".format(fname, self.best_epoch))
            logging.info("Best tracked metric achieved: {:.3f}".format(self.best_state_metric))
        
        # Save metrics history to file
        logging.info("Saving metrics history to file.")
        MH = open(ospj(self.checkpoint_path, "{}_metrics.json".format(self.model_name)), "w")
        MH.write(json.dumps(self.metrics_history, indent=2))
        MH.close()
