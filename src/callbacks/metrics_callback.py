from enum import Enum
from typing import Any, Optional, Dict
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics import Recall, Accuracy, RetrievalMRR

from src.models.components.outputs import ForwardPassOutput
from src.models.components.knn import k_nearest_neighbor
from src import utils

log = utils.get_logger(__name__)


class Metric(Enum):
    MRR = 'MRR'
    ACCURACY = 'Accuracy@k'
    RECALL = 'Recall@k'


class MetricsCallback(Callback):
    """In PyTorch lighning logging metrics is agnostic to the logger used.
    Simply calling self.log() or pl_module.log() will log the metrics to all
    loggers passed to the trainer.
    
    For image-text retrieval tasks we could use the metric Recall@K (R@K) like here:
    https://arxiv.org/pdf/2103.01913v2.pdf
    
    Other callbacks can be used to log files, data, checkpoints and artifacts
    specific to wandb.

    Args:
        logging_interval (int): how often to log
    """
    def __init__(
        self, 
        logging_interval: int = 10,
    ) -> None:
        
        self.logging_interval = logging_interval
        
        # TODO make this configurable for num_classes, HOW?
        # Get the number of classes from the dataset, init the metrics in on_fit_start (??)
        self.val_acc_at_1 = Accuracy(top_k=1)
        self.val_acc_at_5 = Accuracy(top_k=5)
        self.test_acc_at_1 = Accuracy(top_k=1)
        self.test_acc_at_5 = Accuracy(top_k=5)
        
        self.val_recall_at_1 = Recall(top_k=1)
        self.val_recall_at_5 = Recall(top_k=5)
        self.test_recall_at_1 = Recall(top_k=1)
        self.test_recall_at_5 = Recall(top_k=5)
        
        self.val_mrr = RetrievalMRR()
        self.test_mrr = RetrievalMRR()
        
        self.val_student_preds = []
        self.val_teacher_preds = []
        self.val_labels = []
        
        self.test_student_preds = []
        self.test_teacher_preds = []
        self.test_labels = []
        
        self.metric = None
        self.align_fuse = None
        self.output_modalities = None
    
    
    @rank_zero_only
    def on_validation_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Optional[Dict], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int
        ) -> None: 
        #! for metrics that don't require whole val/test set sync_dist=True is enough: https://github.com/Lightning-AI/lightning/discussions/4702
        #! when using torchmetrics, sync_dist=True is not needed, calling compute() will sync the metrics across all processes: https://github.com/Lightning-AI/lightning/discussions/6501#discussioncomment-553152
        
        fwd_outputs: ForwardPassOutput = outputs['forward_pass_output']
        
        #TODO unclutter the dispatch function by getting output_modalities and align_fuse from the batch
        self.output_modalities = fwd_outputs.output_modalities
        self.align_fuse = fwd_outputs.align_fuse
        self.metric = fwd_outputs.metric
        
        self.val_student_preds.append(fwd_outputs.student_output.pooler_output.detach().cpu())
        self.val_teacher_preds.append(fwd_outputs.teacher_output.pooler_output.detach().cpu())
        if utils.exists(fwd_outputs.labels):
            self.val_labels.append(fwd_outputs.labels.detach().cpu())
        else:
            self.val_labels.append(None)
        
    
    @rank_zero_only
    def on_validation_epoch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
        ) -> None:
        
        #? use self.output_modalities to properly asign these variables
        features = torch.cat(self.val_student_preds, dim=0)
        queries = torch.cat(self.val_teacher_preds, dim=0)
        
        if set(self.val_labels) == {None}:
            # we treat retrieval as classification with a unique class per sample
            labels = torch.tensor(list(range(len(features))))
        else:
            labels = torch.cat(self.val_labels)
        
        if self.metric == Metric.MRR.value:
            pl_module.log('val_mrr', self.val_mrr.compute(self.preds, self.labels), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            #TODO implement MRR
            raise NotImplementedError
        elif self.metric == Metric.ACCURACY.value:
            probabilities, _, labels = k_nearest_neighbor(prediction_features=features, labels=labels)
            pl_module.log(
                'val_accuracy@5', 
                self.val_acc_at_5.compute(probabilities, labels), 
                prog_bar=True, 
                on_step=False, 
                on_epoch=True, 
                sync_dist=True
                )
            pl_module.log(
                'val_accuracy@1', 
                self.val_acc_at_1.compute(probabilities, labels), 
                prog_bar=True, 
                on_step=False, 
                on_epoch=True, 
                sync_dist=True
                )
        elif self.metric == Metric.RECALL.value:
            probabilities, _, labels = k_nearest_neighbor(prediction_features=features, query_features=queries, labels=labels)
            pl_module.log(
                'val_recall@5', 
                self.val_recall_at_5(probabilities, labels), #TODO should this be compute()?
                prog_bar=True, 
                on_step=False, 
                on_epoch=True, 
                sync_dist=True)
            
            pl_module.log(
                'val_recall@1', 
                self.val_recall_at_1(probabilities, labels), #TODO should this be compute()?
                prog_bar=True, 
                on_step=False, 
                on_epoch=True, 
                sync_dist=True
                )
        else:
            raise Exception('No metric specified or metric not supported')
        
        self.val_student_preds = []
        self.val_teacher_preds = []
        self.val_labels = []
            
    
    @rank_zero_only
    def on_test_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Optional[Dict], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int
        ) -> None:
        
        fwd_outputs: ForwardPassOutput = outputs['forward_pass_output']
        
        self.output_modalities = fwd_outputs.output_modalities
        self.align_fuse = fwd_outputs.align_fuse
        self.metric = fwd_outputs.metric
        
        self.test_student_preds.append(fwd_outputs.student_output.pooler_output.detach().cpu())
        self.test_teacher_preds.append(fwd_outputs.teacher_output.pooler_output.detach().cpu())
        if utils.exists(fwd_outputs.labels):
            self.test_labels.append(fwd_outputs.labels.detach().cpu()) 
        else:
            self.test_labels.append(None)
    
    
    @rank_zero_only
    def on_test_epoch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
        ) -> None:
        
        #? use self.output_modalities to properly asign these variables
        features = torch.cat(self.test_student_preds, dim=0)
        queries = torch.cat(self.test_teacher_preds, dim=0)
        
        if set(self.test_labels) == {None}:
            # we treat retrieval as classification with a unique class per sample
            labels = torch.tensor(list(range(len(features))))
        else:
            labels = torch.cat(self.test_labels)
        
        if self.metric == Metric.MRR.value:
            pl_module.log('test_mrr', self.val_mrr.compute(self.preds, self.labels), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            #TODO implement MRR
            raise NotImplementedError
        elif self.metric == Metric.ACCURACY.value:
            probabilities, _, labels = k_nearest_neighbor(prediction_features=features, labels=labels)
            pl_module.log(
                'test_accuracy@5', 
                self.test_acc_at_5.compute(probabilities, labels), 
                prog_bar=True, 
                on_step=False, 
                on_epoch=True, 
                sync_dist=True
                )
            pl_module.log(
                'test_accuracy@1', 
                self.test_acc_at_1.compute(probabilities, labels), 
                prog_bar=True, 
                on_step=False, 
                on_epoch=True, 
                sync_dist=True
                )
        elif self.metric == Metric.RECALL.value:
            probabilities, _, labels = k_nearest_neighbor(prediction_features=features, query_features=queries, labels=labels)
            pl_module.log(
                'test_recall@5', 
                self.test_recall_at_5(probabilities, labels), #TODO should this be compute()?
                prog_bar=True, 
                on_step=False, 
                on_epoch=True, 
                sync_dist=True
                )
            pl_module.log(
                'test_recall@1', 
                self.test_recall_at_1(probabilities, labels), #TODO should this be compute()?
                prog_bar=True, 
                on_step=False, 
                on_epoch=True, 
                sync_dist=True
                )
        else:
            raise Exception('No metric specified or metric not supported')
        
        self.val_student_preds = []
        self.val_teacher_preds = []
        self.val_labels = []