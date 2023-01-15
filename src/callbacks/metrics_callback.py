from enum import Enum
from typing import Any, Optional, Dict
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningModule
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
        top_k: list = [1, 10, 100],
        logging_interval: int = 10,
        on_step: bool = False,
        on_epoch: bool = True,
    ) -> None:
        
        self.top_k = top_k
        self.logging_interval = logging_interval
        
        self.on_step = on_step
        self.on_epoch = on_epoch
                
        self.val_accuracy = None
        self.test_accuracy = None
        
        self.val_recall = {k: Recall(top_k=k) for k in top_k} #TODO in case on_step: k might be bigger than the batch size, find fix. -> k=batch_size, chunking=False
        self.test_recall = {k: Recall(top_k=k) for k in top_k}
        
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
        
        # for metrics that don't require whole val/test set sync_dist=True 
        # is enough: https://github.com/Lightning-AI/lightning/discussions/4702
        
        # when using torchmetrics, sync_dist=True is not needed, calling compute() will sync 
        # the metrics across all processes: https://github.com/Lightning-AI/lightning/discussions/6501#discussioncomment-553152
        
        fwd_outputs: ForwardPassOutput = outputs['forward_pass_output']
        
        self.output_modalities = fwd_outputs.output_modalities
        self.align_fuse = fwd_outputs.align_fuse
        self.metric = fwd_outputs.metric
        
        if self.val_accuracy is None and fwd_outputs.num_classes is not None and Metric.ACCURACY.value in self.metric:
            self.val_accuracy = {k: Accuracy(top_k=k, num_classes=fwd_outputs.num_classes) for k in self.top_k}
        
        features = fwd_outputs.student_output.pooler_output.detach().cpu()
        queries = fwd_outputs.teacher_output.pooler_output.detach().cpu()
        labels = fwd_outputs.labels.detach().cpu() if utils.exists(fwd_outputs.labels) else None
        
        self.val_student_preds.append(features)
        self.val_teacher_preds.append(queries)
        self.val_labels.append(labels)
            
        if self.on_step:
            if labels is None:
                # we treat retrieval as classification with a unique class per sample
                labels = torch.tensor(list(range(len(features))))    
            self.compute_metrics(
            predictions=features, 
            queries=queries, 
            labels=labels, 
            mrr_metric=self.val_mrr, 
            accuracy_metric=self.val_accuracy, 
            recall_metric=self.val_recall, 
            pl_module=pl_module, 
            on_step=True, 
            on_epoch=False,
            val_or_test='val'
            )
            
        if self.on_epoch:
            self.val_student_preds.append(features)
            self.val_teacher_preds.append(queries)
            self.val_labels.append(labels)
    
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
            
        if self.on_epoch:
            self.compute_metrics(
                predictions=features, 
                queries=queries, 
                labels=labels, 
                mrr_metric=self.val_mrr, 
                accuracy_metric=self.val_accuracy, 
                recall_metric=self.val_recall, 
                pl_module=pl_module, 
                on_step=False, 
                on_epoch=True,
                val_or_test='val'
                )
        
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
        
        if self.test_accuracy is None and fwd_outputs.num_classes is not None and Metric.ACCURACY.value in self.metric:
            self.test_accuracy = {k: Accuracy(top_k=k, num_classes=fwd_outputs.num_classes) for k in self.top_k}
            
        features = fwd_outputs.student_output.pooler_output.detach().cpu()
        queries = fwd_outputs.teacher_output.pooler_output.detach().cpu()
        labels = fwd_outputs.labels.detach().cpu() if utils.exists(fwd_outputs.labels) else None
            
        self.test_student_preds.append(features)
        self.test_teacher_preds.append(queries)
        self.test_labels.append(labels) 
            
        if self.on_step:
            if labels is None:
                # we treat retrieval as classification with a unique class per sample
                labels = torch.tensor(list(range(len(features))))  
            self.compute_metrics(
                predictions=features, 
                queries=queries, 
                labels=labels, 
                mrr_metric=self.test_mrr, 
                accuracy_metric=self.test_accuracy, 
                recall_metric=self.test_recall, 
                pl_module=pl_module, 
                on_step=True, 
                on_epoch=False,
                val_or_test='test'
                )
        elif self.on_epoch:
            self.test_student_preds.append(features)
            self.test_teacher_preds.append(queries)
            self.test_labels.append(labels)
    
    
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
        
        if self.on_epoch:
            self.compute_metrics(
                predictions=features, 
                queries=queries, 
                labels=labels, 
                mrr_metric=self.test_mrr, 
                accuracy_metric=self.test_accuracy, 
                recall_metric=self.test_recall, 
                pl_module=pl_module, 
                on_step=False, 
                on_epoch=True,
                val_or_test='test'
                )
        
        self.val_student_preds = []
        self.val_teacher_preds = []
        self.val_labels = []
        
        
    def compute_metrics(
        self, 
        predictions, 
        queries, 
        labels, 
        mrr_metric: RetrievalMRR, 
        accuracy_metric: dict, 
        recall_metric: dict, 
        pl_module: LightningModule, 
        on_step: bool = False, 
        on_epoch: bool = True,
        val_or_test: str = 'val'
        ) -> None:
        
        when_to_log = '_on_step' if on_step else '_on_epoch'
        k = max(self.top_k)
        
        if Metric.MRR.value in self.metric:
            probabilities, _, labels = k_nearest_neighbor(prediction_features=predictions, query_features=queries, labels=labels, k=k)
            preds = torch.flatten(probabilities)
            target = torch.eye(len(labels)).flatten()
            indexes = torch.tensor([[n]*len(labels) for n in range(len(labels))], dtype=torch.long).flatten()
            pl_module.log(
                f'{val_or_test}_mrr@{len(labels)}{when_to_log}', 
                mrr_metric.compute(preds, target, indexes=indexes), 
                prog_bar=True, 
                on_step=on_step, 
                on_epoch=on_epoch, 
                sync_dist=True
                )
        if Metric.ACCURACY.value in self.metric:
            probabilities, _, labels = k_nearest_neighbor(prediction_features=predictions, labels=labels, k=k)
            for key, value in accuracy_metric.items():
                if key < len(labels):
                    pl_module.log(
                        f'{val_or_test}_accuracy@{key}{when_to_log}',
                        value(probabilities, labels),
                        prog_bar=True, 
                        on_step=on_step, 
                        on_epoch=on_epoch, 
                        sync_dist=True
                        )
        if Metric.RECALL.value in self.metric:
            probabilities, _, labels = k_nearest_neighbor(prediction_features=predictions, query_features=queries, labels=labels, k=k)
            for key, value in recall_metric.items():
                if key < len(labels):
                    pl_module.log(
                        f'{val_or_test}_recall@{key}{when_to_log}',
                        value(probabilities, labels),
                        prog_bar=True, 
                        on_step=on_step, 
                        on_epoch=on_epoch, 
                        sync_dist=True
                        )
        else:
            raise Exception('No metric specified or metric not supported')
        