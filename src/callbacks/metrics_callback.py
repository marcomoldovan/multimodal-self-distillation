from enum import Enum
from typing import Any, Optional, Dict
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics import Recall, Accuracy, RetrievalMRR

from src.models.components.outputs import ForwardPassOutput


class Metric(Enum):
    MRR = 'MRR'
    ACCURACY = 'Accuracy@k'
    RECALL = 'Recall@k'

class MetricsCallback(Callback):
    """In PyTorch lighning logging metrics is agnostic to the logger used.
    Simply calling self.log() or pl_module.log() will log the metrics to all
    loggers passed to the trainer.
    
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
        
        self.val_acc = Accuracy(top_k=1)
        self.test_acc = Accuracy(top_k=1)
        
        self.val_recall = Recall(top_k=1)
        self.test_recall = Recall(top_k=1)
        
        self.val_mrr = RetrievalMRR()
        self.test_mrr = RetrievalMRR()
        
        self.preds = []
        self.labels = []
        
    
    @rank_zero_only
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.metric = pl_module.datamodule.metric
        self.align_fuse = pl_module.datamodule.align_fuse
        

    @rank_zero_only
    def on_train_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Dict, 
        batch: Any, 
        batch_idx: int
        ) -> None:
        
        pl_module.log('train_loss', outputs['loss'], prog_bar=True, on_step=True, on_epoch=False)
        pl_module.log('train_mrr', outputs['mrr'], prog_bar=True, on_step=True, on_epoch=False)
    
    
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
        
        if self.metric == Metric.MRR:
            self.preds.append(fwd_outputs.student_output.last_hidden_state.detach().cpu())
            self.labels.append(fwd_outputs.labels.detach().cpu()) #TODO implement sending labels from datamodule to to pl_module to output
        elif self.metric == Metric.ACCURACY:
            self.preds.append(fwd_outputs.student_output.last_hidden_state.detach().cpu())
            self.labels.append(fwd_outputs.labels.detach().cpu())
        elif self.metric == Metric.RECALL:
            self.preds.append(fwd_outputs.student_output.last_hidden_state.detach().cpu())
            self.labels.append(fwd_outputs.labels.detach().cpu())
        
        #TODO for imagenet zero-shot we need to get the embeddings of the class names (do it once at the end of the epoch??)
        
        pl_module.log('val_loss', outputs['loss'], prog_bar=True, on_step=True, on_epoch=False)
        pl_module.log('val_mrr', outputs['mrr'], prog_bar=True, on_step=True, on_epoch=False)
        
    
    @rank_zero_only
    def on_validation_epoch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
        ) -> None:
        
        if self.metric == Metric.MRR:
            pl_module.log('val_mrr', self.val_mrr.compute(self.preds, self.labels), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            #TODO implement MRR
            raise NotImplementedError
        elif self.metric == Metric.ACCURACY:
            pl_module.log('val_accuracy', self.val_acc.compute(self.preds, self.labels), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            #TODO implement accuracy
            raise NotImplementedError
        elif self.metric == Metric.RECALL:
            pl_module.log('val_recall', self.val_recall.compute(self.preds, self.labels), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            #TODO implement recall
            raise NotImplementedError
           
    
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
        
        raise NotImplementedError #TODO this should be the same as validation batch end
    
    
    @rank_zero_only
    def on_test_epoch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
        ) -> None:
        raise NotImplementedError #TODO this should be the same as validation batch end