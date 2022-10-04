from typing import Any, Optional, Dict
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only

from src.models.components.outputs import ModelOutputs


class LoggingCallback(Callback):
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
        logging_interval: int = 100,
    ) -> None:
        
        self.logging_interval = logging_interval
        
        
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
        
        pl_module.log('val_loss', outputs['loss'], prog_bar=True, on_step=True, on_epoch=False)
        pl_module.log('val_mrr', outputs['mrr'], prog_bar=True, on_step=True, on_epoch=False)
           
    
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
        
        pl_module.log('val_loss', outputs['loss'], prog_bar=True, on_step=True, on_epoch=False)
        pl_module.log('val_mrr', outputs['mrr'], prog_bar=True, on_step=True, on_epoch=False)