from typing import Any, Union

import torch
import pytorch_lightning as pl

from torchmetrics import MaxMetric
from torchmetrics.retrieval.reciprocal_rank import RetrievalMRR

from src.models.components.ema import EMA
from src.models.components.outputs import TrainingStepOutput
from src.models.components.criterion import LatentPredictionLoss
from src.models.components.metrics import PretrainingMetric
from src.models.components.perceiver import PerceiverModel
from src.models.components.hip import HiPModel
from src.utils import exists


class LatentPredictionPretraining(pl.LightningModule):
    """
    Example of LightningModule for MNIST classification.
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: Union[PerceiverModel, HiPModel],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: LatentPredictionLoss,
        metric: PretrainingMetric,
        ema_decay: float = 0.999,
        ema_end_decay: float = 0.9999,
        ema_anneal_end_step: int = 300000,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        #TODO check if contains a bug
        self.save_hyperparameters(logger=False) 
        
        # student and teacher models is instantiated by Hydra
        self.student = model
        self.teacher = EMA(model, ema_decay)
        
        # set student status for each model in order for masking to be applied only to the student model
        self.student.set_student_status(True)
        self.teacher.model.set_student_status(False)
        
        #TODO when saving the best checkpoint, the teacher model is not saved
        #TODO clarify what val metric could be used during training. or are we only looking at the loss? 
        #TODO Encapsulate metrics as a class so this module is agnostic to val matrics and 
        
        # EMA parameters
        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.ema_anneal_end_step = ema_anneal_end_step
        
        # optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler

        # loss function
        self.criterion = criterion
        
        # metric class that is configured depending on pretraining data
        self.metric = metric
        
        
    def ema_step(self):
        """
        One EMA step for the offline/teacher model until the ending decay value is reached
        """
        if self.ema_decay != self.ema_end_decay:
            if self.teacher.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.teacher.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.teacher.num_updates,
                    self.ema_anneal_end_step,
                )
            self.teacher.decay = decay
        if self.teacher.decay < 1:
            self.teacher.step(self.student)


    def forward(self, batch: Any) -> TrainingStepOutput:
        #TODO adapt this for multimodal alignment
        student_outputs = self.student(batch)
        with torch.no_grad():
            self.teacher.model.eval()
            teacher_outputs = self.teacher.model(batch)
        return TrainingStepOutput(student_outputs, teacher_outputs)


    def step(self, batch: Any):
        # forward pass
        outputs = self.forward(batch)
        
        # compute loss
        loss = self.criterion(outputs.student_output.hidden_states, outputs.teacher_output.hidden_states)
        
        return outputs, loss
    
    
    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if exists(self.teacher):
            self.ema_step(self.student)


    def training_step(self, batch: Any, batch_idx: int):
        outputs, loss = self.step(batch)
        
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}
    

    def validation_step(self, batch: Any, batch_idx: int):
        outputs, loss = self.step(batch)
        
        self.log("val/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}
        

    def test_step(self, batch: Any, batch_idx: int):
        outputs, loss = self.step(batch)
        
        self.log("test/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.optimizer(params=self.parameters())
        scheduler = self.scheduler(optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
        
        
        
class LatentPredictionFinetuning(pl.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        #TODO use pl.callbacks.BaseFineTuningCallback when finetuning on a smaller dataset