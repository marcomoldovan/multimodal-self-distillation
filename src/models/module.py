from typing import Any

import torch
import pytorch_lightning as pl

from torchmetrics import MaxMetric
from torchmetrics.retrieval.reciprocal_rank import RetrievalMRR

from src.models.components.ema import EMA
from src.models.components.outputs import TrainingStepOutput
from src.models.components.criterion import LatentPredictionLoss
from src.utils import exists


class FlatPerceiverData2VecPreTraining(pl.LightningModule):
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
        model: torch.nn.Module,
        ema: EMA,
        criterion: LatentPredictionLoss,
        ema_decay: float = 0.999,
        ema_end_decay: float = 0.9999,
        ema_anneal_end_step: int = 300000,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False) #! appears to contain a bug
        
        # student and teacher models is instantiated by Hydra
        self.student = model
        self.teacher = ema
        #TODO when saving the best checkpoint, the teacher model is not saved
        #TODO clarify what val metric could be used during training. or are we only looking at the loss? 
        #! --> Encapsulate metrics as a class so this module is agnostic to val matrics and 
        #! it's specified according to the type of data we're training on
        
        # EMA parameters
        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.ema_anneal_end_step = ema_anneal_end_step
        
        # optimizer parameters
        self.lr = lr
        self.weight_decay = weight_decay

        # loss function
        self.criterion = criterion
        
        
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


    def forward(self, masked_inputs: torch.Tensor, original_inputs: torch.Tensor) -> TrainingStepOutput:
        student_outputs = self.student(masked_inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher(original_inputs)
        return TrainingStepOutput(student_outputs, teacher_outputs)


    def step(self, batch: Any):
        masked_inputs, original_inputs = batch
        
        # forward pass
        outputs = self.forward(masked_inputs, original_inputs)
        
        # compute loss
        loss = self.criterion(outputs.student_outputs.hidden_states, outputs.teacher_outputs.hidden_states)
        
        return outputs, loss


    def training_step(self, batch: Any, batch_idx: int):
        outputs, loss = self.step(batch)
        
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}
    
    
    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if exists(self.teacher):
            self.ema_step(self.student)


    def validation_step(self, batch: Any, batch_idx: int):
        outputs, loss, indexes, targets, preds = self.step(batch)
        
        self.log("val/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}
        

    def test_step(self, batch: Any, batch_idx: int):
        outputs, loss, indexes, targets, preds = self.step(batch)
        
        self.log("test/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        
        
        
class FlatPerceiverData2VecFineTuning(pl.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        #TODO use pl.callbacks.BaseFineTuningCallback when finetuning on a smaller dataset