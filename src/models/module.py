from typing import Any, Union, Tuple, Dict

import torch
import pytorch_lightning as pl

from src.models.components.ema import EMA
from src.models.components.dispatcher import dispatch_inputs
from src.models.components.outputs import DispatcherOutput, ModelOutput, ForwardPassOutput
from src.models.components.criterion import LatentPredictionLoss
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
        ema_decay: float = 0.999,
        ema_end_decay: float = 0.9999,
        ema_anneal_end_step: int = 300000,
        switch_student_teacher_per_epoch: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True) #, ignore=['criterion']) 
        
        # student and teacher models is instantiated by Hydra
        self.student = model
        self.teacher : EMA = EMA(model, ema_decay)
        
        # set student status for each model in order for masking to be applied only to the student model
        self.student.set_student_status(True)
        self.teacher.model.set_student_status(False)
                
        # EMA parameters
        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.ema_anneal_end_step = ema_anneal_end_step
        
        # optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler

        # loss function
        self.criterion = criterion
        
        # whether to switch student and teacher model every epoch in multimodal training
        self.switch_student_teacher_per_epoch = switch_student_teacher_per_epoch
        
        
    def on_fit_start(self) -> None:
        student_device = next(self.student.parameters()).device
        self.teacher.model.to(student_device) 
        self.teacher.model.eval()
                
        
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


    def forward(
        self, 
        batch: Any, 
    ) -> Tuple[ForwardPassOutput, DispatcherOutput]:
        
        dispatched_inputs = dispatch_inputs(
            batch,
            self.current_epoch,
            self.switch_student_teacher_per_epoch,
        )  
              
        student_outputs: ModelOutput = self.student(
            dispatched_inputs.student_input, 
            apply_mask=dispatched_inputs.apply_mask
        )
        
        outputs = ForwardPassOutput(
            student_output=student_outputs, 
            align_fuse=dispatched_inputs.align_fuse,
            labels=dispatched_inputs.labels, 
            output_modalities=dispatched_inputs.output_modalities,
            metric=dispatched_inputs.metric
        )
        
        return outputs, dispatched_inputs


    def step(
        self, 
        batch: Any, 
    ) -> ForwardPassOutput:
        
        # forward pass student
        outputs, dispatched_inputs = self.forward(batch)
        
        # forward pass teacher
        with torch.no_grad():  
            self.teacher.model.eval()          
            teacher_outputs: ModelOutput = self.teacher.model(
                dispatched_inputs.teacher_inputs, 
                apply_mask=dispatched_inputs.apply_mask
            )
            outputs.set_attributes(**{"teacher_output": teacher_outputs})
        
        # compute loss
        criterion_output = self.criterion(outputs)
        
        outputs.set_attributes(**{"criterion_output": criterion_output})
        
        return outputs
    
    
    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if exists(self.teacher):
            self.ema_step()


    # in training/validation/test_step we can return dict with any tensors
    # and then read it in some callback or in `training/validation/test_epoch_end()`` below
    # remember to always return loss from `training_step()` or else backpropagation will fail!
    
    def training_step(self, batch: Any, batch_idx: int):
        outputs : ForwardPassOutput = self.step(batch)
        
        self.log("train/total_loss", outputs.criterion_output.total_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train/latent_loss", outputs.criterion_output.latent_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train/align_loss", outputs.criterion_output.align_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        return {"loss": outputs.criterion_output.total_loss, "forward_pass_output": outputs}
    

    def validation_step(self, batch: Any, batch_idx: int):
        outputs : ForwardPassOutput = self.step(batch)
        
        self.log("val/total_loss", outputs.criterion_output.total_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("val/latent_loss", outputs.criterion_output.latent_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("val/align_loss", outputs.criterion_output.align_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        return {"loss": outputs.criterion_output.total_loss, "forward_pass_output": outputs}
        

    def test_step(self, batch: Any, batch_idx: int):
        outputs : ForwardPassOutput = self.step(batch)
        
        self.log("test/total_loss", outputs.criterion_output.total_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("test/latent_loss", outputs.criterion_output.latent_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("test/align_loss", outputs.criterion_output.align_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        return {"loss": outputs.criterion_output.total_loss, "forward_pass_output": outputs}


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
                "monitor": "train/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
          
        
        
class LatentPredictionFinetuning(pl.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        #TODO use pl.callbacks.BaseFineTuningCallback when finetuning on a smaller dataset