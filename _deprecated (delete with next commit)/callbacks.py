import sys
import math
from typing import Any, Optional, Union, Sequence
import pytorch_lightning as pl
from torch import Tensor, nn, cuda
from pytorch_lightning.callbacks import Callback, ProgressBarBase, BaseFinetuning
import wandb
from utils import count_parameters


class BYOLMAWeightUpdate(Callback):
    """Weight update rule from BYOL.
    Your model should have:
        - ``self.online_network``
        - ``self.target_network``
    Updates the target_network params using an exponential moving average update rule weighted by tau.
    BYOL claims this keeps the online_network from collapsing.
    .. note:: Automatically increases tau from ``initial_tau`` to 1.0 with every training step
    Example::
        # model must have 2 attributes
        model = Model()
        model.online_network = ...
        model.target_network = ...
        trainer = Trainer(callbacks=[BYOLMAWeightUpdate()])
    """

    def __init__(self, initial_tau: float = 0.996):
        """
        Args:
            initial_tau: starting tau. Auto-updates with every training step
        """
        super().__init__()
        self.initial_tau = initial_tau
        self.current_tau = initial_tau

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # get networks
        #TODO adjust this
        online_net = pl_module.online_network
        target_net = pl_module.target_network

        # update weights
        self.update_weights(online_net, target_net)

        # update tau after
        self.current_tau = self.update_tau(pl_module, trainer)

    def update_tau(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> float:
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs
        tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi * pl_module.global_step / max_steps) + 1) / 2
        return tau

    def update_weights(self, online_net: Union[nn.Module, Tensor], target_net: Union[nn.Module, Tensor]) -> None:
        # apply MA weight update
        for (name, online_p), (_, target_p) in zip(
            online_net.named_parameters(),
            target_net.named_parameters(),
        ):
            target_p.data = self.current_tau * target_p.data + (1 - self.current_tau) * online_p.data




class LoggingCallback(Callback):
  
  def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    print(count_parameters(pl_module))
    
  
  def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    """Logs gradients, parameters and model topology."""
    trainer.logger.watch(pl_module)
    trainer.logger.log_hyperparams(pl_module.parameters())
  
  
  def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
    """Logs the loss and accuracy of the training step."""
    trainer.logger.log_metrics(outputs)
  
  
  def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
    """Logs the MRR@val_batch_size of the validation step."""
    #TODO log table of given text, predicted speech and ground truth speech
    trainer.logger.log_metrics(outputs)
    
    
  def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #TODO log similarity matrix to observe how model becomes more confident over time
    pass
    
    
  def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    """Save model checkpoint at the end of each epoch."""
    pass
  
  
  def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    config = pl_module.config
    model_artifact = wandb.Artifact(name=config.model_name, description=config.run_name, type='model', metadata=config)
  
  
  def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
    """Logs the MRR@test_batch_size of the test step."""
    pass
  
  
  def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    trainer.logger.finalize()
    trainer.logger.unwatch(pl_module)
    
    
    
    
class GPUManagement(Callback):  
  
  def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
    # print(f"mem_get_info(): {cuda.mem_get_info()}\n")
    # print(f"memory_allocated(): {cuda.memory_allocated()}\n")
    # print(f"memory_reserved(): {cuda.memory_reserved()}\n")
    speech_input = batch[0]['input_values']
    text_input = batch[1]['input_ids']
    print(f"speech_input shape: {speech_input.shape}")
    print(f"text_input shape: {text_input.shape})")
    cuda.empty_cache()
    
    
    
    print(f"memory_summary(): \n{cuda.memory_summary()}\n")
    # print(f"max_memory_allocated(): {cuda.max_memory_allocated()}\n")
    # print(f"max_memory_reserved(): {cuda.max_memory_reserved()}\n")
    print("End of GPUManagement.on_train_batch_start()\n")
  
  
  def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
    # print(f"mem_get_info(): {cuda.mem_get_info()}\n")
    # print(f"memory_allocated(): {cuda.memory_allocated()}\n")
    # print(f"memory_reserved(): {cuda.memory_reserved()}\n")
    print(f"memory_summary(): \n{cuda.memory_summary()}\n")
    # print(f"max_memory_allocated(): {cuda.max_memory_allocated()}\n")
    # print(f"max_memory_reserved(): {cuda.max_memory_reserved()}\n")
    print("End of GPUManagement.on_train_batch_end()\n")
    
    
  def on_before_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: Tensor) -> None:
    # print(f"mem_get_info(): {cuda.mem_get_info()}\n")
    # print(f"memory_allocated(): {cuda.memory_allocated()}\n")
    # print(f"memory_reserved(): {cuda.memory_reserved()}\n")
    print(f"memory_summary(): \n{cuda.memory_summary()}\n")
    # print(f"max_memory_allocated(): {cuda.max_memory_allocated()}\n")
    # print(f"max_memory_reserved(): {cuda.max_memory_reserved()}\n")
    print("End of GPUManagement.on_before_backward()\n")
    
    
  def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    # print(f"mem_get_info(): {cuda.mem_get_info()}\n")
    # print(f"memory_allocated(): {cuda.memory_allocated()}\n")
    # print(f"memory_reserved(): {cuda.memory_reserved()}\n")
    print(f"memory_summary(): \n{cuda.memory_summary()}\n")
    # print(f"max_memory_allocated(): {cuda.max_memory_allocated()}\n")
    # print(f"max_memory_reserved(): {cuda.max_memory_reserved()}\n")
    print("End of GPUManagement.on_after_backward()\n")
    
    
  def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
    # print(f"mem_get_info(): {cuda.mem_get_info()}\n")
    # print(f"memory_allocated(): {cuda.memory_allocated()}\n")
    # print(f"memory_reserved(): {cuda.memory_reserved()}\n")
    print(f"memory_summary(): \n{cuda.memory_summary()}\n")
    # print(f"max_memory_allocated(): {cuda.max_memory_allocated()}\n")
    # print(f"max_memory_reserved(): {cuda.max_memory_reserved()}\n")
    print("End of GPUManagement.on_validation_batch_end()\n")
    
    
    
    
class ToggleTrainableParamsPSTM(BaseFinetuning):
  def __init__(self, config):
    pass
    
    

class LitProgressBar(ProgressBarBase):

  def __init__(self):
    super().__init__()
    self.enable = True


  def disable(self):
    self.enable = False


  def on_train_batch_end(self, trainer, pl_module, outputs, batch_idx):
    super().on_train_batch_end(trainer, pl_module, outputs, batch_idx)
    percent = (self.train_batch_idx / self.total_train_batches) * 100
    sys.stdout.flush()
    sys.stdout.write(f'{percent:.01f} percent complete \r')
    
    
#TODO add callback BaseFinetunig
