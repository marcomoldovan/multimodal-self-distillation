import wandb

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from models import SpeechAndTextBiEncoder
from datamodules import LibriSpeechDataModule, SpotifyPredictionDataModule
from callbacks import BYOLMAWeightUpdate, GPUManagement, LoggingCallback, LitProgressBar
from setup import construct_arguments_parser, build_config_from_args, build_run_name_from_config, rebuild_config_object_from_wandb



#TODO add distributed training plugin from Ray: https://docs.ray.io/en/latest/ray-more-libs/ray-lightning.html
def main():
  """
    The main training function.
    To restore either a model from checkpoint or continue training from a previous run:
    --> https://pytorch-lightning.readthedocs.io/en/latest/common/checkpointing.html#checkpoint-loading
  """
  
  # ---------------------
  # args
  # ---------------------
  
  parser = construct_arguments_parser()
  config = build_config_from_args(parser)
  run_name = build_run_name_from_config(config)
  
  with wandb.init(project=config.project_name, entity=config.project_entity, job_type="train", config=config) as run:
    
    run_config = run.config
    run_config_as_dict = run_config.as_dict()
    run_config_readable = rebuild_config_object_from_wandb(run_config_as_dict)
    run_config_readable.run_name = run_name

    wandb_logger = WandbLogger(experiment=run, name=run_name, log_model='all')

    # ---------------------
    # callbacks
    # ---------------------
    
    libri_logging_callback = LoggingCallback()
    progress_bar = LitProgressBar()
    lr_monitor = LearningRateMonitor(logging_interval=None, log_momentum=True)
    early_stopping = EarlyStopping(monitor='mrr_score', min_delta=0.1, patience=run_config_readable.early_stopping_patience, verbose=True)
    checkpoint_callback = ModelCheckpoint(dirpath=run_config_readable.checkpoint_save_path, filename=run_config_readable.run_name, save_top_k=5, verbose=True, monitor='mrr_score', mode='max', save_last=True, every_n_epochs=1)
    byol_weight_update = BYOLMAWeightUpdate()
    gpu_management = GPUManagement()
    if run_config_readable.pretraining_contrastive_loss_fn == 'BYOL':
      callbacks_list = [libri_logging_callback, progress_bar, lr_monitor, early_stopping, checkpoint_callback, byol_weight_update, gpu_management]
    else:
      callbacks_list = [libri_logging_callback, progress_bar, lr_monitor, early_stopping, checkpoint_callback, gpu_management]
    
    # ---------------------
    # data
    # ---------------------

    libri_data_module = LibriSpeechDataModule(run_config_readable)
    # spotify_predict_data_module = SpotifyPredictionDataModule(run_config_readable)
    
    # ---------------------
    # model
    # ---------------------
    
    model = SpeechAndTextBiEncoder(run_config_readable)
    
    # ---------------------
    # trainer
    # ---------------------

    trainer = Trainer(logger=wandb_logger, 
                      callbacks=callbacks_list, 
                      accelerator=run_config_readable.accelerator, 
                      precision=run_config_readable.precision,
                      gpus=run_config_readable.num_gpus, 
                      strategy=run_config_readable.strategy, 
                      accumulate_grad_batches=run_config_readable.accumulate_grad_batches,
                      log_every_n_steps=run_config_readable.log_every_n_steps)
    
    # ---------------------
    # training
    # ---------------------

    trainer.fit(model=model, datamodule=libri_data_module)
    
    # ---------------------
    # testing
    # ---------------------

    trainer.test(model=model, datamodule=libri_data_module)
    
    # ---------------------
    # predict
    # ---------------------
    
    # trainer.predict(model=model, datamodule=spotify_predict_data_module)
    
    # ---------------------
    # save artifact
    # ---------------------

    #TODO save artifact, try to do it in callbacks, example here: https://github.com/wandb/artifacts-examples/blob/master/detectron2/wandb_detectron.py

if __name__ == "__main__":
    main()