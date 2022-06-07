"""
  Serves as a base configuration file for the model.
  Individual arguments can be overwritten by the user during training start.
  Run names are built from the arguments.
"""

import torch
from transformers import PretrainedConfig


class SpeechAndTextBiEncoderConfig(PretrainedConfig):
  """
  This is the configuration class for pre-training the SpeechAndTextBiEncoder. It holds
  all necessary arguments for:
    - the model hyperparameters
    - the dataset to train on and its parameters
    - training parameters for the pl.Trainer
    - strategy related parameters
    - logging and checkpointing related parameters
  Individual parameters can be overwritten with argparse by the user during training start.
  The config file is passed to the wandb.init() context manager to allow for the sweep to
  overwrite default values during hyperparameter search.
  It holds the 'training_mode' attribute as a fixed string that can't be overwritten as a
  helper to build the correct model in setup.py

  Args:
    model_name (str): name of the model to train
    hidden_size (int): hidden size of the model
    bert_pretrained_name (str): name of the pretrained BERT model to use
    hubert_pretrained_name (str): name of the pretrained hubert model to use
    speech_output_pooling_strategy (str): pooling strategy for the speech output
    pretraining_contrastive_loss_fn (str): name of the contrastive loss function to use
    train_last_n_layers (int): number of layers to train
    num_epochs (int): number of epochs to train
    early_stopping_patience (int): patience for early stopping
    accumulate_grad_batches (int): number of batches to accumulate gradients for
    dataset_name (str): name of the dataset to train on
    train_batch_size (int): batch size for training
    val_batch_size (int): batch size for validation
    test_batch_size (int): batch size for testing
    project_name (str): name of the project
    checkpoint_save_path (str): path to save checkpoints to
    project_entity (str): name of the project entity
    run_name (str): name of the run
  """
  def __init__(self, 
               model_name='PSTM',
               hidden_size=768,
               bert_pretrained_name='google/bert_uncased_L-2_H-768_A-12', # 'bert-base-uncased'
               hubert_pretrained_name='ntu-spml/distilhubert', 
               speech_output_pooling_strategy='mean',
               training_mode = 'pretrain',
               pretraining_contrastive_loss_fn='TripletMarginLoss',
               train_last_n_speech_model_layers=1,
               train_last_n_text_model_layers=0,
               num_epochs=10,
               early_stopping_patience=5,
               accumulate_grad_batches=1,
               precision=16,
               dataset_name='librispeech',
               include_conv_module_in_preprocessing=True,
               use_gpu_for_preprocessing=True,
               train_batch_size=64,
               val_batch_size=100,
               test_batch_size=100,
               log_every_n_steps=1,
               project_name='cross-modal-speech-segment-retrieval',
               checkpoint_save_path=None,
               project_entity=None,
               run_name=None):
    
    # model related
    self.model_name = model_name
    self.hidden_size = hidden_size
    self.bert_pretrained_name = bert_pretrained_name
    self.hubert_pretrained_name = hubert_pretrained_name
    self.speech_output_pooling_strategy = speech_output_pooling_strategy
    
    # training related
    self.training_mode = training_mode
    self.pretraining_contrastive_loss_fn = pretraining_contrastive_loss_fn
    self.train_last_n_speech_model_layers = train_last_n_speech_model_layers
    self.train_last_n_text_model_layers = train_last_n_text_model_layers
    self.num_epochs = num_epochs
    self.early_stopping_patience = early_stopping_patience
    self.accumulate_grad_batches = accumulate_grad_batches # can be a dict with an accumulation strategy for each epoch: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.GradientAccumulationScheduler.html#pytorch_lightning.callbacks.GradientAccumulationScheduler
    self.precision = precision
    
    # data related
    self.dataset_name = dataset_name
    self.include_conv_module_in_preprocessing = include_conv_module_in_preprocessing
    self.use_gpu_for_preprocessing = use_gpu_for_preprocessing
    self.train_batch_size = train_batch_size
    self.val_batch_size = val_batch_size
    self.test_batch_size = test_batch_size
    
    # strategy related
    self.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    self.num_gpus = torch.cuda.device_count()
    self.strategy = "ddp" if self.num_gpus > 1 else None
    
    # logging & checkpointing related
    self.log_every_n_steps = log_every_n_steps
    self.project_name = project_name
    self.project_entity = project_entity
    self.run_name = run_name
    self.checkpoint_save_path = checkpoint_save_path if checkpoint_save_path is not None else f'{self.project_name}/{self.run_name}/checkpoints/'
    
    #TODO throw exception if some arguments are incompatible, e.g. training_mode='finetune' and dataset='librispeech'

    
    

class CrossModalLanguageModelConfig(PretrainedConfig):
  def __init__(self,
               model_name='CMLM'):
    
    # model related
    self.model_name=model_name
