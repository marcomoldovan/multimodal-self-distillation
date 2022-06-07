from argparse import ArgumentParser

from config import SpeechAndTextBiEncoderConfig


def construct_arguments_parser():
  parser = ArgumentParser()
  parser.add_argument("--pretraining_contrastive_loss", type=str, default="TripletMarginLoss", help="TripelMarginLoss or SimCLR")
  parser.add_argument("--train_last_n_speech_model_layers", type=int, default=1)
  parser.add_argument("--train_last_n_text_model_layers", type=int, default=0)
  parser.add_argument("--training_mode", type=str, default="pretrain", help="pretrain or finetune")
  parser.add_argument("--train_batch_size", type=int, default=8)
  parser.add_argument("--val_batch_size", type=int, default=10)
  parser.add_argument("--num_epochs", type=int, default=10)
  parser.add_argument("--dataset_name", type=str, default="librispeech", help="librispeech or spotify")
  return parser


def build_config_from_args(parser):
  #TODO have different configs for model, data, trainer?
  #TODO build corresponding config class depending on user input, options: bi-encoder, multimodal encoder
  args = parser.parse_args()
  config = SpeechAndTextBiEncoderConfig(pretraining_contrastive_loss_fn=args.pretraining_contrastive_loss,
                                                        train_last_n_speech_model_layers=args.train_last_n_speech_model_layers,
                                                        train_last_n_text_model_layers=args.train_last_n_text_model_layers,
                                                        training_mode=args.training_mode,
                                                        train_batch_size=args.train_batch_size,
                                                        val_batch_size=args.val_batch_size,
                                                        num_epochs=args.num_epochs,
                                                        dataset_name=args.dataset_name)
  return config


def rebuild_config_object_from_wandb(wandb_config_as_dict):
  """
    This is necessary for wandb sweeps to work. Sweep agent will replace config parameters
    with values from the hyperparameter search space as specified in the sweep configs. 
    Agents needs wandb.config object to do that, our custom config object knows nothing
    about the sweep and doesn't contain the changed hyperparameters. Thats why we have to
    pass the config to the wandb context manager for the sweep changes to take effect, then
    turn the updated config back to our custom config object that our trainer/model/dataset
    can handle.
    NOTICE: This explanation exists because this function may seems nonsensical to a reader
    unfimiliar with wandb sweeps and pytorch-lightning modules (or myself once I forget why
    I've added this function).
  """
  config_readable = SpeechAndTextBiEncoderConfig()
  #TODO fix this
  for key in wandb_config_as_dict:
    config_readable.key = wandb_config_as_dict[key]
  return config_readable


def build_run_name_from_config(config):
  run_name = config.model_name
  run_name += '_' + config.training_mode
  run_name += '_' + config.dataset_name
  run_name += '_los-fn-' + config.pretraining_contrastive_loss_fn
  run_name += '_train-batch-' + str(config.train_batch_size)
  run_name += '_val-batch-' + str(config.val_batch_size)
  run_name += '_epochs-' + str(config.num_epochs)
  return run_name


def build_model_from_config(config):
  if config.training_mode == 'pretrain' and config.model_name == 'PSTM':
    from models import SpeechAndTextBiEncoder
    model = SpeechAndTextBiEncoder(config)
    return model


def build_data_module_from_config(config):
  #TODO build the correct data module from the config, there needs to be an exception to make sure that the model is compatible with the dataset
  pass


def build_trainer_from_config(config):
  #TODO build the correct trainer from the config
  pass


