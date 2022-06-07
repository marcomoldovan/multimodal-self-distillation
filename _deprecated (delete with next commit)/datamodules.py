import os
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset, load_from_disk, concatenate_datasets
from pytorch_lightning import LightningDataModule
from preprocessing import LibriPreprocessor, SpokenSquadPreprocessor, collate_fn_spotify
from utils import assert_presence_of_dataset_on_machine, assert_presence_of_dataset_on_gdrive, download_dataset_from_gdrive


##############################
######## LibriSpeech #########
##############################

class LibriSpeechDataModule(LightningDataModule):
  def __init__(self, config):
    super().__init__()
    
    # this line allows to access init params with 'self.hparams' attribute
    self.save_hyperparameters()
    
    self.config = config
    self.preprocessor = LibriPreprocessor(config)
    
    self.num_workers = os.cpu_count()
    if config.use_gpu_for_preprocessing:
      self.num_proc = 1
    else:
      self.num_proc = os.cpu_count()
      
    self.preprocessed_data_available_on_machine, self.data_path = assert_presence_of_dataset_on_machine()
    self.preprocessed_data_available_on_gdrive, self.drive_path = assert_presence_of_dataset_on_gdrive()
    
    
  def prepare_data(self):
    if self.preprocessed_data_available_on_machine:
      for dir in range(1): # range(len(next(os.walk(self.data_path))[1])):
        load_from_disk(f'{self.data_path}/{dir}')
    elif self.preprocessed_data_available_on_gdrive:
      download_dataset_from_gdrive(self.drive_path)
      load_from_disk(self.data_path)
    else:
      raise Exception("""No preprocessed data available on machine or on Google Drive, 
                      it is STRONGLY recommended to perform all necessary preprocessing 
                      steps before executing the main training script as to avoid any 
                      crashes from unstable data preparation function.""")
        
    
  def setup(self, stage=None):
    
    # Assign train/val datasets for use in dataloaders
    
    if stage == "fit" or stage is None:
      if self.preprocessed_data_available_on_machine or self.preprocessed_data_available_on_gdrive:
        shards_list = []
        for dir in range(1): # range(len(next(os.walk(self.data_path))[1])):
          shard = load_from_disk(f'{self.data_path}/{dir}')
          shards_list.append(shard)
        libri = concatenate_datasets(shards_list)
      else:
        raise Exception("""Execute all necessary preprocessing involving
                        the convolutional feature excractor in a seperate script.""")
        
      libri_full = LibriSpeechDataset(libri)
      self.libri_train, self.libri_val = random_split(libri_full, [int(0.9 * len(libri_full)), len(libri_full) - int(0.9 * len(libri_full))])
    
    # Assign test dataset for use in dataloader(s)
    if stage == "test" or stage is None:
      self.libri_test = LibriSpeechDataset()
    
    if stage == "predict" or stage is None:
      pass
  
  
  def train_dataloader(self):
    return DataLoader(self.libri_train, batch_size=self.config.train_batch_size, shuffle=True, collate_fn=self.preprocessor, num_workers=self.num_workers)
    
    
  def val_dataloader(self):
    return DataLoader(self.libri_val, batch_size=self.config.val_batch_size, shuffle=False, collate_fn=self.preprocessor, num_workers=self.num_workers)
    
    
  def test_dataloader(self):
    return DataLoader(self.libri_test, batch_size=self.hparams.batch_size, shuffle=False, collate_fn=self.preprocessor, num_workers=self.num_workers)
  
  
  
  
class LibriSpeechDataset(Dataset):
  def __init__(self, libri_dataset):
    self.libri_dataset = libri_dataset
  
  
  def __len__(self):
    return len(self.libri_dataset)
  
  
  def __getitem__(self, index):
    return self.libri_dataset[index]
    


##################################
#### SpokenSQuAD Fine-Tuning #####
##################################


class SpokenSquadDataModule(LightningDataModule):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.preprocessor = SpokenSquadPreprocessor(config)
      
      
  def prepare_data(self):
    pass  
    
    
  def setup(self, stage=None):
    if stage == "fit" or stage is None:
      spotify_full = SpokenSquadDataset(self.extractor, self.tokenizer, split='train')
      self.spotify_train, self.spotify_val = random_split(spotify_full, [int(0.9 * len(spotify_full)), len(spotify_full) - int(0.9 * len(spotify_full))])
    
    # Assign test dataset for use in dataloader(s)
    if stage == "test" or stage is None:
      self.spotify_test = SpokenSquadDataset(self.extractor, self.tokenizer, split='train')
    
    if stage == "predict" or stage is None:
      pass
  
  
  def train_dataloader(self):
    return DataLoader(self.spotify_train, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=collate_fn_spotify)
    
    
  def val_dataloader(self):
    return DataLoader(self.spotify_val, batch_size=self.hparams.batch_size, shuffle=False, collate_fn=collate_fn_spotify)
    
    
  def test_dataloader(self):
    return DataLoader(self.spotify_test, batch_size=self.hparams.batch_size, shuffle=False, collate_fn=collate_fn_spotify)
    
    

class SpokenSquadDataset(Dataset):
  def __init__(self, feature_extractor, tokenizer):
    self.spotify_dataset = load_dataset('spotify_podcasts', 'clean')
    self.extractor = feature_extractor
    self.tokenizer = tokenizer
    
  
  def __len__(self):
    return len(self.spotify_dataset)
  
  
  def __getitem__(self, index):
    return self.spotify_dataset[index]
  


##################################
### SpotifyPodcastsPrediction ####
##################################

#TODO is a datamodule necessary? does a simple PyTorch dataloader suffice for this task? Especially since DataModules have training/validation/test dataloaders but none for prediction.
class SpotifyPredictionDataModule(LightningDataModule):
  """
  This DataModule only loads the small subset of labeled query-segment pairs from the Spotify dataset.
  Since it only contains a small subset of the data, it is not used for training, but easy plug-and-play testing.
  Since the speech segment retrieval is the actual main task of this project having a seperate DataModule on
  which we can easily test inference performance is important.
  The DataModule is loaded in any case regardless of the model previously trained. Every model has to implement
  the predict_step function that is compatible with this DataModule.

  Args:
      LightningDataModule (_type_): _description_
  """
  def __init__(self, config):
    super().__init__()
    self.config = config