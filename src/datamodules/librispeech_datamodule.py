import os
import torch
from typing import Optional, Union, Dict, List
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from transformers import Wav2Vec2FeatureExtractor, PerceiverTokenizer
from transformers.utils import logging


class LibriSpeechDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """
    def __init__(
        self,
        data_dir,
        train_batch_size,
        val_batch_size,
        test_batch_size,
        split='train.360',
        pin_memory=True
    ) -> None:
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters()
        
        if os.name == 'nt':
            self.num_workers = 0
        else:
            self.num_workers = os.cpu_count()
            
        self.libri_train: Optional[Dataset] = None
        self.libri_val: Optional[Dataset] = None
        self.libri_test: Optional[Dataset] = None
        
        logging.set_verbosity(logging.CRITICAL)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')
        self.tokenizer = PerceiverTokenizer.from_pretrained('deepmind/language-perceiver')
        logging.set_verbosity(logging.WARNING)
        
        
    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        if os.path.isdir(self.hparams.data_dir):
            print("Data directory already exists, skipping download.")
        else:
            load_dataset('librispeech_asr', 'clean', split=self.hparams.split, cache_dir=self.hparams.data_dir)
            
        
    def setup(self, stage=None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""
        
        # Assign train/val datasets for use in dataloaders
        
        if stage == "fit" or stage is None:
            self.libri_train = load_dataset('librispeech_asr', 'clean', split=self.hparams.split, cache_dir=self.hparams.data_dir)
            self.libri_val = load_dataset('librispeech_asr', 'clean', split='validation', cache_dir=self.hparams.data_dir)
                

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.libri_test = load_dataset('librispeech_asr', 'clean', split='test', cache_dir=self.hparams.data_dir)
        
        if stage == "predict" or stage is None:
            raise Exception("""This DataModule is not designed to be used for prediction.
                            Please use the Spotify DataModule for prediction.""")
    
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.libri_train, 
            batch_size=self.hparams.train_batch_size, 
            shuffle=True, 
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory
        )
        
        
    def val_dataloader(self):
        return DataLoader(
            dataset=self.libri_val, 
            batch_size=self.hparams.val_batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory
        )
        
        
    def test_dataloader(self):
        return DataLoader(
            dataset=self.libri_test, 
            batch_size=self.hparams.test_batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory
        )
        
        
    def collate_fn(
        self,
        batch: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
                
        input_values = [feature["audio"]["array"] for feature in batch]
        text = [feature["text"] for feature in batch]
        
        audio = self.feature_extractor(
            input_values,
            pad_to_multiple_of=96, #TODO read from config
            padding="longest",
            return_tensors="pt",
            sampling_rate=16000, #TODO read from config
        )
        
        tokens = self.tokenizer(
            text,
            padding="longest",
            return_tensors="pt",
        )
        
        return dict(text=tokens["input_ids"], audio=audio["input_values"])
