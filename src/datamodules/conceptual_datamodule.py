import os
import io
import urllib
import PIL.Image
import torch
import datasets

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

from transformers import PerceiverFeatureExtractor, PerceiverTokenizer
from datasets import load_dataset, load_from_disk
from datasets.utils.file_utils import get_datasets_user_agent
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule



USER_AGENT = get_datasets_user_agent()


class ConceptualCaptionsDataModule(LightningDataModule):
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
        pin_memory=True):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters()
                
        self.num_workers = os.cpu_count() * 5
        
        self.tokenizer = PerceiverTokenizer()
            
        self.cc_train: Optional[Dataset] = None
        self.cc_val: Optional[Dataset] = None
        self.cc_test: Optional[Dataset] = None
        
          
    def fetch_single_image(self, image_url, timeout=None, retries=0):
        for _ in range(retries + 1):
            try:
                request = urllib.request.Request(
                    image_url,
                    data=None,
                    headers={"user-agent": USER_AGENT},
                )
                with urllib.request.urlopen(request, timeout=timeout) as req:
                    image = PIL.Image.open(io.BytesIO(req.read()))
                break
            except Exception:
                image = None
        return image


    def fetch_images(self, batch, num_threads, timeout=None, retries=0):
        fetch_single_image_with_args = partial(self.fetch_single_image, timeout=timeout, retries=retries)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
        return batch
        
        
    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        if os.path.isfile(f'{self.hparams.data_dir}/full/dataset.arrow'):
            print('Dataset inculding images already downloaded')
        else:
            load_dataset(
                'conceptual_captions', split='validation', cache_dir=self.hparams.data_dir
                ).map(
                    function=self.fetch_images, batched=True, batch_size=100, fn_kwargs={"num_threads": self.num_workers}
                    ).filter(
                        lambda x: x['image'] is not None and x['image'].mode == 'RGB'
                        ).save_to_disk(
                            f'{self.hparams.data_dir}/full'
                            )
            
            
    def setup(self, stage=None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""
             
        # Assign train/val/ datasets for use in dataloaders, data should already be downloaded
        if stage == "fit" or stage is None:
            self.cc_train = ConceptualCaptionsDataset(
                load_from_disk(f'{self.hparams.data_dir}/full/', split='train', cache_dir=self.hparams.data_dir)
                )
            
        if stage == "validate" or stage is None:
            self.cc_val = ConceptualCaptionsDataset(
                load_from_disk(f'{self.hparams.data_dir}/full/', split='validation', cache_dir=self.hparams.data_dir)
                )
            
        if stage == "test" or stage is None:
            raise Exception("""This dataset's test set it not available.""")
        
        if stage == "predict" or stage is None:
            raise Exception("""This DataModule is not designed to be used for prediction.
                            Please use the Spotify DataModule for prediction.""")
    
    
    def train_dataloader(self):
        return DataLoader(
            self.cc_train, 
            batch_size=self.hparams.train_batch_size, 
            shuffle=True, 
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory
            )
        
        
    def val_dataloader(self):
        return DataLoader(
            self.cc_val, 
            batch_size=self.hparams.val_batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory
            )
        
        
    def test_dataloader(self):
        return DataLoader(
            self.cc_test, 
            batch_size=self.hparams.test_batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory
            )
        
        
    def collate_fn(self, batch):
        return dict(
            text=self.tokenizer([item['caption'] for item in batch], padding=True, return_tensors='pt')['input_ids'],
            image=torch.cat([item['image'] for item in batch]),
        )


class ConceptualCaptionsDataset(Dataset):
    def __init__(
        self, 
        hf_dataset: datasets.arrow_dataset.Dataset
        ) -> None:
        super().__init__()
        self.dataset = hf_dataset
        self.feature_extractor = PerceiverFeatureExtractor()
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x = self.dataset[index]
        x['image'] = self.feature_extractor(x['image'], return_tensors='pt')['pixel_values']
        return x