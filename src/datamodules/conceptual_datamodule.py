import os
import io
import urllib
import PIL.Image

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional


from datasets import load_dataset
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
        collator,
        data_dir,
        train_batch_size,
        val_batch_size,
        test_batch_size,
        version='3M',
        pin_memory=True):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters()
        
        self.collator = collator
        
        self.num_workers = os.cpu_count()
        if self.hparams.load_preprocessed_data:
            self.num_proc = 1
        else:
            self.num_proc = os.cpu_count()
            
        self.cc_train: Optional[Dataset] = None
        self.cc_val: Optional[Dataset] = None
        self.cc_test: Optional[Dataset] = None
        
        
    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        
        load_dataset('conceptual_captions', cache_dir=self.hparams.data_dir)
            
        
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
        
        
    def setup(self, stage=None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""
        
        self.cc = load_dataset('conceptual_captions', cache_dir=self.hparams.data_dir)
        self.cc = self.cc.map(self.fetch_images, batched=True, batch_size=100, fn_kwargs={"num_threads": self.num_workers})
        
        # Assign train/val/test datasets for use in dataloaders
        if stage == "fit" or stage == "test" or stage is None:
            self.cc_train, self.cc_val, self.cc_test = self.cc.split(split_ratio=(0.8, 0.1, 0.1))
        
        if stage == "predict" or stage is None:
            raise Exception("""This DataModule is not designed to be used for prediction.
                            Please use the Spotify DataModule for prediction.""")
    
    
    def train_dataloader(self):
        return DataLoader(
            self.cc_train, 
            batch_size=self.hparams.train_batch_size, 
            shuffle=True, 
            collate_fn=self.collator, 
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory
            )
        
        
    def val_dataloader(self):
        return DataLoader(
            self.cc_val, 
            batch_size=self.hparams.val_batch_size, 
            shuffle=False, 
            collate_fn=self.collator, 
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory
            )
        
        
    def test_dataloader(self):
        return DataLoader(
            self.cc_test, 
            batch_size=self.hparams.test_batch_size, 
            shuffle=False, 
            collate_fn=self.collator, 
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory
            )
        
