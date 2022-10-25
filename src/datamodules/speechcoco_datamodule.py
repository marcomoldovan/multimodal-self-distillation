from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class SpeechCOCODataModule(LightningDataModule):
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
        batch_size, 
        num_workers, 
        pin_memory=True
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.align_fuse = [['audio'], ['image']]
        
    def prepare_data(self) -> None:
        pass
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Dataset()
            self.val_dataset = Dataset()
            
        if stage == "test" or stage is None:
            self.test_dataset = Dataset()
            
        if stage == "predict" or stage is None:
            raise Exception("""This DataModule is not designed to be used for prediction.
                            Please use the Spotify DataModule for prediction.""")
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
            )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
            )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
            )
        
    def collate_fn(self, batch):
        print(type(batch))
        print(batch)