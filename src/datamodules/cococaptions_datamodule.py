import pytorch_lightning as pl
import torchvision.datasets as ds
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class COCOCaptionsDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        pin_memory: bool = True):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters()
        
    def setup(self, stage: str):
        raise NotImplementedError
    
    def prepare_data(self):
        raise NotImplementedError
    
    def train_dataloader(self):
        coco_train = ds.CocoCaptions(
            root='/home/ubuntu/data/coco/',
            annFile='/home/ubuntu/data/coco/annotations/captions_train2014.json',
            transform=transforms.ToTensor(),
            )
        return DataLoader(
            dataset=coco_train,
            batch_size=self.hparams.train_batch_size,
            )
    
    def val_dataloader(self):
        coco_val = ds.CocoCaptions(root='/home/ubuntu/data/coco/',)
        return DataLoader(
            dataset=coco_val,
            batch_size=self.hparams.train_batch_size,
            )
    
    def test_dataloader(self):
        coco_test = ds.CocoCaptions(root='/home/ubuntu/data/coco/',)
        return DataLoader(
            dataset=coco_test,
            batch_size=self.hparams.train_batch_size,
            )

