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
        
        self.align_fuse = [['text'], ['image']]
        self.metric = 'Recall@k'
        
    def setup(self, stage: str):
        raise NotImplementedError
    
    def prepare_data(self):
        raise NotImplementedError
    
    def train_dataloader(self):
        coco_train = ds.CocoCaptions(
            root=f'{self.hparams.data_dir}/train2014',
            annFile=f'{self.hparams.data_dir}/annotations/captions_train2014.json',
            transform=transforms.ToTensor(),
        )
        return DataLoader(
            dataset=coco_train,
            batch_size=self.hparams.train_batch_size,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        coco_val = ds.CocoCaptions(
            root=f'{self.hparams.data_dir}/val2014',
            annFile=f'{self.hparams.data_dir}/annotations/captions_val2014.json',
            transform=transforms.ToTensor(),
        )
        return DataLoader(
            dataset=coco_val,
            batch_size=self.hparams.val_batch_size,
            collate_fn=self.collate_fn,
            )
    
    def test_dataloader(self):
        coco_test = ds.CocoCaptions(
            root=f'{self.hparams.data_dir}/test2014',
            annFile=f'{self.hparams.data_dir}/annotations/captions_train2014.json',
            transform=transforms.ToTensor(),
        )
        return DataLoader(
            dataset=coco_test,
            batch_size=self.hparams.test_batch_size,
            collate_fn=self.collate_fn,
        )


    def collate_fn(self, batch):
        return dict(text=None, image=None, align_fuse=self.align_fuse, metric=self.metric)
