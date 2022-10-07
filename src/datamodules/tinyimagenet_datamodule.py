import os.path
from typing import Callable

import torch
import pytorch_lightning as pl
from classy_vision.dataset.classy_dataset import ClassyDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



class TinyImagenetDataset(ClassyDataset):
    """
    TinyImageNetDataset is a ClassyDataset for the tiny imagenet dataset.
    """

    def __init__(self, data_path: str, transform: Callable[[object], object]) -> None:
        batchsize_per_replica = 16
        shuffle = False
        num_samples = 1000
        dataset = datasets.ImageFolder(data_path)
        super().__init__(
            # pyre-fixme[6]
            dataset,
            batchsize_per_replica,
            shuffle,
            transform,
            num_samples,
        )



class TinyImagenetDataModule(pl.LightningDataModule):
    """
    TinyImageNetDataModule is a pytorch LightningDataModule for the tiny
    imagenet dataset.
    """
    def __init__(
        self, 
        data_dir: str, 
        num_workers: int = 0,
        batch_size: int = 16,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters()
        
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        
        
    def _verify_splits(self, data_dir: str, split: str) -> None:
        dirs = os.listdir(data_dir)

        if split not in dirs:
            raise FileNotFoundError(
                f"a {split} Imagenet split was not found in {data_dir}, make sure the"
                f" folder contains a subfolder named {split}"
            )
            
            
    def prepare_data(self) -> None:
        # imagenet cannot be downloaded... must provide path to folder with the train/val splits
        self._verify_splits(self.data_dir, "train")
        self._verify_splits(self.data_dir, "val")
        self._verify_splits(self.data_dir, "test")
        
        
    def train_dataloader(self) -> DataLoader:
        img_transform = self._default_transforms()
        
        self.train_ds = TinyImagenetDataset(
            data_path=os.path.join(self.data_dir, "train"),
            transform=lambda x: (img_transform(x[0]), x[1]),
        )
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        img_transform = self._default_transforms()
        
        self.val_ds = TinyImagenetDataset(
            data_path=os.path.join(self.data_dir, "val"),
            transform=lambda x: (img_transform(x[0]), x[1]),
        )
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        img_transform = self._default_transforms()
        
        self.test_ds = TinyImagenetDataset(
            data_path=os.path.join(self.data_dir, "test"),
            transform=lambda x: (img_transform(x[0]), x[1]),
        )
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
    
    
    def _default_transforms(self) -> Callable:
        img_transform = transforms.ToTensor()
        return img_transform
    
    def collate_fn(self, batch):
        images, labels = zip(*batch)
        return dict(image=torch.stack(tensors=images, dim=0), label=torch.tensor(labels))

