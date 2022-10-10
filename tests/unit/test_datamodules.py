import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import hydra

from src.datamodules.msmarco_datamodule import MSMARCOPassageDataModule
from src.datamodules.imagenet_datamodule import ImagenetDataModule
from src.datamodules.wikipedia_datamodule import WikipediaDataModule
from src.datamodules.conceptual_datamodule import ConceptualCaptionsDataModule
from src.datamodules.speechcoco_datamodule import SpeechCOCODataModule
from src.datamodules.librispeech_datamodule import LibriSpeechDataModule
from src.datamodules.tinyimagenet_datamodule import TinyImagenetDataModule
from src.datamodules.cococaptions_datamodule import COCOCaptionsDatamodule


def test_wikipedia():
    """
    Test that the model can instantiate a WikipediaDatamodule object.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/datamodule', job_name="test_wikipedia"):
        cfg = hydra.compose(config_name='wikipedia')
        datamodule = hydra.utils.instantiate(cfg)
        assert isinstance(datamodule, WikipediaDataModule)
    
    
def test_msmarco():
    with hydra.initialize(version_base='1.1', config_path='../../configs/datamodule', job_name="test_msmarco"):
        cfg = hydra.compose(config_name='msmarco')
        datamodule = hydra.utils.instantiate(cfg)
        assert isinstance(datamodule, MSMARCOPassageDataModule)
    
    
def test_imagenet():
    """
    Test that the model can instantiate an ImageNetDatamodule object.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/datamodule', job_name="test_imagenet"):
        cfg = hydra.compose(config_name='imagenet')
        datamodule = hydra.utils.instantiate(cfg)
        assert isinstance(datamodule, ImagenetDataModule)
        
def test_tinyimagenet():
    """
    Test that the model can instantiate an ImageNetDatamodule object.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/datamodule', job_name="test_tiny_imagenet"):
        cfg = hydra.compose(config_name='tinyimagenet')
        datamodule = hydra.utils.instantiate(cfg)
        assert isinstance(datamodule, TinyImagenetDataModule)
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch['image'].size() == torch.Size([cfg.batch_size, 3, 64, 64])
        assert train_batch['label'].size() == torch.Size([cfg.batch_size])
            
    
def test_librispeech():
    """
    Test that the model can instantiate a LibrispeechDatamodule object.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/datamodule', job_name="test_librispeech"):
        cfg = hydra.compose(config_name='librispeech')
        datamodule = hydra.utils.instantiate(cfg)
        assert isinstance(datamodule, LibriSpeechDataModule)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch['audio'].size()[0] == cfg.train_batch_size
        assert train_batch['text'].size()[0] == cfg.train_batch_size
        
        
def test_coco_captions():
    """
    Test that the model can instantiate a COCOCaptionsDatamodule object.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/datamodule', job_name="test_coco_captions"):
        cfg = hydra.compose(config_name='cococaptions')
        datamodule = hydra.utils.instantiate(cfg)
        assert isinstance(datamodule, COCOCaptionsDatamodule)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch['text'].size()[0] == cfg.train_batch_size
        assert train_batch['image'].size()[0] == cfg.train_batch_size
            
    
def test_conceptual_captions():
    """
    Test that the model can instantiate a ConceptualCaptionsDatamodule object.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/datamodule', job_name="test_conceptual_captions"):
        cfg = hydra.compose(config_name='conceptual')
        datamodule = hydra.utils.instantiate(cfg)
        assert isinstance(datamodule, ConceptualCaptionsDataModule)
        datamodule.prepare_data()
        datamodule.setup(stage='validate')
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch['text'].size()[0] == cfg.train_batch_size
        assert train_batch['image'].size()[0] == cfg.train_batch_size
        
test_conceptual_captions()
        
        
def test_speechcoco():
    """
    Test that the model can instantiate a SpeechCocoDatamodule object.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/datamodule', job_name="test_speechcoco"):
        cfg = hydra.compose(config_name='speechcoco')
        datamodule = hydra.utils.instantiate(cfg)
        assert isinstance(datamodule, SpeechCOCODataModule)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_batch = next(iter(datamodule.train_dataloader()))
        assert train_batch['audio'].size()[0] == cfg.train_batch_size
        assert train_batch['image'].size()[0] == cfg.train_batch_size
    