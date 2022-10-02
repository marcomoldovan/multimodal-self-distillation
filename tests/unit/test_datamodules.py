import os
import pytest
import torch

from src.datamodules.wikipedia_datamodule import WikipediaDataModule
from src.datamodules.imagenet_datamodule import ImagenetDataModule
from src.datamodules.librispeech_datamodule import LibriSpeechDataModule
from src.datamodules.conceptual_datamodule import ConceptualCaptionsDataModule


def test_wikipedia_datamodule_instantiation():
    """
    Test that the model can instantiate a WikipediaDatamodule object.
    """
    model = WikipediaDataModule()
    assert isinstance(model, WikipediaDataModule)
    
    
def test_imagenet_datamodule_instantiation():
    """
    Test that the model can instantiate an ImageNetDatamodule object.
    """
    model = ImagenetDataModule()
    assert isinstance(model, ImagenetDataModule)
    
    
def test_librispeech_datamodule_instantiation():
    """
    Test that the model can instantiate a LibrispeechDatamodule object.
    """
    model = LibriSpeechDataModule()
    assert isinstance(model, LibriSpeechDataModule)
    
    
def test_conceptual_datamodule_instantiation():
    """
    Test that the model can instantiate a ConceptualCaptionsDatamodule object.
    """
    model = ConceptualCaptionsDataModule()
    assert isinstance(model, ConceptualCaptionsDataModule)
    
    
def test_wikipedia_batch_loading():
    """
    Test that the model can load a batch of Wikipedia data.
    """
    model = WikipediaDataModule()
    batch = model.get_batch(batch_size=1)
    assert batch.size() == (1, 1, 300)
    
    
def test_imagenet_batch_loading():
    """
    Test that the model can load a batch of ImageNet data.
    """
    model = ImagenetDataModule()
    batch = model.get_batch(batch_size=1)
    assert batch.size() == (1, 3, 224, 224)
    
    
def test_librispeech_batch_loading():
    """
    Test that the model can load a batch of LibriSpeech data.
    """
    model = LibriSpeechDataModule()
    batch = model.get_batch(batch_size=1)
    assert batch.size() == (1, 1, 16000)
    
    
def test_conceptual_batch_loading():
    """
    Test that the model can load a batch of ConceptualCaptions data.
    """
    model = ConceptualCaptionsDataModule()
    batch = model.get_batch(batch_size=1)
    assert batch.size() == (1, 1, 300)
    