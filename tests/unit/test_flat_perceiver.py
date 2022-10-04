import sys
import os
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import torch
import hydra

from src.models.module import LatentPredictionPretraining
from src.models.components.perceiver import PerceiverModel
from src.models.components.ema import EMA
from src.models.components.criterion import LatentPredictionLoss
from src.models.components.outputs import ForwardPassOutput, TrainingStepOutput
from src.models.components.masking import mask_hidden_states
from tests.helpers import get_input_features


def test_model_instantiation():
    """
    Test that the model can be instantiated.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/tests', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='flat_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        assert isinstance(model, PerceiverModel)
    
    
def test_ema_instatiation():
    """
    Test that the model can instantiate an EMA object.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/tests', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='flat_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        ema = EMA(model)
        assert isinstance(ema, EMA)
        
        
def test_lightning_module_instantiation():
    """
    Test that the model can instantiate a LightningModule object.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/tests', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='lit_module')
        module = hydra.utils.instantiate(cfg)
        assert isinstance(module, LatentPredictionPretraining)
        
        
def test_text_throughput():
    """
    Test that the model can process text.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/tests', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='flat_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        tokens, _, _, _, token_batch, _, _, _ = get_input_features()
        
        inputs = dict(text=tokens)
        inputs_batch = dict(text=token_batch)
        
        outputs = model(inputs)
        outputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ForwardPassOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, 1, 1, 1)
    
    
def test_audio_throughput():
    """
    Test that the model can process audio.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/tests', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='flat_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        _, _, audio_features, _, _, _, audio_batch, _ = get_input_features()
        
        inputs = dict(audio=audio_features)
        inputs_batch = dict(audio=audio_batch)
        
        outputs = model(inputs)
        ouputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ForwardPassOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, 1, 1, 1)
    
    
def test_image_throughput():
    """
    Test that the model can process images.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/tests', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='flat_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        _, image_features, _, _, _, image_batch, _, _ = get_input_features()
        
        inputs = dict(image=image_features)
        inputs_batch = dict(image=image_batch)
        
        outputs = model(inputs)
        ouputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ForwardPassOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, 1, 1, 1)
    

def test_video_throughput():
    """
    Test that the model can process video.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/tests', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='flat_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        _, _, _, video_features, _, _, _, video_batch = get_input_features()
        
        inputs = dict(image=video_features)
        inputs_batch = dict(image=video_batch)
        
        outputs = model(inputs)
        outputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ForwardPassOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, 1, 1, 1)
        
        
def test_image_text_throughput():
    """
    Test that the model can process image-text pairs.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/tests', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='flat_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        _, _, audio_features, _, _, _, audio_batch, _ = get_input_features()
        
        inputs = dict(audio=audio_features)
        inputs_batch = dict(audio=audio_batch)
        
        outputs = model(inputs)
        ouputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ForwardPassOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, 1, 1, 1)
        

def test_image_audio_throughput():
    """
    Test that the model can process audio-text pairs.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/tests', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='flat_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        _, _, audio_features, _, _, _, audio_batch, _ = get_input_features()
        
        inputs = dict(audio=audio_features)
        inputs_batch = dict(audio=audio_batch)
        
        outputs = model(inputs)
        ouputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ForwardPassOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, 1, 1, 1)


def test_audio_text_throughput():
    """
    Test that the model can process audio-text pairs.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/tests', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='flat_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        _, _, audio_features, _, _, _, audio_batch, _ = get_input_features()
        
        inputs = dict(audio=audio_features)
        inputs_batch = dict(audio=audio_batch)
        
        outputs = model(inputs)
        ouputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ForwardPassOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, 1, 1, 1)
    
    
    
def test_video_audio_throughput():
    """
    Test that the model can process video.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/tests', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='flat_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        _, _, _, video_features, _, _, _, video_batch = get_input_features()
        
        inputs = dict(image=video_features)
        inputs_batch = dict(image=video_batch)
        
        outputs = model(inputs)
        outputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ForwardPassOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, 1, 1, 1)
    
    
def test_video_text_thoughput():
    """
    Test that the model can process multimodal data.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/tests', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='flat_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        tokens, _, audio_features, video_features, token_batch, _, audio_batch, video_batch = get_input_features()
        
        inputs = dict(text=tokens, image=video_features, audio=audio_features)
        inputs_batch = dict(text=token_batch, image=video_batch, audio=audio_batch)
        
        outputs = model(inputs)
        outputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ForwardPassOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, 1, 1, 1)
        
        
def test_video_audio_text_thoughput():
    """
    Test that the model can process multimodal data.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/tests', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='flat_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        tokens, _, audio_features, video_features, token_batch, _, audio_batch, video_batch = get_input_features()
        
        inputs = dict(text=tokens, image=video_features, audio=audio_features)
        inputs_batch = dict(text=token_batch, image=video_batch, audio=audio_batch)
        
        outputs = model(inputs)
        outputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ForwardPassOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, 1, 1, 1)
    
    
# def test_outputs():
#     """
#     Test that the model can process outputs.
#     """
#     model = PerceiverModel()
#     outputs = torch.randn(1, 1, 1, 1)
#     model.forward(outputs)
#     assert model.outputs == 1
    
    
# def test_latent_masking():
#     """
#     Test that the model can process latent masks.
#     """
#     model = PerceiverModel()
#     mask = torch.randn(1, 1, 1, 1)
#     model.forward(mask)
#     assert model.latent_masking == 1
    
    
# def test_loss_function():
#     """
#     Test that the model can process loss functions.
#     """
#     model = PerceiverModel()
#     loss_function = torch.nn.MSELoss()
#     model.forward(loss_function)
#     assert model.loss_function == 1
    

# def test_lightning_module_pipeline():
#     """
#     Test that the model can process a pipeline.
#     """
#     model = PerceiverModel()
#     pipeline = model.pipeline
#     assert isinstance(pipeline, torch.nn.Sequential)