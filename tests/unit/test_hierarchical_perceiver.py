import sys
import os
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import torch
import hydra
from typing import List
from src.models.module import LatentPredictionPretraining
from src.models.components.perceiver import PerceiverModel
from src.models.components.ema import EMA
from src.models.components.outputs import ModelOutput, ForwardPassOutput
from src.models.components.masking import mask_hidden_states
from tests.helpers import get_input_features


def test_model_instantiation():
    """
    Test that the model can be instantiated.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/model', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='hierarchical_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        assert isinstance(model, PerceiverModel)
    
    
def test_ema_instatiation():
    """
    Test that the model can instantiate an EMA object.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/model', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='hierarchical_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        ema = EMA(model)
        assert isinstance(ema, EMA)
        
        
def test_lightning_module_instantiation():
    """
    Test that the model can instantiate a LightningModule object.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/model', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='hierarchical_perceiver')
        module = hydra.utils.instantiate(cfg)
        assert isinstance(module, LatentPredictionPretraining)
        
        
def test_text_throughput():
    """
    Test that the model can process text.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/model', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='hierarchical_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        tokens, _, _, _, token_batch, _, _, _ = get_input_features(cfg.model.preprocessor.modalities.audio.samples_per_patch)
        
        num_latents = cfg.model.hip.block_configs[-1].num_latents
        d_latents  = cfg.model.hip.block_configs[-1].hidden_size
        # num_layers = cfg.model.num_self_attends_per_block
        
        inputs = dict(text=tokens)
        inputs_batch = dict(text=token_batch)
        
        outputs = model(inputs)
        outputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ModelOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert isinstance(outputs.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs.attentions[-1], torch.Tensor)
        assert isinstance(outputs.cross_attentions[0], torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, num_latents, d_latents)
        # assert len(outputs.hidden_states) == num_layers + 1
        
        assert isinstance(outputs_batch, ModelOutput)
        assert isinstance(outputs_batch.last_hidden_state, torch.Tensor)
        assert isinstance(outputs_batch.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs_batch.attentions[-1], torch.Tensor)
        assert isinstance(outputs_batch.cross_attentions[0], torch.Tensor)
        assert outputs_batch.last_hidden_state.size() == (32, num_latents, d_latents)
        # assert len(outputs_batch.hidden_states) == num_layers + 1
        
test_text_throughput()
            
            
def test_audio_throughput():
    """
    Test that the model can process audio.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/model', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='hierarchical_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        _, _, audio_features, _, _, _, audio_batch, _ = get_input_features(cfg.model.preprocessor.modalities.audio.samples_per_patch)
        
        num_latents = cfg.model.num_latents
        d_latents  = cfg.model.d_latents
        num_layers = cfg.model.num_self_attends_per_block
        
        inputs = dict(audio=audio_features)
        inputs_batch = dict(audio=audio_batch)
        
        outputs = model(inputs)
        outputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ModelOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert isinstance(outputs.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs.attentions[-1], torch.Tensor)
        assert isinstance(outputs.cross_attentions[0], torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, num_latents, d_latents)
        assert len(outputs.hidden_states) == num_layers + 1
        
        assert isinstance(outputs_batch, ModelOutput)
        assert isinstance(outputs_batch.last_hidden_state, torch.Tensor)
        assert isinstance(outputs_batch.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs_batch.attentions[-1], torch.Tensor)
        assert isinstance(outputs_batch.cross_attentions[0], torch.Tensor)
        assert outputs_batch.last_hidden_state.size() == (32, num_latents, d_latents)
        assert len(outputs_batch.hidden_states) == num_layers + 1
    
    
def test_image_throughput():
    """
    Test that the model can process images.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/model', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='hierarchical_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        _, image_features, _, _, _, image_batch, _, _ = get_input_features(cfg.model.input_preprocessor.modalities.audio.samples_per_patch)
        
        num_latents = cfg.model.num_latents
        d_latents  = cfg.model.d_latents
        num_layers = cfg.model.num_self_attends_per_block
        
        inputs = dict(image=image_features)
        inputs_batch = dict(image=image_batch)
        
        outputs = model(inputs)
        outputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ModelOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert isinstance(outputs.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs.attentions[-1], torch.Tensor)
        assert isinstance(outputs.cross_attentions[0], torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, num_latents, d_latents)
        assert len(outputs.hidden_states) == num_layers + 1
        
        assert isinstance(outputs_batch, ModelOutput)
        assert isinstance(outputs_batch.last_hidden_state, torch.Tensor)
        assert isinstance(outputs_batch.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs_batch.attentions[-1], torch.Tensor)
        assert isinstance(outputs_batch.cross_attentions[0], torch.Tensor)
        assert outputs_batch.last_hidden_state.size() == (32, num_latents, d_latents)
        assert len(outputs_batch.hidden_states) == num_layers + 1
  

def test_video_throughput():
    """
    Test that the model can process video.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/model', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='hierarchical_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        _, _, _, video_features, _, _, _, video_batch = get_input_features(cfg.model.input_preprocessor.modalities.audio.samples_per_patch)
        
        num_latents = cfg.model.num_latents
        d_latents  = cfg.model.d_latents
        num_layers = cfg.model.num_self_attends_per_block
        
        inputs = dict(image=video_features)
        inputs_batch = dict(image=video_batch)
        
        outputs = model(inputs)
        outputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ModelOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert isinstance(outputs.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs.attentions[-1], torch.Tensor)
        assert isinstance(outputs.cross_attentions[0], torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, num_latents, d_latents)
        assert len(outputs.hidden_states) == num_layers + 1
        
        assert isinstance(outputs_batch, ModelOutput)
        assert isinstance(outputs_batch.last_hidden_state, torch.Tensor)
        assert isinstance(outputs_batch.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs_batch.attentions[-1], torch.Tensor)
        assert isinstance(outputs_batch.cross_attentions[0], torch.Tensor)
        assert outputs_batch.last_hidden_state.size() == (32, num_latents, d_latents)
        assert len(outputs_batch.hidden_states) == num_layers + 1
        
        
def test_image_text_throughput():
    """
    Test that the model can process image-text pairs.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/model', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='hierarchical_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        _, _, audio_features, _, _, _, audio_batch, _ = get_input_features(cfg.model.input_preprocessor.modalities.audio.samples_per_patch)
        
        num_latents = cfg.model.num_latents
        d_latents  = cfg.model.d_latents
        num_layers = cfg.model.num_self_attends_per_block
        
        inputs = dict(audio=audio_features)
        inputs_batch = dict(audio=audio_batch)
        
        outputs = model(inputs)
        outputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ModelOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert isinstance(outputs.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs.attentions[-1], torch.Tensor)
        assert isinstance(outputs.cross_attentions[0], torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, num_latents, d_latents)
        assert len(outputs.hidden_states) == num_layers + 1
        
        assert isinstance(outputs_batch, ModelOutput)
        assert isinstance(outputs_batch.last_hidden_state, torch.Tensor)
        assert isinstance(outputs_batch.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs_batch.attentions[-1], torch.Tensor)
        assert isinstance(outputs_batch.cross_attentions[0], torch.Tensor)
        assert outputs_batch.last_hidden_state.size() == (32, num_latents, d_latents)
        assert len(outputs_batch.hidden_states) == num_layers + 1
        

def test_image_audio_throughput():
    """
    Test that the model can process audio-text pairs.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/model', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='hierarchical_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        _, _, audio_features, _, _, _, audio_batch, _ = get_input_features(cfg.model.input_preprocessor.modalities.audio.samples_per_patch)
        
        num_latents = cfg.model.num_latents
        d_latents  = cfg.model.d_latents
        num_layers = cfg.model.num_self_attends_per_block
        
        inputs = dict(audio=audio_features)
        inputs_batch = dict(audio=audio_batch)
        
        outputs = model(inputs)
        outputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ModelOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert isinstance(outputs.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs.attentions[-1], torch.Tensor)
        assert isinstance(outputs.cross_attentions[0], torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, num_latents, d_latents)
        assert len(outputs.hidden_states) == num_layers + 1
        
        assert isinstance(outputs_batch, ModelOutput)
        assert isinstance(outputs_batch.last_hidden_state, torch.Tensor)
        assert isinstance(outputs_batch.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs_batch.attentions[-1], torch.Tensor)
        assert isinstance(outputs_batch.cross_attentions[0], torch.Tensor)
        assert outputs_batch.last_hidden_state.size() == (32, num_latents, d_latents)
        assert len(outputs_batch.hidden_states) == num_layers + 1


def test_audio_text_throughput():
    """
    Test that the model can process audio-text pairs.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/model', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='hierarchical_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        _, _, audio_features, _, _, _, audio_batch, _ = get_input_features(cfg.model.input_preprocessor.modalities.audio.samples_per_patch)
        
        num_latents = cfg.model.num_latents
        d_latents  = cfg.model.d_latents
        num_layers = cfg.model.num_self_attends_per_block
        
        inputs = dict(audio=audio_features)
        inputs_batch = dict(audio=audio_batch)
        
        outputs = model(inputs)
        outputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ModelOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert isinstance(outputs.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs.attentions[-1], torch.Tensor)
        assert isinstance(outputs.cross_attentions[0], torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, num_latents, d_latents)
        assert len(outputs.hidden_states) == num_layers + 1
        
        assert isinstance(outputs_batch, ModelOutput)
        assert isinstance(outputs_batch.last_hidden_state, torch.Tensor)
        assert isinstance(outputs_batch.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs_batch.attentions[-1], torch.Tensor)
        assert isinstance(outputs_batch.cross_attentions[0], torch.Tensor)
        assert outputs_batch.last_hidden_state.size() == (32, num_latents, d_latents)
        assert len(outputs_batch.hidden_states) == num_layers + 1
    
    
    
def test_video_audio_throughput():
    """
    Test that the model can process video.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/model', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='hierarchical_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        _, _, _, video_features, _, _, _, video_batch = get_input_features(cfg.model.input_preprocessor.modalities.audio.samples_per_patch)
        
        num_latents = cfg.model.num_latents
        d_latents  = cfg.model.d_latents
        num_layers = cfg.model.num_self_attends_per_block
        
        inputs = dict(image=video_features)
        inputs_batch = dict(image=video_batch)
        
        outputs = model(inputs)
        outputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ModelOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert isinstance(outputs.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs.attentions[-1], torch.Tensor)
        assert isinstance(outputs.cross_attentions[0], torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, num_latents, d_latents)
        assert len(outputs.hidden_states) == num_layers + 1
        
        assert isinstance(outputs_batch, ModelOutput)
        assert isinstance(outputs_batch.last_hidden_state, torch.Tensor)
        assert isinstance(outputs_batch.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs_batch.attentions[-1], torch.Tensor)
        assert isinstance(outputs_batch.cross_attentions[0], torch.Tensor)
        assert outputs_batch.last_hidden_state.size() == (32, num_latents, d_latents)
        assert len(outputs_batch.hidden_states) == num_layers + 1
    
    
def test_video_text_thoughput():
    """
    Test that the model can process multimodal data.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/model', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='hierarchical_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        tokens, _, audio_features, video_features, token_batch, _, audio_batch, video_batch = get_input_features(cfg.model.input_preprocessor.modalities.audio.samples_per_patch)
        
        num_latents = cfg.model.num_latents
        d_latents  = cfg.model.d_latents
        num_layers = cfg.model.num_self_attends_per_block
        
        inputs = dict(text=tokens, image=video_features, audio=audio_features)
        inputs_batch = dict(text=token_batch, image=video_batch, audio=audio_batch)
        
        outputs = model(inputs)
        outputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ModelOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert isinstance(outputs.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs.attentions[-1], torch.Tensor)
        assert isinstance(outputs.cross_attentions[0], torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, num_latents, d_latents)
        assert len(outputs.hidden_states) == num_layers + 1
        
        assert isinstance(outputs_batch, ModelOutput)
        assert isinstance(outputs_batch.last_hidden_state, torch.Tensor)
        assert isinstance(outputs_batch.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs_batch.attentions[-1], torch.Tensor)
        assert isinstance(outputs_batch.cross_attentions[0], torch.Tensor)
        assert outputs_batch.last_hidden_state.size() == (32, num_latents, d_latents)
        assert len(outputs_batch.hidden_states) == num_layers + 1
        
        
def test_video_audio_text_thoughput():
    """
    Test that the model can process multimodal data.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/model', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='hierarchical_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        tokens, _, audio_features, video_features, token_batch, _, audio_batch, video_batch = get_input_features(cfg.model.input_preprocessor.modalities.audio.samples_per_patch)
        
        num_latents = cfg.model.num_latents
        d_latents  = cfg.model.d_latents
        num_layers = cfg.model.num_self_attends_per_block
        
        inputs = dict(text=tokens, image=video_features, audio=audio_features)
        inputs_batch = dict(text=token_batch, image=video_batch, audio=audio_batch)
        
        outputs = model(inputs)
        outputs_batch = model(inputs_batch)
        
        assert isinstance(outputs, ModelOutput)
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        assert isinstance(outputs.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs.attentions[-1], torch.Tensor)
        assert isinstance(outputs.cross_attentions[0], torch.Tensor)
        assert outputs.last_hidden_state.size() == (1, num_latents, d_latents)
        assert len(outputs.hidden_states) == num_layers + 1
        
        assert isinstance(outputs_batch, ModelOutput)
        assert isinstance(outputs_batch.last_hidden_state, torch.Tensor)
        assert isinstance(outputs_batch.hidden_states[-1], torch.Tensor)
        assert isinstance(outputs_batch.attentions[-1], torch.Tensor)
        assert isinstance(outputs_batch.cross_attentions[0], torch.Tensor)
        assert outputs_batch.last_hidden_state.size() == (32, num_latents, d_latents)
        assert len(outputs_batch.hidden_states) == num_layers + 1
    
    
def test_pl_module_forward():
    """
    Test that the model can process outputs.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/model', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='hierarchical_perceiver')
        pl_module = hydra.utils.instantiate(cfg)
        
        tokens, _, _, _, token_batch, _, _, _ = get_input_features(cfg.model.input_preprocessor.modalities.audio.samples_per_patch)
        
        inputs = dict(text=tokens)
        inputs_batch = dict(text=token_batch)
        
        forward_outputs = pl_module.forward(inputs)
        forward_outputs_batch = pl_module.forward(inputs_batch)
        
        assert isinstance(forward_outputs, ForwardPassOutput)
        assert isinstance(forward_outputs.student_output, ModelOutput)
        assert isinstance(forward_outputs.teacher_output, ModelOutput)
        
        
        assert isinstance(forward_outputs_batch, ForwardPassOutput)
        assert isinstance(forward_outputs_batch.student_output, ModelOutput)
        assert isinstance(forward_outputs_batch.teacher_output, ModelOutput)
    
    
def test_pl_module_step():
    """
    Test that the model can process loss functions.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/model', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='hierarchical_perceiver')
        pl_module = hydra.utils.instantiate(cfg)
        
        tokens, _, _, _, token_batch, _, _, _ = get_input_features(cfg.model.input_preprocessor.modalities.audio.samples_per_patch)
        
        inputs = dict(text=tokens)
        inputs_batch = dict(text=token_batch)
        
        step_outputs, loss = pl_module.step(inputs)
        step_outputs_batch, loss_batch = pl_module.step(inputs_batch)
        
        assert isinstance(step_outputs, ForwardPassOutput)
        assert loss.size() == torch.Size([])
        assert isinstance(step_outputs_batch, ForwardPassOutput)
        assert loss.size() == torch.Size([])
    

def test_latent_masking():
    """
    Test that the model can process latent masks.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/model', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='hierarchical_perceiver')
        pl_module = hydra.utils.instantiate(cfg)
        
        assert pl_module.student.is_student == True
        assert pl_module.teacher.model.is_student == False
        
        tokens, _, _, _, _, _, _, _ = get_input_features(cfg.model.input_preprocessor.modalities.audio.samples_per_patch)
        inputs = dict(text=tokens)
        step_outputs, _ = pl_module.step(inputs)
        
        assert step_outputs.student_output.last_hidden_state.size() == step_outputs.teacher_output.last_hidden_state.size()
        assert torch.equal(step_outputs.student_output.last_hidden_state, step_outputs.teacher_output.last_hidden_state) == False
    
