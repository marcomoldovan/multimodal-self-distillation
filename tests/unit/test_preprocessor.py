import os
import sys
import hydra

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.components.preprocessor import PerceiverMultimodalPreprocessor
from tests.helpers import get_input_features

inputs = get_input_features()
max_padding = 2


def test_preprocessor_instantiation():
    """
    Test that the model can instantiate a preprocessor.
    """
    with hydra.initialize(version_base='1.2', config_path='../../configs/tests', job_name="test_preprocessor_instantiation"):
        cfg = hydra.compose(config_name='flat_perceiver')
        preprocessor = hydra.utils.instantiate(cfg.model.input_preprocessor)
        assert isinstance(preprocessor, PerceiverMultimodalPreprocessor)
        

def test_preprocessor_text():
    """
    Test all reasonable combinations of inputs to the preprocessor.
    """
    with hydra.initialize(version_base='1.2', config_path='../../configs/tests', job_name="test_preprocessor_outputs"):
        cfg = hydra.compose(config_name='flat_perceiver')
        preprocessor = hydra.utils.instantiate(cfg.model.input_preprocessor)
        
        hidden_size = cfg.model.input_preprocessor.modalities.text.d_model
        
        inputs_txt = dict(text=inputs[0])
        outputs_txt = preprocessor(inputs_txt)
        assert outputs_txt[0].size()[0] == 1
        assert outputs_txt[0].size()[1] == 46
        assert outputs_txt[0].size()[2] == hidden_size + max_padding
        
        inputs_txt_batch = dict(text=inputs[4])
        outputs_txt_batch = preprocessor(inputs_txt_batch)
        assert outputs_txt_batch[0].size()[0] == 32
        assert outputs_txt_batch[0].size()[1] == 46
        assert outputs_txt_batch[0].size()[2] == hidden_size + max_padding
            
    
def test_preprocessor_audio():
    """
    Test all reasonable combinations of inputs to the preprocessor.
    """
    with hydra.initialize(version_base='1.2', config_path='../../configs/tests', job_name="test_preprocessor_outputs"):
        cfg = hydra.compose(config_name='flat_perceiver')
        preprocessor = hydra.utils.instantiate(cfg.model.input_preprocessor)
        
        hidden_size = cfg.model.input_preprocessor.modalities.text.d_model
        
        inputs_aud = dict(audio=inputs[2])
        outputs_aud = preprocessor(inputs_aud)
        assert outputs_aud[0].size()[0] == 1
        assert outputs_aud[0].size()[1] == 4038
        assert outputs_aud[0].size()[2] == hidden_size + max_padding
        
        inputs_aud_batch = dict(audio=inputs[6])
        outputs_aud_batch = preprocessor(inputs_aud_batch)
        assert outputs_aud_batch[0].size()[0] == 32
        assert outputs_aud_batch[0].size()[1] == 4038
        assert outputs_aud_batch[0].size()[2] == hidden_size + max_padding

    
def test_preprocessor_image():
    """
    Test all reasonable combinations of inputs to the preprocessor.
    """
    with hydra.initialize(version_base='1.2', config_path='../../configs/tests', job_name="test_preprocessor_outputs"):
        cfg = hydra.compose(config_name='flat_perceiver')
        preprocessor = hydra.utils.instantiate(cfg.model.input_preprocessor)
        
        hidden_size = cfg.model.input_preprocessor.modalities.text.d_model
        
        inputs_img = dict(image=inputs[1])
        outputs_img = preprocessor(inputs_img)
        assert outputs_img[0].size()[0] == 1
        assert outputs_img[0].size()[1] == 50176
        assert outputs_img[0].size()[2] == hidden_size + max_padding
        
        inputs_img_batch = dict(image=inputs[5])
        outputs_img_batch = preprocessor(inputs_img_batch)
        assert outputs_img_batch[0].size()[0] == 32
        assert outputs_img_batch[0].size()[1] == 50176
        assert outputs_img_batch[0].size()[2] == hidden_size + max_padding
                
    
def test_preprocessor_video():
    """
    Test all reasonable combinations of inputs to the preprocessor.
    """
    with hydra.initialize(version_base='1.2', config_path='../../configs/tests', job_name="test_preprocessor_outputs"):
        cfg = hydra.compose(config_name='flat_perceiver')
        preprocessor = hydra.utils.instantiate(cfg.model.input_preprocessor)
        
        hidden_size = cfg.model.input_preprocessor.modalities.text.d_model
        
        inputs_vid = dict(video=inputs[3])
        outputs_vid = preprocessor(inputs_vid) #! length of 800k - which is 16x the length of the image output length - still too long!!!
        
        inputs_vid_batch = dict(video=inputs[7])
        # outputs_vid_batch = preprocessor(inputs_vid_batch) #! tensor way too big, crashes RAM
        
    
def test_preprocessor_image_text():
    """
    Test all reasonable combinations of inputs to the preprocessor.
    """
    with hydra.initialize(version_base='1.2', config_path='../../configs/tests', job_name="test_preprocessor_outputs"):
        cfg = hydra.compose(config_name='flat_perceiver')
        preprocessor = hydra.utils.instantiate(cfg.model.input_preprocessor)
        
        hidden_size = cfg.model.input_preprocessor.modalities.text.d_model
        
        inputs_txt_img = dict(text=inputs[0], image=inputs[1])
        outputs_txt_img = preprocessor(inputs_txt_img)
        
        inputs_txt_img_batch = dict(text=inputs[4], image=inputs[5])
        outputs_txt_img_batch = preprocessor(inputs_txt_img_batch) 
        
    
def test_preprocessor_image_audio():
    """
    Test all reasonable combinations of inputs to the preprocessor.
    """
    with hydra.initialize(version_base='1.2', config_path='../../configs/tests', job_name="test_preprocessor_outputs"):
        cfg = hydra.compose(config_name='flat_perceiver')
        preprocessor = hydra.utils.instantiate(cfg.model.input_preprocessor)
        
        hidden_size = cfg.model.input_preprocessor.modalities.text.d_model
        
        inputs_img_aud = dict(image=inputs[1], audio=inputs[2])
        outputs_img_aud = preprocessor(inputs_img_aud) 
        
        inputs_img_aud_batch = dict(image=inputs[5], audio=inputs[6])
        outputs_img_aud_batch = preprocessor(inputs_img_aud_batch)
        
    
def test_preprocessor_audio_text():
    """
    Test all reasonable combinations of inputs to the preprocessor.
    """
    with hydra.initialize(version_base='1.2', config_path='../../configs/tests', job_name="test_preprocessor_outputs"):
        cfg = hydra.compose(config_name='flat_perceiver')
        preprocessor = hydra.utils.instantiate(cfg.model.input_preprocessor)
        
        hidden_size = cfg.model.input_preprocessor.modalities.text.d_model
        
        inputs_aud_txt = dict(text=inputs[0], audio=inputs[2])
        outputs_aud_txt = preprocessor(inputs_aud_txt) 
        
        inputs_aud_txt_batch = dict(text=inputs[4], audio=inputs[6])
        outputs_aud_txt_batch = preprocessor(inputs_aud_txt_batch) 
        
    
def test_preprocessor_video_audio():
    """
    Test all reasonable combinations of inputs to the preprocessor.
    """
    with hydra.initialize(version_base='1.2', config_path='../../configs/tests', job_name="test_preprocessor_outputs"):
        cfg = hydra.compose(config_name='flat_perceiver')
        preprocessor = hydra.utils.instantiate(cfg.model.input_preprocessor)
        
        hidden_size = cfg.model.input_preprocessor.modalities.text.d_model
        
        inputs_vid_aud = dict(video=inputs[3], audio=inputs[2])
        outputs_vid_aud = preprocessor(inputs_vid_aud)
        
        inputs_vid_aud_batch = dict(video=inputs[7], audio=inputs[6])
        # outputs_vid_aud_batch = preprocessor(inputs_vid_aud_batch) #! video too large
        
    
def test_preprocessor_video_text():
    """
    Test all reasonable combinations of inputs to the preprocessor.
    """
    with hydra.initialize(version_base='1.2', config_path='../../configs/tests', job_name="test_preprocessor_outputs"):
        cfg = hydra.compose(config_name='flat_perceiver')
        preprocessor = hydra.utils.instantiate(cfg.model.input_preprocessor)
        
        hidden_size = cfg.model.input_preprocessor.modalities.text.d_model
        
        inputs_vid_text = dict(text=inputs[0], video=inputs[3])
        outputs_vid_text = preprocessor(inputs_vid_text)
        
        inputs_vid_text_batch = dict(text=inputs[4], video=inputs[7])
        # outputs_vid_aud_batch = preprocessor(inputs_vid_aud_batch) #! video too large
        
    
def test_preprocessor_video_audio_text():
    """
    Test all reasonable combinations of inputs to the preprocessor.
    """
    with hydra.initialize(version_base='1.2', config_path='../../configs/tests', job_name="test_preprocessor_outputs"):
        cfg = hydra.compose(config_name='flat_perceiver')
        preprocessor = hydra.utils.instantiate(cfg.model.input_preprocessor)
        
        hidden_size = cfg.model.input_preprocessor.modalities.text.d_model
        
        inputs_vid_aud_text = dict(text=inputs[0], video=inputs[3], audio=inputs[2])
        outputs_vid_aud_text = preprocessor(inputs_vid_aud_text) 
        
        inputs_vid_aud_text_batch = dict(text=inputs[4], video=inputs[7], audio=inputs[6])
        # outputs_vid_aud_text_batch = preprocessor(inputs_vid_aud_text_batch) #! video too large
        
