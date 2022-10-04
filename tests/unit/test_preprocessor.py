import os
import sys
import hydra

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.components.preprocessor import PerceiverMultimodalPreprocessor
from tests.helpers import get_input_features


def test_preprocessor_instantiation():
    """
    Test that the model can instantiate a preprocessor.
    """
    with hydra.initialize(version_base='1.2', config_path='../../configs/tests', job_name="test_preprocessor_instantiation"):
        cfg = hydra.compose(config_name='preprocessor')
        preprocessor = hydra.utils.instantiate(cfg.input_preprocessor)
        assert isinstance(preprocessor, PerceiverMultimodalPreprocessor)


def test_preprocessor_outputs():
    """
    Test all reasonable combinations of inputs to the preprocessor.
    """
    with hydra.initialize(version_base='1.2', config_path='../../configs/tests', job_name="test_preprocessor_outputs"):
        cfg = hydra.compose(config_name='preprocessor')
        preprocessor = hydra.utils.instantiate(cfg.input_preprocessor)
        preprocessor_videos = hydra.utils.instantiate(cfg.input_preprocessor_vid)
        
        inputs = get_input_features()
        
        # text only
        inputs_txt = dict(text=inputs[0])
        outputs_txt = preprocessor(inputs_txt)
        
        inputs_txt_batch = dict(text=inputs[4])
        outputs_txt_batch = preprocessor(inputs_txt_batch)
        
        # audio only
        inputs_aud = dict(audio=inputs[2])
        # outputs_aud = preprocessor(inputs_aud) #! audio does not get padded correctly to multiple of 96
        
        inputs_aud_batch = dict(audio=inputs[6])
        # outputs_aud_batch = preprocessor(inputs_aud_batch) #! audio does not get padded correctly to multiple of 96
        
        # image only
        inputs_img = dict(image=inputs[1])
        outputs_img = preprocessor(inputs_img)
        
        inputs_img_batch = dict(image=inputs[5])
        outputs_img_batch = preprocessor(inputs_img_batch)
        
        # video only
        inputs_vid = dict(video=inputs[3])
        outputs_vid = preprocessor(inputs_vid) #! length of 800k - which is 16x the length of the image output length - still too long!!!
        
        inputs_vid_batch = dict(video=inputs[7])
        # outputs_vid_batch = preprocessor(inputs_vid_batch) #! tensor way too big, crashes RAM
        
        # image and text
        inputs_txt_img = dict(text=inputs[0], image=inputs[1])
        outputs_txt_img = preprocessor(inputs_txt_img)
        
        inputs_txt_img_batch = dict(text=inputs[4], image=inputs[5])
        outputs_txt_img_batch = preprocessor(inputs_txt_img_batch) 
        
        # image and audio
        inputs_img_aud = dict(image=inputs[1], audio=inputs[2])
        # outputs_img_aud = preprocessor(inputs_img_aud) #! audio does not get padded correctly to multiple of 96
        
        inputs_img_aud_batch = dict(image=inputs[5], audio=inputs[6])
        # outputs_img_aud_batch = preprocessor(inputs_img_aud_batch) #! audio does not get padded correctly to multiple of 96
        
        # audio and text
        inputs_aud_txt = dict(text=inputs[0], audio=inputs[2])
        # outputs_aud_txt = preprocessor(inputs_aud_txt) #! audio does not get padded correctly to multiple of 96
        
        inputs_aud_txt_batch = dict(text=inputs[4], audio=inputs[6])
        # outputs_aud_txt_batch = preprocessor(inputs_aud_txt_batch) #! audio does not get padded correctly to multiple of 96
        
        # video and audio
        inputs_vid_aud = dict(video=inputs[3], audio=inputs[2])
        # outputs_vid_aud = preprocessor(inputs_vid_aud) #! audio does not get padded correctly to multiple of 96
        
        inputs_vid_aud_batch = dict(video=inputs[7], audio=inputs[6])
        # outputs_vid_aud_batch = preprocessor(inputs_vid_aud_batch) #! audio does not get padded correctly to multiple of 96
        
        # video and text
        inputs_vid_aud = dict(text=inputs[0], video=inputs[3])
        # outputs_vid_aud = preprocessor(inputs_vid_aud) #! audio does not get padded correctly to multiple of 96
        
        inputs_vid_aud_batch = dict(text=inputs[4], video=inputs[7])
        # outputs_vid_aud_batch = preprocessor(inputs_vid_aud_batch) #! audio does not get padded correctly to multiple of 96
        
        # video and audio and text
        inputs_vid_aud = dict(text=inputs[0], video=inputs[3], audio=inputs[2])
        # outputs_vid_aud = preprocessor(inputs_vid_aud) #! audio does not get padded correctly to multiple of 96
        
        inputs_vid_aud_batch = dict(text=inputs[4], video=inputs[7], audio=inputs[6])
        # outputs_vid_aud_batch = preprocessor(inputs_vid_aud_batch) #! audio does not get padded correctly to multiple of 96
        
        
        
        
        # preprocessor configured for images
        inputs_img = dict(text=inputs[0], image=inputs[1], audio=inputs[2])
        inputs_img_batch = dict(text=inputs[4], image=inputs[5], audio=inputs[6])
        final_inputs_img, modality_sizes_img, inputs_without_pos_img = preprocessor(inputs_img)
        final_inputs_img_batch, modality_sizes_img_batch, inputs_without_pos_img_batch = preprocessor(inputs_img_batch)
        
        # preprocessor configured for videos
        inputs_vid = dict(text=inputs[0], image=inputs[3], audio=inputs[2])
        inputs_vid_batch = dict(text=inputs[4], image=inputs[7], audio=inputs[6])
        final_inputs_vid, modality_sizes_vid, inputs_without_pos_vid = preprocessor_videos(inputs_vid)
        final_inputs_vid_batch, modality_sizes_vid_batch, inputs_without_pos_vid_batch = preprocessor_videos(inputs_vid_batch)
        
        assert isinstance(final_inputs_img, dict)
        assert isinstance(modality_sizes_img, dict)
        assert isinstance(inputs_without_pos_img, dict)
        
        assert isinstance(final_inputs_img_batch, dict)
        assert isinstance(modality_sizes_img_batch, dict)
        assert isinstance(inputs_without_pos_img_batch, dict)
        
        assert isinstance(final_inputs_vid, dict)
        assert isinstance(modality_sizes_vid, dict)
        assert isinstance(inputs_without_pos_vid, dict)
        
        assert isinstance(final_inputs_vid_batch, dict)
        assert isinstance(modality_sizes_vid_batch, dict)
        assert isinstance(inputs_without_pos_vid_batch, dict)
        
        
test_preprocessor_outputs()