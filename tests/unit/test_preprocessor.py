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
    with hydra.initialize(version_base='1.1', config_path='../../configs/tests', job_name="test_preprocessor_instantiation"):
        cfg = hydra.compose(config_name='preprocessor')
        preprocessor = hydra.utils.instantiate(cfg.input_preprocessor)
        assert isinstance(preprocessor, PerceiverMultimodalPreprocessor)


def test_preprocessor_outputs():
    """
    Test whether the preprocessor outputs are correct.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/tests', job_name="test_preprocessor_outputs"):
        cfg = hydra.compose(config_name='preprocessor')
        preprocessor_images = hydra.utils.instantiate(cfg.input_preprocessor_img)
        preprocessor_videos = hydra.utils.instantiate(cfg.input_preprocessor_vid)
        
        inputs = get_input_features()
        
        inputs_img = dict(text=inputs[0], image=inputs[1], audio=inputs[2])
        inputs_img_batch = dict(text=inputs[4], image=inputs[5], audio=inputs[6])
        final_inputs_img, modality_sizes_img, inputs_without_pos_img = preprocessor_images(inputs_img)
        final_inputs_img_batch, modality_sizes_img_batch, inputs_without_pos_img_batch = preprocessor_images(inputs_img_batch)
        
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