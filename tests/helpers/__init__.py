import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import io
import requests
import hydra
import torch
import numpy as np
import soundfile as sf
from PIL import Image
from decord import VideoReader, cpu
from einops import rearrange, reduce
from transformers import PerceiverTokenizer, PerceiverFeatureExtractor, Wav2Vec2FeatureExtractor

from src.utils import get_logger

log = get_logger(__name__)


def get_model_hidden_size() -> int:
    with hydra.initialize(version_base='1.2', config_path='../../configs/tests', job_name="test_preprocessor_instantiation"):
        cfg = hydra.compose(config_name='preprocessor')
        return cfg.get('input_preprocessor').get('modalities').get('text').get('d_model')


def get_input_features():
    tokenizer = PerceiverTokenizer.from_pretrained('deepmind/language-perceiver')
    feature_extractor = PerceiverFeatureExtractor.from_pretrained('deepmind/vision-perceiver-conv')
    audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')
    
    text = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer(text, return_tensors='pt')['input_ids']
    
    try:
        url_image = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url_image, stream=True).raw)
        image_features = reduce(feature_extractor(image, return_tensors='pt')['pixel_values'].unsqueeze(0), 'i b c h w -> b c h w', 'max')
        
        url_audio = "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg.wav"
        audio = np.array(sf.read(io.BytesIO(requests.get(url_audio).content)))[0] #TODO deprecated, fix this
        audio_features = audio_feature_extractor(audio, pad_to_multiple_of=96, padding='longest', return_tensors='pt', sampling_rate=16000)['input_values'].unsqueeze(2) 
        
        url_video = "https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4"
        video = rearrange(VideoReader(requests.get(url_video, stream=True).raw, ctx=cpu(0)).get_batch(range(0, 16)).asnumpy(), 'f h w c -> f c h w')
        video_features = feature_extractor(list(video), return_tensors='pt')['pixel_values'].unsqueeze(0)
        
        log.info("Successfully downloaded sample data and computed input features.")
    
    except Exception as e:
        log.error(f"Failed to create input features: {e}")
        log.error("Could not download the test files. Initializing with random tensors.")
        image_features = torch.randn((1, 3, 224, 224))
        audio_features = torch.randn((1, 85248, 1))
        video_features = torch.randn((1, 16, 3, 224, 224))
        
    token_batch = tokens.expand(32, -1)
    image_batch = image_features.expand(32, -1, -1, -1)
    audio_batch = audio_features.expand(32, -1, -1)
    video_batch = video_features.expand(32, -1, -1, -1, -1)
        
    return tokens, image_features, audio_features, video_features, token_batch, image_batch, audio_batch, video_batch
