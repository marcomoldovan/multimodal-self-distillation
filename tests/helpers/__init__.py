import io
import requests
import torch
import numpy as np
import soundfile as sf
from PIL import Image
from transformers import PerceiverTokenizer, PerceiverFeatureExtractor, Wav2Vec2FeatureExtractor


def get_input_features():
    tokenizer = PerceiverTokenizer.from_pretrained('deepmind/language-perceiver')
    image_feature_extractor = PerceiverFeatureExtractor.from_pretrained('deepmind/vision-perceiver-conv')
    audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')
    
    text = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer(text, return_tensors='pt')['input_ids']
    
    try:
        url_image = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url_image, stream=True).raw)
        image_features = image_feature_extractor(image, return_tensors='pt')['pixel_values'].unsqueeze(0)
        
        url_audio = "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg.wav"
        audio = np.array(sf.read(io.BytesIO(requests.get(url_audio).content)))
        audio_features = audio_feature_extractor(audio, return_tensors='pt', sampling_rate=16000)['input_values'].unsqueeze(2)
        
        url_video = "https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4"
        video = 'something'
        video_features = 'something else'
    
    except:
        image_features = torch.randn((1, 3, 224, 22))
        audio_features = torch.randn((1, 85248, 1))
        video_features = torch.randn((1, 16, 3, 224, 224))
        
    token_batch = tokens.expand(32, -1)
    image_batch = image_features.expand(32, -1, -1, -1)
    audio_batch = audio_features.expand(32, -1, -1)
    video_batch = video_features.expand(32, -1, -1, -1, -1)
        
    return tokens, image_features, audio_features, video_features, token_batch, image_batch, audio_batch, video_batch