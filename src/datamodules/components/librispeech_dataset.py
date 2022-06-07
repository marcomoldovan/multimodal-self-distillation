import torch
import numpy as np

from typing import Dict, List, Union
from torch.utils.data import Dataset

from transformers import BertTokenizerFast, Wav2Vec2FeatureExtractor



class LibriSpeechDataset(Dataset):
    def __init__(self, libri_dataset):
        self.libri_dataset = libri_dataset
    
    
    def __len__(self):
        return len(self.libri_dataset)
    
    
    def __getitem__(self, index):
        return self.libri_dataset[index]
    
  
  
class LibriCollator:
  def __init__(
    self,
    load_preprocessed_data=False,
    pretrained_speech_model="ntu-spml/distilhubert",
    speech_max_length=80000,
    pretrained_text_model="google/bert_uncased_L-2_H-768_A-12",
    text_max_length=32,
    ):
    
    self.load_preprocessed_data = load_preprocessed_data
    self.speech_max_length = speech_max_length
    self.text_max_length = text_max_length
    
    self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_speech_model)
    self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_text_model)
    
    
  def pad_latent_features(self, latent_features, padding='longest', return_tensors="pt"):
    padding_value = 0.0
    if padding == 'longest':
      longest_latent_feature = max(len(item['latent_features']) for item in latent_features)

    padded_features = []
    for item in latent_features:
      latent_features_as_ndarray = np.array(item['latent_features']).astype(np.float32)
      padded_item = np.pad(latent_features_as_ndarray, 
                           ((0, longest_latent_feature - latent_features_as_ndarray.shape[0]), (0, 0)), 
                           mode='constant', 
                           constant_values=padding_value)
      if return_tensors == "pt":
        padded_item = torch.from_numpy(padded_item).to(torch.float32)
      padded_features.append(padded_item)
      
    if return_tensors == "pt":
      padded_features = torch.stack(padded_features)
      
    return padded_features
  
  
  def collate_fn_for_latent_features(
    self, 
    batch: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
    """
    Collate function to be used when training with PyTorch Lightning.
    Args:
        batch(:obj:`List[Dict[str, Union[List[int], torch.Tensor]]]`):
            A list of features to be collated.
    Returns:
        :obj:`Dict[str, torch.Tensor]`: A dictionary of tensors containing the collated features.
    """ 
    latent_features = [{"latent_features": feature["latent_features"]} for feature in batch]
    input_sentences = [{"input_ids": feature["input_ids"]} for feature in batch]
    
    text_batch = self.tokenizer.pad(
        input_sentences,
        padding='longest',
        return_tensors='pt'
    )
    speech_batch = self.pad_latent_features(
        latent_features,
        padding='longest',
        return_tensors="pt",
        )
    
    return speech_batch, text_batch
  
  
  def collate_fn_for_input_values(
    self, 
    batch: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
    """
    Collate function to be used when training with PyTorch Lightning.
    Args:
        batch (:obj:`List[Dict[str, Union[List[int], torch.Tensor]]]`):
            A list of features to be collated.
    Returns:
        :obj:`Dict[str, torch.Tensor]`: A dictionary of tensors containing the collated features.
    """ 
    input_values = [{"input_values": feature["audio"]["array"]} for feature in batch]
    input_sentences = [feature["text"] for feature in batch]
    # input_sentences = [{"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask_text"]} for feature in batch]
    
    speech_batch = self.extractor.pad(
        input_values,
        padding='longest',
        max_length=self.speech_max_length,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
        )
    text_batch = self.tokenizer(
        input_sentences,
        padding='longest',
        max_length=self.text_max_length,
        truncation=True,
        return_tensors='pt'
    )
    
    return speech_batch, text_batch
  
  
  def __call__(self, batch):
    if self.load_preprocessed_data:
      speech_batch, text_batch = self.collate_fn_for_latent_features(batch)
    else:
      speech_batch, text_batch = self.collate_fn_for_input_values(batch)
    
    return speech_batch, text_batch
  