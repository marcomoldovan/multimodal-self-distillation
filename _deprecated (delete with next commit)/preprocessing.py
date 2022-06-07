from typing import Dict, List, Union
import torch
import numpy as np
import soundfile as sf
from transformers import BertTokenizerFast, Wav2Vec2FeatureExtractor, Wav2Vec2Model

class LibriPreprocessor:
  def __init__(self, config):
    self.config = config
    self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    self.extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')
    if self.config.include_conv_module_in_preprocessing:
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.feature_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base').feature_extractor
      if torch.cuda.is_available() and self.config.use_gpu_for_preprocessing:
        self.feature_encoder.to(self.device)
      self.feature_encoder.eval()
  
  
  def speech_file_to_array_fn(self, data):
    speech_array, sampling_rate = sf.read(data["file"])
    data["speech"] = speech_array
    data["sampling_rate"] = sampling_rate
    data["target_text"] = data["text"]
    return data
    
    
  def prepare_dataset(self, data):    
    # check that all files have the correct sampling rate
    assert (
        len(set(data["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {self.extractor.feature_extractor.sampling_rate}."

    data["input_values"] = self.extractor(data["speech"], sampling_rate=data["sampling_rate"][0]).input_values
    
    tokenized_batch = self.tokenizer(data["target_text"], padding='longest', max_length=128, pad_to_max_length=False)
    data['input_ids'] = tokenized_batch['input_ids']
    data['attention_mask_text'] = tokenized_batch['attention_mask']
    data['token_type_ids_text'] = tokenized_batch['token_type_ids']
    
    return data
  
  
  def prepare_dataset_with_conv_module(self, data):
    # check that all files have the correct sampling rate
    assert (
        len(set(data["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {self.extractor.feature_extractor.sampling_rate}."

    input_values = self.extractor(data["speech"], sampling_rate=data["sampling_rate"][0])
    data["input_values"] = input_values.input_values
    padded_input_values = self.extractor.pad(input_values, return_tensors="pt")
    
    import torch
    with torch.no_grad():
      input_values = padded_input_values['input_values'].to(self.device)
      latent_features = self.feature_encoder(input_values).transpose(1, 2)
      latent_features = latent_features.cpu().numpy()
      data['latent_features'] = latent_features
    
    tokenized_batch = self.tokenizer(data["target_text"], padding='longest', max_length=128, pad_to_max_length=False)
    data['input_ids'] = tokenized_batch['input_ids']
    data['attention_mask_text'] = tokenized_batch['attention_mask']
    data['token_type_ids_text'] = tokenized_batch['token_type_ids']
    
    return data
  
  
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
    input_features = [{"input_values": feature["input_values"]} for feature in batch]
    input_sentences = [{"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask_text"]} for feature in batch]
    
    speech_batch = self.extractor.pad(
        input_features,
        padding='longest',
        return_tensors="pt",
        )
    text_batch = self.tokenizer.pad(
        input_sentences,
        padding='longest',
        return_tensors='pt'
    )
    
    return speech_batch, text_batch
  
  
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


  def __call__(
    self,
    batch: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
    """
    Collate function to be used when training with PyTorch Lightning.
    Args:
        extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        tokenizer (:class:`~transformers.BertTokenizerFast`)
            The tokenizer used for proccessing the data.
        features (:obj:`List[Dict[str, Union[List[int], torch.Tensor]]]`):
            A list of features to be collated.
    Returns:
        :obj:`Dict[str, torch.Tensor]`: A dictionary of tensors containing the collated features.
    """ 
    if self.config.include_conv_module_in_preprocessing:
      speech_batch, text_batch = self.collate_fn_for_latent_features(batch)
    else:
      speech_batch, text_batch = self.collate_fn_for_input_values(batch)
      
    return speech_batch, text_batch
  
  
 
#TODO use inheritence for differnt preprocessors? 
class  SpokenSquadPreprocessor:
  def __init__(self, config):
    pass
  


def collate_fn_spotify(batch):
  pass