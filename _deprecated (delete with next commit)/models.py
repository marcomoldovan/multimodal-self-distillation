from typing import Any, Optional
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn.modules.transformer import TransformerEncoder
from transformers import BertModel, Conv1D, HubertModel
from sentence_transformers.util import semantic_search
from torch.nn import TripletMarginWithDistanceLoss, Linear, Tanh
from metrics import reciprocal_ranks, mean_reciprocal_rank
from losses import InfoNceLoss, BYOL
from utils import freeze_model_except_last_n_layers



class HubertPooler(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.dense = Linear(self.config.hidden_size, self.config.hidden_size)
    self.activation = Tanh()
  
  
  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, _ = hidden_states.size()
    attention_mask = torch.ones(batch_size, sequence_length)
    
    # ? Does torch.mean(outputs['last_hidden_state'], dim=1) also work?
    
    output_vectors = []
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    output_vectors.append(sum_embeddings / sum_mask)
    output_vector = torch.cat(output_vectors, 0)
    
    if self.config.speech_output_pooling_strategy == 'pooling_layer':
      output_vector = self.activation(self.dense(output_vector))
      
    return output_vector



class HubertModelWithoutFeatureEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.hubert = HubertModel.from_pretrained(config.hubert_pretrained_name)
    self.feature_projection = self.hubert.feature_projection
    self.encoder = self.hubert.encoder
    self.pooler = HubertPooler(config)
    
  
  def forward(self, speech_features: torch.Tensor) -> torch.Tensor:
    # ? Maybe we need _mask_hidden_states() between projector and encoder 
    outputs = self.feature_projection(speech_features)
    outputs = self.encoder(outputs)
    outputs = self.pooler(outputs.last_hidden_state)
    
    return outputs



class SpeechAndTextBiEncoder(pl.LightningModule):
  def __init__(self, config):
    super().__init__()
    self.config = config
    
    # this line allows to access init params with 'self.hparams' attribute
    # it also ensures init params will be stored in ckpt
    self.save_hyperparameters()
    
    # models and model heads
    self.speech_model = HubertModelWithoutFeatureEncoder(config)
    self.text_model = BertModel.from_pretrained(config.bert_pretrained_name)
    freeze_model_except_last_n_layers(self.speech_model, self.text_model, config.train_last_n_speech_model_layers, config.train_last_n_text_model_layers)
    
    # loss function
    if config.pretraining_contrastive_loss_fn == 'TripletMarginLoss':
      #TODO set margin and distance function config
      self.triplet_loss = TripletMarginWithDistanceLoss()  
    elif config.pretraining_contrastive_loss_fn == 'InfoNceLoss':
      self.info_nce_loss = InfoNceLoss()
    elif config.pretraining_contrastive_loss_fn == 'BYOL':
      self.byol_loss = BYOL(target_network=self.text_model, online_network=self.speech_model)
    
    
    
  def forward(self, speech_input, text_input):
    speech_output = self.speech_model(speech_features=speech_input)
    text_output = self.text_model(input_ids=text_input['input_ids'], attention_mask=text_input['attention_mask'])
    return text_output, speech_output
  
  
  def training_step(self, batch, batch_idx):
    speech_input = batch[0]
    text_input = batch[1]
    
    text_anchors, speech_positives = self(speech_input, text_input)
    
    if self.config.pretraining_contrastive_loss_fn == 'TripletMarginLoss':
      speech_negatives = speech_positives['pooler_outputs'][torch.randperm(speech_positives['pooler_outputs'].shape[0]),:]
      loss = self.triplet_loss(text_anchors['pooler_outputs'], speech_positives['pooler_outputs'], speech_negatives)
    elif self.config.pretraining_contrastive_loss_fn == 'SimCLR':
      loss = self.info_nce_loss(text_anchors['pooler_outputs'], speech_positives['pooler_outputs'])
    elif self.config.pretraining_contrastive_loss_fn == 'BYOL':
      loss = self.byol_loss(text_anchors['pooler_outputs'], speech_positives['pooler_outputs'])
    
    return {f'{self.config.pretraining_contrastive_loss_fn}': loss}
  
  
  def validation_step(self, batch, batch_idx):
    speech_input = batch[0]
    text_input = batch[1]
    
    text_output, speech_output = self(speech_input, text_input)
    pairwise_similarity_results = semantic_search(text_output['pooler_output'], speech_output['pooler_output'])
    rs = reciprocal_ranks(pairwise_similarity_results)
    mrr_score = mean_reciprocal_rank(rs)
    
    return {f'MRR@{self.config.val_batch_size}': mrr_score}
  
  
  def test_step(self, batch, batch_idx):
    speech_input = batch[0]
    text_input = batch[1]
    
    text_output, speech_output = self(speech_input, text_input)
    pairwise_similarity_results = semantic_search(text_output['pooler_output'], speech_output['pooler_output'])
    rs = reciprocal_ranks(pairwise_similarity_results)
    mrr_score = mean_reciprocal_rank(rs)
    
    return {f'MRR@{self.config.test_batch_size}': mrr_score}
  
  
  def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
    #TODO implement so it's compatible with the SpotifyPredictionDataModule
      return super().predict_step(batch, batch_idx, dataloader_idx)
  
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    return (
      {
        "optimizer": optimizer,
        "lr_scheduler": 
          {
            "scheduler": scheduler,
            "interval": "epoch",
            "monitor": "mrr_score"
          }
      }
    )    
    
  
  
  
class CrossModalLanguageModel(pl.LightningModule):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.conv_feature_extractor = Conv1D()
    self.multimodal_encoder = TransformerEncoder()
    
    self.save_hyperparameters()
    
    
  def forward(self, speech_input, text_input):
    speech_output = self.speech_model(speech_input)
    text_output = self.language_model(text_input)
    return text_output, speech_output
  
  
  def training_step(self, batch, batch_idx):
    if self.config["contrastive_loss"] == "SimCLR":
      pass
    elif self.config["contrastive_loss"] == "TripletMarginWithDistance":
      pass
    else:
      raise ValueError("Invalid contrastive loss")
    
    language_input, speech_input = batch
    text_output, speech_output = self(language_input, speech_input)
    loss = torch.mean(torch.abs(text_output - speech_output))
    return {'train_loss': loss}
  
  
  def validation_step(self, batch, batch_idx):
    language_input, speech_input = batch
    text_output, speech_output = self(language_input, speech_input)
    loss = torch.mean(torch.abs(text_output - speech_output))
    return {'val_loss': loss}
  
  
  def test_step(self, batch, batch_idx):
    language_input, speech_input = batch
    text_output, speech_output = self(language_input, speech_input)
    loss = torch.mean(torch.abs(text_output - speech_output))
    return {'test_loss': loss}
  
  
  def configure_optimizers(self):
    #TODO add scheduler to optimizer config or return list of optimizers and schedulers
    return torch.optim.Adam(self.parameters(), lr=1e-3)
  
  
  def save_hyperparameters():
    #TODO check whether this is necessary when using wandb
    pass
    
    
  def load_from_checkpoint(self, checkpoint_path: str) -> None:
    #TODO is this redundant? does the default checkpoint loading work?
    return super().load_from_checkpoint(checkpoint_path)
  
