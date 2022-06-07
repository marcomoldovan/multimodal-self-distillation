from typing import Optional, Union, Tuple
import torch
from torch import nn
from transformers import HubertModel, BertModel, PretrainedConfig
from transformers.models.hubert.modeling_hubert import HubertPreTrainedModel, HubertFeatureEncoder, HubertFeatureProjection, HubertEncoder
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from src.models.components.outputs import ModelOutputs


    
class BertEmbeddingsWrapper(BertPreTrainedModel):
    def __init__(
        self,
        config
        ):
        super().__init__(config)
        
        self.embeddings = BertEmbeddings(config)
        
        self.post_init()
    
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
        ) -> torch.Tensor:
        
        return self.embeddings(input_ids, token_type_ids, position_ids, inputs_embeds, past_key_values_length)
    
    
    
class BertEncoderWrapper(BertPreTrainedModel):
    def __init__(
        self,
        config
        ):
        super().__init__(config)
        
        self.encoder = BertEncoder(config)
        
        self.post_init()
    
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        
        return encoder_outputs
    
    
    
class BertPoolerWrapper(BertPreTrainedModel):
    def __init__(
        self, 
        config):
        super().__init__(config)
        
        self.pooler = BertPooler(config)
        
        self.post_init()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.pooler(hidden_states)
    
    

class HubertConvFeatureExtractorWrapper(HubertPreTrainedModel):
    # named HubertFeatureEncoder on huggingface
    def __init__(
        self,
        config
        ):
        super().__init__(config)
        
        self.feature_extractor = HubertFeatureEncoder(config)
        
        self.post_init()
    
    
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(input_values)
    
    
class FeatureProjection(nn.Module):
    def __init__(
        self, 
        speech_config, 
        text_config
        ):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(speech_config.conv_dim[0])
        self.projection = nn.Linear(speech_config.conv_dim[0], text_config.hidden_size)
        self.dropout = nn.Dropout(speech_config.feat_proj_dropout)    
    
    
    def forward(self):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states
    
    
    
class HubertFeatureProjectionWrapper(HubertPreTrainedModel):
    def __init__(
        self,
        config
        ):
        super().__init__(config)
        
        self.feature_projection = HubertFeatureProjection(config)
        
        self.post_init()
    
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.feature_projection(hidden_states)
    
    

class HubertEncoderWrapper(HubertPreTrainedModel):
    def __init__(
        self,
        config
        ):
        super().__init__(config)
        
        self.encoder = HubertEncoder(config)
        
        self.post_init()
    
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True):
        
        outputs = self.encoder(
            hidden_states,
            attention_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        
        return outputs
    
    
    
class HubertPooler(nn.Module):
    def __init__(
        self,
        hidden_size_in: int = 768,
        hidden_size_out: int = 768,
        ):
        
        super().__init__()
        
        self.dense = nn.Linear(hidden_size_in, hidden_size_out)
        self.activation = nn.Tanh()
    
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
        ) -> torch.Tensor:
        
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
        
        #TODO condition whether to execture following line on config setting
        output_vector = self.activation(self.dense(output_vector))
        
        return output_vector



class HubertModelWithoutFeatureEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model: str = 'ntu-spml/distilhubert',
        hidden_size_in: int = 768,
        hidden_size_out: int = 768
        ):
        
        super().__init__()
        
        self.projector = HubertFeatureProjectionWrapper.from_pretrained(pretrained_model)
        self.encoder = HubertEncoderWrapper.from_pretrained(pretrained_model)
        self.pooler = HubertPooler(hidden_size_in, hidden_size_out)
        
    
    def forward(self, speech_features: torch.Tensor) -> torch.Tensor:
        #TODO adjust inputs, feed attention mask correctly
        outputs = self.projector(speech_features)
        outputs = self.encoder(outputs)
        outputs = self.pooler(outputs.last_hidden_state)
        
        return outputs
    
    

class HubertModelWithPooler(nn.Module):
    def __init__(
        self,
        pretrained_model: str = 'ntu-spml/distilhubert',
        pooler_output_size: int = 768,
        ):
        
        super().__init__()
        
        self.hubert = HubertModel.from_pretrained(pretrained_model)
        self.pooler = HubertPooler(self.hubert.config.hidden_size, pooler_output_size)
        
        
    def forward(self, speech_features: torch.Tensor) -> torch.Tensor:
        outputs = self.hubert(input_values=speech_features['input_values'], attention_mask=speech_features['attention_mask'])
        outputs = self.pooler(hidden_states=outputs.last_hidden_state, attention_mask=speech_features['attention_mask'])
        
        return outputs
    
    
    
class BiEncoderSpeechTextModelWithoutFeatureEncoder(nn.Module):
    def __init__(
        self,
        pretrained_speech_model: str = 'ntu-spml/distilhubert',
        pretrained_text_model: str = 'google/bert_uncased_L-2_H-768_A-12',
        ):
        
        super().__init__()
        
        self.text_model = BertModel.from_pretrained(pretrained_text_model)
        self.speech_model = HubertModelWithoutFeatureEncoder(pretrained_speech_model, hidden_size_out=self.text_model.config.hidden_size)
        
    
    def forward(self, speech, text):
        #TODO outsource this to before_sanity_check callback
        try:
            assert 'latent_features' in speech.keys()
        except:
            print("No latent features passed, use BiEncoderSpeechTextModel or MultiModalSpeechTextEncoder instead.")
        
        speech_representations = self.speech_model(speech)
        text_representations = self.text_model(
            input_ids=text['input_ids'], 
            attention_mask=text['attention_mask'], 
            token_type_ids=text['token_type_ids']
            ).pooler_outputs
        
        outputs = ModelOutputs(speech_pooler_output=speech_representations, 
                               text_pooler_output=text_representations)
        
        return outputs
    
    
    
class BiEncoderSpeechTextModel(nn.Module):
    def __init__(
        self,
        pretrained_speech_model: str = 'ntu-spml/distilhubert',
        pretrained_text_model: str = 'google/bert_uncased_L-2_H-768_A-12',
    ):
        
        super().__init__()
        
        self.text_model = BertModel.from_pretrained(pretrained_text_model)
        self.speech_model = HubertModelWithPooler(pretrained_speech_model, self.text_model.config.hidden_size)
        
        
    def forward(self, speech, text):
        speech_representations = self.speech_model(speech)
        text_representations = self.text_model(
            input_ids=text['input_ids'], 
            attention_mask=text['attention_mask'], 
            token_type_ids=text['token_type_ids']
            ).pooler_output
        
        outputs = ModelOutputs(speech_pooler_output=speech_representations, 
                               text_pooler_output=text_representations)
        
        return outputs
    
    
    
    
class MultiModalSpeechTextEncoder(nn.Module):
    def __init__(
        self, 
        pretrained_embedding_model: str = 'google/bert_uncased_L-8_H-256_A-4',
        pretrained_feature_extractor_model: str = 'ntu-spml/distilhubert',
        pretrained_transformer_model: str = 'google/bert_uncased_L-8_H-256_A-4',
        pooler_output_size: int = 256,
        ):
        
        super().__init__()
        
        self.pretrained_bert_config = PretrainedConfig.from_pretrained(pretrained_transformer_model)
        self.pretrained_hubert_config = PretrainedConfig.from_pretrained(pretrained_feature_extractor_model)
        
        self.token_embedding = BertEmbeddingsWrapper.from_pretrained(pretrained_embedding_model)
        self.feature_extractor = HubertConvFeatureExtractorWrapper.from_pretrained(pretrained_feature_extractor_model)
        
        if self.pretrained_bert_config.hidden_size != self.pretrained_hubert_config.hidden_size:
            self.feature_projection = FeatureProjection(self.pretrained_hubert_config, self.pretrained_bert_config)
        else:
            self.feature_projection = HubertFeatureProjectionWrapper.from_pretrained(pretrained_feature_extractor_model)
                
        self.transformer = BertEncoderWrapper.from_pretrained(pretrained_transformer_model)
        
        self.speech_pooler = HubertPooler(self.pretrained_bert_config.hidden_size, pooler_output_size)
        self.text_pooler = BertPoolerWrapper.from_pretrained(pretrained_transformer_model)
        

    def forward(self, speech, text):
        speech_hidden_states = self.feature_extractor(speech)
        speech_hidden_states = self.feature_projection(speech_hidden_states)
        
        text_hidden_states = self.token_embedding(text)
        
        speech_encoder_outputs = self.transformer(speech_hidden_states[0])
        text_encoder_outputs = self.transformer(text_hidden_states)
                
        speech_pooler_output = self.speech_pooler(speech_encoder_outputs.last_hidden_state)
        text_pooler_output = self.text_pooler(text_encoder_outputs.last_hidden_state)
        
        outputs = ModelOutputs(speech_pooler_output=speech_pooler_output, text_pooler_output=text_pooler_output)
        
        return outputs
