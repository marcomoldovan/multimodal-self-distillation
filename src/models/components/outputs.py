import torch

# output classes for bi-encoder and mm-encoder account for flexibility in case of additional byol or data2vec outputs

class ModelOutputs:
    def __init__(
        self,
        speech_pooler_output: torch.Tensor = None,
        speech_model_hidden_states: torch.Tensor = None,
        speech_model_attentions: torch.Tensor = None,
        text_pooler_output: torch.Tensor = None,
        text_model_hidden_states: torch.Tensor = None,
        text_model_attentions: torch.Tensor = None,
        multimodal_model_hidden_states: torch.Tensor = None,
        multimodal_model_attentions: torch.Tensor = None,
    ):
        # speech encoder outputs
        self.speech_pooler_output = speech_pooler_output
        self.speech_model_hidden_states = speech_model_hidden_states
        self.speech_model_attentions = speech_model_attentions
        # text encoder outputs
        self.text_pooler_output = text_pooler_output
        self.text_model_hidden_states = text_model_hidden_states
        self.text_model_attentions = text_model_attentions
        # multimodal encoder outputs
        self.multimodal_model_hidden_states = multimodal_model_hidden_states
        self.multimodal_model_attentions = multimodal_model_attentions