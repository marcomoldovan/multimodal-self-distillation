from turtle import forward
from typing import Optional, Union, Tuple
import torch
from torch import nn
from transformers import PerceiverModel
from transformers.models.perceiver.modeling_perceiver import PerceiverModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from src.models.components.outputs import ModelOutputs


class PerceiverEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        
    def forward(self):
        raise NotImplementedError