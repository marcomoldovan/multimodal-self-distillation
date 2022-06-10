import torch
from torch import nn

class LatentPredictionLoss(nn.Module):
    def __init__(
        self,
        num_hidden_layers_to_predict: int,
        reduction: str = "mean",
        beta: float = 1.0        
        ) -> None:
        
        self.loss_fn = nn.SmoothL1Loss(reduction=reduction, beta=beta)
        
        self.num_hidden_layers_to_predict = num_hidden_layers_to_predict
        
    
    def forward(
        self, 
        hidden_states_student: torch.Tensor, 
        hidden_states_teacher: torch.Tensor
        ) -> torch.Tensor:
        raise NotImplementedError