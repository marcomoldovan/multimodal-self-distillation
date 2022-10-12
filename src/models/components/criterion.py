import torch
from torch import nn

class LatentPredictionLoss(nn.Module):
    def __init__(
        self,
        num_hidden_layers_to_predict: int,
        reduction: str = "mean",
        beta: float = 1.0        
        ) -> None:
        super().__init__()
        
        self.loss_fn = nn.SmoothL1Loss(reduction=reduction, beta=beta)
        
        self.num_hidden_layers_to_predict = num_hidden_layers_to_predict
        
    
    def forward(
        self, 
        hidden_states_student: torch.Tensor, 
        hidden_states_teacher: torch.Tensor,
        pooler_output_student: torch.Tensor,
        pooler_output_teacher: torch.Tensor,
        ) -> torch.Tensor:
        
        x = hidden_states_student[-1:][0]
        
        with torch.no_grad():
            # take the last k transformer layers from the teacher
            y = hidden_states_teacher[-self.num_hidden_layers_to_predict:] 
            # Follow the same layer normalization for all modalities
            y = [torch.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]
            y = sum(y) / len(y)
            if True: # normalize targets
                y = torch.layer_norm(y.float(), y.shape[-1:])
                
            hidden_states_loss = self.loss_fn(x, y)
            pooler_loss = self.loss_fn(pooler_output_student, pooler_output_teacher) #TODO: check if this is correct
            
            loss = hidden_states_loss + pooler_loss
                
        return loss