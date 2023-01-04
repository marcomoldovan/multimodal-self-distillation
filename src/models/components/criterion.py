import torch
from torch import nn

from src.models.components.outputs import ForwardPassOutput

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
        fwd_output: ForwardPassOutput,
        ) -> torch.Tensor:
        
        # take the last transformer layers from the student
        x = fwd_output.student_output.hidden_states[-1:][0] 
        # Follow the same layer normalization for all modalities
        x = [torch.layer_norm(tl.float(), tl.shape[-1:]) for tl in x]
        x = sum(x) / len(x)
        # normalize targets
        x = torch.layer_norm(x.float(), x.shape[-1:])
    
        
        
        with torch.no_grad():
            # take the last k transformer layers from the teacher
            y = fwd_output.teacher_output.hidden_states[-self.num_hidden_layers_to_predict:]
            # Follow the same layer normalization for all modalities
            y = [torch.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]
            y = sum(y) / len(y)
            # normalize targets
            y = torch.layer_norm(y.float(), y.shape[-1:])
                
        hidden_states_loss = self.loss_fn(x, y)
        
        x_pooler = fwd_output.student_output.pooler_output
        y_pooler = fwd_output.teacher_output.pooler_output
        pooler_loss = self.loss_fn(x_pooler, y_pooler)
        
        loss = hidden_states_loss + pooler_loss
        
        return loss
                