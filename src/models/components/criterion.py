import torch
from torch import nn
import torch.nn.functional as F

from src.models.components.gather import gather
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
        
        #TODO is this the same as the mean pooling in the pooler?
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
                
        hidden_states_loss = self.loss_fn(x, y) #TODO should x be the student pooler output? Here x is the output of the regression head: https://github.com/arxyzan/data2vec-pytorch/blob/main/data2vec/data2vec.py
        
        x_pooler = fwd_output.student_output.pooler_output
        y_pooler = fwd_output.teacher_output.pooler_output
        pooler_loss = self.loss_fn(x_pooler, y_pooler)
        
        loss = hidden_states_loss + pooler_loss
        
        return loss
                
                

class VICRegLoss(nn.Module):
    # https://github.com/vturrisi/solo-learn/blob/main/solo/losses/vicreg.py
    def __init__(self) -> None:
        super().__init__()

    
    def invariance_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes mse loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: invariance loss (mean squared error).
        """

        return F.mse_loss(z1, z2)


    def variance_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes variance loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: variance regularization loss.
        """

        eps = 1e-4
        std_z1 = torch.sqrt(z1.var(dim=0) + eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + eps)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
        return std_loss


    def covariance_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes covariance loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: covariance regularization loss.
        """

        N, D = z1.size()

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (N - 1)
        cov_z2 = (z2.T @ z2) / (N - 1)

        diag = torch.eye(D, device=z1.device)
        cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D
        return cov_loss


    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        sim_loss_weight: float = 25.0,
        var_loss_weight: float = 25.0,
        cov_loss_weight: float = 1.0,
    ) -> torch.Tensor:
        """Computes VICReg's loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
            sim_loss_weight (float): invariance loss weight.
            var_loss_weight (float): variance loss weight.
            cov_loss_weight (float): covariance loss weight.
        Returns:
            torch.Tensor: VICReg loss.
        """

        sim_loss = self.invariance_loss(z1, z2)

        # vicreg's official code gathers the tensors here
        # https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
        z1, z2 = gather(z1), gather(z2)

        var_loss = self.variance_loss(z1, z2)
        cov_loss = self.covariance_loss(z1, z2)

        loss = sim_loss_weight * sim_loss + var_loss_weight * var_loss + cov_loss_weight * cov_loss
        return loss