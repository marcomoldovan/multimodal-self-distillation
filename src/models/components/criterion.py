import math
import torch
from torch import nn
import torch.nn.functional as F

from src.models.components.gather import gather
from src.models.components.outputs import ForwardPassOutput, CriterionOutput

class LatentPredictionLoss(nn.Module):
    def __init__(
        self,
        num_hidden_layers_to_predict: int,
        reduction: str = "none",
        aggregation: str = "mean",
        beta: float = 1.0,
        latent_loss_scale: float = 1.0,
        batch_norm_target_layer:bool = True,
        instance_norm_target_layer: bool = True,
        layer_norm_target_layer: bool = True,
        layer_norm_targets: bool = True,
        instance_norm_targets: bool = True,
        sim_loss_weight: float = 25.0,
        var_loss_weight: float = 25.0,
        cov_loss_weight: float = 1.0,
        ) -> None:
        super().__init__()
        
        self.has_faiss_format = False
        self.batch_norm_target_layer = batch_norm_target_layer
        self.instance_norm_target_layer = instance_norm_target_layer
        self.layer_norm_target_layer = layer_norm_target_layer
        self.layer_norm_targets = layer_norm_targets
        self.instance_norm_targets = instance_norm_targets
        
        self.reduction = reduction
        self.aggregation = aggregation
        self.beta = beta
        self.latent_loss_scale = latent_loss_scale
        
        self.align_loss_fn = VICRegLoss(sim_loss_weight, var_loss_weight, cov_loss_weight)
        
        self.k = num_hidden_layers_to_predict
        
    
    def forward(
        self,
        fwd_output: ForwardPassOutput,
        ) -> CriterionOutput:
        
        # take the last transformer layers from the student: (batch size, sequence length, hidden size)
        x = fwd_output.student_output.hidden_states[-1:][0] 
        #TODO optionally: x = regression_head(x)
    
        with torch.no_grad():
            
            # (batch_size, sequence_length, hidden_size) * attention_layers
            y = fwd_output.teacher_output.hidden_states[-self.k:]

            # B: batch size, T: sequence length, C: hidden size

            if not self.has_faiss_format:
                y = [tl.permute(1, 0, 2) for tl in y] # BTC -> TBC

            permuted = False
            if  self.batch_norm_target_layer or self.instance_norm_target_layer:
                y = [tl.permute(1, 2, 0) for tl in y]  # TBC -> BCT
                permuted = True

            if self.batch_norm_target_layer:
                y = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in y
                ]

            if self.instance_norm_target_layer:
                y = [F.instance_norm(tl.float()) for tl in y]

            if permuted:
                y = [tl.transpose(1, 2) for tl in y]  # BCT -> BTC

            if self.layer_norm_target_layer:
                y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]

            y = sum(y) / len(y)

            if not permuted:
                y = y.transpose(0, 1)

            if self.layer_norm_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])

            if self.instance_norm_targets:
                y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)
                
        sz = x.size(-1)

        latent_loss = F.smooth_l1_loss(
                        x.float(), y.float(), reduction=self.reduction, beta=self.beta
                    ).sum(dim=-1)
        
        if self.aggregation == 'mean':
            latent_loss = latent_loss.mean() / math.sqrt(sz) if self.latent_loss_scale <= 0 else latent_loss.mean() * self.latent_loss_scale
        elif self.aggregation == 'sum':
            latent_loss = latent_loss.sum() / math.sqrt(sz) if self.latent_loss_scale <= 0 else latent_loss.sum() * self.latent_loss_scale
        
        #TODO add option of not using pooler loss at all to see if it's dominating convergence
        # pooler loss (batch size, hidden size)                
        x_pooler = fwd_output.student_output.pooler_output
        y_pooler = fwd_output.teacher_output.pooler_output
        align_loss = self.align_loss_fn(x_pooler, y_pooler)
        
        total_loss = latent_loss + align_loss
        
        criterion_output = CriterionOutput(total_loss=total_loss, latent_loss=latent_loss, align_loss=align_loss)
        
        return criterion_output
                
                

class VICRegLoss(nn.Module):
    # https://github.com/vturrisi/solo-learn/blob/main/solo/losses/vicreg.py
    def __init__(
        self,
        sim_loss_weight: float = 25.0,
        var_loss_weight: float = 25.0,
        cov_loss_weight: float = 1.0,
        ) -> None:
        """_summary_

        Args:
            sim_loss_weight (float, optional): _description_. Defaults to 25.0.
            var_loss_weight (float, optional): _description_. Defaults to 25.0.
            cov_loss_weight (float, optional): _description_. Defaults to 1.0.
        """
        super().__init__()
        
        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight

    
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
        z2: torch.Tensor
    ) -> torch.Tensor:
        """Computes VICReg's loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: VICReg loss.
        """

        sim_loss = self.invariance_loss(z1, z2)

        # vicreg's official code gathers the tensors here
        # https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
        z1, z2 = gather(z1), gather(z2)

        var_loss = self.variance_loss(z1, z2)
        cov_loss = self.covariance_loss(z1, z2)

        loss = self.sim_loss_weight * sim_loss + self.var_loss_weight * var_loss + self.cov_loss_weight * cov_loss
        
        return loss
    