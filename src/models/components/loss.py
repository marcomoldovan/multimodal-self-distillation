import torch

import torch.nn as nn
import torch.nn.functional as F

from src.models.components.outputs import ModelOutputs




class AdaptiveCriterion(nn.Module):
    def __init__(
        self, 
        criterion: str = 'TripletLoss'
        ):
        """_summary_

        Args:
            criterion (str, optional): What criterion to generate.
                Options: 'TripletLoss', 'InfoNCE', 'BYOL', 'data2vec'
                Default: 'TripletLoss'
        """
        
        super().__init__()
        
        self.criterion = criterion
        
        if criterion == 'TripletLoss':
            self.loss_fn = nn.TripletMarginWithDistanceLoss()
        elif criterion == 'InfoNCE':
            self.loss_fn = InfoNCELoss()
    
    
    def forward(self, model_outputs: ModelOutputs) -> torch.Tensor:
        if self.criterion == 'TripletLoss':
            anchors = model_outputs.text_pooler_output
            positives = model_outputs.speech_pooler_output
            negatives = positives[torch.randperm(positives.shape[0]),:]
            loss = self.loss_fn(anchors, positives, negatives)
        elif self.criterion == 'InfoNCE':
            loss = self.loss_fn(anchors, positives)
        
        return loss
    
    
    
class InfoNCELoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """
    def __init__(
        self, 
        batch_size, 
        temperature=0.5
        ):
        
        super().__init__()
        
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
        
        
    def device_as(t1, t2):
        """
        Moves t1 to the device of t2
        """
        return t1.to(t2.device)
    

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)


    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = self.device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss