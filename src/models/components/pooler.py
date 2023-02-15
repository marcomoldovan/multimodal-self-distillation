import torch
from torch import nn


class Pooler(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        projection_size: int,
        widening_factor: int = 4,
        use_simsiam_mlp: bool = False
        ):
        #TODO there's a difference in the regression head from faiss data2vec and a BERT style pooler
        #NOTE BatchNorm1d is not used in the faiss regression head and does not work on 3D tensors
        super().__init__()
        
        hidden_size = dim_in * widening_factor
        
        #TODO add option of not using a projection head and just returning the mean-pooled vector
        if use_simsiam_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(dim_in, hidden_size, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, projection_size, bias=False),
                nn.BatchNorm1d(projection_size, affine=False)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim_in, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, projection_size)
            )
    
    
    def forward(
        self, 
        last_hidden_state: torch.Tensor = None,
        ) -> torch.Tensor:
                
        batch_size, sequence_length, _ = last_hidden_state.size()
        attention_mask = torch.ones(batch_size, sequence_length)
        
        output_vectors = []
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()).float().to(last_hidden_state.device
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        
        output_vectors.append(sum_embeddings / sum_mask)
        output_vector = torch.cat(output_vectors, 0)
        
        return self.mlp(output_vector) # batch size, projection size
        