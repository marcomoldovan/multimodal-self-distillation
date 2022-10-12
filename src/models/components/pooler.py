import torch


class Pooler(torch.nn.Module):
    def __init__(
        self,
        hidden_size_in: int,
        hidden_size_out: int,
        ):
        
        super().__init__()
        
        self.dense = torch.nn.Linear(hidden_size_in, hidden_size_out)
        self.activation = torch.nn.Tanh()
    
    
    def forward(
        self, 
        last_hidden_state: torch.Tensor = None,
        ) -> torch.Tensor:
                
        batch_size, sequence_length, _ = last_hidden_state.size()
        attention_mask = torch.ones(batch_size, sequence_length)
        
        output_vectors = []
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float().to(last_hidden_state.device)
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        
        output_vectors.append(sum_embeddings / sum_mask)
        output_vector = torch.cat(output_vectors, 0)
        
        output_vector = self.activation(self.dense(output_vector))
        
        return output_vector