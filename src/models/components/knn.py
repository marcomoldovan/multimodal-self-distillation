from typing import Tuple
import torch
from torch.nn import functional as F


def k_nearest_neighbor(
    prediction_features: torch.Tensor, 
    query_features: torch.Tensor = None, 
    labels: torch.Tensor = None, 
    num_classes: int = 1000, 
    k: int = 20, 
    temperature: float = 0.1
    ) -> Tuple(torch.Tensor, torch.Tensor, torch.Tensor):
    
    probabilities = []
    predictions = []
        
    num_classes = len(set(list(labels.numpy())))
    
    if query_features is None:
        # means that similarity is computed between prediction features and itself
        query_features = prediction_features
        zero_diagonal = True
        trim_preds = False
    else:
        zero_diagonal = False
        trim_preds = True
        
    num_chunks = 100
    num_test_samples = query_features.size()[0]
    samples_per_chunk = num_test_samples // num_chunks
        
    for idx in range(0, num_test_samples, samples_per_chunk):
        
        chunk_features = query_features[
            idx : min((idx + samples_per_chunk), num_test_samples), :
        ]
        chunk_labels = labels[
            idx : min((idx + samples_per_chunk), num_test_samples)
        ]
        
        batch_size = chunk_labels.shape[0]
        
        similarity = F.normalize(chunk_features) @ F.normalize(prediction_features).t() 
        torch.diagonal(similarity, 0).zero_() if zero_diagonal else None
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)
        
        retrieval_one_hot = torch.zeros(batch_size * k, num_classes)
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = (distances / temperature).exp_()
        
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        probs.div_(probs.sum(dim=1, keepdim=True))
        probs_sorted, preds = probs.sort(1, True)
        
        probabilities.append(probs)
        predictions.append(preds)
    
    probabilities = torch.cat(probabilities, dim=0)
    predictions = torch.cat(predictions, dim=0)
    
    return probabilities, predictions, labels
