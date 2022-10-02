import torch


def reciprocal_ranks(pairwise_similarity_results):
    indexes = []
    targets = []
    for i, result in enumerate(pairwise_similarity_results):
        for entry in result:
            indexes.append(i)
            if entry['corpus_id'] == i:
                targets.append(1)
            else:
                targets.append(0)
    
    preds = [0] * len(targets)
    
    indexes_tensor = torch.LongTensor(indexes)
    targets_tenosr = torch.Tensor(targets)
    preds_tensor = torch.Tensor(preds)
    
    return indexes_tensor, targets_tenosr, preds_tensor

#TODO is it possible to use the same class for all metrics?
# https://torchmetrics.readthedocs.io/en/stable/pages/overview.html
# https://torchmetrics.readthedocs.io/en/stable/wrappers/metric_tracker.html
class PretrainingMetric:
    def __init__(self):
        self.train
        self.val
        self.test
        