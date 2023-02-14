import torch

# output classes for bi-encoder and mm-encoder account for flexibility in case of additional byol or data2vec outputs

class DispatcherOutput:
    def __init__(
        self,
        student_input, 
        teacher_inputs, 
        align_fuse, 
        apply_mask: bool, 
        labels: torch.Tensor, 
        output_modalities: dict, 
        metric: str, 
        num_classes: int,
    ) -> None:
        self.student_input = student_input
        self.teacher_inputs = teacher_inputs
        self.align_fuse = align_fuse
        self.apply_mask = apply_mask
        self.labels = labels
        self.output_modalities = output_modalities
        self.metric = metric
        self.num_classes = num_classes
        
    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class ModelOutput:
    def __init__(
        self,
        pooler_output: torch.Tensor,
        last_hidden_state: torch.Tensor,
        hidden_states: torch.Tensor,
        attentions: torch.Tensor,
        cross_attentions: torch.Tensor    
    ) -> None:
        self.pooler_output = pooler_output
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.cross_attentions = cross_attentions
        
    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        
class CriterionOutput:
    def __init__(
        self,
        total_loss: torch.Tensor,
        latent_loss: torch.Tensor = None,
        align_loss: torch.Tensor = None,
    ) -> None:
        self.total_loss = total_loss
        self.latent_loss = latent_loss
        self.align_loss = align_loss
        
    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class ForwardPassOutput:
    def __init__(
        self,
        student_output: ModelOutput = None,
        teacher_output: ModelOutput = None,
        align_fuse: dict = None,
        labels: torch.Tensor = None,
        output_modalities: dict = None,
        metric: str = None,
        num_classes: int = None,
        criterion_output: CriterionOutput = None,
    ) -> None:
        self.student_output = student_output
        self.teacher_output = teacher_output
        self.align_fuse = align_fuse
        self.labels = labels
        self.output_modalities = output_modalities
        self.metric = metric
        self.num_classes = num_classes,
        self.criterion_output = criterion_output
        
    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            