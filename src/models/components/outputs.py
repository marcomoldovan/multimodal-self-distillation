import torch

# output classes for bi-encoder and mm-encoder account for flexibility in case of additional byol or data2vec outputs

class ModelOutput:
    def __init__(
        self,
        last_hidden_state: torch.Tensor,
        hidden_states: torch.Tensor,
        attentions: torch.Tensor,
        cross_attentions: torch.Tensor    
    ) -> None:
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.cross_attentions = cross_attentions
        

class ForwardPassOutput:
    def __init__(
        self,
        student_output: ModelOutput = None,
        teacher_output: ModelOutput = None,
        align_fuse: dict = None,
    ) -> None:
        self.student_output = student_output
        self.teacher_output = teacher_output
        self.align_fuse = align_fuse