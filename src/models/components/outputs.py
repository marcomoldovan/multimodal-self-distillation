import torch

# output classes for bi-encoder and mm-encoder account for flexibility in case of additional byol or data2vec outputs

class ForwardPassOutput:
    def __init__(
        self,
        last_hidden_state: torch.Tensor,
        hidden_states: torch.Tensor,    
    ) -> None:
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states
        

class TrainingStepOutput:
    def __init__(
        self,
        student_output: ForwardPassOutput = None,
        teacher_output: ForwardPassOutput = None
    ) -> None:
        self.student_output = student_output
        self.teacher_output = teacher_output