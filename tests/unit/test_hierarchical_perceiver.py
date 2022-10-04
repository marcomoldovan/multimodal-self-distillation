import sys
import os
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import torch
import hydra

from src.models.components.hip import HiP
from src.models.components.ema import EMA
from src.models.components.criterion import LatentPredictionLoss
from src.models.components.outputs import ForwardPassOutput, TrainingStepOutput
from src.models.components.masking import mask_hidden_states


def test_model_instantiation():
    """
    Test that the model can process text.
    """
    with hydra.initialize(version_base='1.2', config_path='../../configs/tests', job_name="test_hip_instantiation"):
        cfg = hydra.compose(config_name='hierarchical_perceiver')
        model = hydra.utils.instantiate(cfg.hip_only)
        assert isinstance(model, HiP)
        
        input = torch.randn(64, 65536, 32)
        output = model(input)
        print(output.size())
        
def test_ema_instatiation():
    """
    Test that the model can instantiate an EMA object.
    """
    with hydra.initialize(version_base='1.1', config_path='../../configs/tests', job_name="test_perceiver_instantiation"):
        cfg = hydra.compose(config_name='hierarchical_perceiver')
        model = hydra.utils.instantiate(cfg.model)
        ema = EMA(model)
        assert isinstance(ema, EMA)
        
test_model_instantiation()