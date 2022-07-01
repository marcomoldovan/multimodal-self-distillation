import torch.nn as nn


class EMA:
    """
    Modified version of class fairseq.models.ema.EMAModule.
    Args:
        model (nn.Module):
        cfg (DictConfig):
        device (str):
        skip_keys (list): The keys to skip assigning averaged weights to.
    """

    def __init__(
        self, 
        model: nn.Module, 
        ema_decay: float = 0.999, 
        skip_keys=None
    ):
        self.model = model
        self.model.requires_grad_(False)
        self.decay = ema_decay
        self.skip_keys = skip_keys or set()
        self.num_updates = 0

    def step(self, new_model: nn.Module):
        """
        One EMA step
        Args:
            new_model (nn.Module): Online model to fetch new weights from
        """
        ema_state_dict = {}
        ema_params = self.model.state_dict()
        for key, param in new_model.state_dict().items():
            ema_param = ema_params[key].float()
            if key in self.skip_keys:
                ema_param = param.to(dtype=ema_param.dtype).clone()
            else:
                ema_param.mul_(self.decay)
                ema_param.add_(param.to(dtype=ema_param.dtype), alpha=1 - self.decay)
            ema_state_dict[key] = ema_param
        self.model.load_state_dict(ema_state_dict, strict=False)
        self.num_updates += 1

    def restore(self, model: nn.Module):
        """
        Reassign weights from another model
        Args:
            model (nn.Module): model to load weights from.
        Returns:
            model with new weights
        """
        d = self.model.state_dict()
        model.load_state_dict(d, strict=False)
        return model

    def state_dict(self):
        return self.model.state_dict()

    @staticmethod
    def get_annealed_rate(start, end, curr_step, total_steps):
        """
        Calculate EMA annealing rate
        """
        r = end - start
        pct_remaining = 1 - curr_step / total_steps
        return end - r * pct_remaining