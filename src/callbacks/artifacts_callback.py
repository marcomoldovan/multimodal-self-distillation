import subprocess
from pathlib import Path
from typing import List

import wandb
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only

from src.utils import get_wandb_logger

class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = 'all', log_freq: int = 10):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log='all', log_freq=self.log_freq, log_graph=True)


class UploadCodeAsArtifact(Callback):
    """Upload all code files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str, use_git: bool = True):
        """
        Args:
            code_dir: the code directory
            use_git: if using git, then upload all files that are not ignored by git.
            if not using git, then upload all '*.py' file
        """
        self.code_dir = code_dir
        self.use_git = use_git

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment 

        code = wandb.Artifact("project-source", type="code")

        if self.use_git:
            # get .git folder path
            git_dir_path = Path(
                subprocess.check_output(["git", "rev-parse", "--git-dir"]).strip().decode("utf8")
            ).resolve()

            for path in Path(self.code_dir).resolve().rglob("*"):

                # don't upload files ignored by git
                # https://alexwlchan.net/2020/11/a-python-function-to-ignore-a-path-with-git-info-exclude/
                command = ["git", "check-ignore", "-q", str(path)]
                not_ignored = subprocess.run(command).returncode == 1

                # don't upload files from .git folder
                not_git = not str(path).startswith(str(git_dir_path))

                if path.is_file() and not_git and not_ignored:
                    code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        else:
            for path in Path(self.code_dir).resolve().rglob("*.py"):
                code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        experiment.log_artifact(code)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)
