import logging
import warnings
from typing import List, Sequence

import torch

import torch
import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning import Trainer
from transformers import BertModel
from transformers.models.hubert.modeling_hubert import HubertPositionalConvEmbedding


from src.models.components.encoder_components import (
    BertEmbeddingsWrapper, 
    HubertConvFeatureExtractorWrapper, 
    HubertFeatureProjectionWrapper, 
    BertEncoderWrapper, 
    HubertModelWithoutFeatureEncoder, 
    HubertModelWithPooler, 
    HubertModel, 
    BertPoolerWrapper, 
    HubertPooler, 
    HubertEncoderWrapper
)


count_parameters = lambda model : {'requires_grad':sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6,
                                   'does_not_require_grad':sum(p.numel() for p in model.parameters() if not p.requires_grad)/1e6}


def freeze_model(model, trainable_layers=0):
    """Trainable layers refers to the number of trainable attention layers
        in the network. If trainable layers > 0, then the corresponding projection
        head will also be trainable. In case of a Bi-Encoder only components of
        speech model will be trainable, the text model will always be frozen.

    Args:
        model (
            BiEncoderSpeechTextModelWithoutFeatureEncoder,
            BiEncoderSpeechTextModel,
            MultiModalSpeechTextEncoder
            ): The model to be frozen.
        trainablelayers (int, optional): How many attention layers in the speech or
            multimodal encoder to train. Defaults to 0.
    """
    print(f"Parameters before freezing: {count_parameters(model)}")
    
    for _, child in model.named_children():
        
        # standard BERT as text model
        if isinstance(child, BertModel):
            freeze_module(child)
        
        # modules for the multimodal encoder
        elif isinstance(child, BertEmbeddingsWrapper):
            freeze_module(child)
        elif isinstance(child, HubertConvFeatureExtractorWrapper):
            freeze_module(child)
        elif isinstance(child, HubertFeatureProjectionWrapper):
            freeze_module(child)
        elif isinstance(child, BertEncoderWrapper):          
            for na, ch in child.named_children():
                for n, c in ch.named_children():
                    if isinstance(c, torch.nn.ModuleList):
                        for i, _ in enumerate(c._modules):
                                if i < (len(c._modules) - trainable_layers):
                                    freeze_module(c[i])
        elif isinstance(child, HubertPooler) or isinstance(child, BertPoolerWrapper):
            pass
        
        # modules for the speech encoder without convolution
        elif isinstance(child, HubertModelWithoutFeatureEncoder): # done
            for na, ch in child.named_children():
                if isinstance(ch, HubertFeatureProjectionWrapper):
                    freeze_module(ch)
                elif isinstance(ch, HubertEncoderWrapper):
                    for n, c in ch.named_children():
                        for n_enc, c_enc in c.named_children():
                            if isinstance(c_enc, torch.nn.LayerNorm):
                                freeze_module(c_enc)
                            elif isinstance(c_enc, torch.nn.Dropout):
                                freeze_module(c_enc)
                            elif isinstance(c_enc, torch.nn.ModuleList):
                                for i, _ in enumerate(c_enc._modules):
                                    if i < (len(c_enc._modules) - trainable_layers):
                                        freeze_module(c_enc[i])
                elif isinstance(ch, HubertPooler):
                    pass
        
        # modules for the HuBERT speech encoder with convolution and pooler             
        elif isinstance(child, HubertModelWithPooler): # done
            for na, ch in child.named_children():
                if isinstance(ch, HubertModel):
                    freeze_module(ch.feature_extractor)
                    freeze_module(ch.feature_projection)
                    for n, c in ch.encoder.named_children():
                        if isinstance(c, HubertPositionalConvEmbedding):
                            freeze_module(c)
                        elif isinstance(c, torch.nn.LayerNorm):
                            freeze_module(c)
                        elif isinstance(c, torch.nn.Dropout):
                            freeze_module(c)
                        elif isinstance(c, torch.nn.ModuleList):
                            for i, _ in enumerate(c._modules):
                                if i < (len(c._modules) - trainable_layers):
                                    freeze_module(c[i])
                if isinstance(ch, HubertPooler):
                    pass
                
    print(f"Parameters after freezing: {count_parameters(model)}")
    

def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.
    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        log.info("Printing config tree with Rich! <config.print_config=True>")
        print_config(config, resolve=True)


@rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    quee = []

    for field in print_order:
        quee.append(field) if field in config else log.info(f"Field '{field}' not found in config")

    for field in config:
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as file:
        rich.print(tree, file=file)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Controls which config parts are saved by Lightning loggers.
    Additionaly saves:
    - number of model parameters
    """

    if not trainer.logger:
        return

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()
            
            
def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )