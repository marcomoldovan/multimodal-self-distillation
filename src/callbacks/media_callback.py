from typing import Any, Dict, Optional, Tuple

import torch
import wandb
import PIL

from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from transformers import PerceiverTokenizer
from transformers.utils import logging

from src.models.components.knn import k_nearest_neighbor
from src.models.components.outputs import ForwardPassOutput
from src.utils import get_wandb_logger, exists



class MediaCallback(Callback):
    def __init__(self, log_every_n_steps) -> None:
        self.log_every_n_steps = log_every_n_steps
        logging.set_verbosity(logging.CRITICAL)
        self.tokenizer : PerceiverTokenizer = PerceiverTokenizer.from_pretrained('deepmind/language-perceiver')
        logging.set_verbosity(logging.WARNING)
    
    def on_fit_start(
        self, trainer: Trainer, 
        pl_module: LightningModule
    ) -> None:
        self.logger : WandbLogger = get_wandb_logger(trainer=trainer)
    
    @rank_zero_only
    def on_validation_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs: Optional[Dict], 
        batch: dict, 
        batch_idx: int, 
        dataloader_idx: int
    ) -> None:
        self.log_media(batch, outputs['forward_pass_output'], batch_idx)

    @rank_zero_only
    def on_test_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs: Optional[Dict], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int
    ) -> None:
        self.log_media(batch, outputs['forward_pass_output'], batch_idx)
    
    
    def log_media(self, batch: dict, outputs: ForwardPassOutput, step: int) -> None:
        if step % self.log_every_n_steps == 0:
            
            text = self.tokenizer.batch_decode(batch['text'].detach().cpu(), skip_special_tokens=True) if 'text' in batch else None
            audio = batch['audio'].detach().cpu() if 'audio' in batch else None
            image = batch['image'].detach().cpu() if 'image' in batch else None
            video = batch['video'].detach().cpu() if 'video' in batch else None
            
            align_fuse = outputs.align_fuse
            features = outputs.student_output.pooler_output.detach().cpu()
            queries = outputs.teacher_output.pooler_output.detach().cpu()
            labels = outputs.labels.detach().cpu() if exists(outputs.labels) else torch.tensor(list(range(len(features))))
            
            table = wandb.Table(columns=['query', 'ground truth', 'similarity ground truth', '#1 prediction', 'similarity #1 prediction'])
            
            #TODO we're searching only one batch so make k = batch_size
            _, similarity_gt, top_k_dist, top_k_ids, probs, _, _ = k_nearest_neighbor(
                prediction_features=features, 
                query_features=queries, 
                labels=labels, 
                k=3, 
                chunking=False)
                        
            for i, sim_gt in enumerate(similarity_gt):
                
                # unimodal cases
                if align_fuse[0] == align_fuse[1]:
                    if align_fuse == [['text'],['text']]:
                        pass
                    elif align_fuse == [['image'],['image']]:
                        pass
                    elif align_fuse == [['audio'],['audio']]:
                        pass
                    elif align_fuse == [['video'],['video']]:
                        pass
                    elif align_fuse == [['video', 'audio'],['video', 'audio']]:
                        pass
                    else:
                        raise NotImplementedError(f'Unimodal alignment and/or fusion case: <<{align_fuse}>> not implemented')
                
                # multimodal cases
                else:
                    if align_fuse == [['text'],['audio']]:
                        text_query = text[i]
                        audio_gt = audio[i]
                        audio_pred = audio[top_k_ids[i][0]]
                        audio_pred_caption = text[top_k_ids[i][0]]
                        table.add_data(
                            text_query, 
                            wandb.Audio(audio_gt, sample_rate=16000, caption=text_query), 
                            sim_gt, 
                            wandb.Audio(audio_pred, sample_rate=16000, caption=audio_pred_caption), 
                            top_k_dist[i][0]
                        )
                    elif align_fuse == [['text'],['image']]:
                        table.add_data(
                            batch['text'][i], 
                            wandb.Image(image[i], caption=text[i]), 
                            sim_gt, 
                            wandb.Image(image[top_k_ids[i][0]], caption=text[top_k_ids[i][0]]), 
                            top_k_dist[i][0]
                        )
                    elif align_fuse == [['audio'],['image']]:
                        pass
                    elif align_fuse == [['text'],['video']]:
                        pass
                    elif align_fuse == [['text'],['video', 'audio']]:
                        pass
                    else:
                        raise NotImplementedError(f'Multimodal alignment and/or fusion case: <<{align_fuse}>> not implemented')
                
            self.logger.experiment.log({f'predictions': table})
                
