from typing import Optional, List
from dataclasses import dataclass

import torch
from torch import nn

from src.models.components.preprocessor import PreprocessorType
from src.models.components.masking import mask_hidden_states
from src.models.components.outputs import ModelOutput
from src.models.components.pooler import Pooler


class MultiHeadAttention(nn.Module):
    """Multi-head attention"""
    def __init__(
        self,
        kv_dim: int,
        q_dim: int,
        *,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_heads: int = 1,
        dropout: float = 0.0
    ):
        """Constructor.
        Args:
            kv_dim: Size of input key and value vectors.
            q_dim: Size of input query vector.
            qk_out_dim: Size of Query and Key matrices last dimension.
                If None, it will be equal to q_dim. Defaults to None.
            v_out_dim: Size of Value matrix last dimension.
                If None, it will be equal to qk_out_dim. Defaults to None.
            output_dim: Size of output after the QKV attention.
                If none, it will be equal to v_out_dim. Defaults to None.
            num_heads: Number of heads. Defaults to 1.
            dropout: Dropout probability. Defaults to 0.0.
        """
        super().__init__()

        if qk_out_dim is None:
            qk_out_dim = q_dim
        if v_out_dim is None:
            v_out_dim = qk_out_dim
        if output_dim is None:
            output_dim = v_out_dim

        self.num_heads = num_heads
        self.qk_head_dim = qk_out_dim // num_heads
        self.v_head_dim = v_out_dim // num_heads

        self.k = nn.Linear(kv_dim, qk_out_dim)
        self.q = nn.Linear(q_dim, qk_out_dim)
        self.v = nn.Linear(kv_dim, v_out_dim)
        self.projection = nn.Linear(v_out_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.qk_head_dim ** -0.5

    def transform_for_scores(self, x: torch.Tensor, head_dim: int):
        # (..., seq_len, dim) -> (..., n_heads, seq_len, head_dim)
        *dims, seq, hid = x.size()
        x = x.view(*dims, seq, self.num_heads, head_dim)
        return x.transpose(-3, -2)

    def forward(
        self,
        inputs_kv: torch.Tensor,
        inputs_q: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            inputs_kv: Key/Value embeddings of shape (B, ..., M, C).
            inputs_q: Query embeddings of shape (B, ..., N, D)
            attention_mask: Tensor of shape (B, ..., N, M).
        Returns:
            Tensor of shape (B, ..., N, D)
        """
        keys, queries, values = self.k(inputs_kv), self.q(inputs_q), self.v(inputs_kv)
        keys = self.transform_for_scores(keys, self.qk_head_dim)
        queries = self.transform_for_scores(queries, self.qk_head_dim)
        values = self.transform_for_scores(values, self.v_head_dim)
        attention = (queries @ keys.transpose(-2, -1) * self.scale)
        if attention_mask is not None:
            min_value = torch.finfo(attention.dtype).min
            extended_mask = (1 - attention_mask) * min_value
            attention = attention + extended_mask
        attention = attention.softmax(dim=-1)
        attention = self.dropout(attention)
        if attention_mask is not None:
            attention = attention.masked_fill(1 - attention_mask, value=0)
        weighted = attention @ values
        # (..., n_heads, seq_len, head_dim) -> (..., seq_len, hid)
        *dims, n_heads, seq, hid = weighted.size()
        weighted = weighted.transpose(-3, -2)
        weighted = weighted.reshape(*dims, seq, n_heads * hid)
        return self.projection(weighted)


class FeedForward(nn.Module):
    """Transformer Feed-Forward network."""
    def __init__(
        self,
        dim: int,
        widening_factor: int = 4,
        dropout: float = 0.0
    ):
        """Constructor.
        Args:
            dim: Dimension of input tensor.
            widening_factor: Widening factor. Defaults to 4.
            dropout: Dropout probability. Defaults to 0.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * widening_factor),
            nn.GELU(),
            nn.Linear(dim * widening_factor, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class SelfAttention(nn.Module):
    """Self-attention module."""
    def __init__(
        self,
        *,
        hidden_dim: int,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        widening_factor: int = 4,
        num_heads: int = 1,
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        """Constructor.
        Args:
            hidden_dim: Dimension of input tensor.
            qk_out_dim: Size of Query and Key matrices last dimension.
                Defaults to None.
            v_out_dim: Size of Value matrix last dimension.
                Defaults to None.
            widening_factor: Feed-forward network widening factor.
                Defaults to 4.
            num_heads: Number of attention heads. Defaults to 1.
            dropout: Dropout probability. Defaults to 0.
            attention_dropout: Attention scores probability. Defaults to 0.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.qkv_layer_norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(
            kv_dim=hidden_dim,
            q_dim=hidden_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.mlp = FeedForward(hidden_dim, widening_factor, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x: Input tensor of shape (B, ..., M, C).
            attention_mask: Input mask tensor of shape (B, ..., M, M).
                Mask values selected in [0, 1]. Defaults to None.
        """
        x_norm = self.layer_norm(x)
        attention = self.attention(
            inputs_kv=x_norm,
            inputs_q=x_norm,
            attention_mask=attention_mask
        )
        attention = self.dropout(attention)
        x = x + attention
        x = x + self.mlp(self.qkv_layer_norm(x))
        return x


class CrossAttention(nn.Module):
    """Cross-attention module."""
    def __init__(
        self,
        *,
        kv_dim: int,
        q_dim: int,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        widening_factor: int = 1,
        num_heads: int = 1,
        use_query_residual: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        """Constructor.
        Args:
            kv_dim: Dimension of key/value input tensor.
            q_dim: Dimension of query input tensor.
            qk_out_dim: Size of Query and Key matrices last dimension.
                Defaults to None.
            v_out_dim: Size of Value matrix last dimension.
                Defaults to None.
            widening_factor: Feed-forward network widening factor.
                Defaults to 4.
            num_heads: Number of attention heads. Defaults to 1.
            use_query_residual: Indicates whether to use query residual in
                cross-attention. Defaults to True.
            dropout: Dropout probability. Defaults to 0.
            attention_dropout: Attention scores probability. Defaults to 0.
        """
        super().__init__()
        self.use_query_residual = use_query_residual
        self.kv_layer_norm = nn.LayerNorm(kv_dim)
        self.q_layer_norm = nn.LayerNorm(q_dim)
        self.qkv_layer_norm = nn.LayerNorm(q_dim)
        self.attention = MultiHeadAttention(
            kv_dim=kv_dim,
            q_dim=q_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            output_dim=q_dim,
            num_heads=num_heads,
            dropout=attention_dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.mlp = FeedForward(q_dim, widening_factor, dropout)

    def forward(
        self,
        inputs_kv: torch.Tensor,
        inputs_q: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            inputs_kv: Key/Value embeddings of shape (B, ..., M, C).
            inputs_q: Query embeddings of shape (B, ..., N, D)
            attention_mask: Tensor of shape (B, ..., N, M). Mask values selected
                in [0, 1]. Defaults to None.
        """
        attention = self.attention(
            inputs_kv=self.kv_layer_norm(inputs_kv),
            inputs_q=self.q_layer_norm(inputs_q),
            attention_mask=attention_mask
        )
        attention = self.dropout(attention)
        if self.use_query_residual:
            x = inputs_q + attention
        else:
            x = attention
        x = x + self.mlp(self.qkv_layer_norm(x))
        return x
    

class PerceiverBlock(nn.Module):
    """Basic Hierarchical Perceiver block. Consists of learned set of latent vectors (one for each group),
    cross-attention encoding layer and number of self-attention processing layers.
    All parameters of cross- and self-attention layers are shared.
    """
    def __init__(
        self,
        input_dim: int,
        num_groups: int,
        num_latents: int,
        hidden_size: int,
        num_self_attn_layers: int = 1,
        num_cross_attn_heads: int = 1,
        num_self_attn_heads: int = 1,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        cross_attn_widening_factor: int = 1,
        self_attn_widening_factor: int = 1,
        use_query_residual: bool = True,
        dropout: float = 0.0,
        cross_attn_dropout: float = 0.0,
        self_attn_dropout: float = 0.0
    ):
        super().__init__()
        self.num_groups = num_groups
        self.hidden_size = hidden_size

        self.latents = nn.Parameter(torch.randn(num_groups, num_latents, hidden_size))
        self.cross_attention = CrossAttention(
            kv_dim=input_dim,
            q_dim=hidden_size,
            num_heads=num_cross_attn_heads,
            dropout=dropout,
            attention_dropout=cross_attn_dropout,
            widening_factor=cross_attn_widening_factor,
            use_query_residual=use_query_residual,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim
        )
        self.self_attention_layers = nn.ModuleList([
            SelfAttention(
                hidden_dim=hidden_size,
                num_heads=num_self_attn_heads,
                dropout=dropout,
                attention_dropout=self_attn_dropout,
                widening_factor=self_attn_widening_factor,
                qk_out_dim=qk_out_dim,
                v_out_dim=v_out_dim
            ) for _ in range(num_self_attn_layers)
        ])

    def forward(self, inputs, attention_mask=None):
        *dims, seq_len, input_dim = inputs.size()
        if attention_mask is not None:
            # (bs, seq_len) -> (bs, num_groups, group_len)
            attention_mask = attention_mask.view(*dims, self.num_groups, -1)
            # (bs, num_groups, group_len) -> (bs, num_groups, num_heads, q_seq_len, kv_seq_len)
            # num_groups and q_seq_len are broadcast
            # group_len is the same as kv_seq_len
            attention_mask = attention_mask[:, :, None, None, :]

        # (..., seq_len, hid_dim) -> (..., num_groups, group_len, hid_dim)
        inputs = inputs.view(*dims, self.num_groups, -1, input_dim)
        latents = self.cross_attention(inputs, self.latents, attention_mask)
        for self_attention in self.self_attention_layers:
            latents = self_attention(latents)

        # (.., num_groups, group_len, latent_dim) -> (.., seq_len, hid_dim)
        *_, latents_dim = latents.size()
        outputs = latents.view(*dims, -1, latents_dim)
        return outputs
    
    
@dataclass
class BlockConfig:
    num_groups: int
    num_self_attn_layers: int
    num_self_attn_heads: int
    num_latents: int
    hidden_size: int

    
class HiP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        block_configs: List[BlockConfig]
    ):
        super().__init__()
        layers = []
        input_dim = input_dim
        for cfg in block_configs:
            layer = PerceiverBlock(
                input_dim=input_dim,
                num_groups=cfg.num_groups,
                num_self_attn_layers=cfg.num_self_attn_layers,
                num_self_attn_heads=cfg.num_self_attn_heads,
                num_latents=cfg.num_latents,
                hidden_size=cfg.hidden_size
            )
            layers.append(layer)
            input_dim = cfg.hidden_size
        self.layers = nn.ModuleList(layers)

    def forward(self, x, attention_mask=None):
        hidden_states = []
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask)
            hidden_states.append(x)
            attention_mask = None
        return x, hidden_states
    
    
class HiPModel(nn.Module):
    def __init__(
        self,
        preprocessor: PreprocessorType,
        hip: HiP,
        is_student: bool = False,
        is_training: bool = False,
        mask_time_prob: float = 0.05,
        mask_time_length: int = 10,
        use_simsiam_mlp: bool = False
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.hip = hip
        
        self.pooler = Pooler(
            dim_in=self.hip.layers[-1].hidden_size, 
            projection_size=self.hip.layers[-1].hidden_size,
            use_simsiam_mlp=use_simsiam_mlp
        )
        
        self.is_student = is_student
        self.is_training = is_training
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        
    def set_student_status(self, is_student: bool):
        self.is_student = is_student

    def forward(self, x, attention_mask=None, apply_mask=True):
        x, _, _ = self.preprocessor(x)
        
        batch_size, seq_length, _ = x.size()
        
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length))) #TODO move to CUDA
        
        if self.is_student and apply_mask:
            x = mask_hidden_states(
                hidden_states=x,
                attention_mask=attention_mask,
                mask_time_prob=self.mask_time_prob,
                mask_time_length=self.mask_time_length,
                training=self.is_training
            )
            
        x, hidden_states = self.hip(x, attention_mask)
        
        pooler_output = self.pooler(x)
        
        return ModelOutput(
            pooler_output=pooler_output,
            last_hidden_state=x,
            hidden_states=hidden_states
        )
    