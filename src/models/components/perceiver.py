import math
import torch
from torch import nn
from typing import Any, Callable, Mapping, Optional, Tuple, Union, Dict

from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer

from src.models.components.masking import mask_hidden_states
from src.models.components.outputs import ModelOutput
from src.models.components.pooler import Pooler
from src.utils import get_logger, get_parameter_dtype




ModalitySizeType = Mapping[str, int]
PreprocessorOutputType = Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
PreprocessorType = Callable[..., PreprocessorOutputType]
PredictionHeadType = Callable[..., Any]
PostprocessorType = Callable[..., Any]

logger = get_logger(__name__)


class PerceiverEmbeddings(nn.Module):
    """Construct the latent embeddings."""

    def __init__(
        self,
        num_latents: int,
        d_latents: int,
        ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, d_latents))

    def forward(self, batch_size: int):
        return self.latents.expand(batch_size, -1, -1)  # Thanks, Phil Wang


class PerceiverSelfAttention(nn.Module):
    """Multi-headed {cross, self}-attention. Can be used both in the encoder as well as in the decoder."""

    def __init__(
        self,
        attention_probs_dropout_prob,
        is_cross_attention,
        qk_channels,
        v_channels,
        num_heads,
        q_dim,
        kv_dim
    ):
        super().__init__()
        self.num_heads = num_heads
        # Q and K must have the same number of channels.
        # Default to preserving Q's input's shape.
        if qk_channels is None:
            qk_channels = q_dim
        # V's num_channels determines the shape of the output of QKV-attention.
        # Default to the same number of channels used in the key-query operation.
        if v_channels is None:
            v_channels = qk_channels
        if qk_channels % num_heads != 0:
            raise ValueError(f"qk_channels ({qk_channels}) must be divisible by num_heads ({num_heads}).")
        if v_channels % num_heads != 0:
            raise ValueError(f"v_channels ({v_channels}) must be divisible by num_heads ({num_heads}).")

        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.qk_channels_per_head = self.qk_channels // num_heads
        self.v_channels_per_head = self.v_channels // num_heads

        # Layer normalization
        self.layernorm1 = nn.LayerNorm(q_dim)
        self.layernorm2 = nn.LayerNorm(kv_dim) if is_cross_attention else nn.Identity()

        # Projection matrices
        self.query = nn.Linear(q_dim, qk_channels)
        self.key = nn.Linear(kv_dim, qk_channels)
        self.value = nn.Linear(kv_dim, v_channels)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x, channels_per_head):
        new_x_shape = x.size()[:-1] + (self.num_heads, channels_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        hidden_states = self.layernorm1(hidden_states)
        inputs = self.layernorm2(inputs)

        # Project queries, keys and values to a common feature dimension. If this is instantiated as a cross-attention module,
        # the keys and values come from the inputs; the attention mask needs to be such that the inputs's non-relevant tokens are not attended to.
        is_cross_attention = inputs is not None
        queries = self.query(hidden_states)

        if is_cross_attention:
            keys = self.key(inputs)
            values = self.value(inputs)
            attention_mask = inputs_mask
        else:
            keys = self.key(hidden_states)
            values = self.value(hidden_states)

        # Reshape channels for multi-head attention.
        # We reshape from (batch_size, time, channels) to (batch_size, num_heads, time, channels per head)
        queries = self.transpose_for_scores(queries, self.qk_channels_per_head)
        keys = self.transpose_for_scores(keys, self.qk_channels_per_head)
        values = self.transpose_for_scores(values, self.v_channels_per_head)

        # Take the dot product between the queries and keys to get the raw attention scores.
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))

        batch_size, num_heads, seq_len, q_head_dim = queries.shape
        _, _, _, v_head_dim = values.shape
        hiddens = self.num_heads * v_head_dim

        attention_scores = attention_scores / math.sqrt(q_head_dim)

        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in PerceiverModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, values)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hiddens,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class PerceiverSelfOutput(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.dense = nn.Linear(input_channels, output_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return hidden_states


class PerceiverAttention(nn.Module):
    """Attention module, including a dense block."""

    def __init__(
        self,
        attention_probs_dropout_prob=0.1,
        cross_attention_shape_for_attention="kv",
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
        use_query_residual=True,
    ):
        super().__init__()
        # MultiHead attention
        if is_cross_attention and qk_channels is None:
            if cross_attention_shape_for_attention == "q":
                qk_channels = q_dim
            elif cross_attention_shape_for_attention == "kv":
                qk_channels = kv_dim
            else:
                raise ValueError(
                    f"Unknown value {cross_attention_shape_for_attention} for "
                    "cross_attention_shape_for_attention."
                )
        else:
            if qk_channels is None:
                qk_channels = q_dim
            if v_channels is None:
                v_channels = qk_channels
        self.self = PerceiverSelfAttention(
            attention_probs_dropout_prob,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim
        )
        # dense block
        output_channels = None
        if is_cross_attention:
            output_channels = q_dim
        else:
            if output_channels is None:
                output_channels = v_channels
        self.output = PerceiverSelfOutput(input_channels=self.self.v_channels, output_channels=output_channels)
        self.use_query_residual = use_query_residual
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads) 
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
        )

        # Output projection
        attention_output = self.output(self_outputs[0])

        # Optionally include a residual to the original queries.
        # Consider omitting the residual if the semantics of query and output
        # are different, e.g. if queries are positions and outputs are pixels.
        if self.use_query_residual:
            attention_output = attention_output + hidden_states

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class PerceiverMLP(nn.Module):
    """A Transformer-style dense module to follow attention."""

    def __init__(self, input_size, widening_factor, hidden_act='gelu'):
        super().__init__()
        self.dense1 = nn.Linear(input_size, widening_factor * input_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = nn.GELU()
        else:
            self.intermediate_act_fn = hidden_act
        self.dense2 = nn.Linear(widening_factor * input_size, input_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states


class PerceiverLayer(nn.Module):
    def __init__(
        self,
        chunk_size_feed_forward=0, # from PretrainedConfig
        attention_probs_dropout_prob=0.1,
        cross_attention_shape_for_attention="kv",
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
        widening_factor=4,
        use_query_residual=True,
    ):
        super().__init__()
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PerceiverAttention(
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            cross_attention_shape_for_attention=cross_attention_shape_for_attention,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
            use_query_residual=use_query_residual,
        )
        self.layernorm = nn.LayerNorm(q_dim)
        self.mlp = PerceiverMLP(input_size=q_dim, widening_factor=widening_factor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
        )
        attention_output = attention_outputs[0]

        outputs = attention_outputs[1:]  # add attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        layer_output = layer_output + attention_output  # residual connection

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        layer_output = self.layernorm(attention_output)
        layer_output = self.mlp(layer_output)
        return layer_output


class PerceiverEncoder(nn.Module):
    """The Perceiver Encoder: a scalable, fully attentional encoder."""

    def __init__(
        self,
        d_latents,
        num_blocks,
        num_self_attention_heads,
        num_self_attends_per_block,
        num_cross_attention_heads,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        attention_probs_dropout_prob=0.1,
        chunk_size_feed_forward=0, # found in PretrainedConfig        
        kv_dim=None,
        use_query_residual=True,
        ):
        super().__init__()
        
        self.num_blocks = num_blocks
        
        # Check that we can use multihead-attention with these shapes.
        if d_latents % num_self_attention_heads != 0:
            raise ValueError(
                f"num_z_channels ({d_latents}) must be divisible by"
                f" num_self_attend_heads ({num_self_attention_heads})."
            )
        if d_latents % num_cross_attention_heads != 0:
            raise ValueError(
                f"num_z_channels ({d_latents}) must be divisible by"
                f" num_cross_attend_heads ({num_cross_attention_heads})."
            )

        # Construct the cross attention layer.
        self.cross_attention = PerceiverLayer(
            chunk_size_feed_forward=chunk_size_feed_forward,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            cross_attention_shape_for_attention=cross_attention_shape_for_attention,
            is_cross_attention=True,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_cross_attention_heads,
            q_dim=d_latents,
            kv_dim=kv_dim,
            widening_factor=cross_attention_widening_factor,
            use_query_residual=use_query_residual,
        )

        # Construct a single block of self-attention layers.
        # We get deeper architectures by applying this block more than once.
        self_attention_layers = []
        for _ in range(num_self_attends_per_block):
            layer = PerceiverLayer(
                chunk_size_feed_forward=chunk_size_feed_forward,
                attention_probs_dropout_prob=attention_probs_dropout_prob,                
                is_cross_attention=False,
                qk_channels=qk_channels,
                v_channels=v_channels,
                num_heads=num_self_attention_heads,
                q_dim=d_latents,
                kv_dim=d_latents,
                widening_factor=self_attention_widening_factor,
            )
            self_attention_layers.append(layer)

        self.self_attends = nn.ModuleList(self_attention_layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        # Apply the cross-attention between the latents (hidden_states) and inputs:
        layer_outputs = self.cross_attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            inputs=inputs,
            inputs_mask=inputs_mask,
            output_attentions=output_attentions,
        )
        hidden_states = layer_outputs[0]

        if output_attentions:
            all_cross_attentions = all_cross_attentions + (layer_outputs[1],)

        # Apply the block of self-attention layers more than once:
        for _ in range(self.num_blocks):
            for i, layer_module in enumerate(self.self_attends):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
        
        

class PerceiverModel(nn.Module):
    def __init__(
        self,
        is_training=False,
        is_student=False,
        d_model=704,
        num_latents=784,
        d_latents=512,
        num_blocks=1,
        num_self_attention_heads=8,
        num_self_attends_per_block=8,
        num_cross_attention_heads=1,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        attention_probs_dropout_prob=0.1,
        chunk_size_feed_forward=0, # found in PretrainedConfig        
        kv_dim=None,
        use_query_residual=True,
        mask_time_prob=0.05,
        mask_time_length=10,
        use_projection_head=True,
        use_simsiam_projector=False,
        input_preprocessor: PreprocessorType = None,
    ):
        """
        This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
        it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
        behavior.
        Parameters:
            config ([`PerceiverConfig`]): Model configuration class with all the parameters of the model.
                Initializing with a config file does not load the weights associated with the model, only the
                configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
            input_preprocessor (*PreprocessorType*, *optional*):
                Optional input preprocessor to use. Examples include
                *transformers.models.perceiver.modeling_perceiver.PerceiverImagePreprocessor*,
                *transformers.models.perceiver.modeling_perceiver.PerceiverAudioPreprocessor*,
                *transformers.models.perceiver.modeling_perceiver.PerceiverTextPreprocessor*,
                *transformers.models.perceiver.modeling_perceiver.PerceiverMultimodalPreprocessor*.
            Note that you can define your own decoders, preprocessors and/or postprocessors to fit your use-case.
        """
        super().__init__()
        
        self.is_training = is_training
        self.is_student = is_student
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_self_attends_per_block = num_self_attends_per_block
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        
        # initialized by Hydra
        self.input_preprocessor = input_preprocessor
        
        self.embeddings = PerceiverEmbeddings(num_latents, d_latents)
        
        self.encoder = PerceiverEncoder(
            d_latents=d_latents,
            num_blocks=num_blocks,
            num_self_attention_heads=num_self_attention_heads,
            num_self_attends_per_block=num_self_attends_per_block,
            num_cross_attention_heads=num_cross_attention_heads,
            qk_channels=qk_channels,
            v_channels=v_channels,
            cross_attention_shape_for_attention=cross_attention_shape_for_attention,
            self_attention_widening_factor=self_attention_widening_factor,
            cross_attention_widening_factor=cross_attention_widening_factor,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            chunk_size_feed_forward=chunk_size_feed_forward, # found in PretrainedConfig        
            kv_dim=input_preprocessor.num_channels if input_preprocessor is not None else d_model,
            use_query_residual=use_query_residual,
        )
        
        self.pooler = Pooler(
            dim_in=d_latents, 
            projection_size=d_latents, 
            widening_factor=self_attention_widening_factor, 
            use_projection_head=use_projection_head,
            use_simsiam_mlp=use_simsiam_projector
        )
    
    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)
        
        
    def set_student_status(self, is_student: bool):
        self.is_student = is_student
        
        
    def invert_attention_mask(self, encoder_attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).
        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.
        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        if self.dtype == torch.float16:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
        elif self.dtype in [torch.bfloat16, torch.float32]:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            raise ValueError(
                f"{self.dtype} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`"
            )
            
        # if device is on GPU, convert to CUDA tensor:
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(next(self.encoder.parameters()).device)

        return encoder_extended_attention_mask
    
    
    def get_head_mask(
        self, head_mask: Optional[torch.Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> torch.Tensor:
        """
        Prepare the head mask if needed.
        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.
        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask


    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask


    def forward(
        self,
        inputs: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        apply_mask: Optional[bool] = True,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
    ) -> ModelOutput:
        r"""
        Args:
            inputs (`torch.FloatTensor`):
                Inputs to the perceiver. Can be anything: images, text, audio, video, etc.
            attention_mask (`torch.FloatTensor` of shape `{0}`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        if self.input_preprocessor is not None:
            inputs, _, _ = self.input_preprocessor(inputs)
        else:

            if inputs.size()[-1] != self.d_model:
                raise ValueError(
                    f"Last dimension of the inputs: {inputs.size()[-1]} doesn't correspond to d_model: {self.d_model}. "
                    "Make sure to set d_model appropriately."
                )
                
        if self.is_student and apply_mask:
            inputs = mask_hidden_states(
                hidden_states=inputs,
                attention_mask=attention_mask,
                mask_time_prob=self.mask_time_prob,
                mask_time_length=self.mask_time_length,
                training=self.is_training
            )
            

        batch_size, seq_length, _ = inputs.size()

        # If no attention mask is provided, make them all ones
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)))
        # Make the attention mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = self.invert_attention_mask(attention_mask)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_blocks x num_heads]
        # and head_mask is converted to shape [num_blocks x batch x num_heads x N x N]
        head_mask = self.get_head_mask(head_mask, self.num_blocks * self.num_self_attends_per_block)

        embedding_output = self.embeddings(batch_size=batch_size)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=None,
            head_mask=head_mask,
            inputs=inputs,
            inputs_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        
        pooler_output = self.pooler(sequence_output)

        return ModelOutput(
            pooler_output=pooler_output,
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
