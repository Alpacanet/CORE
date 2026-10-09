"""Pure-ID, continuous-input LLM adapter; not a strict CORE RCE encoder."""

import hashlib
import json

import torch
from torch import nn
import torch.nn.functional as F

from core_ave import COREave
from core_trm import TransNet


def _value(config, name, default=None):
    value = config[name] if name in config else None
    return default if value is None else value


class COREllm(COREave):
    """Share the ID table/scorer with CORE; train an LLM sequence branch.

    No tokenizer, item metadata, natural-language prompt, or generated item ID
    is used. Frozen parameters must still participate in autograd: gradients
    through the backbone train the input projection and item embeddings.
    """

    def __init__(self, config, dataset, backbone=None):
        super().__init__(config, dataset)
        # CORE's initializer must run BEFORE attaching a pretrained backbone.
        self.core_branch = _value(config, 'llm_core_branch', 'ave')
        self.fusion = _value(config, 'llm_fusion', 'gate')
        self.tuning = _value(config, 'llm_tuning', 'lora')
        self.init_mode = _value(config, 'llm_init', 'pretrained')
        self.llm_seed = int(_value(config, 'llm_seed', 2020))
        if self.core_branch not in {'ave', 'trm'}:
            raise ValueError('llm_core_branch must be ave or trm')
        if self.fusion not in {'gate', 'llm_only'}:
            raise ValueError('llm_fusion must be gate or llm_only')
        if self.tuning not in {'frozen', 'lora', 'full'}:
            raise ValueError('llm_tuning must be frozen, lora, or full')
        if self.init_mode not in {'pretrained', 'random', 'tiny_random'}:
            raise ValueError('llm_init must be pretrained, random, or tiny_random')
        dtype_name = _value(config, 'llm_dtype', 'bfloat16')
        dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}[dtype_name]
        if dtype == torch.float16:
            raise ValueError('Use bfloat16 on supported GPUs, or float32; no FP16 scaler is provided.')

        # Separate RNG: paired random/pretrained trials initialize adapters and
        # projections identically, independent of backbone initialization cost.
        if backbone is None:
            from transformers import AutoConfig, AutoModel, Qwen3Config, Qwen3Model
            kwargs = dict(cache_dir=_value(config, 'llm_cache_dir'),
                          revision=_value(config, 'llm_revision'),
                          local_files_only=bool(_value(config, 'llm_local_files_only', False)),
                          trust_remote_code=False)
            if self.init_mode == 'tiny_random':
                llm_config = Qwen3Config(vocab_size=16, hidden_size=32,
                                        intermediate_size=64, num_hidden_layers=2,
                                        num_attention_heads=2, num_key_value_heads=2,
                                        head_dim=16, max_position_embeddings=128)
            else:
                llm_config = AutoConfig.from_pretrained(config['llm_name'], **kwargs)
            llm_config.use_cache = False
            llm_config._attn_implementation = 'sdpa'
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.llm_seed)
                if self.init_mode == 'pretrained':
                    backbone = AutoModel.from_pretrained(
                        config['llm_name'], config=llm_config, dtype=dtype,
                        attn_implementation='sdpa', **kwargs)
                elif self.init_mode == 'tiny_random':
                    backbone = Qwen3Model(llm_config).to(dtype=dtype)
                else:
                    backbone = AutoModel.from_config(llm_config).to(dtype=dtype)
        else:
            backbone = backbone.to(dtype=dtype)
        backbone.config.use_cache = False
        backbone.requires_grad_(self.tuning == 'full')
        # The language token table is never used by inputs_embeds, even in full
        # tuning. Do not optimize or save its unused parameters.
        backbone.get_input_embeddings().requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.llm_seed)
            if self.tuning == 'lora':
                from peft import LoraConfig, get_peft_model
                backbone = get_peft_model(backbone, LoraConfig(
                    r=int(_value(config, 'lora_rank', 8)),
                    lora_alpha=int(_value(config, 'lora_alpha', 16)),
                    lora_dropout=float(_value(config, 'lora_dropout', 0.05)),
                    target_modules=_value(config, 'lora_target_modules',
                                          ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
                    bias='none'))
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.llm_seed + 1)
            self.input_projection = nn.Linear(self.embedding_size, hidden)
            self.input_norm = nn.LayerNorm(hidden)
            self.output_projection = nn.Linear(hidden, self.embedding_size)
            self.gate = nn.Linear(2 * self.embedding_size, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, float(_value(config, 'gate_bias', 1.0)))
            if self.core_branch == 'trm' and self.fusion == 'gate':
                self.net = TransNet(config, dataset)
        if bool(_value(config, 'llm_gradient_checkpointing', False)):
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})

        tokens = [str(x) for x in dataset.field2id_token[self.ITEM_ID]]
        vocab_hash = hashlib.sha256(json.dumps(tokens, ensure_ascii=False,
                                               separators=(',', ':')).encode()).hexdigest()
        self._signature = {
            'format_version': 1, 'item_vocab_sha256': vocab_hash,
            'n_items': self.n_items, 'embedding_size': self.embedding_size,
            'llm_name': _value(config, 'llm_name'),
            'llm_revision': _value(config, 'llm_revision'),
            'llm_init': self.init_mode, 'llm_tuning': self.tuning,
            'llm_seed': self.llm_seed, 'llm_dtype': dtype_name,
            'llm_core_branch': self.core_branch, 'llm_fusion': self.fusion,
            'temperature': self.temperature,
            'sess_dropout': float(config['sess_dropout']),
            'item_dropout': float(config['item_dropout']),
            'core_trm_settings': {
                key: _value(config, key) for key in
                ('n_layers', 'n_heads', 'inner_size', 'hidden_dropout_prob',
                 'attn_dropout_prob', 'hidden_act', 'layer_norm_eps', 'initializer_range')
            } if self.core_branch == 'trm' and self.fusion == 'gate' else None,
            'backbone_config': backbone.config.to_dict(),
            'lora_rank': int(_value(config, 'lora_rank', 8)),
            'lora_alpha': int(_value(config, 'lora_alpha', 16)),
            'lora_dropout': float(_value(config, 'lora_dropout', 0.05)),
            'lora_target_modules': _value(config, 'lora_target_modules',
                                          ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
        }
        # Environment/path and Transformers metadata are not model semantics.
        for key in ('_name_or_path', 'transformers_version', '_commit_hash'):
            self._signature['backbone_config'].pop(key, None)

    def forward(self, item_seq):
        # Avoid sending the fixed dataset padding width through 28 LM layers.
        # Causal attention makes trimming trailing padding prediction-invariant.
        width = max(1, int(item_seq.ne(0).sum(dim=1).max().item()))
        item_seq = item_seq[:, :width]
        mask = item_seq.ne(0)
        x = self.sess_dropout(self.item_embedding(item_seq))
        projected = self.input_norm(self.input_projection(x))
        backbone_dtype = self.backbone.get_input_embeddings().weight.dtype
        states = self.backbone(inputs_embeds=projected.to(backbone_dtype),
                               attention_mask=mask.long(), use_cache=False,
                               return_dict=True).last_hidden_state
        # RecBole benchmark sequences are right-padded. Gather the last real
        # position (rather than the final padded column).
        positions = torch.arange(item_seq.size(1), device=item_seq.device)
        last = (mask.long() * (positions + 1)).amax(dim=1) - 1
        last = last.clamp_min(0)
        final_state = states[torch.arange(item_seq.size(0), device=item_seq.device), last]
        llm_session = F.normalize(self.output_projection(final_state.float()), dim=-1)
        if self.fusion == 'llm_only':
            return llm_session
        alpha = self.ave_net(item_seq) if self.core_branch == 'ave' else self.net(item_seq, x)
        core_session = F.normalize(torch.sum(alpha * x, dim=1), dim=-1)
        # gate=1 means CORE, initially sigmoid(1) ~= 0.73.
        gate = torch.sigmoid(self.gate(torch.cat([core_session, llm_session], dim=-1)))
        return F.normalize(gate * core_session + (1 - gate) * llm_session, dim=-1)

    def predict(self, interaction):
        session = self.forward(interaction[self.ITEM_SEQ])
        item = F.normalize(self.item_embedding(interaction[self.ITEM_ID]), dim=-1)
        return (session * item).sum(dim=-1) / self.temperature

    def get_extra_state(self):
        return dict(self._signature, compact=False)

    def set_extra_state(self, state):
        signature = {k: v for k, v in state.items() if k != 'compact'}
        if signature != self._signature:
            differing = sorted(k for k in set(signature) | set(self._signature)
                               if signature.get(k) != self._signature.get(k))
            raise ValueError('Incompatible LLM checkpoint / item mapping: ' + ', '.join(differing))

    def checkpoint_state_dict(self):
        """Omit reconstructible frozen weights; keep projections/items/LoRA."""
        state = self.state_dict()
        for name, parameter in self.named_parameters():
            if name.startswith('backbone.') and not parameter.requires_grad:
                state.pop(name, None)
        state['_extra_state'] = dict(self._signature, compact=True)
        return state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        metadata = state_dict.get('_extra_state')
        if metadata is None:
            raise ValueError('LLM checkpoint is missing its item mapping/model metadata')
        self.set_extra_state(metadata)  # validate BEFORE modifying parameters
        if not metadata.get('compact', False):
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        permitted_missing = {name for name, p in self.named_parameters()
                             if name.startswith('backbone.') and not p.requires_grad}
        invalid_missing = set(result.missing_keys) - permitted_missing
        if invalid_missing or result.unexpected_keys:
            raise RuntimeError(f'Invalid compact checkpoint: missing={sorted(invalid_missing)}, '
                               f'unexpected={result.unexpected_keys}')
        return result
