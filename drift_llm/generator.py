"""
One-Step Text Generator wrapper.

Takes a pretrained bidirectional Transformer (LLaDA) and wraps it for
one-step generation: prompt_emb + noise → single forward → logits.

Mirrors image Drifting's Generator: noise + class_embed → DiT → image.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from einops import repeat


class OneStepGenerator(nn.Module):
    """
    Wraps a pretrained LLaDA model for one-step text generation.

    Input:  prompt token ids + Gaussian noise for response positions
    Output: logits [B, T_r, V] over the full vocabulary at each response position

    The model is bidirectional (all positions attend to all positions),
    so it can generate all response tokens in a single forward pass.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: A pretrained LLaDAModelLM (or compatible HuggingFace model)
                   that supports inputs_embeds and output_hidden_states.
        """
        super().__init__()
        self.model = model

    @property
    def embedding(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    @property
    def hidden_dim(self) -> int:
        return self.embedding.weight.shape[1]

    @property
    def vocab_size(self) -> int:
        return self.embedding.weight.shape[0]

    def forward(
        self,
        prompt_ids: torch.Tensor,
        response_length: int,
        noise: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> dict:
        """
        One-step generation.

        Args:
            prompt_ids:      [B, T_p]  prompt token ids
            response_length: int       number of response tokens to generate
            noise:           [B, T_r, D] optional pre-generated noise
                             (if None, sampled from N(0, I))
            output_hidden_states: if True, return intermediate hidden states

        Returns:
            dict with:
                logits:        [B, T_r, V]  response logits
                hidden_states: [B, T_p+T_r, D]  last hidden states (if requested)
        """
        B = prompt_ids.shape[0]
        T_p = prompt_ids.shape[1]
        T_r = response_length
        D = self.hidden_dim
        device = prompt_ids.device

        prompt_emb = self.embedding(prompt_ids)  # [B, T_p, D]

        if noise is None:
            noise = torch.randn(B, T_r, D, device=device, dtype=prompt_emb.dtype)

        input_emb = torch.cat([prompt_emb, noise], dim=1)  # [B, T_p+T_r, D]

        outputs = self.model(
            inputs_embeds=input_emb,
            output_hidden_states=output_hidden_states,
        )

        logits = outputs.logits[:, T_p:, :]  # [B, T_r, V]

        result = {"logits": logits}
        if output_hidden_states and outputs.hidden_states is not None:
            result["hidden_states"] = outputs.hidden_states[-1]  # last layer
            result["all_hidden_states"] = outputs.hidden_states
        return result

    def generate_candidates(
        self,
        prompt_ids: torch.Tensor,
        response_length: int,
        num_candidates: int,
        output_hidden_states: bool = False,
    ) -> dict:
        """
        Generate G candidates per prompt with different noise.

        Args:
            prompt_ids:      [B, T_p]
            response_length: T_r
            num_candidates:  G

        Returns:
            dict with:
                logits:        [B*G, T_r, V]
                hidden_states: [B*G, T_p+T_r, D] (if requested)
        """
        B = prompt_ids.shape[0]
        G = num_candidates

        prompt_rep = prompt_ids.repeat_interleave(G, dim=0)  # [B*G, T_p]

        return self.forward(
            prompt_ids=prompt_rep,
            response_length=response_length,
            output_hidden_states=output_hidden_states,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        response_length: int,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference: one-step generation → argmax → token ids.

        Args:
            prompt_ids:      [B, T_p]
            response_length: T_r

        Returns:
            token_ids: [B, T_r]
        """
        result = self.forward(prompt_ids, response_length, noise=noise)
        return result["logits"].argmax(dim=-1)
