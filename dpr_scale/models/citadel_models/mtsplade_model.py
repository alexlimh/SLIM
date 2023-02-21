#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional
import torch.nn as nn
import torch
from dpr_scale.utils.utils import PathManager

# @manual=//python/wheel/transformers3:transformers3
from transformers import AutoModelForMaskedLM, AutoConfig

class MTSPLADEEncoder(nn.Module):
    def __init__(
        self,
        model_path: str = "bert-base-uncased",
        dropout: float = 0.1,
        sparse_mode: bool = False,
    ):
        super().__init__()
        # remove recursive argument which is not supported now
        local_model_path = PathManager.get_local_path(model_path)
        cfg = AutoConfig.from_pretrained(local_model_path)
        cfg.output_hidden_states = True
        cfg.attention_probs_dropout_prob = dropout
        cfg.hidden_dropout_prob = dropout
        self.sparse_mode = sparse_mode
        self.transformer = AutoModelForMaskedLM.from_pretrained(local_model_path, config=cfg)

    def forward(self, tokens, topk=1):
        ret = {}
        # make it transformer 4.x compatible
        outputs = self.transformer(**tokens, return_dict=True)
        attention_mask = tokens["attention_mask"][:, 1:]
        logits = outputs.logits[:, 1:, :]
        
        # routing, assign every token to top-k expert
        full_router_repr = torch.log(1 + torch.relu(logits)) * attention_mask.unsqueeze(-1)
        expert_weights, expert_ids = torch.topk(full_router_repr, dim=2, k=topk) # B x T x topk
        min_expert_weight = torch.min(expert_weights, -1, True)[0]
        sparse_expert_weights = torch.where(full_router_repr >= min_expert_weight, full_router_repr, 0)

        ret["attention_mask"] = attention_mask.clone()
        if self.sparse_mode:
            # some training stat.
            router_mask = torch.zeros_like(full_router_repr)
            router_mask.scatter_(dim=2, index=expert_ids, src=(expert_weights > 0.).to(expert_weights.dtype)) # B x T x E 
            # average number of experts per input 
            ret["avg_cond_num_experts"] = router_mask.sum(1).sum(1, keepdim=True).mean(0, keepdim=True)
            # average number of distinct experts per batch
            ret["avg_marg_num_experts"] = router_mask.sum(1).max(0, keepdim=True).values.sum(1, keepdim=True)
            router_softmax_repr = torch.softmax(logits, dim=-1)
            ret["router_mask"] = router_mask.sum(1).clone()
            ret["router_softmax_repr"] = router_softmax_repr.sum(1).clone()
            ret["expert_weights"] = sparse_expert_weights.clone()
        else:
            ret["expert_ids"] = expert_ids.clone()
            ret["expert_weights"] = expert_weights.clone()
            ret["sparse_weights"] = sparse_expert_weights.clone()
        return ret
