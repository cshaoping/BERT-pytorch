#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on 2020-12-15 17:36

@author: caoshaoping
"""
import torch

from bert_pytorch.model.embedding.position import PositionalEmbedding


if __name__ == "__main__":
    positionalEmbedding = PositionalEmbedding(d_model=32, max_len=512)
    input = torch.ones(1, 512)
    out = positionalEmbedding(input)
    print(out.shape)
