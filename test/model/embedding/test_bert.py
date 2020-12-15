#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on 2020-12-15 17:43

@author: caoshaoping
"""
import torch

from bert_pytorch.model.embedding.bert import BERTEmbedding


if __name__ == "__main__":
    embedding = BERTEmbedding(vocab_size=1000, embed_size=32)
    sequence = torch.ones(1, 512)
    segment_label = torch.ones(1, 3)

    out = embedding(sequence, segment_label)
    print(out)
