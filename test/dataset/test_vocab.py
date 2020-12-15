#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on 2020-12-15 17:06

@author: caoshaoping
"""
from bert_pytorch.dataset.vocab import WordVocab

if __name__ == "__main__":
    with open("../data/vocab.txt", "r") as f:
        vocab = WordVocab(f)
        print(vocab.freqs)
