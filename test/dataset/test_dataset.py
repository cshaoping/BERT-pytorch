#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on 2020-12-15 17:19

@author: caoshaoping
"""
from bert_pytorch.dataset.dataset import BERTDataset
from bert_pytorch.dataset.vocab import WordVocab


if __name__ == "__main__":
    corpus_path = "../data/corpus.txt"
    vocab_path = "../data/vocab.txt"
    vocab = WordVocab(vocab_path)
    dataset = BERTDataset(corpus_path, vocab, seq_len=10)

