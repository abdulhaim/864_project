#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 20:43:43 2021

@author: Onslaught
"""

from gensim.models.keyedvectors import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
model.save_word2vec_format('GoogleNews-vectors-negative300.txt', binary=False)