import argparse
from sentences import Sentences
import streamlit as st

import jax
jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

import jax.numpy as np
from transformers import BartConfig, BartTokenizer, BertTokenizer
import sys

from lib.Generator import Generator
from lib.param_utils.load_params import load_params
from lib.en_kfw_nmt.fwd_transformer_encoder_part import fwd_transformer_encoder_part

# ass
from flask import Flask, render_template, request, redirect, url_for
from sentences import Sentences

import os
    
@st.cache_data
def load_model():
    return load_params("atomic-thunder-15-7.dat")

@st.cache_data     
def load_en():
    return BartTokenizer.from_pretrained('facebook/bart-base')

@st.cache_data
def load_yue():
    return BertTokenizer.from_pretrained('Ayaka/bart-base-cantonese')

@st.cache_data
def load_config():
    return BartConfig.from_pretrained('Ayaka/bart-base-cantonese')

def translate():
    # params = load_params(sys.argv[1])
    params = load_model()
    params = jax.tree_map(np.asarray, params)

    tokenizer_en = load_en()
    tokenizer_yue = load_yue() 

    config = load_config()
    generator = Generator({'embedding': params['decoder_embedding'], **params}, config=config)

    thing_to_generate = Sentences.get_sentence()
    print('second: ', thing_to_generate)
    sentences = [
        thing_to_generate,
    ]

    inputs = tokenizer_en(sentences, return_tensors='jax', padding=True)
    src = inputs.input_ids.astype(np.uint16)

    mask_enc_1d = inputs.attention_mask.astype(np.bool_)
    mask_enc = np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]

    encoder_last_hidden_output = fwd_transformer_encoder_part(params, src, mask_enc)
    generate_ids = generator.generate(encoder_last_hidden_output, mask_enc_1d, num_beams=5, max_length=128)


    decoded_sentences = tokenizer_yue.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(decoded_sentences)
    answer = ""
    for sentence in decoded_sentences:
        sentence = sentence.replace(' ', '')
        answer += sentence
        print(sentence)
    Sentences.set_cantonese(answer)

def generate(input_data):
    # sentence = request.form['filename']
    Sentences.set_sentence(input_data)
    print('first:', Sentences.get_sentence())

    translate()


