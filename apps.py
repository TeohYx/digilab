import streamlit as st
from sentences import Sentences
import file
from sentences import Sentences
import streamlit as st

import jax

import jax.numpy as np
from transformers import BartConfig, BartTokenizer, BertTokenizer

from lib.Generator import Generator
from lib.param_utils.load_params import load_params
from lib.en_kfw_nmt.fwd_transformer_encoder_part import fwd_transformer_encoder_part

from sentences import Sentences

import pickle

import gdown
import gc


@st.cache_data
def download():
    url = "https://drive.google.com/file/d/1fTUGsXiIz_egy4qM_R4Jujg2gCNFuZsX/view?usp=sharing"
    file_id = url.split('/')[-2]
    prefix = 'https://drive.google.com/uc?/export=download&id='
    return gdown.download(prefix+file_id) 
    # file_url = 'https://drive.google.com/file/d/1fTUGsXiIz_egy4qM_R4Jujg2gCNFuZsX/view?usp=sharing'



    
@st.cache_data
def load_model():
    # return load_params("atomic-thunder-15-7.dat")
    with open('model', 'rb') as f:
        model = pickle.load(f)

    return load_params(model)

@st.cache_data     
def load_en():
    return BartTokenizer.from_pretrained('facebook/bart-base')

@st.cache_data
def load_yue():
    return BertTokenizer.from_pretrained('Ayaka/bart-base-cantonese')

@st.cache_data
def load_config():
    return BartConfig.from_pretrained('Ayaka/bart-base-cantonese')
    
# Translate text
def translate(input_data):
    Sentences.set_sentence(input_data)
    print('first:', Sentences.get_sentence())
    # params = load_params(sys.argv[1])
    params = load_params("atomic-thunder-15-7.dat")
    print("test1")
    params = jax.tree_map(np.asarray, params)
    print("test2")
    tokenizer_en = load_en()
    print("test3")
    tokenizer_yue = load_yue() 
    print("test4")

    config = load_config()
    print("test5")
    generator = Generator({'embedding': params['decoder_embedding'], **params}, config=config)
    print("test6")
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

    gc.collect()

jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

st.set_page_config(page_title="Dialect Translator", page_icon=":tada:")

# download()
# print(test)


spinner = None
with st.container():
    st.write("---")
    st.header("English to Cantonese Translator")
    st.write("##")

    input_data = st.text_input('English Sentences:', value="")
    submit_button = st.button('Generate')
    audio_button = st.button('Record Audio')

    if audio_button:
        file.record_audio()
        if Sentences.get_sentence() != "":   
            st.write("#")
            st.write("Translated Cantonese:")
            with st.spinner():
                translate(Sentences.get_sentence())
                while Sentences.is_translated() is False:
                    st.write('Translating...')
            # input_data.write(Sentences.get_sentence())
            # st.text_input('English Sentences:', value=Sentences.get_sentence())      
            print("input data: " + input_data)
        else:
            # st.write(Sentences.get_popupmsg())
            st.error(Sentences.get_popupmsg())

    if submit_button:
        # Call the function in other_file.py and pass the input_data as an argument
        st.write("#")
        st.write("Translated Cantonese:")
        with st.spinner():
            translate(input_data)
            while Sentences.is_translated() is False:
                st.write('Translating...')

    if Sentences.is_translated() is True:
        st.write(Sentences.get_cantonese())
