import argparse
from sentences import Sentences

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


def translate():
    params = load_params(sys.argv[1])
    params = jax.tree_map(np.asarray, params)

    tokenizer_en = BartTokenizer.from_pretrained('facebook/bart-base')
    tokenizer_yue = BertTokenizer.from_pretrained('Ayaka/bart-base-cantonese')

    config = BartConfig.from_pretrained('Ayaka/bart-base-cantonese')
    generator = Generator({'embedding': params['decoder_embedding'], **params}, config=config)



    # parser = argparse.ArgumentParser()
    # parser.add_argument('filename', type=str)
    # parser.add_argument('--sentence', type=str)
    # args = parser.parse_args()

    # Sentences.set_sentence(args.sentence)
    # thing_to_generate = Sentences.get_sentence().replace("'", "")
    thing_to_generate = Sentences.get_sentence()
    print('second: ', thing_to_generate)
    sentences = [
        thing_to_generate,
        # 'The sky is so blue.',
        # 'I am gonna done this project by today!',
        # 'enter the spotlight',
        # 'Are you feeling unwell?',
        # 'How long have you been waiting?',
        # 'There are many bubbles in soda water.',
        # 'He feels deeply melancholic for his past.',
        # 'She prepared some mooncakes for me to eat.',
        # 'This gathering only allows adults to join.',
        # "Do you know it's illegal to recruit triad members?",
        # "Today I'd like to share some tips about making a cake.",
        # 'An annual fee is required if one wants to use the bank counter service.',
        # 'You guys have no discipline, how can you be part of the disciplined services?',
        # 'We need to offer young people drifting into crime an alternative set of values.',
        # 'A tiger is put with equal probability behind one of two doors, while treasure is put behind the other one.',
        # "Recently he's been working so hard to get rid of his nerdy way of living, now he looks so different than before.",
        # 'Running the marathon race in such a hot day, I almost collapsed as I arrived the destination.',
        # 'He says flowery and pretty words, no one knows when his praise is real and when is not.',
        # 'After 10 years of investigation, the police can finally figure out the man behind this murder case.',
        # "She has vowed for many times that she would slim down and thus never eat anything fatty, but she still has a fried chicken thigh every week. She can't keep her commitment.",
        # 'Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles.',
        # # sentences in the training dataset
        # 'Adults should protect children so as to avoid them being sexually abused.',
        # 'Clerks working on a construction site are also construction site workers, engineers are also construction site workers.',
        # "A taxi opposite me didn't use its signal light, and when it was driving on my right front side it suddenly turned in toward my left. Luckily I was able to stop in time. Otherwise I would have been hit hard!",
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


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    sentence = request.form['filename']
    Sentences.set_sentence(sentence)
    print('first:', Sentences.get_sentence())

    # command = "python generate.py atomic-thunder-15-7.dat --sentence '{}'".format(Sentences.get_sentence())
    # subprocess.run(command, shell=True)
    translate()

    return render_template('index.html', content=Sentences.get_cantonese())

if __name__ == '__main__':
    app.run()


# ass
