from flask import Flask, render_template, request
import subprocess
from sentences import Sentences

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    sentence = request.form['filename']
    Sentences.set_sentence(sentence)
    print('first:', Sentences.get_sentence())

    command = "python generate.py atomic-thunder-15-7.dat --sentence '{}'".format(Sentences.get_sentence())
    subprocess.run(command, shell=True)

    return Sentences.get_cantonese()

if __name__ == '__main__':
    app.run()