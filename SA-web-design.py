# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify

from code.input_sentence import input_sentence
from code.train import train

app = Flask(__name__)


# @app.route('/')
# def hello_world():
#     return 'Hello World!'


@app.route('/train')
def lstm_train():
    trainStr = request.args.get('trainStr')
    print trainStr
    model, word2index = train()
    train_result = input_sentence(trainStr, model, word2index)
    return jsonify(result=train_result)


@app.route('/_add_numbers')
def add_numbers():
    print '计算'
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    print a + b
    return jsonify(result=a + b)


@app.route('/')
def index():
    from flask import render_template
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
