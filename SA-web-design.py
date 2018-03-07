# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify

from code.input_sentence import input_sentence
from code.train import train

app = Flask(__name__)


@app.route('/')
def index():
    from flask import render_template
    return render_template('index.html')


# Keras训练
@app.route('/train')
def lstm_train():
    trainStr = request.args.get('trainStr')
    print trainStr
    # train_result = input_sentence(trainStr, model, word2index)
    return jsonify(result=1)


# 导入情感训练结果
@app.route('/upload')
def upload():
    print '计算'
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    print a + b
    return jsonify(result=a + b)


# 情感分析
@app.route('/SA')
def s_a():
    print '计算'
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    print a + b
    return jsonify(result=a + b)


if __name__ == '__main__':
    app.run()
