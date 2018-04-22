# -*- coding: utf-8 -*-
import os
from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from code.input_sentence import input_sentence
from code.train import train
from utils.constant import image_urls

app = Flask(__name__)

UPLOAD_FOLDER = '/Users/jianglingjun/Document/PycharmProjects/SA-web-design/data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # 设置文件上传的目标文件夹
basedir = os.path.abspath(os.path.dirname(__file__))  # 获取当前项目的绝对路径
ALLOWED_EXTENSIONS = set(['txt', 'png', 'jpg', 'xls', 'JPG', 'PNG', 'xlsx', 'gif', 'GIF', 'h5'])  # 允许上传的文件后缀


@app.route('/')
def index():
    return render_template('main.html')


# Keras训练
@app.route('/train')
def lstm_train():
    train()
    return jsonify(result=1)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# 上传文件
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('index.html')


# 下载功能
@app.route("/download/<path:filename>")
def downloader(filename):
    dirpath = os.path.join(app.root_path, 'data')
    return send_from_directory(dirpath, filename, as_attachment=True)


# 情感分析预测
@app.route('/predict')
def S_A():
    predict_text = request.args.get('q')
    if predict_text == '':
        return jsonify(image_url=image_urls[0])
    result = input_sentence(predict_text)
    print image_urls[result]
    return jsonify(image_url=image_urls[result])


if __name__ == '__main__':
    app.run()
