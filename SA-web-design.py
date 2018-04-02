# -*- coding: utf-8 -*-
import os
from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from code.input_sentence import input_sentence
from code.train import train

app = Flask(__name__)

UPLOAD_FOLDER = '/Users/jianglingjun/Document/PycharmProjects/SA-web-design/data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # 设置文件上传的目标文件夹
basedir = os.path.abspath(os.path.dirname(__file__))  # 获取当前项目的绝对路径
ALLOWED_EXTENSIONS = set(['txt', 'png', 'jpg', 'xls', 'JPG', 'PNG', 'xlsx', 'gif', 'GIF'])  # 允许上传的文件后缀


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


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        # if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_url = url_for('uploaded_file', filename=filename)
        print file_url


# 下载功能
@app.route("/download/<path:filename>")
def downloader(filename):
    dirpath = os.path.join(app.root_path, 'data')  # 这里是下在目录，从工程的根目录写起，比如你要下载static/js里面的js文件，这里就要写“static/js”
    return send_from_directory(dirpath, filename, as_attachment=True)  # as_attachment=True 一定要写，不然会变成打开，而不是下载


# 情感分析
@app.route('/predict')
def s_a():
    print '分析'
    q = request.args.get('q')
    result = input_sentence(q)
    if result == 1:
        return (render_template('main.html', image_name='laugh.jpeg'))
    elif result == 0:
        return (render_template('main.html', image_name='cry.jpeg'))


if __name__ == '__main__':
    app.run()
