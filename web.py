import sys
import os
import numpy as np
import json
import urllib
import pandas as pd
from flask import Flask, request, redirect, url_for
from flask import send_from_directory, render_template
from datetime import datetime
from mercari import PriceModel

sys.path.append(os.curdir)  # カレントファイルをインポートするための設定
app = Flask(__name__, static_url_path='/static', static_folder='assets/static')

model = PriceModel()

model.load_data('train.10000.tsv')

model.train()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/api/predict', methods=['GET', 'POST'])
def predict():
    item = request.args.get('item', "")
    print("get args item:", item)

    if(item == ""):
        item = request.get_json(force=True)
        print("get body json item:", item)
    else:
        item = json.loads(item)

    X = pd.DataFrame().append(item, ignore_index=True)

    if(len(X) == 0):
        return json.dumps([0.0], ensure_ascii=False)

    print("X:", X)

    result = model.predict(X)

    result = [{'yen': int(price * 113.13), 'dollar': price}
              for price in result]

    ret = json.dumps(result, ensure_ascii=False)
    print("ret:", ret)
    return ret


@app.route('/api/train')
def train():
    filename = request.args.get('filename', "train.10000.tsv")
    model.load_data(filename)
    return model.train()


@app.route('/')
def serve_index():
    return send_from_directory('assets', 'index.html')


@app.route('/<filename>', defaults={'filename': 'index.html'})
def serve_assets(filename):
    return send_from_directory('assets', filename)


if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)
