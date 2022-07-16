import base64
import os
import numpy as np
import paddlex
from PIL import Image
from flask import Flask, url_for, request, render_template, redirect, current_app
from flask_cors import CORS
from matplotlib import pyplot as plt
from paddle.dataset.image import cv2
from paddlers.deploy import Predictor


app = Flask(__name__)
CORS(app, resources=r'/*')

#数据库登录与注册


@app.route('/')
def hello_world():
    return render_template("html/test.html")

@app.route('/main')
def main():
    return render_template("html/main.html")

# 地物分类
@app.route('/classification', methods=['GET', 'POST'])
def classification():
    if request.method == "POST":
        name = pic_doing('Classification/')
        predictor = Predictor("model/static model/Classification of features")
        res = predictor.predict("static/Classification/" + name)
        cm_1024x1024 = res['label_map']
        img = cm_1024x1024.astype('uint8')
        print("地物分类👌")
        filepath = 'static/result/Classification.png'

        cv2.imwrite(filepath, img)
        # 参考 https://stackoverflow.com/a/68209152
        fig = plt.figure(figsize=(10.24, 10.24), constrained_layout=True)
        subfigs = fig.subfigures(nrows=1, ncols=1)
        axs = subfigs.subplots(nrows=1, ncols=1)
        im = filepath
        lut = get_lut()
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.spines['bottom'].set_visible(False)
        axs.spines['left'].set_visible(False)
        axs.get_xaxis().set_ticks([])
        axs.get_yaxis().set_ticks([])
        if isinstance(im, str):
            im = cv2.imread(im, cv2.IMREAD_COLOR)
        if lut is not None:
            if im.ndim == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = lut[im]
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # file = Image.open(im)
        axs.imshow(im)
        # axs.imshow(a)
        fig.canvas.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        image = np.asarray(image)
        # 保存图片
        Image.fromarray(np.uint8(image)).convert('RGB').save(filepath)

        # cv2.imwrite(filepath, img)
        result = turn_web(filepath)
        return result
    else:
        return render_template("html/Classification.html")


# 变化检测
@app.route('/changing', methods=['GET', 'POST'])
def changing():
    if request.method == "POST":
        name1 = pic_doing('Changing/')
        name2 = pic_doing1('Changing/')
        predictor = Predictor("model/static model/Changing detection")
        res = predictor.predict(("static/Changing/" + name1, "static/Changing/" + name2))
        cm_1024x1024 = res[0]['label_map']
        img = (cm_1024x1024 * 255).astype('uint8')
        print("变化检测👌")
        filepath = 'static/result/Changing.png'

        # 参考 https://stackoverflow.com/a/68209152
        fig = plt.figure(figsize=(10.24, 10.24), constrained_layout=True)
        subfigs = fig.subfigures(nrows=1, ncols=1)
        axs = subfigs.subplots(nrows=1, ncols=1)
        cv2.imwrite(filepath, img)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.spines['bottom'].set_visible(False)
        axs.spines['left'].set_visible(False)
        axs.get_xaxis().set_ticks([])
        axs.get_yaxis().set_ticks([])
        file = Image.open(filepath)
        axs.imshow(file)
        fig.canvas.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        image = np.asarray(image)
        # 保存图片
        Image.fromarray(np.uint8(image * 255)).convert('RGB').save(filepath)
        result = turn_web(filepath)
        return result
    else:
        return render_template("html/changing.html")


# 目标提取
@app.route('/extraction', methods=['GET', 'POST'])
def extraction():
    if request.method == "POST":
        name = pic_doing('extraction/')
        # basepath = os.path.dirname(__file__)
        predictor = Predictor("model/static model/Object extraction")
        res = predictor.predict("static/extraction/" + name)
        cm_1024x1024 = res['label_map']
        img = (cm_1024x1024 * 255).astype('uint8')
        print("目标提取👌")
        filepath = 'static/result/extraction.png'

        # 参考 https://stackoverflow.com/a/68209152
        fig = plt.figure(figsize=(10.24, 8.75), constrained_layout=True)
        subfigs = fig.subfigures(nrows=1, ncols=1)
        axs = subfigs.subplots(nrows=1, ncols=1)
        cv2.imwrite(filepath, img)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.spines['bottom'].set_visible(False)
        axs.spines['left'].set_visible(False)
        axs.get_xaxis().set_ticks([])
        axs.get_yaxis().set_ticks([])
        file = Image.open(filepath)
        axs.imshow(file)
        fig.canvas.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        image = np.asarray(image)
        # 保存图片
        Image.fromarray(np.uint8(image * 255)).convert('RGB').save(filepath)

        result = turn_web(filepath)
        return result
    else:
        return render_template("html/Extraction.html")


basedir = os.path.abspath(os.path.dirname(__file__))
# 目标检测
@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if request.method == "POST":
        name = pic_doing('detection/')
        predictor = Predictor("model/static model/Object detection")
        res = predictor.predict("static/detection/" + name)

        # 保存图片
        paddlex.det.visualize(basedir + "/static/detection/" + name, res, threshold=0.1,
                              save_dir=basedir + '/static/result/')
        # img = Image.open(basedir + "/static/result/visualize_" + name)
        # img = cv2.imdecode(np.fromfile(res, dtype=np.uint8), cv2.IMREAD_COLOR)
        # cm_1024x1024 = res['bbox']
        # img = (cm_1024x1024 * 255).astype('uint8')
        print("目标检测👌")
        filepath = 'static/result/visualize_' + name
        # cv2.imwrite(filepath, img)
        result = turn_web(filepath)
        return result
    else:
        return render_template("html/Detection.html")

@app.route('/cannot')
def cannot():
    return render_template("cannot.html")


#地物分类根据类别进行上色
def get_lut():
    lut = np.zeros((256,3), dtype=np.uint8)
    lut[0] = [255, 0, 0]     #红 建筑
    lut[1] = [30, 255, 142]  #绿 树林
    lut[2] = [60, 0, 255]    #蓝 道路
    lut[3] = [255, 222, 0]   #黄 湖泊
    lut[4] = [0, 0, 0]       #黑 地面
    return lut


# 实现上传图片到a文件夹
def pic_doing(a):
    img = request.files.get('files')
    print(img)
    print(img.filename)
    path = basedir + "/static/"
    file_path = path + a + img.filename
    img.save(file_path)
    return img.filename


def pic_doing1(a):
    img = request.files.get('files1')
    print(img)
    print(img.filename)
    path = basedir + "/static/"
    file_path = path + a + img.filename
    img.save(file_path)
    return img.filename


# 返回图片给前端
def turn_web(filepath):
    # filename = file_name;
    # filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream


if __name__ == '__main__':
    # app.debug=True
    app.run(host='0.0.0.0',port=443)
