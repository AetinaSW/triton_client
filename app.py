# -*- coding: UTF-8 -*-
from asyncio.log import logger
from distutils.log import debug
import shutil
from unicodedata import name
from flask import Flask, config, request, redirect, url_for, render_template, Response
from werkzeug.utils import secure_filename
from flask import send_from_directory
from flask.logging import default_handler

import cv2, os, json, sys, random, string, base64, io
import numpy as np
from PIL import Image
import threading, time

from innotis.client import Client
from innotis.common.json_parser import JsonParser
from innotis.common.custom_logger import conf_for_flask
from innotis.web_utils import camera_open, cv2base64, clear_images, gen_uuid, allowed_file, parse_results
from logging.config import dictConfig

# -----------------------------------------------------------------------------------------------------------------------------
# initial logger
dictConfig(conf_for_flask(write_mdoe='w'))

# initial Flask application
app = Flask(__name__)

# -----------------------------------------------------------------------------------------------------------------------------
''' 串流模式，只能寫在這裡，後續可以考慮 WebRTC 去整 '''
def stream_infer():

    client = app.config['CLIENT']
    cap = app.config['CAP']
    model = app.config['MODEL']

    while(app.config['STREAM']):
        
        ret, frame = cap.read()
        
        if not ret: break

        res, image_draw =  client.image_infer(frame, model['width'], model['height'])

        ret, jpg = cv2.imencode('.jpg', image_draw )

        yield ( b'--frame\r\n' + 
                b'Content-Type: image/jpeg\r\n\r\n' + 
                jpg.tobytes() + 
                b'\r\n')    

    app.config['STREAM'] = False

# -----------------------------------------------------------------------------------------------------------------------------
'''根目錄 : 通常會放置首頁'''
@app.route('/', methods=['GET', 'POST'])
def index():
    
    app.config['STREAM'] = False    # 初始化 STREAM 參數
    app.logger.info('Clear directory ({}) '.format(app.config['UPLOAD_FOLDER']))     # 每次進首頁都會清除所有檔案
    clear_images(app.config['UPLOAD_FOLDER'], app.config['ALLOWED_EXTENSIONS'])

    return render_template('index.html', models=app.config['MODEL_INFO'].items())   # 使用 render_template 會自動去 templates 抓取 index.html

# -----------------------------------------------------------------------------------------------------------------------------
'''不斷餵新的畫面'''
@app.route("/_feedFrame") 
def feed_frame():
    return Response(stream_infer(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------------------------------------------------------------------------------------------------------
'''取得選擇的模型'''
@app.route('/_sel', methods=['GET', 'POST'])
def get_sel():
    
    app.config['MODEL'] = app.config['MODEL_INFO'][request.get_json()]  # 取得特定的模型資訓

    app.config['CLIENT_SETUP'] = app.config['MODEL_INFO'][request.get_json()]["option"] # 該模型的相關 Infer 參數
    
    app.config['YOLO'] = 'yolo' in app.config['MODEL']['name'] or 'usb_detector' in app.config['MODEL']['name'] # 確認是否是 YOLO

    app.logger.info('Selected Model ... {}'.format(app.config['MODEL']['name']))
    
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

# -----------------------------------------------------------------------------------------------------------------------------
'''取得選擇到的模式'''
@app.route('/_mode', methods=['GET', 'POST'])
def get_mode():
    app.config['MODE'] = request.get_json()
    app.logger.info('Selected Mode ... {}'.format(app.config['MODE']))
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

# -----------------------------------------------------------------------------------------------------------------------------
'''註冊 upload 位置'''
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    
    # 取得 IP 後建立 Client 端物件
    model = app.config['MODEL']
    mode = app.config['MODE']
    client = Client( url=request.form.get('ip'), model_info=model )

    app.logger.info('Check mode ... {}'.format(mode))
    if mode == 'camera':
        uploaded_files = camera_open()
    else:
        uploaded_files = request.files.getlist("file[]")    # 取得上傳的檔案

    # -------------------------------------------------------------------------
    # Image 讀取到檔案後直接轉成 Base64 的形式回傳到網頁上 ( 沒有進行存檔 )
    if mode == 'image':      
        
        app.logger.info("IMAGE MODE")
        data = []
        
        for file in uploaded_files:

            if not (file and allowed_file(file.filename, ALLOWED_EXTENSIONS)) : 
                print("Not allowed extensions or no such file ... {}".format(file))
                continue

            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)      # 轉換成 cv image

            filename = secure_filename(file.filename)                       # 檢查檔案 & 儲存到自訂的目錄
            out_name = gen_uuid(filename)                                   # 取得獨特的UUID，避免 HTML 讀取舊的檔案
            out_pth = os.path.join(app.config['UPLOAD_FOLDER'], out_name)   # 取得除群位置
            
            results, image_draw =  client.image_infer(img, model['width'], model['height'], out_pth)    # 進行 Inference

            
            # info = parse_results(res=results, isYOLO=app.config['YOLO'])  # 解析結果
            
            # data.append(  (out_name if app.config['YOLO'] else filename, info, cv2base64(image_draw)) ) # 將資料記錄下來     
            data.append(  (out_name if app.config['YOLO'] else filename, results, cv2base64(image_draw) ) ) # 將資料記錄下來     

        # 更新到 result 頁面
        return render_template('result.html', mode=mode, data=data, wid=len(data))
    
    # -------------------------------------------------------------------------
    # Video 不能透過file.read讀取圖片，所以只能用存取的方式
    elif mode == 'video':       
        
        # 如果非本地端則傳送過去儲存下來
        file = uploaded_files[0]
        filename = secure_filename(file.filename) 

        save_pth = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_pth)

        # 取得 UUID 的輸出
        out_name = gen_uuid(filename)       
        out_pth = os.path.join(app.config['UPLOAD_FOLDER'], out_name)

        # 進行推論取得結果
        ret, info = client.video_infer(save_pth, model['width'], model['height'], out_pth)

        return render_template('result.html', mode=mode, out_pth=out_name, info=info) 
    
    # -------------------------------------------------------------------------
    # Stream 每一幀讀取完就直接回傳 
    elif mode=='stream':    
        
        app.logger.info("VIDEO MODE")
        file = uploaded_files[0]                    # 一次只能一個檔案，直接限定
        filename = secure_filename(file.filename)   # 檢查檔名
        save_path = os.path.abspath(filename)       # 取得路徑
        # -------------------------------------------------------------------------
        if not os.path.exists(filename):            # 如果非本地端則傳送過去儲存下來
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            app.logger.warning("Sending file to server ... ({})".format(save_path))

        # -------------------------------------------------------------------------
        cv2.VideoCapture(0).release()
        cv2.destroyAllWindows()
        app.config['CLIENT'] = client                                       
        app.config['CAP'] = cv2.VideoCapture(str(save_path)) # 將相機物件儲存進 app.config
        app.config['STREAM']  = True    # 允許串流
        return render_template('result.html', mode=mode)


    # -------------------------------------------------------------------------    
    elif mode == 'camera':
        app.logger.info("VIDEO MODE")
        file = uploaded_files[0]                    # 一次只能一個檔案，直接限定
        # filename = secure_filename(file.filename)   # 檢查檔名
        # save_path = os.path.abspath(filename)       # 取得路徑
        # -------------------------------------------------------------------------
        # if not os.path.exists(filename):            # 如果非本地端則傳送過去儲存下來
        #     save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #     file.save(save_path)
        #     app.logger.warning("Sending file to server ... ({})".format(save_path))

        # -------------------------------------------------------------------------
        app.config['CLIENT'] = client                                       
        app.config['CAP'] = cv2.VideoCapture(0) # 將相機物件儲存進 app.config
        app.config['STREAM']  = True    # 允許串流
        return render_template('result.html', mode=mode)
    # -------------------------------------------------------------------------

    
    else:
        app.logger.error("Error")
        sys.exit(1)

# -------------------------------------------------------------------------

if __name__ == '__main__':

    # ----------------------------------------------------------------------------------------
    # initial basic parameters
    app.logger.info("Initialize flask application.")    # Flask logger 的使用方法



    UPLOAD_FOLDER = './static'  # 上傳到的資料夾 ( Server端 ) 
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'mp4']) # 設定 副檔名 規範
    MAX_FILE_SIZE = 16
    JSON_FILE_PATH = "./configs/models.json"
    UPLOAD_FOLDER = './static'  # 上傳到的資料夾 ( Server端 ) 
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'mp4']) # 設定 副檔名 規範
    MAX_FILE_SIZE = 16
    JSON_FILE_PATH = "./configs/models.json"

    # ----------------------------------------------------------------------------------------
    # set up variable of flask application
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER     # 定義 APP 的全域變數
    app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
    model_info = JsonParser(JSON_FILE_PATH).get_data()  # 解析 模型資訊 (JSON)
    app.config['MODEL_INFO'] = model_info
    app.config.update(  CLIENT = [],                # 當有多個變數的時候就需要使用 update       
                        MODEL = None,
                        MODE = 'image',
                        YOLO = False,
                        SEED = 1,
                        STREAM = False,
                        CAP = [],
                        OUT_PATH = '')
    # app.config['MAX_FILE_SIZE'] = MAX_FILE_SIZE * 1024 * 1024  # 設定上傳最大檔案的限制 16MB
    # app.config['CLIENT_SETUP'] = model_info["option"] # 其他的 client 設定

    if not os.path.exists(UPLOAD_FOLDER): os.mkdir(UPLOAD_FOLDER)   # 檢查暫存用的資料夾是否存在
    
    # ----------------------------------------------------------------------------------------
    # run application
    app.logger.info("Run flask application.")
    if len(sys.argv)>1 and sys.argv[1].lower()=="debug":
        app.run(host='0.0.0.0', port='5000', threaded=True, debug=True)
    else:
        app.run(host='0.0.0.0', port='5000', threaded=True)
