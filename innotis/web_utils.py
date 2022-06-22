import ast, sys, os, random, string, base64, cv2, io
import logging
from PIL import Image

SEED = 0


# -----------------------------------------------------------------------------------------------------------------------------
''' 檢查副檔名：有副檔名 以及 副檔名有在制定的規範內 回傳 mode '''
def allowed_file(filename, ext):
    return '.' in filename and filename.rsplit('.', 1)[1] in ext

# -----------------------------------------------------------------------------------------------------------------------------
''' 取得模型資訊之字串 '''
def get_model_info(model):

    info = ''
    [ info + f'{item}: {val}\n' for item, val in model.items() ] 
    return info

# -----------------------------------------------------------------------------------------------------------------------------
'''隨機產生UUID'''
def gen_uuid(img_path):
    
    global SEED
    random.seed( SEED )
    
    name, ext = img_path.rsplit('.', 1)
    fakeid='?uuid='
       
    for i in range(3):
        fakeid += '{}{}'.format(random.randint(0,9), string.ascii_letters[random.randint(0,26)])  # 取得 隨機 數值 與 字母

    SEED += 1

    return '{}{}.{}'.format(name, fakeid, ext)

# -----------------------------------------------------------------------------------------------------------------------------
'''清除資料夾中的圖片'''
def clear_images(dir, ext): 
    [ os.remove(os.path.join(dir, file)) for file in os.listdir(dir) if file.endswith(tuple(ext)) ]

# -----------------------------------------------------------------------------------------------------------------------------
'''解析辨識結果'''
def parse_results(res, isYOLO):
    
    if isYOLO:
        index=0
        det_nums = 0 if res[0][0] is 'None' else len(res)   # 如果第一筆資料是 None 代表沒有辨識到結果

        results ='Detected Objects: {} \n'.format(det_nums)
        for classes, confidence in res.values():
            index+=1
            results = results + '[{}] {} , {:.3f}\n'.format(index, classes, confidence)
    else:
        import numpy as np
        [logging.info(cnt) for cnt in ['\n\n\n\n\n', '-'*10, res, '\n\n', type(res), '\n\n', np.array(res).shape, '\n\n\n\n\n']]
        index, classes, confidence = res[0]
        results = '[{}] {} , {:.3f}\n'.format(index, classes, confidence)

    return results

# -----------------------------------------------------------------------------------------------------------------------------
'''Convet cv to base64'''
def cv2base64(img):
    # cv to pillow
    img = Image.fromarray(cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2RGB))
    
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)        # clear temp
    img_base64 = base64.b64encode(rawBytes.getvalue()).decode('ascii')
    mime = "image/jpeg"
    uri = "data:%s;base64,%s"%(mime, img_base64)
    return uri




'''open camera '''
def camera_open():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    data = cv2base64(frame)

    return data