from ..common.boundingbox import BoundingBox
import cv2
import numpy as np
import logging
from innotis.common.custom_logger import conf_for_flask
from logging.config import dictConfig
from PIL import Image
# -----------------------------------------------------------------------------------------------------------------------------
# initial logger
dictConfig(conf_for_flask(write_mdoe='w'))

""" 各自圖片進行撿均值（突顯該 圖像 重點） """
def subtract_avg(img):
    
    trg_img = img
    for i in range(3):
        avg=np.average(trg_img[:,:,i])
        trg_img[:,:,i]-=avg
    return trg_img

""" 整個數據集進行撿均值（突顯該 類別 重點） """
def subtract_offset(img, offset=( 103.939, 116.779, 123.68 )):
    trg_img = img
    for i in range(3):
        trg_img[:,:,i]=img[:,:,i]-offset[i]
    return trg_img

""" caffe mode of image processing """
def preprocess(img, input_shape, dtype=np.float32):
    
    img_resize = cv2.resize(img, (input_shape[0], input_shape[1])).astype(dtype)    
    img_avg = subtract_offset(img_resize)
    img_chw = img_avg.transpose( (2, 0, 1) ).astype(dtype)    
    return img_chw 

def peoplenet_process_image(img, input_shape, dtype=np.float32):
    logging.info('PEOPLE NET PRE-PROCESS')
    
    # PIL Way
    # w, h = input_shape[0], input_shape[1]
    # image = Image.fromarray(np.uint8(img))
    # img_resized = image.resize(size=(w, h), resample=Image.BILINEAR)
    # img_resized = np.array(img_resized)
    # img_np = (1.0 / 255.0) * img_resized
    # img_chw = img_np.transpose((2, 1, 0))
    
    # OpenCV Way ( resize: w, h)
    img_resized = cv2.resize(img, (input_shape[0], input_shape[1])).astype(dtype)
    # img_avg=subtract_offset(img_resized) 
    img_avg = img_resized
    img_np = img_avg / 255.0
    # h , w, c
    # c , h, w
    img_chw = img_np.transpose((2, 0, 1))

    return img_chw

def postprocess(output, img_w, img_h, input_shape, conf_th=0.8, nms_threshold=0.5, letter_box=False, model_type=""):
    """
    if is objected detection in yolo ( with 200 TopK ):
        # TopK 是當初 TAO 訓練的時候給予的，代表每張圖最後最多輸出 TopK 個 BBOX
        (-1,200), (-1,200,4), (-1,200), (-1,200)
        num_detections: A [batch_size] tensor containing the INT32 scalar indicating the number of valid detections per batch item. It can be less than keepTopK. Only the top num_detections[i] entries in nmsed_boxes[i], nmsed_scores[i] and nmsed_classes[i] are valid
        nmsed_boxes: A [batch_size, keepTopK, 4] float32 tensor containing the coordinates of non-max suppressed boxes
        nmsed_scores: A [batch_size, keepTopK] float32 tensor containing the scores for the boxes
        nmsed_classes: A [batch_size, keepTopK] float32 tensor containing the classes for the boxes
    """
    (detections, bboxes, scores, labels) = tuple([ np.squeeze(out) for out in output ])
    
    results= []
    for idx in range(detections):
        # print('scores:',scores[idx], '\t')
        x1, y1, x2, y2 = map(float, bboxes[idx]) # x1, y1, x2, y2
        x1, y1, x2, y2 = x1*img_w, y1*img_h, x2*img_w, y2*img_h
        if scores[idx]>=conf_th:
            results.append(BoundingBox(labels[idx], scores[idx], x1, x2, y1, y2, img_h, img_w))
            break
    return results
    # else:
    #     (labels, bboxes) = tuple([ np.squeeze(out) for out in output ]) 
    #     for idx in range(labels):
    #         print(idx)   
    # print('\n\n\n')      
    # [ print(type(item), np.shape(item)) for item in [labels, bboxes] ]
    # print('\n\n\n', labels, bboxes)


def applyBoxNorm(o1, o2, o3, o4, x, y, grid_centers_w, grid_centers_h, box_norm):
    """
    Applies the GridNet box normalization
    Args:
        o1 (float): first argument of the result
        o2 (float): second argument of the result
        o3 (float): third argument of the result
        o4 (float): fourth argument of the result
        x: row index on the grid
        y: column index on the grid
    Returns:
        float: rescaled first argument
        float: rescaled second argument
        float: rescaled third argument
        float: rescaled fourth argument
    """
    o1 = (o1 - grid_centers_w[x]) * -box_norm
    o2 = (o2 - grid_centers_h[y]) * -box_norm
    o3 = (o3 + grid_centers_w[x]) * box_norm
    o4 = (o4 + grid_centers_h[y]) * box_norm
    return o1, o2, o3, o4

def peoplenet_postprocess(outputs, min_confidence, analysis_classes, image_shape, model_shape, wh_format=True):
    """
    Postprocesses the inference output
    Args:
        outputs (list of float): inference output
        min_confidence (float): min confidence to accept detection
        analysis_classes (list of int): indices of the classes to consider
    Returns: list of list tuple: each element is a two list tuple (x, y) representing the corners of a bb
    """
    print("in post")
    # model_h = 544
    # model_w = 960
    model_h = model_shape[0]
    model_w = model_shape[1]
    img_w = image_shape[1]
    img_h = image_shape[0]
    stride = 16
    box_norm = 35.0
    grid_h = int(model_h / stride)
    grid_w = int(model_w / stride)
    grid_size = grid_h * grid_w
    grid_centers_w = []
    grid_centers_h = []

    for i in range(grid_h):
        value = (i * stride + 0.5) / box_norm
        grid_centers_h.append(value)
        
    for i in range(grid_w):
        value = (i * stride + 0.5) / box_norm
        grid_centers_w.append(value)

    logging.info('start doing post process')
    bbs = []
    class_ids = []
    scores = []
    print("grid_h:", grid_h)
    print("grid_w:", grid_w)
    for c in analysis_classes:
    
        x1_idx = c * 4 * grid_size
        y1_idx = x1_idx + grid_size
        x2_idx = y1_idx + grid_size
        y2_idx = x2_idx + grid_size
        boxes = outputs[0]
        
        for h in range(grid_h):
            
            for w in range(grid_w):
    
                i = w + h * grid_w
    
                score = outputs[1][c * grid_size + i]
    
                if score >= min_confidence:
    
                    o1 = boxes[x1_idx + w + h * grid_w]
                    o2 = boxes[y1_idx + w + h * grid_w]
                    o3 = boxes[x2_idx + w + h * grid_w]
                    o4 = boxes[y2_idx + w + h * grid_w]
    
                    o1, o2, o3, o4 = applyBoxNorm(o1, o2, o3, o4, w, h, grid_centers_w, grid_centers_h, box_norm)


                    logging.info('before image_shape')
                    
                    scale_x = int(img_w)/int(model_w)
                    scale_y = int(img_h)/int(model_h)
                    # print("scale:",scale)
                    logging.debug(image_shape)

                    # scale = 2

                    # scale_x = scale
                    # scale_y = scale
                    
                    logging.info('before scale')
                    print("scale_x:", scale_x)
                    print("scale_y:", scale_y)
                    # print("scale_x:",scale_x)
                    # print("o1:", o1)
                    # print("scale_x*o1:", scale_x*o1)
                    xmin = int(scale_x*o1)
                    ymin = int(scale_y*o2)
                    xmax = int(scale_x*o3)
                    ymax = int(scale_y*o4)
                    print("xmin:", xmin, type(xmin))
                    print("ymin:", ymin, type( ymin))
                    print("xmax:", xmax, type(xmax))
                    print("ymax:", ymax, type(ymax))


                    if wh_format:
                        bbs.append([xmin, ymin, xmax - xmin, ymax - ymin])
                    else:
                        bbs.append([xmin, ymin, xmax, ymax])
                    
                    class_ids.append(c)
                    scores.append(float(score))

    return bbs, class_ids, scores