from ..common.boundingbox import BoundingBox
import cv2
import numpy as np

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
def preprocess(img, input_shape, letter_box=False):
    """Preprocess an image before TRT YOLO inferencing.
    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)
        letter_box: boolean, specifies whether to keep aspect ratio and
                    create a "letterboxed" image for inference
    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img

def postprocess(output, img_w, img_h, input_shape, conf_th=0.8, nms_threshold=0.5, letter_box=False):
    """
    (-1,200), (-1,200,4), (-1,200), (-1,200)
    num_detections: A [batch_size] tensor containing the INT32 scalar indicating the number of valid detections per batch item. It can be less than keepTopK. Only the top num_detections[i] entries in nmsed_boxes[i], nmsed_scores[i] and nmsed_classes[i] are valid
    nmsed_boxes: A [batch_size, keepTopK, 4] float32 tensor containing the coordinates of non-max suppressed boxes
    nmsed_scores: A [batch_size, keepTopK] float32 tensor containing the scores for the boxes
    nmsed_classes: A [batch_size, keepTopK] float32 tensor containing the classes for the boxes
    """

    (detections, bboxes, scores, labels) = output

    results= []
    for idx, det in enumerate(detections):
        # print(scores[idx], '\t')
        x1, y1, x2, y2 = map(float, bboxes[idx]) # x1, y1, x2, y2
        x1, y1, x2, y2 = x1*img_w, y1*img_h, x2*img_w, y2*img_h
        if scores[idx]>=conf_th:
            results.append(BoundingBox(labels[idx], scores[idx], x1, x2, y1, y2, img_h, img_w))

    return results