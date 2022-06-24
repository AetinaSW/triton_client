#!/usr/bin/env python
# -*- coding: utf-8 -*-
from unicodedata import name
import numpy as np
import logging, sys, cv2, os, time, shlex, ast
import subprocess as sp

import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from innotis.common.render import render_dets
import time

class Client:
    # -----------------------------------------------------------------------------------------------------------------------------
    ''' 初始化相關參數與設定 '''
    def __init__(self, url:str, model_info:dict):
        
        # initialize 
        logging.info('Initialize')
        self.url = url

        self.model_info = model_info
        self.model = model_info['name'] # 模型名稱
        self.task = model_info['task']  # 任務 ( classification, objected detection ... )
        self.is_tao = model_info['TAO'] # 是否是 TAO 訓練出來的，darknet 跟 TAO 的 object detection 解析程式不同

        self.label = self.get_label(model_info['label'])         # 透過 get_label 取得 標籤檔的內容
        self.confidence = model_info['option']['conf']      # 取得 confidence 的閥值
        self.nms = model_info['option']['nms'] if 'nms' in model_info['option'].keys() else None    # 如果沒給 nms 就是 None，只有 darkent 會用到
        
        self.get_info = model_info['option']['info']            # 是否要印出相關資訓
        self.client_timeout = model_info['option']['timeout']   # 是否要開啟 timeout 功能
        
        # -------------------------------------------------------------------------------------------------
        # set up flags
        self.tensorrt = False                                                                  # 是否為 tensorrt
        self.darknet_yolo = 'yolo' in self.model and (not self.is_tao)                         # 是否是 darknet 的 yolo
        self.multi_output = True if (('object' in self.task) and self.is_tao ) else False      # 是否有多個輸出「層」

        # -------------------------------------------------------------------------------------------------
        # load different module based on the sources    
        if self.darknet_yolo:
            logging.info('Load darknet module')
            import innotis.darknet.processing as proc
        elif self.is_tao:
            logging.info('Load tao module')
            import innotis.tao.processing as proc
        else:
            logging.info('Load other module')
            import innotis.default.processing as proc
        
        if 'peoplenet' in model_info['name'] or 'DashCamNet' in model_info['name']:
            import innotis.tao.processing as proc
            logging.debug("Using DetectNet_v2 Preprocess ...")
            self.prep = proc.peoplenet_process_image
            self.postp = proc.peoplenet_postprocess
        else:
            self.prep, self.postp = proc.preprocess, proc.postprocess

        # -------------------------------------------------------------------------------------------------
        # Create Triton Client
        try:
            self.triton_client = grpcclient.InferenceServerClient(url=url,verbose=False)
            logging.info('Created triton client !')
        except Exception as e:
            logging.error("Context creation failed: " + str(e))
            sys.exit()

        # -------------------------------------------------------------------------------------------------
        # Check Health and Model Information
        self.check_triton_status()
        if self.get_info==True : self.get_model_info()

        # -------------------------------------------------------------------------------------------------
        # define & get model parameters
        self.input_name, self.output_name, self.input_dims, self.input_size, self.dtype = self.parse_model()
        
        c, h, w = self.input_size
        self.inputs_shape = [c, h, w] if self.input_dims==3 else [1, c, h, w] 

        logging.debug('Initialize ... Done')

    # -----------------------------------------------------------------------------------------------------------------------------
    ''' 取得標籤物件 '''
    def get_label(self, label_name):
        
        label_file_root = 'labels'

        label_path = os.path.join(label_file_root, label_name)

        with open(label_path, 'r') as f:
            cnt = f.read()
            label = ast.literal_eval(cnt)

        return label

    # -----------------------------------------------------------------------------------------------------------------------------
    ''' 解析模型資訊，取得輸入層 輸出層 資訊 '''
    def parse_model(self):
        
        # -----------------------------------------------------------------------------------
        logging.info('Parsing model parameters ... ')

        channel = grpc.insecure_channel(self.url)
        grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

        metadata_request = service_pb2.ModelMetadataRequest( name=self.model, version="")
        model_metadata = grpc_stub.ModelMetadata(metadata_request)

        config_request = service_pb2.ModelConfigRequest(name=self.model,version="")
        model_config = grpc_stub.ModelConfig(config_request).config

        platform = model_config.platform
        input_max_batch_size = model_config.max_batch_size

        # 判斷是否維 tensorrt
        self.tensorrt = True if 'tensorrt' in platform else False

        # -----------------------------------------------------------------------------------
        # 解析輸入
        input_name = [ input.name for input in model_metadata.inputs ]
        input_datatype = [ input.datatype for input in model_metadata.inputs ]
        input_shape = [ input.shape for input in model_metadata.inputs ]
        logging.debug('Input: {} {} {}'.format(input_name, input_datatype, input_shape))

        # 解析輸出的名稱
        output_name = [ output.name for output in model_metadata.outputs ]
        output_shape = [ output.shape for output in model_metadata.outputs ]
        logging.debug('Name: {}'.format(output_name))
        logging.debug('Shape: {}'.format(output_shape))
        
        # -----------------------------------------------------------------------------------
        # 取得預期輸入的維度
        input_batch_dim = (input_max_batch_size > 0) # not set is not using batching
        expected_input_dims = 3 + ( 1 if input_batch_dim else 0 )
        logging.info('Ecpected dim of input is {}'.format(expected_input_dims))
        
        # -----------------------------------------------------------------------------------
        # 判斷資料維度是否正確
        if len(input_shape[0]) != expected_input_dims:
            raise Exception("Expecting input to have {} dimensions, model '{}' input has {}".format(
                expected_input_dims, model_metadata.name, len(input_shape[0])))

        # -----------------------------------------------------------------------------------
        # 判斷資料型態是否正確
        if input_datatype[0] != "FP32":
            raise Exception("expecting output datatype to be FP32, model '" +
                            model_metadata.name + "' output type is " +
                            input_datatype[0])

        # -----------------------------------------------------------------------------------
        # 輸入通常是 nchw 理當要處理，但是 前處理應該會處理好，所以直接輸出
        c = input_shape[0][1 + (-1 if expected_input_dims==3 else 0 )]
        h = input_shape[0][2 + (-1 if expected_input_dims==3 else 0 )]
        w = input_shape[0][3 + (-1 if expected_input_dims==3 else 0 )]
        logging.info('Get NCHW {} '.format((c, h, w)))

        return (input_name, output_name, expected_input_dims, (c, h, w), input_datatype[0])

    # -----------------------------------------------------------------------------------------------------------------------------
    ''' 檢查 Triton 的狀態 '''
    def check_triton_status(self):

        if not self.triton_client.is_server_live():
            logging.info("FAILED : is_server_live")
            sys.exit(1)
        if not self.triton_client.is_server_ready():
            logging.info("FAILED : is_server_ready")
            sys.exit(1)
        if not self.triton_client.is_model_ready(self.model):
            logging.info("FAILED : is_model_ready")
            sys.exit(1)   

    # -----------------------------------------------------------------------------------------------------------------------------
    ''' 取得 Triton Server 的模型資訊 '''
    def get_model_info(self):

        try:
            metadata = self.triton_client.get_model_metadata(self.model)
            logging.info(metadata)
        except InferenceServerException as ex:
            if "Request for unknown model" not in ex.message():
                logging.info("FAILED : get_model_metadata")
                logging.info("Got: {}".format(ex.message()))
                sys.exit(1)
            else:
                logging.info("FAILED : get_model_metadata")
                sys.exit(1)

        # Model configuration
        try:
            config = self.triton_client.get_model_config(self.model)
            if not (config.config.name == self.model):
                logging.info("FAILED: get_model_config")
                sys.exit(1)
            logging.info(config)
        except InferenceServerException as ex:
            logging.info("FAILED : get_model_config")
            logging.info("Got: {}".format(ex.message()))
            sys.exit(1)     

    # -----------------------------------------------------------------------------------------------------------------------------
    ''' 取得訓練的狀況 '''
    def get_infer_stats(self):
        statistics = self.triton_client.get_inference_statistics(model_name=self.model)
        if len(statistics.model_stats) != 1:
            logging.info("FAILED: get_inference_statistics")
            sys.exit(1)
        logging.info(statistics)

    # -----------------------------------------------------------------------------------------------------------------------------
    ''' 取得影片的相關資訊 '''
    def get_video_info(self, cap):
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))             # 取得影像寬
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))            # 取得影像高
        fps = cap.get(cv2.CAP_PROP_FPS)                             # 取得FPS
        # cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)                       # 設定   影片結尾
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        legnth = num_frames/fps
        # legnth = cap.get(cv2.CAP_PROP_POS_MSEC)/1000                # 取得時間郵戳 ( 毫秒 )
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO,0)                       # 調整回 影片開頭
        
        return width, height, legnth, fps

    # -----------------------------------------------------------------------------------------------------------------------------
    """ 圖片模式 """
    def image_infer(self, image, width, height, output='', input_type='FP32'):
        logging.debug("{} {}".format('width:',width))
        logging.debug("{} {}".format('height:',height))

        inputs, outputs = list(), list()

        if image is None:
            logging.error("FAILED: no input image")
            sys.exit(1)

        # ---------------------------------------------------------------------------------
        # Add meta data into grpc client
        [ inputs.append( grpcclient.InferInput(name, self.inputs_shape, self.dtype) ) for name in self.input_name ]
        [ outputs.append( grpcclient.InferRequestedOutput(name) ) for name in self.output_name ]
        logging.debug("{} {} {}".format('input:',len(inputs), inputs))
        logging.debug("{} {} {}".format('output:', len(outputs), outputs))

        # ---------------------------------------------------------------------------------
        # Preprocessing: insert image buffer
        logging.info("Creating buffer from image file...")
        if not self.tensorrt:
            image_buffer = self.prep(image, [width, height])   # image_buffer is for inference
        else:
            logging.info('Pre-processing with Caffe mode')
            image_buffer = self.prep(image, [width, height])

        """ 新增危險區域 """
        # image_zone = cv2.rectangle(image,(0,int(image.shape[1]/2)),(int(image.shape[0]/2),image.shape[1]),(10,0,255),-1)
        # image = cv2.addWeighted(image, 1.0, image_zone, 0.3, 0)
        """ 新增危險區域 """

        image_buffer = np.expand_dims(image_buffer, axis=0) if self.input_dims==4 else image_buffer
        logging.info(np.shape(image_buffer))
        logging.info('Create Image buffer .... ')
        try:
            inputs[0].set_data_from_numpy(image_buffer)    
        except Exception as e:
            logging.error(e)
        # ---------------------------------------------------------------------------------
        # Do inference
        logging.info("Invoking inference...")
        t1 = time.time()
        results = self.triton_client.infer( model_name=self.model,
                                            inputs=inputs,
                                            outputs=outputs )
        t2 = time.time()
        infer_time = t2 - t1
        if self.get_info: self.get_infer_stats()  # 取得訓練的狀態，太多直接註解掉
        # print('results:', results)

        # ---------------------------------------------------------------------------------
        # Parsing result
        logging.info("Parsing Results...")
        if results is not None:
            result = [ results.as_numpy(name) for name in self.output_name ]
            result = result[0] if not self.multi_output else result
            
        else:
            logging.info('Error Parsing')
            
        # ---------------------------------------------------------------------------------
        # Post Processing
        # 如果是物件辨識的話，由於已經在載入模型的時候區分好不同的平台(darknet, TAO)，所以使用相同的程式碼即可。
        if 'object' in self.model_info['task']:
            
            logging.warning('Object Detection')
            thres = 0.15

            if 'peoplenet' in self.model_info['name'] or 'DashCamNet' in self.model_info['name']:
                
                logging.warning('DetectNet_v2')
                detection_out, keepCount_out = result[0], result[1]
                # logging.warning(np.shape(detection_out))
                # logging.warning(np.shape(keepCount_out))
                
                detection_out, keepCount_out = np.reshape(detection_out, (-1,)), np.reshape( keepCount_out, (-1,))

                # logging.warning(np.shape(detection_out))
                # logging.warning(np.shape(keepCount_out))
                
                logging.info('Do post process')
                if 'peoplenet' in self.model_info['name']:
                    NUM_CLASSES = 2
                elif 'DashCamNet' in self.model_info['name']:
                    NUM_CLASSES = 4
                    # peoplenet_postprocess(outputs, min_confidence, analysis_classes, image_shape, wh_format
                detected_objects = self.postp(  outputs= [detection_out, keepCount_out], 
                                                                                min_confidence=thres, 
                                                                                analysis_classes=list(range(NUM_CLASSES)),
                                                                                image_shape=image.shape )
            else:
                logging.info(f"The objected detection model from TAO.")
                
                detected_objects = self.postp(result, image.shape[1], image.shape[0], [width, height], thres, self.nms)
                logging.info("Rendering Bounding Box...")

            logging.warning('After Post process')
            
            if 'peoplenet' in self.model_info['name'] or 'DashCamNet' in self.model_info['name']:
                image_draw = image.copy()
                
                bboxes, class_ids, scores = detected_objects
                if bboxes != []:
                    logging.info('detected something')
                    logging.info(bboxes)
                    indexes = cv2.dnn.NMSBoxes(bboxes, scores, thres, 0.5)
                    for idx in indexes:
                        idx = int(idx)
                        xmin, ymin, w, h = bboxes[idx]
                        #進入危險區域判斷
                        center = [(xmin+w)/2, (ymin+h)/2]
                        class_id = class_ids[idx]
                        color = [255, 0, 0] if class_id else [0, 255, 0]
                        if center[0] <= image.shape[0]/2 and center[1] >= image.shape[1]/2:
                            color = [0, 0, 255]
                        if not class_id:
                            cv2.rectangle(image_draw, (xmin, ymin), (xmin + w, ymin + h), color, 2)
                        parsed_results =  "inference time : {} \n fps : {}". format(str(infer_time), str(1/infer_time))
                else:
                    logging.info('nothing')
                    image_draw = image_draw
                    parsed_results = None

            else:
                parsed_results, image_draw = render_dets(image.copy(), self.label, detected_objects)
                # print('parsed_results:',parsed_results)
                # print('image_draw:',image_draw)
                print(self.label)
                
            if output:
                cv2.imwrite(output, image_draw)
                logging.info(f"Saved result to {output}")

            return parsed_results, image_draw

        # ---------------------------------------------------------------------------------
        # 如果是圖片分類，則 TAO 與 常見的 TensorRT 解析方式相同
        elif 'classi' in self.model_info['task']:   
            logging.info('The classification model from TAO')
            parsed_results = self.parse_cls_results(result)
            return parsed_results, image



    # -----------------------------------------------------------------------------------------------------------------------------
    """ 用於解析分類模型的函式庫 """
    def parse_cls_results(self, result):

        parsed_results = list() # 存放結果
        result = np.squeeze(result) # 去除空白的維度
        
        index = np.argmax(result)   # 取得最大值所在的index
        conf = result[index]        # 取得該信心指數
        classes = self.label[index] # 取得標籤名稱

        logging.info('Result is : [{}] {} {}'.format(index, classes, conf)) # 印出結果
        parsed_results.append( [index, classes, conf] )
        return parsed_results

    # -----------------------------------------------------------------------------------------------------------------------------
    """ 影像模式 """
    def video_infer(self, input ,width , height, output, input_type='FP32'):
        """
        input: input video,     output: output video
        """
        # logging.info(output)
        inputs, outputs = list(), list()
        thres = 0.15

        # ---------------------------------------------------------------------------------
        # Add meta data into grpc client
        [ inputs.append( grpcclient.InferInput(name, self.inputs_shape, self.dtype) ) for name in self.input_name ]
        [ outputs.append( grpcclient.InferRequestedOutput(name) ) for name in self.output_name ]

        # ---------------------------------------------------------------------------------
        # Capture the video
        logging.info("Capture the input video ...")     
        cap = cv2.VideoCapture(input)
        
        if not cap.isOpened():                              # Check status
            logging.info(f"Cannot open video {input}")
            sys.exit(1)
        
        logging.info("Get information of the video")    # Get information
        vwidth,vheight,vlegnth,vfps = self.get_video_info(cap)
        logging.info('{}_{}_{}_{}'.format(vwidth, vheight, vlegnth, vfps))

        # ---------------------------------------------------------------------------------
        # Create a subprocess for using ffmpeg record the results
        counter = 0
        if counter == 0 and output:        
            logging.info("Create a subprocess for using ffmpeg record the results.")
            process = sp.Popen(shlex.split(f'ffmpeg -y -s {vwidth}x{vheight} -pixel_format bgr24 -f rawvideo -i pipe: -r {vfps} -vcodec libx264 -pix_fmt yuv420p -crf 24 {output}'), stdin=sp.PIPE)

        # ---------------------------------------------------------------------------------
        # Start stream and inference
        logging.info("Invoking inference...")
        t_start = time.time()
        infer_time = 0
        while cap.isOpened():
            
            # ---------------------------------------------------------------------------------
            # read and check image
            ret, frame = cap.read()
            if not ret:
                logging.info("Failed to fetch next frame")
                break

            # ---------------------------------------------------------------------------------
            # Preprocessing: insert image buffer
            # logging.info("Creating buffer from image file...")
            input_image_buffer = self.prep(frame, [width, height])
            input_image_buffer = np.expand_dims(input_image_buffer, axis=0) if self.input_dims==4 else input_image_buffer
            inputs[0].set_data_from_numpy(input_image_buffer)

            # ---------------------------------------------------------------------------------
            # Do inference
            t1 = time.time()
            results = self.triton_client.infer(model_name=self.model,
                                    inputs=inputs,
                                    outputs=outputs)
            t2 = time.time()
            infer_time = infer_time + (t2 - t1)
            # ---------------------------------------------------------------------------------
            # Parsing result
            # logging.info("Parsing Results...")
            result = [ results.as_numpy(name) for name in self.output_name ]
            result = result[0] if not self.multi_output else result
            # ---------------------------------------------------------------------------------
            # Post Processing
            # 如果是物件辨識的話，由於已經在載入模型的時候區分好不同的平台(darknet, TAO)，所以使用相同的程式碼即可。
            
            if 'object' in self.model_info['task']:
                counter = counter + 1
                try:

                    if 'peoplenet' in self.model_info['name'] or 'DashCamNet' in self.model_info['name']:
                    
                        logging.warning('DetectNet_v2')
                        detection_out, keepCount_out = result[0], result[1]
                        # logging.warning(np.shape(detection_out))
                        # logging.warning(np.shape(keepCount_out))
                    
                        detection_out, keepCount_out = np.reshape(detection_out, (-1,)), np.reshape( keepCount_out, (-1,))

                        # logging.warning(np.shape(detection_out))
                        # logging.warning(np.shape(keepCount_out))
                    
                        logging.info('Do post process')
                        if 'peoplenet' in self.model_info['name']:
                            NUM_CLASSES = 2
                        elif 'DashCamNet' in self.model_info['name']:
                            NUM_CLASSES = 4
                        # outputs, min_confidence, analysis_classes, image_shape
                        print("frame.shape:", frame.shape)
                        detected_objects = self.postp( 
                            outputs = [detection_out, keepCount_out], 
                            min_confidence = thres, 
                            analysis_classes= list(range(NUM_CLASSES)), 
                            image_shape = frame.shape )
                        logging.info('out')
                        image_draw = frame.copy()
                    
                        bboxes, class_ids, scores = detected_objects
                        if bboxes != []:
                            logging.info('detected something')
                            logging.info(bboxes)
                            indexes = cv2.dnn.NMSBoxes(bboxes, scores, thres, 0.5)
                            for idx in indexes:
                                idx = int(idx)
                                xmin, ymin, w, h = bboxes[idx]
                                #進入危險區域判斷
                                center = [(xmin+w)/2, (ymin+h)/2]
                                class_id = class_ids[idx]
                                color = [255, 0, 0] if class_id else [0, 255, 0]
                                if center[0] <= frame.shape[0]/2 and center[1] >= frame.shape[1]/2:
                                    color = [0, 0, 255]
                                if not class_id:
                                    cv2.rectangle(image_draw, (xmin, ymin), (xmin + w, ymin + h), color, 2)
                                parsed_results = None
                        else:
                            logging.info('nothing')
                            image_draw = image_draw
                            parsed_results = None                
                    else:
                    
                        detected_objects = self.postp(result, frame.shape[1], frame.shape[0], [width, height], self.confidence, self.nms, "peoplenet")         
                        results, image_draw = render_dets(frame.copy(), self.label, detected_objects)
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    logging.error(exc_type, fname, exc_tb.tb_lineno)


                ################################################
                process.stdin.write(image_draw.tobytes())   # 輸出

            else:
                logging.info("It's not objected detection !!!")
                break

        # ---------------------------------------------------------------------------------
        # Clear buffer
        cap.release()
        process.stdin.close()
        process.wait()
        process.terminate()
        logging.info("Clear buffer")
        
        # count time and return information
        t_infer = time.time()-t_start

        info =  f'Video Mode \n'+\
                f'Video Length: {vlegnth:.3f}s\n'+\
                f'Cost Time: {t_infer:.3f}s\n'+\
                f'average inference time: {(infer_time/counter):.3f}s\n'+\
                f'average fps: {(counter/infer_time):.3f}'        
        logging.info("All Done")
        return True, info
    # -----------------------------------------------------------------------------------------------------------------------------
    
       
    # -----------------------------------------------------------------------------------------------------------------------------

    """ 進行 Inference 的入口 -> 棄用 """
    def infer(self, mode, input, width, height, output):
        '''
        進行推論 可以選擇模式 : [ 'image', 'video' ]
        '''
        logging.info(f'Running in {mode} mode')

        if mode == 'image':
            image = cv2.imread(str(input))
            self.image_infer(image, width, height, output)
        elif mode == 'video':
            self.video_infer(str(input), width, height, output)
