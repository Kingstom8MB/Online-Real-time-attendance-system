#2.0版本，修复了人脸识别较慢的问题，有效识别帧率在10~5f/s左右,低分辨率视频下表现不佳,建议1080P源视频画面建议摄像头。
#多人同屏时性能表现不佳（i7-6700HQ-16GB-GTX965M 4G）,显存使用0.6G
#网络摄像头请使用高速网络实时网速10MB/S,否则由于网络卡顿在摄像头识别时会重复识别同一帧,显著降低防抖效果
import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

import common
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

HAS_CUDA = torch.cuda.is_available()

from PyFaceDet import facedetectcnn

from PySide2.QtCore import Qt,QTimer,Signal,QObject #,QDateTime
from PySide2.QtWidgets import QApplication, QMessageBox, QMainWindow, QTextBrowser
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QIcon, QImage, QPixmap #,QMouseEvent
from PySide2.QtWidgets import QTableWidgetItem

import threading
#from threading import Thread    #多线程库
from multiprocessing import Manager

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

class MySignals(QObject):

    # 定义一种信号，两个参数 类型分别是： QTextBrowser 和 字符串
    # 调用 emit方法 发信号时，传入参数 必须是这里指定的 参数类型
    ArriveID_Update = Signal(QTextBrowser,str)
    NowNum_Update = Signal(QTextBrowser,str)
    ActualNum_Update = Signal(QTextBrowser,str)
    ShouldNum_Update = Signal(QTextBrowser,str)
    ArriveID_Clear = Signal(QTextBrowser)

    # 还可以定义其他种类的信号
    update_table = Signal(str)

# 实例化
global_ms = MySignals()

class Stats(QMainWindow):

    def __init__(self):

        loader = QUiLoader()
        self.ui = QUiLoader().load('main.ui')

        self.ui.ON.clicked.connect(self.videoStart)
        self.ui.OFF.clicked.connect(self.videoClose)

        global_ms.ArriveID_Update.connect(self.ArriveID_Update)
        global_ms.NowNum_Update.connect(self.NowNum_Update)
        global_ms.ActualNum_Update.connect(self.ActualNum_Update)
        global_ms.ShouldNum_Update.connect(self.ShouldNum_Update)

        global_ms.ArriveID_Clear.connect(self.ArriveID_Clear)



    def videoStart(self):
        global Video_Flag
        Video_Flag = True
        if repeat_thread_detection("Video"):
            QMessageBox.critical(self.ui, '错误' , '视频识别已开始，请勿点击开始按钮')
            
        else:
            threading.Thread(target=threadFunc,
                args=(argss, stats),
                name = "Video"
                ).start()#新建一个子线程，计划用于视频识别
            

    def videoClose(self):
        global Video_Flag
        Video_Flag = False
        print(threading.enumerate())
        print("close")

    def ArriveID_Update(self,fb,text):
        fb.append(str(text))
        fb.ensureCursorVisible()

    def NowNum_Update(self,fb,text):
        fb.setPlainText(str(text))

    def ActualNum_Update(self,fb,text):
        fb.setPlainText(str(text))

    def ShouldNum_Update(self,fb,text):
        fb.setPlainText(str(text))

    def ArriveID_Clear(self,fb):
        fb.clear()


def TextInit(stats,opt):
    stats.ui.output.setText(opt.output)
    stats.ui.source.setText(opt.source)
    stats.ui.IDmodel.setText(opt.IDweights)
    stats.ui.Model.setText(opt.weights)
    stats.ui.Img_size.setText(str(opt.img_size))
    stats.ui.view_img.setText(str(opt.view_img))
    stats.ui.save_txt.setText(str(opt.save_txt))
    stats.ui.NowNum.setPlaceholderText("请开始识别")
    stats.ui.ShouldNum.setPlaceholderText("请开始识别")
    stats.ui.ActualNum.setPlaceholderText("请开始识别")


def threadFunc(arg1,arg2):
    arg1.output = stats.ui.output.text()
    arg1.source = stats.ui.source.text()
    arg1.IDweights = stats.ui.IDmodel.text()
    arg1.weights = stats.ui.Model.text()
    arg1.img_size = int(stats.ui.Img_size.text())
    arg1.view_img = bool(stats.ui.view_img.text())
    arg1.save_txt = bool(stats.ui.save_txt.text())
    print('子线程 开始')
    print(f'线程函数参数是：{arg1}')
    detect(arg1)
    print('子线程 结束')

def repeat_thread_detection(tName):
    # 判断 tName线程是否处于活动状态
    for item in threading.enumerate():
        print(item)
        if tName == item.name: # 如果名字相同，说明tName线程在活动的线程列表里面
            return True
    return False

def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def getname(id):
    id_list = ["Unknown","ZXH"]
    return id_list[id]

def draw_boxes(img, all_dict, name, bbox, identities=None, List=[], offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)

        if all_dict.get(id) != None:
            confidence = str(all_dict[id]["confidence"])
            ID = all_dict[id]["ID"]
        else:
            confidence = None
            ID = 0
        ID = name[int(ID)]

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        cv2.putText(img, confidence, (x1, y1 + t_size[1]*2 + 6), cv2.FONT_HERSHEY_PLAIN, 2, [0, 255, 0], 2)   #更新到7ID时无法追踪,已修复
        cv2.putText(img, ID, (x1, y1 + t_size[1]*3 + 6), cv2.FONT_HERSHEY_PLAIN, 2, [0, 0, 255], 2)


    for i in List:
        #print(i)
        #print(len(i))
        #i=i[1]#框出人脸，这里只取坐标后在图上作画
        #print("i:",i)
        cv2.rectangle(img, (i[1], i[2]), (i[3], i[4]), color, 2)

    return img

def faceSearch(im1, bbox, modelOpt):
    List=[]

    #print(im1.shape)
    #print(modelopt.img_size)
    ratio = im1.shape[1]/modelOpt.img_size
    im2 = cv2.resize(im1,(int(im1.shape[1]/ratio),int(im1.shape[0]/ratio)))#坐标：Y,X，将图片缩放至640，加速识别

    for XYID in bbox:   
        x1, y1, x2, y2 ,ID= [int(i) for i in XYID]  #获取人物识别坐标，原画面坐标
        x1 = int(x1/ratio)  #人物识别坐标缩放至640
        y1 = int(y1/ratio)
        x2 = int(x2/ratio)
        y2 = int(y2/ratio)
        #print(x1, y1, x2, y2, ID)  #获取追踪对象的XY坐标和ID
        person_img = im2[y1:y2,x1:x2]   
        
        faces_XY = facedetectcnn.facedetect_cnn(person_img)     #Py版libfacedetection运行时更稳定，不会出现卡顿等情况，但有时人脸识别结果会偏移，不影响人脸检测
        if len(faces_XY) == 1:#理论上一人一张脸，但有时因重叠等原因导致一张图片上有多个人脸，可能导致识别不准确，放弃这张图像
            for test_info in faces_XY:
                #print("info前:",test_info)
                if test_info[4]>00:#根据置信度过滤掉过低的人脸
                    face_info=[x1+test_info[0]  ,  y1+test_info[1]  ,  x1+test_info[0]+test_info[2]  ,  y1+test_info[1]+test_info[3]  ,  test_info[4]] #将数据由裁剪图片横坐标、纵坐标、长度、宽度、置信度转换为原始图片的X1,Y1,X2,Y2,置信度
                    List +=[[ID , int(face_info[0]*ratio), int(face_info[1]*ratio), int(face_info[2]*ratio), int(face_info[3]*ratio), face_info[4]]]    #还原坐标至原始图像
    #cv2.imshow("person_img",person_img)
    print()

    #print("List:",List)
    return List

def IDmate(faceID,faceIist,im1,Wx,Hx): #pred_FaceID,Face,outputs;Wx是宽度的缩放倍数，画面是缩放到img-size宽度后进行推测的，比例约为1.66 ID配对，匹配人脸检测结果和人脸识别结果
    ID_confidence = {}
    faceIDcenters = []
    #print(Wx,Hx)   查看XY缩放倍率
    if len(faceID[0])>0:
        for faceIDs in faceID[0]:
            faceIDcenter = [(faceIDs[0]*Wx+(faceIDs[2]*Wx-faceIDs[0]*Wx)/2,faceIDs[1]*Hx+(faceIDs[3]*Hx-faceIDs[1]*Hx)/2),faceIDs[4],faceIDs[5]]#人脸识别结果质心坐标、置信度、识别结果
            faceIDcenters += [faceIDcenter]
            cv2.rectangle(im1, (faceIDs[0]*Wx, faceIDs[1]*Hx), (faceIDs[2]*Wx, faceIDs[3]*Hx), (00,00,255), 2)#画一个框
            
    #print(len(faceIDcenters),"faceIDcenters:",faceIDcenters)

    for faces in faceIist:
        for faceIDs in faceIDcenters:

            #print("faces:",faces)
            #print("faceIDs:",faceIDs)
            if faceIDs[0][0]>faces[1] and faceIDs[0][0]<faces[3] and faceIDs[0][1]>faces[2] and faceIDs[0][1]<faces[4]:#判断身份识别结果质心是否在人脸识别结果范围内
                cv2.circle(im1, (faceIDs[0][0],faceIDs[0][1]), 3, (00,00,255),thickness = -1,)#画出识别结果的质心
                #当前结果：faceIDs：人脸识别结果，faces：人脸检测结果及ID
                #print(faces[0])
                #print(faces)
                if faceIDs[1]>0.8:#过滤，当置信度大于多少时视为有效识别
                    ID_confidence[faces[0]] = faceIDs[2]

    return ID_confidence



def ID_link(serial_number, submeter_List):  #将ID配对结果和output信息关联
    new_list = {}
    #print("serial_number",serial_number)
    #print("submeter_List",submeter_List)
    for a in serial_number:
        for b in submeter_List :
            if b[-1] == a:      #识别结果可能比output信息少，通过ID判断于多人识别中出错    计划修复方式：对调serial_number与submeter_List的遍历顺序，多的查对比少的，对比后删除少的已对比信息
                new_list[a] = {"X1":b[0],
                               "Y1":b[1],
                               "X2":b[2],
                               "Y2":b[3],
                               "ID":serial_number[a]}
                break
            #else:
            #    new_list[a] = {"X1":b[0],
            #                   "Y1":b[1],
            #                   "X2":b[2],
            #                   "Y2":b[3],
            #                   "ID":0}
    #print(len(new_list))
    #print('new_list',new_list)
    return new_list


def list_update(All_dict,B_dict):   #列表更新，表明未在图像上探测到人脸，此环节仅更新坐标数据
    
    for iii in B_dict:     #iii[-1]:output中最后一个元素，为ID，ID做key便于搜索
        #print(iii[-1])
        #print(All_dict.get(iii[-1]))
        if All_dict.get(iii[-1]) == None:
            All_dict[iii[-1]] = {"X1":iii[0],
                                    "Y1":iii[1],
                                    "X2":iii[2],
                                    "Y2":iii[3],
                                    "ID":0,   #从旧表原样放回去，下同
                                    "confidence":1}
        else:
            All_dict[iii[-1]] = {"X1":iii[0],
                                    "Y1":iii[1],
                                    "X2":iii[2],
                                    "Y2":iii[3],
                                    "ID":All_dict[iii[-1]]["ID"],   #从旧表原样放回去，下同
                                    "confidence":All_dict[iii[-1]]["confidence"]}
            #print("{",iii[-1],":{",'"X1"',":",iii[0],',"Y1"',":",iii[1],',"X2"',":",iii[2],',"Y2"',":",iii[3],"}}")
    #print(All_dict)
    return All_dict

def dict_update(All_dict,B_dict):  #列表更新，表明在图像上探测到人脸，更新坐标和身份信息
    
    #print("All_dict",All_dict)
    #print("B_dict",B_dict)
    for iii in B_dict:     #iii[-1]:output中最后一个元素，为ID，ID做key便于搜索

        #print(iii)
        #print(All_dict.get(iii))
        
        if All_dict.get(iii) == None:
            All_dict[iii] = {"X1":B_dict[iii]["X1"],
                             "Y1":B_dict[iii]["Y1"],
                             "X2":B_dict[iii]["X2"],
                             "Y2":B_dict[iii]["Y2"],
                             "ID":B_dict[iii]["ID"],
                             "confidence":1}
        else:
            if All_dict[iii]["ID"] == B_dict[iii]["ID"]:
                
                if All_dict[iii]["confidence"] > 4 :  #
                    ID_num = 4+1
                else:
                    ID_num = All_dict[iii]["confidence"]+1
                ID = B_dict[iii]["ID"]#ID相同，无需按条件判断
                
            else:

                if All_dict[iii]["confidence"] <= 1 :
                    ID_num = 1
                    ID = B_dict[iii]["ID"]
                else:
                    ID_num = All_dict[iii]["confidence"]-1
                    ID = All_dict[iii]["ID"]

            All_dict[iii] = {"X1":B_dict[iii]["X1"],
                             "Y1":B_dict[iii]["Y1"],
                             "X2":B_dict[iii]["X2"],
                             "Y2":B_dict[iii]["Y2"],
                             "ID":ID,
                             "confidence":ID_num}
    return All_dict

def detect(opt, save_img=False):
    global Video_Flag
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    
    #初始化跟踪对象总表
    person_dict = {}
    dict_type_flag = type(person_dict)
    #print("type:",dict_type_flag)
    #print(person_dict.get(1))
    personID_list = []  #初始化历史在线目标总表
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # 删除输出文件夹
    os.makedirs(out)  #新建输出文件夹
    half = device.type != 'cpu'  # 只有在CUDA上运行时支持半精度，驱动不为cpu时打开半精度

    # Load model
    model = torch.load(weights, map_location=device)[
        'model'].float()  # 载入FP32
    model.to(device).eval()

    model_faseID = torch.load(opt.IDweights, map_location=device)[
        'model'].float()  # 载入FP32
    model_faseID.to(device).eval()


    if half:
        model.half()  # 加速，将FP32转为FP16
        model_faseID.half()


    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # 对于网络摄像头，设为true可以加速处理，若不是网络摄像头，设为true可能影响处理结果
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors获取模型里的class名
    names = model.module.names if hasattr(model, 'module') else model.names
    names_faceID = model_faseID.module.names if hasattr(model_faseID, 'module') else model_faseID.names

    global_ms.ShouldNum_Update.emit(stats.ui.ShouldNum, str(len(names_faceID) - 1) + "人")   #通过信号更新主线程界面，否则会出问题，下同

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None


    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        if Video_Flag == False:
            #print("结束线程")
            break
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()

        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        
        

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            global_ms.NowNum_Update.emit(stats.ui.NowNum, str(len(det)) + "人")  #在ＱＴ上更新当前画面人数

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                #print(det)
                #print(det[:, :4])

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # 将本次识别结果送入深度排序算法进行跟踪
                outputs = deepsort.update(xywhs, confss, im0)
                #print(outputs)

                # 处理输出信息，并进行人脸检测后确认是否进行人脸识别
                if len(outputs) > 0:

                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]

                    Face = faceSearch(im0,outputs,opt)
                    if len(Face)>0: #若在当前图象上侦测到人脸，否则就不要再执行下面身份识别动作，节约性能

                        #print("face:",Face) #调试信息，输出人脸位置信息

                        pred_FaceID = model_faseID(img, augment=opt.augment)[0]
                        pred_FaceID = non_max_suppression(
                            pred_FaceID, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)
                        #print("pred_FaceID:",pred_FaceID)   #输出身份识别结果
                        
                        serial_number = IDmate(pred_FaceID,Face,im0,im0.shape[1]/img.shape[3],im0.shape[0]/img.shape[2])#取得ID和身份的关联信息

                        #print("serial_number",serial_number)

                        outputs = ID_link(serial_number,outputs)

                        print("outups:",outputs)
                        
                        

                    if type(outputs) != dict_type_flag :    #把当前帧识别到的人物信息添加到一个总表里
                        print("list_update")

                        if len(person_dict)>0:

                            person_dict = list_update(person_dict,outputs)

                        else:   #初始时字典为空，直接塞就行
                            for iii in outputs:     #iii[-1]:output中最后一个元素，为ID，ID做key便于搜索
                                person_dict[iii[-1]] = {"X1":iii[0],
                                                        "Y1":iii[1],
                                                        "X2":iii[2],
                                                        "Y2":iii[3],
                                                        "ID":0,
                                                        "confidence":1}
                                #print("{",iii[-1],":{",'"X1"',":",iii[0],',"Y1"',":",iii[1],',"X2"',":",iii[2],',"Y2"',":",iii[3],"}}")

                    elif type(outputs) == dict_type_flag :
                        print("dict_update")

                        if len(person_dict)>0:

                            person_dict = dict_update(person_dict,outputs)

                        else:   #初始时字典为空，直接塞就行
                            print(outputs)
                            for iii in outputs:     #iii[-1]:output中最后一个元素，为ID，ID做key便于搜索
                                person_dict[iii] = {"X1":outputs[iii]["X1"],
                                                        "Y1":outputs[iii]["Y1"],
                                                        "X2":outputs[iii]["X2"],
                                                        "Y2":outputs[iii]["Y2"],
                                                        "ID":0,
                                                        "confidence":1}


                    for iii in person_dict: #输出当前帧总表信息
                        if int(person_dict[iii]["ID"]) != 0 and person_dict[iii]["confidence"] == 5 :   #把符合条件的ID加到ID总表里
                            personID_list.append(int(person_dict[iii]["ID"]))
                        print(iii,":",person_dict[iii])
                    draw_boxes(im0, person_dict, names_faceID, bbox_xyxy, identities, Face)
                    personID_list = list(set(personID_list))    #列表去重，上面是无脑加，可能会有重复ID加入
                    print("已到用户信息",personID_list)
                    global_ms.ActualNum_Update.emit(stats.ui.ActualNum, str(len(personID_list)) + "人")   #通过信号更新主线程界面，否则会出问题，下同
                    global_ms.ArriveID_Clear.emit(stats.ui.ArriveID)    #清除已到人员文本框，简单粗暴，想到的其他更新方法太麻烦，这样做相当于刷新
                    for i in personID_list :
                        global_ms.ArriveID_Update.emit(stats.ui.ArriveID, names_faceID[i])


                # Write MOT compliant results to file
                #if save_txt and len(outputs) != 0:
                #    for j, output in enumerate(outputs):
                #        bbox_left = output[0]
                #        bbox_top = output[1]
                #        bbox_w = output[2]
                #        bbox_h = output[3]
                #        identity = output[-1]
                #        with open(txt_path, 'a') as f:
                #            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                #                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()

            t2 = time_synchronized()
            # 性能检测，输出耗时
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            
            #CV2和ＱＴ图片格式不一样，需要转换

            

            image = cv2.resize(im0, (1600, 900))    #缩小图片以适应QT框,比例应为16:9/4:3等标准比例，否则转换后图片会出问题
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            stats.ui.showImage.setPixmap(QPixmap.fromImage(image))  #在ＱＴ上显示图片   #危险操作——直接在子线程操作主线程界面，因为没出什么问题所以没改，

            ## Stream results
            #if view_img:
            #    #cv2.imshow(p, im0)
            #    if cv2.waitKey(1) == ord('q'):  # q to quit
            #        raise StopIteration

            # Save results (image with detections)
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/yolov5s.pt', help='model.pt path')#yolov5/weights/yolov5x.pt
    parser.add_argument('--IDweights', type=str,
                        default='yolov5/runs/train/exp4/weights/last.pt', help='FaceID模型的位置，应为Yolov5模型格式的.pt文件')#yolov5/weights/yolov5x.pt
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='yolov5/videos/class3.mp4', help='source') # http://192.168.0.100:4747/mjpegfeed?1920x1080  yolov5/videos/class3.mp4 http://192.168.0.103:4747/mjpegfeed?1920x1080
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # 官方权重包中人的类是0
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    argss = parser.parse_args()
    argss.img_size = check_img_size(argss.img_size)
    print(argss)

    with torch.no_grad():
        #detect(argss)
        
        Video_Flag = True   #创建一个Flag,为假时引发错误退出线程
        manager = Manager()
        
        app = QApplication([])
        app.setWindowIcon(QIcon('logo.ico'))
        stats = Stats()
        TextInit(stats,argss)
        stats.ui.show()
        app.exec_()