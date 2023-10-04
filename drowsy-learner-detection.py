# -*- coding: utf-8 -*-
# 导入所需的库
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np # 数据处理的库 numpy
import argparse
import imutils
import time
import dlib
import playsound
import cv2

'''
定义函数,参数和常数
'''
# 定义三个函数
def sound_alarm(path):  # 播放告警声
	playsound.playsound(path)

def eye_aspect_ratio(eye):  # 获取眼睛的长宽比
    A = dist.euclidean(eye[1], eye[5])  # 计算两组垂直眼宽坐标之间的欧几里得距离
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])  # 计算水平眼长坐标之间的欧几里得距离
    ear = (A + B) / (2.0 * C)  # 眼睛长宽比的计算
    return ear  # 返回眼睛的长宽比

def mouth_aspect_ratio(mouth):  # 获取嘴巴的长宽比
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59  # 计算两组垂直嘴宽坐标之间的欧几里得距离
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55  # 计算水平嘴长坐标之间的欧几里得距离
    mar = (A + B) / (2.0 * C)
    return mar

# 构建参数
ap = argparse.ArgumentParser() 
ap.add_argument("-p", "--shape_predictor", default="./dat/shape_predictor_68_face_landmarks.dat",  # 人脸关键点预测器
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="./wav/alarm.wav",  # 告警声音频文件
	help="path to alarm.wav file")
ap.add_argument("-w", "--webcam", type=int, default=0,   # 视频输入摄像头
	help="index of webcam on system")
args = vars(ap.parse_args())

# 定义常数
EAR_THRESH = 0.2  # 眼睛长宽比阀值
EYE_AR_CONSEC_FRAMES = 20  # 连续满足EAR阀值的帧数阀值
EYE_CLOSE_THRESH = 60  # 闭眼总计数阀值,超出后发出告警
MAR_THRESH = 0.8  # 打哈欠长宽比阀值
MOUTH_AR_CONSEC_FRAMES = 20  # 连续满足MAR阀值的帧数阀值
# 初始化闭眼帧计数器和闭眼总数
eCOUNTER = 0
eTOTAL = 0
# 初始化打哈欠帧计数器和打哈欠总数
mCOUNTER = 0
mTOTAL = 0

'''
主函数
'''
print("[INFO] loading facial landmark predictor...") 
detector = dlib.get_frontal_face_detector()  # 获取正脸位置检测器
predictor = dlib.shape_predictor(args["shape_predictor"])  # 获取人脸关键点预测器

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]  # 获取左眼标志的索引
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # 获取右眼标志的索引
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]  # 获取嘴部标志的索引

print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()  # 从摄像头读取视频流
time.sleep(1.0)

while True:  # 从视频流循环视频帧
    frame = vs.read()      # 读取视频帧
    frame = imutils.resize(frame, width=720)  # 改变视频框尺寸
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 进行灰度化
    rects = detector(gray, 0) #获取脸部方框
    
    for rect in rects:  # 循环脸部位置信息
        shape = predictor(gray, rect)  # 获取脸部特征位置的信息
        shape = face_utils.shape_to_np(shape)  #转换为数组的格式
        
        leftEye = shape[lStart:lEnd]   # 提取左眼坐标
        rightEye = shape[rStart:rEnd]  # 提取右眼坐标
        mouth = shape[mStart:mEnd]  # 提取嘴巴坐标
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0   # 计算左右眼的长宽比，使用平均值作为最终的长宽比EAR
        
        mar = mouth_aspect_ratio(mouth)  # 计算嘴巴的长宽比MAR
        
        leftEyeHull = cv2.convexHull(leftEye)  # 获取左眼凸包位置
        rightEyeHull = cv2.convexHull(rightEye)  # 获取右眼凸包位置
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)  # 画出左眼轮廓
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)  # 画出右眼轮廓
        mouthHull = cv2.convexHull(mouth)  # 获取嘴巴凸包位置
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)  # 画出嘴巴轮廓
        
        left = rect.left()  
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)  # 进行画图操作,用矩形框标注人脸
        '''
            分别计算左眼和右眼的评分求平均作为最终的评分,如果小于阈值,则加1,如果连续EYE_AR_CONSEC_FRAMES次都小于阈值,则表示进行了一次闭眼活动,闭眼数到达阀值后发出告警
        '''
        if ear < EAR_THRESH:  # 循环,满足条件的,闭眼帧计数器+1
            eCOUNTER += 1
            if eCOUNTER >= EYE_AR_CONSEC_FRAMES:  # 如果连续EYE_AR_CONSEC_FRAMES帧都闭眼,则表示进行了一次眨眼活动
                eTOTAL += 1
                if eTOTAL >= EYE_CLOSE_THRESH :        
                    cv2.putText(frame, "Drowsiness!!!", (100, 200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)  # 在屏幕上显示Drowsiness告警
                    if not ALARM_ON:
                        ALARM_ON = True  # 开启告警
                        if args["alarm"] !="":
                            t = Thread(target=sound_alarm, args=(args["alarm"],))   # 播放告警音
                            t.deamon = True
                            t.start()                
            else:
                eTOTAL = 0
                ALARM_ON = False  # 关闭告警
        else:
            eCOUNTER = 0   # 重置闭眼帧计数器  
            
        cv2.putText(frame, "Faces: {}".format(len(rects)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)   # 在屏幕上显示捕获的人脸数量
        cv2.putText(frame, "Blinks: {}".format(eTOTAL), (150, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)   # 在屏幕上显示闭眼总数,告警后会清零
        cv2.putText(frame, "eCOUNTER: {}".format(eCOUNTER), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)    # 在屏幕上显示闭眼帧计数器,告警后会清零
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (450, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)   # 在屏幕上实时显示眼睛长宽比   
        '''
            计算张嘴评分,如果小于阈值,则加1,如果连续MOUTH_AR_CONSEC_FRAMES次都小于阈值,则表示打了一次哈欠
        '''
        if mar > MAR_THRESH: # 循环,满足条件的,打哈欠帧计数器+1
            mCOUNTER += 1           
            if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:  # 如果连续MOUTH_AR_CONSEC_FRAMES次都小于阈值，则表示打了一次哈欠
                cv2.putText(frame, "Yawning!", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)   # 在屏幕上显示Yawning告警
                mTOTAL += 1
            else:
                mTOTAL = 0    
        else:            
            mCOUNTER = 0  # 重置嘴帧计数器
            
        cv2.putText(frame, "Yawning: {}".format(mTOTAL), (150, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # 在屏幕上显示打哈欠帧计数器
        cv2.putText(frame, "mCOUNTER: {}".format(mCOUNTER), (300, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 在屏幕上显示打哈欠总数
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (480, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # 在屏幕上实时显示嘴巴长宽比 
        
        for (x, y) in shape:  # 进行画图操作，标记68个特征点标识
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1) 
            
#    print('眼睛实时长宽比:{:.2f} '.format(ear)+"\t是否眨眼:"+str([False,True][eTOTAL>=60]))
#    print('嘴巴实时长宽比:{:.2f} '.format(mar)+"\t是否张嘴:"+str([False,True][mTOTAL>=60]))
    
    cv2.putText(frame, "Press 'q': Quit", (20, 500),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 255, 159), 2)      # 按q退出
    cv2.imshow("Frame", frame)  # 显示视频框，名Frame
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 输入q就退出
        break
    
vs.stop()  # 停止捕捉视频流
cv2.destroyAllWindows()  #清理
