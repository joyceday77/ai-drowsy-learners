# 基于深度学习的网课走神监测和自动叫醒系统

## 背景介绍

视频会议软件在全球一体化的大背景下被广泛应用，近几年疫情的叠加效应又进一步促进了远程会议软件的发展，各种在线视频会议软件也越来越多的应用于学校和培训机构的日常教学任务。比如今年春季学期，上海中小学的绝大多数教学任务都是通过类似腾讯会议这样的远程视频会议软件来实现。

在线教学和商业会议的侧重点不同，学生在与电子学习平台互动时的清醒和专注与教学成果密切相关，但仅从软件自带的日志中检测昏昏欲睡的学习行为并非易事，往往需要老师和家长志愿者等额外人力投入才能实时发现学生打瞌睡等走神的情况，再通过各种方式提醒。这样不但增加了老师的负担，统计的出错率和滞后性都会影响整体的教学质量。

## 项目摘要

通过捕捉摄像头下学生的面部表情来判断上课的专注度，一旦发现走神的情况，即时发出提醒声叫醒学生，也可以自动触发短信通知预先设置的短信接收者，例如老师或家长等。

项目的实现采用了python编程，调用opencv，dlib等预制库，基于训练好的人脸识别68特征关键点模型来绘制面部特征，并利用深度学习算法来判断学生是否走神，进而触发告警，提醒等一系列操作。值得一提的是，程序输出的视频流可以作为在线视频会议软件例如腾讯会议的输入，从而可以无缝集成到日常网课教学中，无需额外投入任何硬件。

## 关键词

深度学习，网课，走神监测，自动叫醒，python

## 项目创新点

程序输出的视频流可以作为任何支持虚拟摄像头的在线视频会议软件的输入，从而可以无缝集成到日常网课教学中，无需额外投入任何硬件。程序的可扩展性强，对运行的电脑硬件要求不高，兼容Windows，Mac和Linux等各种操作系统，经过进一步的集成和包装可以扩展到各类远程教学系统中，有大规模商用的潜力。

## 项目实现

### 准备环境

#### Mac 操作系统（以此为例）

- [Python](https://www.python.org/) 3.7 或以上，并安装如下必备预制库：

  - opencv-python

  - dlib

  - imutils

  - scipy

  - numpy

  - tensorflow

  - keras

  - pygame

- 下载训练好的人脸识别68特征关键点[模型](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)，解压后放入data目录下。
- 下载和安装[CamTwist](https://camtwiststudio.com/)程序来创建虚拟摄像头
- 下载和安装[腾讯会议](https://meeting.tencent.com/)

#### 运行步骤

1. 走神监测和自动叫醒程序会启动系统内可用的摄像头，使用opencv和dlib库提取学生面部特征，通过深度学习算法实时监测学生的专注程度，在眼睛闭合超过给定帧时发出警报声，并在屏幕上打印出 “DROWSINESS ALERT！” 字样。

   `python3 drowsy_learners.py `

2. 运行CamTwist，并将走神监测程序的输出frame设置为新的虚拟摄像头视频流。
   - 具体步骤待补充

3. 用腾讯会议接入网课教学会议，并指定CamTwist的虚拟摄像头作为视频输入。
   - 视频会议软件可以用其他支持虚拟摄像头的软件代替，例如Teams，Zoom，Skype等等。

##### To Do List

- [ ] 这一节需要补充一个流程图
- [ ] 每一步可以加一些截图方便展示步骤和效果
- [ ] 这一节可以录制一个展示视频



## 参考文献



