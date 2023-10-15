# 基于人工智能的在线学习疲劳监测系统

## 背景介绍

在全球一体化的大环境下，视频会议软件在各行各业被广泛地应用于商务沟通、远程办公和培训之中。近年来，它也日益成为教育领域的重要工具，特别是在疫情期间，远程在线教学几乎成为常态。但是，与商业会议不同，网课教学对学生持续专注提出了更高的要求，目前仅依靠软件自身的功能很难准确检测学生的疲劳状况，为此往往需要投入额外的人力资源，增加了教师的工作负担，也降低了教学质量。因此，开发一种基于人工智能的网课学生疲劳检测与提醒系统已刻不容缓。

## 项目摘要

本研究构建了这样一个系统原型，通过摄像头实时捕捉学生的面部表情，使用机器学习算法判断疲劳状态，在检测到疲劳时及时发出声音及画面形式的提醒，以帮助学生管理状态，提高学习效果。系统使用Python语言实现，调用了OpenCV、Dlib等预制库和工具，基于人脸识别模型来判断疲劳。系统输出的视频流可以作为支持虚拟摄像头的各种在线视频会议软件的输入源，因此可以无缝集成到日常的在线学习中，无需额外购置硬件，且兼容多种操作系统，具有重要的实用价值。本研究验证了人工智能技术在提升在线教育质量方面的应用潜力，也为后续研发奠定了框架与经验基础。

## 关键词

人工智能，在线学习，走神监测，Python，Dlib，OpenCV

## 项目创新点

当前系统采用Python语言实现，并采用了模块化设计，具有良好的拓展性。对运行的电脑硬件要求不高，兼容Windows，Mac和Linux等各种操作系统，经过进一步的集成和包装可以扩展到各类远程教学系统中，有较好的商业前景。

**高度自动化**：系统能够自动监测学生的走神状态，并及时触发提醒和通知，减轻教师的负担。

**无需额外硬件**：系统利用摄像头获取数据，轻量级实现无需新增专用硬件设备，实用性强。

**可扩展性强**：系统的架构和实现方式使其能够与各种在线视频会议软件集成，适用于不同的远程教学系统。

该研究不仅可服务于在线教育，也为未来的技术创新提供了基础与借鉴。总体而言，这项研究成果将有望提升在线学习效率，并推动人工智能技术在教育领域的发展与应用。

## 项目实现

### 准备环境

#### Mac 操作系统（以此为例）

- [Python](https://www.python.org/) 3.7 或以上，并安装如下必备预制库。考虑到国内下载速度较慢，建议切换到国内镜像，例如： `python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple dlib`


 ```
  cmake
  dlib
  imutils
  scipy
  numpy
  opencv-python
  keras
  pygame
  argparse
  playsound
  threading
  cv2
 ```

- 下载训练好的人脸识别68特征关键点[模型](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)，解压后放入dat目录下。
- 选择合适的告警音频，放入wav目录下，音频文件缺省名为alarm.wav 
- 下载和安装[OBS](https://obsproject.com/download)程序来创建虚拟摄像头
- 下载和安装[腾讯会议](https://meeting.tencent.com/)或其他视频会议软件。

#### 运行步骤

1. 在安装了所有必要的依赖库的环境下，执行如下代码。程序会启动系统内可用的摄像头，提取学生面部特征，实时监测学生的专注程度，在眼睛闭合超过给定帧时发出警报声，并在屏幕上打印出 “Drowsiness!!!” 字样。也可以在监测学生打哈欠时在屏幕上打印出 “Yawning” 字样。

   `python3 drowsy_learner_detection.py `

2. 运行OBS，开启虚拟摄像头场景，将走神监测程序的输出的`[pyhon3]Frame`设置为新的虚拟摄像头视频流。

3. 用腾讯会议接入网课教学会议，并指定OBS的虚拟摄像头作为视频输入。
   - 视频会议软件可以用其他支持虚拟摄像头的软件代替，例如Teams，Zoom，Skype等等。


## 参考文献

- http://dlib.net/
- [Research on a Real-Time Driver Fatigue Detection Algorithm Based on Facial Video Sequences](https://www.mdpi.com/2076-3417/12/4/2224)
- [Real-Time Eye Blink Detection using Facial Landmarks](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)




