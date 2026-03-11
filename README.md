# Facial-recognition
轻量级 Python 人脸识别项目，聚焦人脸检测、特征提取与匹配核心能力，代码简洁易上手，适合入门学习，也可快速落地小型非商业化场景（如简易考勤、人脸门禁原型等）。

## 📖 项目介绍
本项目基于 Python 生态经典人脸识别工具链，实现「人脸录入→检测→特征提取→匹配识别」全流程，无需复杂框架依赖，纯 CPU 即可运行，支持图片/摄像头实时流识别，适合人脸识别入门学习与小型非商业化场景落地。

## 🛠️ 核心技术栈
项目依托以下工具/库实现核心能力，各组件分工明确：

| 工具/库         | 核心作用                                                                 |
|-----------------|--------------------------------------------------------------------------|
| Python          | 核心开发语言，兼顾易用性与生态丰富性                                     |
| face_recognition | 高层级人脸识别库，封装 dlib 核心逻辑，简化人脸特征提取/匹配开发           |
| dlib            | 底层机器学习库（C++编写），提供人脸检测（HOG/CNN）、关键点提取能力       |
| OpenCV (cv2)    | 图像/视频流的读取、预处理（缩放/色彩转换）、显示与摄像头交互             |
| numpy           | 数值计算，处理人脸128维特征向量的比对（欧氏距离计算）                    |

## ✨ 核心功能
- 🎯 人脸检测：从图片/摄像头实时流中定位单/多人脸区域；
- 📝 人脸特征提取：将检测到的人脸转换为128维特征向量（唯一表征人脸特征）；
- 🔍 人脸匹配/识别：通过计算特征向量欧氏距离（阈值≈0.6）判断是否为同一人；
- 📂 人脸库管理：支持摄像头采集/图片导入录入人脸，按人名分类存储特征；
- 🎥 实时识别：基于摄像头流的实时人脸检测与匹配，可视化显示识别结果。

## 📋 环境要求
- Python 版本：3.6 ~ 3.9（dlib 对 Python 3.10+ 兼容性较差）；
- 操作系统：Windows/macOS/Linux（Windows 需安装 Visual Studio 构建工具）；
- 硬件：无需 GPU，纯 CPU 即可运行（推荐单核主频 ≥ 2.0GHz）。

## 🚀 安装步骤
### 1. 克隆仓库
```bash
git clone https://github.com/RicerChen/Facial-recognition.git
cd Facial-recognition
```

### 2. 安装依赖
优先安装基础库，再处理核心依赖 `dlib`（新手易踩坑）：

#### 基础依赖
```bash
pip install opencv-python numpy pillow
pip install face_recognition
```

#### dlib 安装（关键）
- Windows（推荐预编译版，无需手动编译）：
  ```bash
  pip install dlib-bin
  ```
- Linux/macOS（需先安装编译环境）：
  ```bash
  # Ubuntu/Debian
  sudo apt-get install build-essential cmake python3-dev
  # CentOS
  sudo yum install gcc gcc-c++ cmake python3-devel
  # macOS
  brew install cmake
  
  # 安装 dlib
  pip install dlib
  ```

> ⚠️ 若 dlib 安装失败：
> - Windows：安装 [Visual Studio 2019+ 构建工具](https://visualstudio.microsoft.com/visual-cpp-build-tools/)（勾选「C++ 构建工具」）；
> - Linux：确认已安装 `python3-dev`（系统级 Python 开发依赖）；
> - 所有系统：若仍失败，可下载对应 Python 版本的 dlib 预编译包（如 [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#dlib)）手动安装。

## 📝 快速开始
### 1. 人脸录入（采集基准特征）
运行人脸采集脚本，输入人名后，摄像头将采集你的人脸（按 `q` 键结束采集）：
```bash
python capture_face.py
```
- 采集的人脸图片会自动存入 `face_dataset/[你的名字]` 目录，作为识别的基准库；
- 建议录入 5+ 张不同角度/光线的人脸，提升识别准确率。

### 2. 实时人脸识别
运行实时识别脚本，摄像头会实时检测人脸并匹配已录入的特征库，屏幕可视化显示识别结果（按 `q` 退出）：
```bash
python realtime_recognition.py
```

### 3. 静态图片识别
修改 `face_recognition.py` 中的 `IMAGE_PATH` 为目标图片路径，运行脚本识别图片中的人脸：
```bash
# 先修改 face_recognition.py 中的 IMAGE_PATH 为实际图片路径
python face_recognition.py
```

## 📂 项目结构
```
Facial-recognition/
├── face_dataset/          # 人脸特征库（录入的人脸图片，按人名分类）
├── capture_face.py        # 人脸录入脚本（摄像头采集/图片导入）
├── face_recognition.py    # 静态图片识别核心逻辑（检测+特征匹配）
├── realtime_recognition.py# 摄像头实时识别脚本
├── requirements.txt       # 依赖清单（可直接 pip install -r requirements.txt）
└── README.md              # 项目说明文档
```

## ⚠️ 注意事项
1. 准确率限制：依赖 HOG 检测算法，对暗光、侧脸、遮挡场景鲁棒性差，建议在光线充足的正面场景使用；
2. 性能优化：纯 CPU 运行时，建议降低摄像头分辨率（如 320×240）提升帧率；
3. 安全限制：无活体检测功能，易被照片/视频欺骗，**禁止用于商业化/高安全要求场景**；
4. 库规模限制：人脸库人数建议不超过 100 人，人数过多会显著降低匹配效率；
5. 隐私提示：录入的人脸图片仅存储在本地，请勿上传至公共服务器。

## ❓ 常见问题
### Q1: 运行脚本提示「找不到人脸」？
A1：检查光线是否充足、人脸是否正对摄像头；确保 `face_dataset` 目录下已录入有效人脸；调整摄像头焦距/角度。

### Q2: 识别结果错误/不精准？
A2：增加人脸录入样本数量（建议 5+ 张）；调整 `face_recognition.py` 中的匹配阈值（默认 0.6，值越小识别越严格）；确保录入与识别场景光线一致。

### Q3: 摄像头无法调用/闪退？
A3：检查摄像头是否被其他程序占用；Windows/macOS 需授予 Python 摄像头访问权限；Linux 需确认当前用户有摄像头设备权限（`sudo chmod 777 /dev/video0`）。

### Q4: dlib 安装耗时过长？
A4：优先使用预编译包（如 Windows 下的 `dlib-bin`）；Linux/macOS 可提前安装 `cmake` 并使用国内 PyPI 源（如清华源）。

## 📄 许可证
本项目仅供**学习和非商业使用**，禁止用于商业/盈利场景，未经授权不得二次分发。

## 🙏 致谢
- 基于 `face_recognition` 库封装核心逻辑，感谢该库开发者的开源贡献；
- 底层依赖 `dlib` 库实现人脸检测与特征提取，感谢开源社区的技术支撑。
```

### 总结
1. 文档覆盖项目全生命周期信息：从「是什么（介绍/技术栈）」→「怎么装（环境/安装）」→「怎么用（快速开始）」→「注意什么（注意事项/常见问题）」，新手可一站式上手；
2. 核心痛点（如 dlib 安装、识别准确率）均做重点标注，降低踩坑概率；
3. 结构符合开源项目规范，逻辑清晰，可直接作为仓库的 README 投入使用。
