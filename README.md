YOLOV8的部署方式汇总
1、ultralytics方式运行YOLOV8
来自 <https://github.com/ultralytics/ultralytics> 
1.1依赖库的版本要求
 Python>=3.8 
 PyTorch>=1.8.
ubuntu18.04安装python3.8
参考网址： https://blog.51cto.com/u_16175452/6915660
1.1.1 安装依赖项Python 3.8
（1）在安装Python 3.8之前，我们需要安装一些依赖项。打开终端并执行以下命令：
 
# sudo apt update
# sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
上述命令将更新系统软件包，并安装一些必要的开发库和工具。
 
（2）下载Python 3.8
我们可以从Python官方网站下载Python 3.8的源代码。打开Web浏览器并访问以下链接：
https://www.python.org/downloads/release/python-3818/
 
找到Python 3.8.0的源代码下载链接，右键点击并选择“复制链接地址”。
https://www.python.org/ftp/python/3.8.18/Python-3.8.18.tgz
 
回到终端，执行以下命令：
 
# cd /tmp
# wget  https://www.python.org/ftp/python/3.8.18/Python-3.8.18.tgz
上述命令将下载Python 3.8的源代码到/tmp目录下。
 
（3）编译和安装Python 3.8
下载完成后，我们需要解压缩并编译Python 3.8源代码。执行以下命令：
 
# tar -xf Python-3.8.0.tar.xz
# cd Python-3.8.0
# ./configure --enable-optimizations
# make -j 4
# sudo make altinstall
 
上述命令将解压缩源代码，配置编译选项，使用4个线程编译源代码，并通过make altinstall命令将Python 3.8安装到系统中。注意使用altinstall而不是install命令，以避免覆盖系统默认的Python版本。
（4）验证安装
安装完成后，我们可以验证Python 3.8是否正确安装。在终端中执行以下命令：
python3.8 --version
如果一切顺利，您将看到类似如下输出：
 
Python 3.8.0
这意味着Python 3.8已成功安装并可用于使用。
 
（5）设置Python 3.8为默认版本
如果您希望在使用python命令时默认使用Python 3.8，可以通过以下命令更改：
# sudo update-alternatives --install /usr/bin/python python /usr/local/bin/python3.8 1
上述命令将/usr/local/bin/python3.8设置为python命令的备选版本，并将其优先级设置为1。如果您希望使用其他版本的Python作为默认版本，可以通过该命令进行更改。
（6）更新pip版本命令
# python -m pip install --upgrade pip
# pip --version 
#  pip list

 
1.1.2 ubuntu18.04安装pytorch
 PyTorch>=1.8.
注意：PyTorch版本不易选择过高，最好选择1.11，此坑以帮你踩过，过高会造成程序编译错误
方法一： 
pip install torch torchvision
 
方法二：离线安装
《1》下载对应的版本
torch安装1.11.0版本，对应的torchversion版本0.12.0
版本匹配网址： https://pytorch.org/get-started/previous-versions/

下载地址： https://download.pytorch.org/whl/torch_stable.html
Torch-1.11.0
 
Torchvision-0.12.0

注意：cp38---python3.8
Aarch64---芯片架构：# arch
linux
下载后安装对应的文件 pip install XXX.whl
（1）x86_64
 
$ pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
（2）aarch64(NVIDIA Jetson AGX Xavier)
<2>安装完成
查看torch版本
$ python
$ import torch
$ torch.__version__
# 输出                        

原文链接： ttps://blog.csdn.net/Elio_LosEr/article/details/106196700
1.2 yolo运行指令说明
模型训练：
yolo train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01    # 训练检测模型 10 次，初始学习率为 0.01 

（1）yolo会把第一次运行的路径设置为默认路径，默认路径的配置文件【/root/.config/Ultralytics/settings.yaml】


（2）训练数据的配置 【data=coco128.yaml】在ultralyutics的python库包内，可配置绝对路径

 

（3）错误类型提示及解决办法

 
 
2、onnxruntime-cpp方式运行YOLOV8
2.1 安装onnxruntime
onnxruntime版本和CUDA版本的对应：
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

《1》python版本库
$  pip install onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
《2》c++版本
方法一：使用官方release的库
https://link.zhihu.com/?target=https%3A//github.com/microsoft/onnxruntime/tags

 
下载onnxruntime-linux-x64-1.13.1.tgz和onnxruntime-linux-x64-gpu-1.13.1.tgz
存储路径：~/ultralytics_old/examples/YOLOv8-ONNXRuntime-CPP/
$ tar -xzvf onnxruntime-linux-x64-1.13.1.tgz
$ tar -xzvf onnxruntime-linux-x64-gpu-1.13.1.tgz
$ 
2.2下载coco.yaml和***.onnx模型
$ cd ~/ultralytics/examples/YOLOv8-ONNXRuntime-CPP
$ sudo find / -name yolov8n.onnx -type f
$ cp /opt/weixiaolin/yolov8n.onnx ./
$  sudo find / -name coco.yaml -type f
$  cp /opt/perception/majicAI/ultralytics/ultralytics/cfg/datasets/coco.yaml ./
2.3 编译代码
$ cd ~/ultralytics_old/examples/YOLOv8-ONNXRuntime-CPP/
$  mkdir build && cd build
$ cmake ..
注意：需要在项目路径下的CMakeList.txt中添加一行：
link_libraries(stdc++fs)

$ sudo make -j4
<1> 编译时出现如下错误
 
错误分析：
在Ubuntu 18.04上支持C++17的std::filesystem，需要进行一些步骤。首先，你需要一个支持C++17的编译器。GCC 7及以上版本和Clang 5及以上版本都支持C++17，但std::filesystem是在GCC 9和Clang 9中首次完全支持的。Ubuntu 18.04默认的GCC版本是7.4.0，所以你需要升级你的编译器。
首先，我们需要添加新的PPA（Personal Package Archive）来获取更新的GCC版本。打开终端，输入以下命令：
 
来自 <https://www.8kiz.cn/archives/6172.html> 
g++-9 --version
若显示版本信息，则表示：已经安装了支持std::filesystem的编译器。接下来，你需要在编译你的C++代码时，指定C++17标准和链接到正确的库。你可以在g++命令行中添加 -std=c++17和 -lstdc++fs参数
 

解决方法：
（1）以在g++命令行中添加 -std=c++17和 -lstdc++fs参数。例如：g++-9 -std=c++17 -lstdc++fs your_file.cpp这里，-std=c++17告诉编译器使用C++17标准，-lstdc++fs告诉链接器链接到std::filesystem库。
（2）若使用的是CMake构建你的项目，可以在CMakeLists.txt文件中添加以下内容：
set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
	link_libraries(stdc++fs)
这样，CMake会自动为你的项目添加正确的编译和链接选项。
2.4 运行代码
$ cd build
$ mkdir images 
$ cp -r ~/ultralytics_old/datasets/coco128/images/train2017/* images/
image下全部是图片
$ ./Yolov8OnnxRuntimeCPPInference
