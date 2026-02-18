# CUDA

在一般情况下，本程序默认使用 CPU 作为基础的计算资源，默认的并发数为CPU核心数，尽可能保证最大的资源利用效率

但是，如果能够使用 CUDA 作为基础的计算资源，整体的计算效率会更加强大

## 环境配置

首先，需要确保系统中具备有 CUDA 和 CUDA ToolKits

```shell
PS D:\_Code\opencv-python> nvidia-smi
Wed Feb 18 18:10:30 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 591.86                 Driver Version: 591.86         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   45C    P8              2W /   70W |     441MiB /   8188MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           16088    C+G   ...ram Files\Tencent\QQNT\QQ.exe      N/A      |
|    0   N/A  N/A           28376    C+G   ...ogram Files\cursor\Cursor.exe      N/A      |
+-----------------------------------------------------------------------------------------+


```

## 配置依赖环境

假定 Python 虚拟环境位于 `D:\_Code\opencv-cuda`

```shell
# 创建虚拟环境
PS D:\_Code> python -m venv opencv-cuda
# 配置 Python 基础库
PS D:\_Code> D:\_Code\opencv-cuda\Scripts\Activate.ps1
(opencv-cuda) PS D:\_Code> python -m pip install --upgrade pip setuptools wheel numpy
```

获取源码

```shell
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

## 配置编译参数

进入目录

```shell
cd C:\dev\src\opencv
mkdir build && cd build
```

**重要**：RTX 4060 需显式指定 CUDA 架构，否则会出现 `list GET given empty list` 错误。

**使用 VS 2026**：CUDA 官方只支持到 VS 2022，用 VS 2026 编译需加 **`CUDA_NVCC_FLAGS`**（OpenCV 实际传给 nvcc 的变量），否则 nvcc 会报 “unsupported Microsoft Visual Studio version”。若之前用过 `CMAKE_CUDA_FLAGS` 仍报错，请改为本变量并**重新配置**（见下方）。

**让 cv2 装进 venv**：当前 `CMAKE_INSTALL_PREFIX` 是项目根目录，头文件/库会装到 FaceSeeker；若希望 **Python 的 cv2 直接装进虚拟环境**，需加 **`OPENCV_PYTHON3_INSTALL_PATH`** 指向 venv 的 `Lib/site-packages`，否则 OpenCV 可能误用系统 Python 路径。

```shell
cmake -G "Visual Studio 18 2026" -A x64 `
  -D CMAKE_BUILD_TYPE=Release `
  -D CMAKE_INSTALL_PREFIX="D:/_Code/FaceAtlas" `
  -D BUILD_opencv_python3=ON `
  -D PYTHON3_EXECUTABLE="D:/_Code/FaceAtlas/.venv/Scripts/python.exe" `
  -D OPENCV_PYTHON3_INSTALL_PATH="D:/_Code/FaceAtlas/.venv/Lib/site-packages" `
  -D OPENCV_ENABLE_NONFREE=ON `
  -D WITH_CUDA=ON `
  -D CUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1" `
  -D CUDA_ARCH_BIN=8.9 `
  -D CUDA_NVCC_FLAGS="-allow-unsupported-compiler" `
  -D OPENCV_DNN_CUDA=ON `
  -D WITH_CUDNN=ON `
  -D CUDNN_INCLUDE_DIR="C:/Program Files/NVIDIA/CUDNN/v9.19/include/13.1" `
  -D CUDNN_LIBRARY="C:/Program Files/NVIDIA/CUDNN/v9.19/lib/13.1/x64/cudnn.lib" `
  -D BUILD_EXAMPLES=OFF `
  -D BUILD_TESTS=OFF `
  -D BUILD_PERF_TESTS=OFF `
  -D OPENCV_EXTRA_MODULES_PATH="../../opencv_contrib/modules" `
  ..
```

## 开始编译与安装

CMake 配置成功只生成了解决方案，**还没有 .lib/.dll**。需要先编译，再安装。

若之前配置用过 `CMAKE_CUDA_FLAGS` 仍报 “unsupported Visual Studio version”，请**重新配置**：在 `opencv\build` 下执行 `Remove-Item CMakeCache.txt -ErrorAction SilentlyContinue`，再执行一次上面的完整 cmake 命令（含 `-D CUDA_NVCC_FLAGS="-allow-unsupported-compiler"`），然后再编译。

**1. 编译**（在 OpenCV 的 build 目录，如 `D:\_Code\opencv\build`）：

```shell
cmake --build . --config Release -- /m
```

**2. 安装到 FaceAtlas**（复制到 `CMAKE_INSTALL_PREFIX` 即 `D:\_Code\FaceAtlas`）：

```shell
cmake --build . --config Release --target INSTALL -- /m
```

## 编译产物在哪里

- **仅做了 CMake 配置**：产物还没有生成，需要执行上面的**编译**和**安装**。
- **已编译、未安装**：DLL/库在 build 目录下，例如 `build\bin\Release\`、各模块输出目录；Python 的 `cv2` 在 `build\lib\python3\` 下某处。使用起来不如安装后方便。
- **已执行 INSTALL 后**：头文件、库、DLL 在 **CMAKE_INSTALL_PREFIX**（如 `D:\_Code\FaceAtlas`）下；Python 的 cv2 安装位置由 **OPENCV_PYTHON3_INSTALL_PATH** 决定：
  - 若配置时加了 `-D OPENCV_PYTHON3_INSTALL_PATH="D:/_Code/FaceAtlas/.venv/Lib/site-packages"`，则 cv2 会装到 **`.venv\Lib\site-packages\cv2\`**，激活 venv 后可直接 `import cv2`。
  - 若未设置该变量，OpenCV 可能按系统 Python 或错误路径安装，导致“编译产物没有自动进 venv”；解决办法见下方常见报错。
  - 其他产物：`FaceAtlas\include\`、`FaceAtlas\x64\vc18\lib\`、`FaceAtlas\x64\vc18\bin\` 等。

**小结**：要得到“编译后的产物”，需要先 `cmake --build . --config Release`，再 `cmake --build . --config Release --target INSTALL`；安装完成后到 **FaceAtlas**（或你的安装前缀）下找即可。

## 常见报错与处理

### 1. `cudnn.h: No such file or directory` / `Could NOT find CUDNN`

**原因**：CMake 在指定路径下找不到 `cudnn.h`，多为 cuDNN 未正确安装或路径不一致。

**处理步骤：**

1. 确认 cuDNN 是否已安装到该路径，在 PowerShell 中执行：

   ```powershell
   Get-ChildItem "C:\Program Files\NVIDIA\CUDNN\v9.19\include\13.1" -ErrorAction SilentlyContinue
   ```

   若目录不存在或其中没有 `cudnn.h`（或 `cudnn_version.h`），说明路径错误或未安装。**注意**：cuDNN v9 按 CUDA 版本分子目录，若使用 CUDA 13.1，头文件在 `include\13.1` 下，不是 `include`。

2. 若尚未安装 cuDNN：
   - 从 [NVIDIA cuDNN 归档](https://developer.nvidia.com/cudnn-archive) 下载与 CUDA 13.x 匹配的 Windows 版本（如 v9.x）。
   - 解压后按安装说明放置；若包内按 CUDA 版本分子目录（如 `include\13.1`），则 `CUDNN_INCLUDE_DIR` 应指向含 `cudnn.h` 的那一层（例如 `.../include/13.1`）。

3. 若 cuDNN 安装在其他路径（例如 `D:\cudnn\include`），则把 CMake 中的路径改为实际路径：

   ```text
   -D CUDNN_INCLUDE_DIR="D:/cudnn/include/13.1" `
   -D CUDNN_LIBRARY="D:/cudnn/lib/13.1/x64/cudnn.lib" `
   ```
   **注意**：cuDNN v9 的库也在版本子目录下，CUDA 13.1 对应 `lib/13.1/x64/cudnn.lib`，不是 `lib/x64/cudnn.lib`。

### 2. `unsupported Microsoft Visual Studio version! Only 2019–2022 are supported`

**原因**：CUDA 的 nvcc 目前只“官方”支持 VS 2019–2022，用 **VS 2026**（或更新）会触发 `host_config.h` 的版本检查报错。

**处理**：OpenCV 传给 nvcc 的是 **CUDA_NVCC_FLAGS**（不是 CMAKE_CUDA_FLAGS）。在 CMake 配置里加上：

```text
-D CUDA_NVCC_FLAGS="-allow-unsupported-compiler"
```

把上面这一行加进完整 cmake 命令（本文主配置示例已包含），然后**必须重新配置**再编译，否则不会生效：

1. 在 `opencv\build` 下删掉缓存再配置（推荐）：  
   `Remove-Item CMakeCache.txt -ErrorAction SilentlyContinue; cmake -G "Visual Studio 18 2026" -A x64 ...`（后面参数同主示例）
2. 或直接再执行一遍完整 cmake 命令（含 `-D CUDA_NVCC_FLAGS=...`），再 `cmake --build . --config Release -- /m`。

使用未官方支持的编译器可能有兼容性风险，但多数情况下可正常使用。

### 3. `LNK1181: 无法打开输入文件 "cudnn.lib"`

**原因**：链接器找不到 `cudnn.lib`。cuDNN v9 的库在 **`lib\13.1\x64\cudnn.lib`**（按 CUDA 版本分子目录），若 CMake 里写成了 `lib\x64\cudnn.lib` 会报此错。

**处理**：重新配置 CMake，把库路径改为（CUDA 13.1 时）：
```text
-D CUDNN_LIBRARY="C:/Program Files/NVIDIA/CUDNN/v9.19/lib/13.1/x64/cudnn.lib"
```
然后重新编译（无需删 build，在 build 目录再跑一遍上面的 cmake 命令即可覆盖配置，再 `cmake --build . --config Release`）。

### 4. `list GET given empty list`（OpenCVDetectCUDAUtils.cmake）

**原因**：未检测到 CUDA 架构列表，常见于 Windows + Visual Studio 构建。

**处理**：在 cmake 命令中显式指定 GPU 架构（见上方示例）：

- RTX 4060：`-D CUDA_ARCH_BIN=8.9`
- 其他显卡可查 [NVIDIA 计算能力表](https://developer.nvidia.com/cuda-gpus)，如 RTX 3080 为 8.6，RTX 4090 为 8.9。

### 5. `DLL load failed while importing cv2: 找不到指定的模块`

**原因**：cv2.pyd 依赖 OpenCV DLL（`x64/vc18/bin`）、CUDA 与 **cuDNN** 的 DLL。安装时生成的 `cv2/config.py` 里通常只加入了 OpenCV 和 CUDA 的路径，没有加 cuDNN 的 `bin` 目录，运行时加载 `opencv_dnn*.dll` 时找不到 `cudnn64_9.dll` 等就会报此错。

**处理**：在 **`.venv\Lib\site-packages\cv2\config.py`** 的 `BINARIES_PATHS` 中增加 cuDNN 的 bin 路径（路径按本机 cuDNN 版本调整），例如：

```python
os.path.join(os.getenv('CUDNN_PATH', 'C:/Program Files/NVIDIA/CUDNN/v9.19'), 'bin/13.1/x64'),
```

若 cuDNN 装在其他盘或版本不同，可设置环境变量 `CUDNN_PATH` 或把上面路径改成实际路径。**注意**：重新执行 OpenCV 的 INSTALL 可能会覆盖该 `config.py`，若再次出现此错误，需重新加上 cuDNN 路径。

### 6. 编译产物没有自动装进 venv / 找不到 cv2（ModuleNotFoundError）

**原因**：未设置 **OPENCV_PYTHON3_INSTALL_PATH** 时，OpenCV 在 Windows 上会从 `python.exe` 所在目录（如 `Scripts`）推断 site-packages，而 venv 的 site-packages 在 `.venv\Lib\site-packages`，推断会失败并可能用系统 Python 路径，所以 cv2 不会装进当前 venv。

**处理**：重新配置 CMake，在命令中加上（路径按你的项目与 venv 实际位置修改）：

```text
-D OPENCV_PYTHON3_INSTALL_PATH="D:/_Code/FaceAtlas/.venv/Lib/site-packages"
```

然后重新执行 **INSTALL** 目标；完成后在激活的 venv 里应能 `import cv2`。若 venv 不在 FaceAtlas 下，把上面的路径改成你的 `.venv\Lib\site-packages` 绝对路径即可。

### 7. 暂时不启用 DNN CUDA（仅要 OpenCV + CUDA，不需要 cuDNN）

若暂时无法解决 cuDNN 路径或版本问题，可先关闭 DNN 的 CUDA 后端，仅用 CUDA 模块编译通过：

```shell
cmake -G "Visual Studio 18 2026" -A x64 `
  -D CMAKE_BUILD_TYPE=Release `
  -D CMAKE_INSTALL_PREFIX="D:/_Code/FaceAtlas" `
  -D BUILD_opencv_python3=ON `
  -D PYTHON3_EXECUTABLE="D:/_Code/FaceAtlas/.venv/Scripts/python.exe" `
  -D OPENCV_PYTHON3_INSTALL_PATH="D:/_Code/FaceAtlas/.venv/Lib/site-packages" `
  -D OPENCV_ENABLE_NONFREE=ON `
  -D WITH_CUDA=ON `
  -D CUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1" `
  -D CUDA_ARCH_BIN=8.9 `
  -D CUDA_NVCC_FLAGS="-allow-unsupported-compiler" `
  -D OPENCV_DNN_CUDA=OFF `
  -D WITH_CUDNN=OFF `
  -D BUILD_EXAMPLES=OFF `
  -D BUILD_TESTS=OFF `
  -D BUILD_PERF_TESTS=OFF `
  -D OPENCV_EXTRA_MODULES_PATH="../../opencv_contrib/modules" `
  ..
```

这样会得到带 CUDA 的 OpenCV，但 DNN 模块不会使用 GPU 加速。