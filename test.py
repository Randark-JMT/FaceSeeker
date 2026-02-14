import cv2
import numpy as np
import time

print(f"OpenCV 版本: {cv2.__version__}")
print("=" * 60)

# ---- 1. CUDA 检测 ----
count = cv2.cuda.getCudaEnabledDeviceCount()
if count > 0:
    print(f"[CUDA] 检测到 {count} 个 CUDA 设备")
else:
    print("[CUDA] 当前 OpenCV 版本不支持 CUDA")

# ---- 2. OpenCL 检测 ----
print()
print(f"[OpenCL] haveOpenCL: {cv2.ocl.haveOpenCL()}")
print(f"[OpenCL] useOpenCL:  {cv2.ocl.useOpenCL()}")
if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)
    # 获取 OpenCL 设备信息
    try:
        dev = cv2.ocl.Device.getDefault()
        print(f"[OpenCL] 设备名称: {dev.name()}")
        print(f"[OpenCL] 供应商:   {dev.vendorName()}")
        print(f"[OpenCL] 版本:     {dev.version()}")
    except Exception as e:
        print(f"[OpenCL] 无法获取设备信息: {e}")

# ---- 3. DNN 后端可用性检测 ----
print()
print("=" * 60)
print("DNN 后端检测:")
backends_to_check = [
    ("CUDA",       cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA),
    ("OpenCL",     cv2.dnn.DNN_BACKEND_OPENCV,  cv2.dnn.DNN_TARGET_OPENCL),
    ("OpenCL_FP16",cv2.dnn.DNN_BACKEND_OPENCV,  cv2.dnn.DNN_TARGET_OPENCL_FP16),
    ("CPU",        cv2.dnn.DNN_BACKEND_OPENCV,  cv2.dnn.DNN_TARGET_CPU),
]
for name, backend, target in backends_to_check:
    try:
        targets = cv2.dnn.getAvailableTargets(backend)
        ok = target in targets
        print(f"  {name:15s} -> {'可用' if ok else '不可用'}")
    except Exception as e:
        print(f"  {name:15s} -> 错误: {e}")

# ---- 4. 模型实际推理测试 ----
print()
print("=" * 60)
print("模型推理测试 (YuNet 人脸检测 + SFace 人脸识别):")

det_model = "models/face_detection_yunet_2023mar.onnx"
rec_model = "models/face_recognition_sface_2021dec.onnx"

# 生成一张带简单图案的测试图（纯黑图可能太简单，加点噪声更真实）
test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

configs = [
    ("CPU",        cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU),
    ("OpenCL",     cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_OPENCL),
]

for name, backend, target in configs:
    print(f"\n  --- {name} ---")
    try:
        # 检测模型
        detector = cv2.FaceDetectorYN.create(det_model, "", (640, 640), 0.7, 0.3, 5000, backend, target)
        
        # 预热一次（OpenCL 首次运行会编译内核）
        detector.detect(test_img)
        
        # 计时多次推理
        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            _, faces = detector.detect(test_img)
        t1 = time.perf_counter()
        avg_det = (t1 - t0) / N * 1000
        face_count = 0 if faces is None else len(faces)
        print(f"  YuNet 检测:  {avg_det:.1f} ms/帧 (平均 {N} 次), 检出 {face_count} 张人脸")

        # 识别模型
        recognizer = cv2.FaceRecognizerSF.create(rec_model, "", backend, target)
        
        # 用检测模型输出做特征提取（如果有人脸的话用真实数据，没有就跳过）
        # 构造一个假的人脸区域来测试识别模型是否能运行
        fake_face = np.array([100, 100, 120, 120,  # bbox
                              130, 140, 170, 140,   # 右眼、左眼
                              150, 165,             # 鼻子
                              135, 185, 165, 185,   # 嘴角
                              0.99], dtype=np.float32)
        aligned = recognizer.alignCrop(test_img, fake_face)
        
        # 预热
        recognizer.feature(aligned)
        
        t0 = time.perf_counter()
        for _ in range(N):
            feat = recognizer.feature(aligned)
        t1 = time.perf_counter()
        avg_rec = (t1 - t0) / N * 1000
        print(f"  SFace 识别:  {avg_rec:.1f} ms/帧 (平均 {N} 次), 特征维度 {feat.shape}")
        print(f"  合计:        {avg_det + avg_rec:.1f} ms/帧")

    except Exception as e:
        print(f"  失败: {e}")

print()
print("=" * 60)
print("测试完成")
