"""PyTorch 计算后端与 CUDA 资源测试脚本"""
import sys

try:
    import torch
except ImportError:
    print("错误: 未安装 PyTorch，请运行 pip install torch")
    sys.exit(1)


def main():
    print("=" * 60)
    print("PyTorch 计算后端与 CUDA 资源测试")
    print("=" * 60)

    # 1. PyTorch 版本
    print(f"\n[PyTorch 版本] {torch.__version__}")

    # 2. CUDA 可用性
    cuda_available = torch.cuda.is_available()
    print(f"\n[CUDA 可用性] {'是' if cuda_available else '否'}")

    if not cuda_available:
        print("\nCUDA 不可用，当前将使用 CPU 作为计算设备")
        print("如需 GPU 加速，请检查:")
        print("  - NVIDIA 驱动是否正确安装")
        print("  - 是否安装了 CUDA 版本的 PyTorch")
        print("  - pip 安装示例: pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return

    # 3. CUDA 版本
    print(f"[CUDA 版本 (编译)] {torch.version.cuda or 'N/A'}")

    # 4. cuDNN 状态
    cudnn_available = torch.backends.cudnn.is_available()
    cudnn_enabled = torch.backends.cudnn.enabled
    print(f"[cuDNN 可用] {'是' if cudnn_available else '否'}")
    print(f"[cuDNN 已启用] {'是' if cudnn_enabled else '否'}")
    if cudnn_available:
        print(f"[cuDNN 版本] {torch.backends.cudnn.version()}")

    # 5. 计算后端概览
    print("\n" + "-" * 60)
    print("计算后端状态")
    print("-" * 60)
    mps_available = False
    if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "is_available"):
        mps_available = torch.backends.mps.is_available()

    backends = [
        ("CUDA", torch.cuda.is_available()),
        ("cuDNN", torch.backends.cudnn.is_available()),
        ("MPS (Apple)", mps_available),
    ]
    for name, available in backends:
        status = "可用" if available else "不可用"
        print(f"  {name}: {status}")

    # 6. GPU 列表与资源
    print("\n" + "-" * 60)
    print("可用 GPU")
    print("-" * 60)
    device_count = torch.cuda.device_count()
    print(f"GPU 数量: {device_count}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\n  [GPU {i}] {props.name}")
        print(f"    计算能力: {props.major}.{props.minor}")
        print(f"    总显存:   {props.total_memory / 1024**3:.2f} GB")
        print(f"    多处理器: {props.multi_processor_count}")

        # 当前显存使用情况
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        mem_free = (props.total_memory - torch.cuda.memory_reserved(i)) / 1024**3
        print(f"    已分配:   {mem_allocated:.3f} GB")
        print(f"    已保留:   {mem_reserved:.3f} GB")
        print(f"    可用:     {mem_free:.2f} GB")

    # 7. 当前设备
    if device_count > 0:
        current = torch.cuda.current_device()
        print(f"\n[当前设备] cuda:{current}")

    # 8. 快速功能测试
    print("\n" + "-" * 60)
    print("快速功能测试")
    print("-" * 60)
    try:
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        print("  GPU 矩阵运算测试: 通过")
    except Exception as e:
        print(f"  GPU 矩阵运算测试: 失败 - {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
