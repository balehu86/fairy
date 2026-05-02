# device_utils.py
import torch
import time

# ── 快速切换：改这一行即可 ──────────────────────────
# "auto"   : 自动选最快 (会跑微型benchmark)
# "cpu"    : 强制CPU
# "cuda"   : 强制GPU
# "cuda:0" : 指定GPU编号
FORCE_DEVICE = "cpu"
# ──────────────────────────────────────────────────────

def _benchmark(device, n=200):
    """微型benchmark: 逐token推理模拟, 衡量小模型在device上的实际速度"""
    dim = 64
    linear = torch.nn.Linear(dim, dim).to(device)
    x = torch.randn(dim, device=device)
    s = torch.zeros(dim, device=device)

    # warmup
    for _ in range(20):
        s = 0.9 * s + linear(x)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.perf_counter()
    for _ in range(n):
        x = torch.randn(dim, device=device)
        s = 0.9 * s + linear(x)
        s = s.detach()  # 模拟step-by-step推理, 不累积计算图
    torch.cuda.synchronize() if device.type == 'cuda' else None
    return time.perf_counter() - t0

def get_device(force=None):
    req = force or FORCE_DEVICE

    if req == "cpu":
        print("[device] 强制使用 CPU")
        return torch.device("cpu")

    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[device] 请求GPU但不可用, 回退CPU")
            return torch.device("cpu")
        dev = torch.device(req)
        print(f"[device] 强制使用 GPU: {torch.cuda.get_device_name(dev)}")
        return dev

    # auto: benchmark后选最快的
    cpu_dev = torch.device("cpu")
    t_cpu = _benchmark(cpu_dev)
    print(f"[device] benchmark CPU: {t_cpu:.3f}s")

    if torch.cuda.is_available():
        gpu_dev = torch.device("cuda")
        t_gpu = _benchmark(gpu_dev)
        print(f"[device] benchmark GPU: {t_gpu:.3f}s  ({torch.cuda.get_device_name(0)})")

        if t_gpu < t_cpu * 0.8:  # GPU至少快20%才选, 否则CPU更稳
            print(f"[device] 选择 GPU (快 {t_cpu/t_gpu:.1f}x)")
            return gpu_dev
        else:
            print(f"[device] 选择 CPU (小模型GPU开销不值得)")
            return cpu_dev
    else:
        print("[device] GPU不可用, 使用 CPU")
        return cpu_dev

DEVICE = get_device()