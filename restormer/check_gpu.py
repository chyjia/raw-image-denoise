"""Quick GPU capability and bf16 sanity check for the training environment."""

import torch


def main() -> None:
    print("torch", torch.__version__)
    print("cuda_runtime", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device is not available inside this environment.")

    device = torch.device("cuda")
    print("device_name", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
    print("bf16_supported", torch.cuda.is_bf16_supported())

    a = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    c = (a @ b).float().sum().item()
    torch.cuda.synchronize()
    print("bf16_matmul_finite", bool(c == c))
    print("peak_mem_mb", round(torch.cuda.max_memory_allocated() / 1e6, 1))


if __name__ == "__main__":
    main()
