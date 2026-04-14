import torch


def main() -> None:
    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"current_device={torch.cuda.current_device()}")
        print(f"device_name={torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("device_name=cpu")


if __name__ == "__main__":
    main()
