import os
import re
import urllib.request
import subprocess
import argparse

# Architecture Configuration Mapping
ARCH_MAP = {
    "nano": {
        "config": "exps/customConfigs/yolox_voc_n.py",
        "weight": "yolox_nano.pth",
        "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth"
    },
    "tiny": {
        "config": "exps/customConfigs/yolox_voc_t.py",
        "weight": "yolox_tiny.pth",
        "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth"
    },
    "small": {
        "config": "exps/customConfigs/yolox_voc_s.py",
        "weight": "yolox_s.pth",
        "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
    },
    "medium": {
        "config": "exps/customConfigs/yolox_voc_m.py",
        "weight": "yolox_m.pth",
        "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth"
    },
    "large": {
        "config": "exps/customConfigs/yolox_voc_l.py",
        "weight": "yolox_l.pth",
        "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth"
    }
}


def modify_config(config_path, input_size, max_epoch):
    """
    Read, modify, and save the YOLOX configuration file using regex.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r") as file:
        content = file.read()

    content = re.sub(
        r"self\.input_size\s*=\s*\([^)]+\)",
        f"self.input_size = ({input_size}, {input_size})",
        content
    )

    content = re.sub(
        r"self\.test_size\s*=\s*\([^)]+\)",
        f"self.test_size = ({input_size}, {input_size})",
        content
    )

    content = re.sub(
        r"self\.max_epoch\s*=\s*\d+",
        f"self.max_epoch = {max_epoch}",
        content
    )

    with open(config_path, "w") as file:
        file.write(content)

    print(
        f"[INFO] Updated {config_path}: "
        f"input_size=({input_size}, {input_size}), "
        f"max_epoch={max_epoch}"
    )


def download_weight(weight_name, url):
    """
    Download pretrained weights if they do not already exist.
    """
    if not os.path.exists(weight_name):
        print(f"[INFO] Weight file '{weight_name}' not found. Downloading...")
        urllib.request.urlretrieve(url, weight_name)
        print(f"[INFO] Download completed: {weight_name}")
    else:
        print(f"[INFO] Weight file already exists: {weight_name}")


def run_training(
    config_path,
    batch_size,
    weight_name,
    devices,
    fp16,
    resume
):
    """
    Build and execute the YOLOX training command dynamically.
    """
    command = [
        "python",
        "tools/train.py",
        "-f",
        config_path,
        "-d",
        str(devices),
        "-b",
        str(batch_size),
        "-c",
        weight_name,
        "-o"
    ]

    if fp16:
        command.append("--fp16")

    if resume:
        command.append("--resume")

    print(
        "\n[INFO] Executing training command:\n"
        + " ".join(command)
        + "\n"
    )

    subprocess.run(command)


def main():
    parser = argparse.ArgumentParser(
        description="YOLOX Training Automation Script"
    )

    # Required arguments
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=ARCH_MAP.keys(),
        help="Model architecture: nano, tiny, small, medium, large"
    )

    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Input and test image size (e.g. 416)"
    )

    parser.add_argument(
        "--batch",
        type=int,
        required=True,
        help="Training batch size"
    )

    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
        help="Maximum number of training epochs"
    )

    # Optional arguments
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed precision (FP16) training"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint"
    )

    args = parser.parse_args()

    selected_arch = ARCH_MAP[args.arch]

    try:
        modify_config(
            selected_arch["config"],
            args.size,
            args.epoch
        )

        download_weight(
            selected_arch["weight"],
            selected_arch["url"]
        )

        run_training(
            config_path=selected_arch["config"],
            batch_size=args.batch,
            weight_name=selected_arch["weight"],
            devices=args.devices,
            fp16=args.fp16,
            resume=args.resume
        )

    except Exception as e:
        print(f"[ERROR] Execution failed: {e}")


if __name__ == "__main__":
    main()