import os
import re
import glob
import urllib.request
import subprocess
import argparse
import sys

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


def find_annotations_dir(config_path):
    """
    Cari lokasi folder Annotations dari config atau fallback ke direktori default YOLOX.
    """
    default_dir = os.path.join("datasets", "VOCdevkit", "VOC2012", "Annotations")
    if not os.path.exists(config_path):
        return default_dir

    with open(config_path, "r") as f:
        content = f.read()

    match = re.search(r"data_dir\s*=\s*r?['\"]([^'\"]+)['\"]", content)
    if match:
        base_dir = match.group(1)
        possible_ann_dir = os.path.join(base_dir, "VOC2012", "Annotations")
        if os.path.exists(possible_ann_dir):
            return possible_ann_dir
        
        possible_ann_dir_2 = os.path.join(base_dir, "Annotations")
        if os.path.exists(possible_ann_dir_2):
            return possible_ann_dir_2

    return default_dir


def update_voc_classes(config_path):
    """
    Fast-scan seluruh file XML di folder Annotations menggunakan regex, 
    urutkan kelas secara alfabetis, lalu perbarui yolox/data/datasets/voc_classes.py
    """
    annotations_dir = find_annotations_dir(config_path)
    print(f"[INFO] Scanning XML files in: {annotations_dir}")

    xml_files = glob.glob(os.path.join(annotations_dir, "*.xml"))
    if not xml_files:
        print(f"[WARNING] Tidak ditemukan file .xml di {annotations_dir}!")
        return 0

    detected_classes = set()
    # Tag <name>...</name> dibaca via regex langsung tanpa parse tree utuh
    class_pattern = re.compile(r"<name>(.*?)</name>")

    for xml_file in xml_files:
        try:
            with open(xml_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                matches = class_pattern.findall(content)
                for name in matches:
                    detected_classes.add(name.strip())
        except Exception as e:
            print(f"[WARNING] Error reading {xml_file}: {e}")

    # Urutkan sesuai abjad (A-Z)
    sorted_classes = sorted(list(detected_classes))

    if not sorted_classes:
        print("[WARNING] Tidak ada kelas objek yang terdeteksi dari XML!")
        return 0

    print(f"[INFO] Detected {len(sorted_classes)} class(es) (sorted): {sorted_classes}")

    # Path file voc_classes.py
    voc_classes_file = os.path.join("yolox", "data", "datasets", "voc_classes.py")

    # Format isi file voc_classes.py
    formatted_classes = ",\n    ".join([f'"{cls}"' for cls in sorted_classes])
    
    file_content = f"""#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# VOC_CLASSES = ( '__background__', # always index 0
VOC_CLASSES = (
    {formatted_classes},
)
"""

    os.makedirs(os.path.dirname(voc_classes_file), exist_ok=True)
    with open(voc_classes_file, "w") as f:
        f.write(file_content)

    print(f"[INFO] Successfully updated {voc_classes_file}")
    return len(sorted_classes)


def modify_config(config_path, input_size, max_epoch, num_classes):
    """
    Read, modify, and save the YOLOX configuration file using regex.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r") as file:
        content = file.read()

    # Modify self.num_classes
    content = re.sub(
        r"self\.num_classes\s*=\s*\d+",
        f"self.num_classes = {num_classes}",
        content
    )

    # Modify self.input_size
    content = re.sub(
        r"self\.input_size\s*=\s*\([^)]+\)",
        f"self.input_size = ({input_size}, {input_size})",
        content
    )

    # Modify self.test_size
    content = re.sub(
        r"self\.test_size\s*=\s*\([^)]+\)",
        f"self.test_size = ({input_size}, {input_size})",
        content
    )

    # Modify self.max_epoch
    content = re.sub(
        r"self\.max_epoch\s*=\s*\d+",
        f"self.max_epoch = {max_epoch}",
        content
    )

    with open(config_path, "w") as file:
        file.write(content)

    print(
        f"[INFO] Updated {config_path}: "
        f"num_classes={num_classes}, "
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
        sys.executable,
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

    try:
        subprocess.run(command, check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Training dihentikan oleh pengguna (Ctrl+C detected). Exiting cleanly...")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Training process exited with error code: {e.returncode}")
        sys.exit(e.returncode)


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
        "--classNumber",
        type=int,
        default=None,
        help="Number of target classes (Optional, defaults to scanned XML classes)"
    )

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

    # Continue & Model Path Arguments
    parser.add_argument(
        "--continue",
        dest="continue_training",
        action="store_true",
        help="Continue training from a custom model checkpoint"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to custom model checkpoint file (.pth) (REQUIRED when --continue is active)"
    )

    args = parser.parse_args()

    # Validasi logika: Jika --continue aktif, --model_path WAJIB diisi
    if args.continue_training:
        if not args.model_path:
            parser.error("You need to give --model_path if you use flag --continue!")
        if not os.path.exists(args.model_path):
            parser.error(f"File checkpoint model not found in this path: {args.model_path}")

    selected_arch = ARCH_MAP[args.arch]

    try:
        # 1. Selalu jalankan Fast XML Scan & Update voc_classes.py
        scanned_num_classes = update_voc_classes(selected_arch["config"])

        # Tentukan nilai final num_classes
        if args.classNumber is not None:
            final_num_classes = args.classNumber
        else:
            final_num_classes = scanned_num_classes if scanned_num_classes > 0 else 1

        # 2. Modify config .py
        modify_config(
            selected_arch["config"],
            args.size,
            args.epoch,
            final_num_classes
        )

        # 3. Tentukan weight yang dipakai
        if args.continue_training:
            # Menggunakan custom checkpoint path hasil training kamu sebelumnya
            weight_to_use = args.model_path
            print(f"[INFO] Continuing training using checkpoint: {weight_to_use}")
        else:
            # Menggunakan pretrained base weight bawaan
            download_weight(
                selected_arch["weight"],
                selected_arch["url"]
            )
            weight_to_use = selected_arch["weight"]

        # 4. Run Training
        run_training(
            config_path=selected_arch["config"],
            batch_size=args.batch,
            weight_name=weight_to_use,
            devices=args.devices,
            fp16=args.fp16,
            resume=args.continue_training
        )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted before training execution. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Execution failed: {e}")


if __name__ == "__main__":
    main()
