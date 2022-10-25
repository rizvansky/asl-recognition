from argparse import ArgumentParser
from glob import glob
import json
import shutil
from pathlib import Path
import os
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser(description="Convert COCO formatted data to YOLOv5 format")
    parser.add_argument(
        "--coco_path",
        help="Path to the COCO dataset with train, valid and test folders",
        required=True,
    )
    parser.add_argument(
        "--yolo_path", help="Path for converted YOLOv5 dataset", default="YOLO_Dataset"
    )
    return parser.parse_args()


def prepare_yolo(yolo_path):
    if not glob(yolo_path):
        os.mkdir(yolo_path)
    yolo_dir = Path(yolo_path)
    for split in ["train", "valid", "test"]:
        split_dir = str(yolo_dir / split)
        if glob(split_dir):
            shutil.rmtree(split_dir)
        os.mkdir(split_dir)

        for folder in ["images", "labels"]:
            os.mkdir(yolo_dir / split / folder)


def coco_to_yolo(coco_path, yolo_path):
    coco_dir = Path(coco_path)
    yolo_dir = Path(yolo_path)
    for split in ["train", "valid", "test"]:
        with open(coco_dir / split / "_annotations.coco.json") as json_anno:
            annotation_file = json.load(json_anno)
        annotations = annotation_file["annotations"]
        images = annotation_file["images"]

        for anno in tqdm(annotations, desc=f"{split} split"):
            image = images[anno["image_id"]]
            cl = anno["category_id"] - 1
            x1, y1, x2, y2 = anno["bbox"]
            x, w = (x1 + x2 / 2) / image["width"], x2 / image["width"]
            y, h = (y1 + y2 / 2) / image["height"], y2 / image["height"]
            detection = " ".join([str(i) for i in [cl, x, y, w, h]])
            filename = image["file_name"][:-3] + "txt"
            filename = str(yolo_dir / split / "labels" / filename)
            with open(filename, "a") as f:
                f.write(detection)

        for image in images:
            shutil.copy(
                coco_dir / split / image["file_name"],
                yolo_dir / split / "images" / image["file_name"],
            )


if __name__ == "__main__":
    args = parse_args()
    prepare_yolo(args.yolo_path)
    coco_to_yolo(args.coco_path, args.yolo_path)
