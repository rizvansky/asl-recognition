# Hide TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
import json
from pathlib import Path

import tensorflow as tf
import hydra
import torch
import cv2
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm


def load_classifier(weights_path):
    return tf.keras.models.load_model(weights_path)


def predict(model, classes, images):
    """ images shape: batchx32x32x3 """
    return np.vectorize(lambda x: classes[x])(
        np.argmax(model.predict(images), axis=1)
    )


@hydra.main(version_base=None, config_path='../configs', config_name='run_pipeline.yaml')
def main(config: DictConfig):

    # Initialize the object detection model
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        object_detector = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=config.models.object_detector.weights_path, force_reload=True
        )
    object_detector.to(config.device)

    if config.use_classifier:

        # Initialize the classification model
        classifier = load_classifier(config.models.classifier.weights_path)

    test_video_path = Path(config.test_video_path)
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    test_video_cap = cv2.VideoCapture(str(test_video_path))
    test_video_length = int(test_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    result_cap = cv2.VideoWriter(
        str(save_dir / f"{test_video_path.stem}.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (int(test_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(test_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    prog_bar = tqdm(total=test_video_length, position=0)
    while True:
        success, img = test_video_cap.read()
        if success:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            break

        detections = json.loads(object_detector(img).pandas().xyxy[0].to_json(orient='records'))
        for det in detections:
            if det['confidence'] >= config.models.object_detector.conf_thresh:
                x_min, y_min, x_max, y_max = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])

                if config.use_classifier:

                    # Crop the image
                    crop = img[y_min: y_max, x_min: x_max]

                    # Resize the crop before passing it to the classifier
                    crop = cv2.resize(crop, dsize=(32, 32))

                    # Create a batch
                    crop = np.expand_dims(crop, 0)
                    class_name = predict(classifier, list(config.models.classifier.classes), crop)[0]
                else:
                    class_name = det['name']

                img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                text_width, text_height = cv2.getTextSize(
                    class_name, cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=1)[0]
                y_center = (y_min + y_max) // 2
                text_origin = (x_min + 17, y_center + 17)
                box_coords = (
                    (text_origin[0], text_origin[1]),
                    (text_origin[0] + text_width + 2, text_origin[1] - text_height - 2)
                )
                background_color = (0, 0, 0)
                img = cv2.rectangle(img, box_coords[0], box_coords[1], background_color, cv2.FILLED)
                color = (255 - background_color[0], 255 - background_color[1], 255 - background_color[2])
                cv2.putText(
                    img, class_name, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 2, tuple(color), 1, cv2.LINE_AA
                )

        result_cap.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        prog_bar.update(1)

    result_cap.release()


if __name__ == '__main__':
    main()
