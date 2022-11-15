import argparse
from sklearn.pipeline import Pipeline
from classifier.models.vgg16 import VGG16
from classifier.models.mobile_net_v2 import MobileNetV2
from classifier.main import training, evaluation
from classifier.config import splitting_configuration, training_configuration
from classifier.preprocessing import merge_datasets, default_train_test_split


def loading_stage(model_name, weights_path=None):
    match model_name:
        case 'vgg16':
            _classifier = VGG16()
            if weights_path is None:
                _classifier.model.load(weights=weights_path)
        case 'mobile':
            _classifier = MobileNetV2()
        case _:
            raise ValueError('Invalid model name')
    if weights_path is None:
        _classifier.model.load(weights=weights_path)
    return _classifier


def preprocessing_stage(_classifier, _mode, dirs):
    dataset = merge_datasets(dataset_dirs=dirs)
    _splits = default_train_test_split(dataset, stratify=dataset['label'], **splitting_configuration)
    if _mode == 'inference':
        _splits['test'] = dataset
    # TODO detection on splits (returns tf.keras.preprocessing.image.Iterator)
    data = _classifier.data_generator(
        _splits
    )
    return data


def running_stage(_classifier, _mode, images):
    match _mode:
        case 'train':
            return training(
                model=_classifier.model,
                train_images=images['train_images'],
                val_images=images['val_images'],
                **training_configuration
            )
        case 'inference':
            return evaluation(
                _classifier.model,
                images['test_images']
            )
        case _:
            raise ValueError('Invalid mode')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        help='train or inference',
        required=True,
    )
    parser.add_argument(
        '--classifier_name',
        help='vgg16 or mobile',
        default='vgg16'
    )
    parser.add_argument(
        '--datasets_dir',
        help='dir of datasets in the following format "path1,path2,..."',
        required=True,
    )
    parser.add_argument(
        '--weights_path',
        help='path to model weights',
    )
    args = parser.parse_args()

    #TODO use sklearn Pipeline
    classifier = loading_stage(args.classifier_name, args.weights_path)
    data = preprocessing_stage(classifier, args.mode, args.datasets_dir.split(','))
    result = running_stage(classifier, args.mode, data)
