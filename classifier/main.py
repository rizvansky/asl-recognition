from typing import Any
import tensorflow as tf

from classifier.preprocessing import default_train_test_split, merge_datasets
from classifier.models.mobile_net_v2 import MobileNetV2
from classifier.models.vgg16 import VGG16
import numpy as np
from classifier.config import training_configuration


def training(
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        train_images: tf.keras.preprocessing.image.Iterator,
        val_images: tf.keras.preprocessing.image.Iterator,
        epochs: int = 10,
        loss: str = 'categorical_crossentropy',
        save_checkpoint_path: str = 'checkpoint',
        model_path: str = 'pretrained_model'
) -> None:
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    res = model.fit(
        train_images,
        validation_data=val_images,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=save_checkpoint_path,
                # save_weights_only=True,
                save_best_only=True,
                verbose=1
            )
        ]
    )
    model.save(model_path)
    return res


def evaluation(
        model: tf.keras.Model,
        test_images: tf.keras.preprocessing.image.Iterator
) -> Any:
    results = model.evaluate(test_images, verbose=0)
    print(f'Test Loss: {results[0]}')
    print(f'Test Accuracy: {results[1] * 100}')
    return results


def predict(
    model: tf.keras.Model,
    test_images: tf.keras.preprocessing.image.Iterator
) -> Any:
    return model.predict(test_images)


if __name__ == '__main__':
    dataset = merge_datasets(['asl_alphabet_train'])
    split = default_train_test_split(dataset, train_size=0.8, stratify=dataset['label'])

    # MobileNetV2 Training #
    mobile_net = MobileNetV2()
    adam = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        name='Adam'
    )
    mobile_net_data = mobile_net.data_generator(
        split['train'],
        split['test']
    )
    _ = training(
        model=mobile_net.model,
        optimizer=adam,
        train_images=mobile_net_data['train_images'],
        val_images=mobile_net_data['val_images']
    )

    # MobileNetV2 Evaluation
    evaluation(
        mobile_net.model,
        mobile_net_data['test_images']
    )

    # VGG16 Training #
    vgg16 = VGG16()
    adam = tf.keras.optimizers.Adam(
        learning_rate=0.0001,
        name='Adam'
    )
    vgg16_data = vgg16.data_generator(
        split['train'],
        split['test']
    )
    _ = training(
        model=vgg16.model,
        optimizer=adam,
        train_images=vgg16_data['train_images'],
        val_images=vgg16_data['val_images']
    )

    # VGG16 Evaluation
    evaluation(
        vgg16.model,
        vgg16_data['test_images']
    )
