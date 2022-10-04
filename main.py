from typing import Dict, List

import preprocessing
from models.mobile_net_v2 import MobileNetV2

import tensorflow as tf


def training(
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        train_images: tf.keras.preprocessing.image.Iterator,
        val_images: tf.keras.preprocessing.image.Iterator,
        epochs: int = 10,
        loss: str = 'categorical_crossentropy'
) -> Dict[str, List[float]]:
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    return model.fit(
        train_images,
        validation_data=val_images,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True
            )
        ]
    ).history


if __name__ == '__main__':
    dataset = preprocessing.merge_datasets(['asl_alphabet_train'])
    split = preprocessing.default_train_test_split(dataset, train_size=0.8, stratify=dataset['label'])

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
    history = training(
        model=mobile_net.model,
        optimizer=adam,
        train_images=mobile_net_data['train_images'],
        val_images=mobile_net_data['val_images']
    )

    # MobileNetV2 Evaluation
    results = mobile_net.model.evaluate(mobile_net_data['test_images'], verbose=0)
    print(f'Test Loss: {results[0]}')
    print(f'Test Accuracy: {results[1] * 100}')
