from typing import Dict

import pandas as pd
import tensorflow as tf


class VGG16:
    def __init__(
            self,
            input_shape=(224, 224, 3),
            image_path_col_name='image_path',
            label_col_name='label',
            batch_size=32,
            num_classes=29
    ):
        self.model = None
        self.input_shape = input_shape
        self.image_path_col_name = image_path_col_name
        self.label_col_name = label_col_name
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.init_model()

    def init_model(self) -> None:
        pretrained_model = tf.keras.applications.vgg16.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        pretrained_model.trainable = False

        # model tuning section #
        flatten_layer = tf.keras.layers.Flatten()
        dense_layer_1 = tf.keras.layers.Dense(512, activation='relu')
        dropout_layer_1 = tf.keras.layers.Dropout(0.5)
        prediction_layer = tf.keras.layers.Dense(self.num_classes, activation='softmax')
        self.model = tf.keras.models.Sequential([
            pretrained_model,
            flatten_layer,
            dense_layer_1,
            dropout_layer_1,
            prediction_layer
        ])

    def data_generator(
            self,
            splits: Dict[str, pd.DataFrame],
            validation_split: int | None = 0.2
    ) -> Dict[str, tf.keras.preprocessing.image.Iterator]:
        test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
        test_images = test_generator.flow_from_dataframe(
            dataframe=splits['test'],
            x_col=self.image_path_col_name,
            y_col=self.label_col_name,
            target_size=self.input_shape[:2:],
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=False,
            subset='validation'
        )
        if 'train' not in splits:
            return {
                'test_images': test_images
            }

        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1/255.0,
            validation_split=validation_split
        )

        train_images = train_generator.flow_from_dataframe(
            dataframe=splits['train'],
            x_col=self.image_path_col_name,
            y_col=self.label_col_name,
            target_size=self.input_shape[:2:],
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=True,
            subset='training'
        )

        val_images = train_generator.flow_from_dataframe(
            dataframe=splits['train'],
            x_col=self.image_path_col_name,
            y_col=self.label_col_name,
            target_size=self.input_shape[:2:],
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=True,
            subset='validation'
        )

        return {
            'train_images': train_images,
            'val_images': val_images,
            'test_images': test_images
        }
