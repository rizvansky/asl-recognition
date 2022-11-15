from typing import Dict

import pandas as pd
import tensorflow as tf


class MobileNetV2:
    def __init__(
            self,
            input_shape=(224, 224, 3),
            image_path_col_name='image_path',
            label_col_name='label',
            batch_size=32,
            num_classes = 29
    ):
        self.model = None
        self.input_shape = input_shape
        self.image_path_col_name = image_path_col_name
        self.label_col_name = label_col_name
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.init_model()

    def init_model(self) -> None:
        pretrained_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        pretrained_model.trainable = False
        inputs = pretrained_model.input

        # model tuning section #
        x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
        x = tf.keras.layers.Dense(128, activation='relu')(x)

        outputs = tf.keras.layers.Dense(29, activation='softmax')(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def data_generator(
            self,
            splits: Dict[str, pd.DataFrame],
            validation_split: int = 0.2
    ) -> Dict[str, tf.keras.preprocessing.image.Iterator]:
        test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
        )
        test_images = test_generator.flow_from_dataframe(
            dataframe=splits['test'],
            x_col=self.image_path_col_name,
            y_col=self.label_col_name,
            target_size=self.input_shape[:2:],
            color_mode='rgb',
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=False
        )
        if 'train' not in splits:
            return {
                'test_images': test_images
            }
        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
            validation_split=validation_split
        )

        train_images = train_generator.flow_from_dataframe(
            dataframe=splits['train'],
            x_col=self.image_path_col_name,
            y_col=self.label_col_name,
            target_size=self.input_shape[:2:],
            color_mode='rgb',
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
            color_mode='rgb',
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
