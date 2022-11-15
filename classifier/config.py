import tensorflow as tf


training_configuration = {
    'optimizer': tf.keras.optimizers.Adam(
        learning_rate=0.001,
        name='Adam'
    ),
    'epochs': 10,
    'loss': 'categorical_crossentropy',
    'save_checkpoint_path': 'checkpoint.ckpt',
}

splitting_configuration = {
    'train_size': 0.8,
}
