from typing import Dict, List

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
plt.interactive(False)


def display_images(dataset: pd.DataFrame, rows: int, cols: int) -> None:
    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, figsize=(20, 17),
        subplot_kw={'xticks': [], 'yticks': []}
    )

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(dataset['image_path'][i]))
        ax.set_title(dataset['label'][i])
    plt.tight_layout(pad=.5)
    plt.show()


def display_train_val_accuracy_history(history: Dict[str, List[float]]) -> None:
    acc = history['accuracy']
    val_acc = history['val_accuracy']

    epochs = range(len(acc))
    _ = plt.figure(figsize=(14, 7))
    plt.plot(epochs, acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc='lower right')
    plt.show()


def display_train_val_loss_history(history: Dict[str, List[float]]) -> None:
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(len(loss))

    _ = plt.figure(figsize=(14, 7))
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')
