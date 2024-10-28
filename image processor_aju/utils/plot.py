"""
Módulo de Plotagem de Imagens

Este módulo contém funções para plotagem de imagens e histogramas usando matplotlib.

Dependências:
    - matplotlib

As funções neste módulo operam em imagens representadas como arrays NumPy.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_image(image: np.ndarray) -> None:
    """
    Plota uma única imagem.

    Args:
        image (np.ndarray): A imagem a ser plotada.

    Returns:
        None

    Example:
        plot_image(my_image)
    """
    plt.figure(figsize=(12, 4))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


def plot_result(*args: np.ndarray) -> None:
    """
    Plota múltiplas imagens lado a lado.

    Args:
        *args: Sequência variável de arrays NumPy representando imagens.

    Returns:
        None

    Example:
        plot_result(image1, image2, result_image)
    """
    number_images = len(args)
    fig, axis = plt.subplots(nrows=1, ncols=number_images, figsize=(12, 4))
    names_lst = [f'Image {i}' for i in range(1, number_images)]
    names_lst.append('Result')

    for ax, name, image in zip(axis, names_lst, args):
        ax.set_title(name)
        ax.imshow(image, cmap='gray')
        ax.axis('off')

    fig.tight_layout()
    plt.show()


def plot_histogram(image: np.ndarray) -> None:
    """
    Plota histogramas de cores para uma imagem RGB.

    Args:
        image (np.ndarray): A imagem RGB para a qual plotar histogramas.

    Returns:
        None

    Raises:
        ValueError: Se a imagem não for uma imagem RGB 3D.

    Example:
        plot_histogram(rgb_image)
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("A imagem deve ser RGB (3 canais).")

    fig, axis = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex=True, sharey=True)
    color_lst = ['red', 'green', 'blue']

    for index, (ax, color) in enumerate(zip(axis, color_lst)):
        ax.set_title(f'{color.title()} histogram')
        ax.hist(image[:, :, index].ravel(), bins=256, color=color, alpha=0.8)

    fig.tight_layout()
    plt.show()
