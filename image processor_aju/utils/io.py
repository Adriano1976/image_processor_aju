"""
Módulo de Processamento de Imagem

Este módulo contém funções para processamento, análise, leitura e gravação de imagens.

Dependências:
    - numpy
    - scikit-image

As funções neste módulo operam em imagens representadas como arrays NumPy.
"""

import numpy as np
from skimage.io import imread, imsave


# ... [Funções anteriores permanecem inalteradas] ...

def read_image(path: str, is_gray: bool = False) -> np.ndarray:
    """
    Lê uma imagem de um arquivo.

    Esta função carrega uma imagem do caminho especificado e opcionalmente
    a converte para escala de cinza.

    Args:
        path (str): Caminho para o arquivo de imagem.
        is_gray (bool, opcional): Se True, carrega a imagem em escala de cinza.
                                  Padrão é False.

    Returns:
        np.ndarray: Imagem carregada como um array NumPy.

    Example:
        img = read_image('caminho/para/imagem.jpg', is_gray=True)
    """
    image = imread(path, as_gray=is_gray)
    return image


def save_image(image: np.ndarray, path: str) -> None:
    """
    Salva uma imagem em um arquivo.

    Esta função grava uma imagem representada como um array NumPy em um arquivo
    no caminho especificado.

    Args:
        image (np.ndarray): Imagem a ser salva.
        path (str): Caminho onde a imagem será salva.

    Returns:
        None

    Example:
        save_image(img, 'caminho/para/salvar/imagem.jpg')
    """
    imsave(path, image)
