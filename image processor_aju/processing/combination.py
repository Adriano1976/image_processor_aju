"""
Módulo de Processamento de Imagem

Este módulo contém funções para processamento e análise de imagens.

Dependências:
    - numpy
    - scikit-image

As funções neste módulo operam em imagens representadas como arrays NumPy.
"""

import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity


def find_difference(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Calcula a diferença estrutural entre duas imagens.

    Esta função converte as imagens para escala de cinza, calcula a similaridade
    estrutural e retorna uma imagem de diferença normalizada.

    Args:
        image1 (np.ndarray): Primeira imagem para comparação.
        image2 (np.ndarray): Segunda imagem para comparação.

    Returns:
        np.ndarray: Imagem de diferença normalizada.

    Raises:
        AssertionError: Se as imagens não tiverem a mesma forma.

    Example:
        diff_image = find_difference(img1, img2)
    """
    assert image1.shape == image2.shape, "Especifique 2 imagens com a mesma forma."

    gray_image1 = rgb2gray(image1)
    gray_image2 = rgb2gray(image2)

    score, difference_image = structural_similarity(gray_image1, gray_image2, full=True)
    print(f"Similaridade das imagens: {score}")

    normalized_difference_image = (difference_image - np.min(difference_image)) / (
            np.max(difference_image) - np.min(difference_image)
    )

    return normalized_difference_image


def transfer_histogram(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Transfere o histograma de uma imagem para outra.

    Esta função ajusta o histograma da primeira imagem para corresponder
    ao histograma da segunda imagem.

    Args:
        image1 (np.ndarray): Imagem fonte para transferência de histograma.
        image2 (np.ndarray): Imagem de referência para o histograma.

    Returns:
        np.ndarray: Imagem com o histograma ajustado.

    Example:
        matched_img = transfer_histogram(source_img, reference_img)
    """
    matched_image = match_histograms(image1, image2, multichannel=True)
    return matched_image
