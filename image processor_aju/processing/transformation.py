"""
Módulo de Redimensionamento de Imagem

Este módulo contém uma função para redimensionar imagens.

Dependências:
    - numpy
    - scikit-image

A função neste módulo opera em imagens representadas como arrays NumPy.
"""

import numpy as np
from skimage.transform import resize


def resize_image(image: np.ndarray, proportion: float) -> np.ndarray:
    """
    Redimensiona uma imagem de acordo com uma proporção especificada.

    Esta função redimensiona a imagem mantendo sua proporção original,
    usando anti-aliasing para melhor qualidade.

    Args:
        image (np.ndarray): Imagem a ser redimensionada.
        proportion (float): Fator de escala para o redimensionamento.
                            Deve ser um valor entre 0 e 1.

    Returns:
        np.ndarray: Imagem redimensionada.

    Raises:
        AssertionError: Se a proporção não estiver entre 0 e 1.

    Example:
        import numpy as np
        from skimage import data

        # Carrega uma imagem de exemplo
        original_image = data.astronaut()

        # Redimensiona a imagem para 50% do tamanho original
        resized_image = resize_image(original_image, 0.5)
    """
    assert 0 <= proportion <= 1, "Especifique uma proporção válida entre 0 e 1."

    height = round(image.shape[0] * proportion)
    width = round(image.shape[1] * proportion)

    image_resized = resize(image, (height, width), anti_aliasing=True)
    return image_resized
