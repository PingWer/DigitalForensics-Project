import cv2
import numpy as np

def apply_non_local_means(image, h=10, hColor=10):
    """
    Applica il filtro Non-Local Means all'immagine con parametri specificati.

    :param image: Immagine di input.
    :param h: Parametro per il denoising.
    :param hColor: Parametro per il denoising del colore.
    :return: Immagine denoised.
    """
    return cv2.fastNlMeansDenoisingColored(image, None, h=h, hColor=hColor, templateWindowSize=7, searchWindowSize=21)

def apply_median_filter(image, ksize=5):
    """
    Applica il filtro mediano all'immagine con dimensione del kernel specificata.

    :param image: Immagine di input.
    :param ksize: Dimensione del kernel (deve essere un numero dispari).
    :return: Immagine denoised.
    """
    return cv2.medianBlur(image, ksize=ksize)

def apply_bilateral_filter(image, d=9, sigmaColor=75, sigmaSpace=75):
    """
    Applica il filtro bilaterale all'immagine con parametri specificati.

    :param image: Immagine di input.
    :param d: Diametro del pixel vicino.
    :param sigmaColor: Sigma per il filtro colore.
    :param sigmaSpace: Sigma per la distanza spaziale.
    :return: Immagine denoised.
    """
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

def denoise_image(image, filter_type='non_local_means', **kwargs):
    """
    Applica il filtro specificato all'immagine con parametri variabili.

    :param image: Immagine di input.
    :param filter_type: Tipo di filtro da applicare ('non_local_means', 'median', 'bilateral').
    :param kwargs: Parametri del filtro.
    :return: Immagine processata.
    """
    if filter_type == 'non_local_means':
        return apply_non_local_means(image, **kwargs)
    elif filter_type == 'median':
        return apply_median_filter(image, **kwargs)
    elif filter_type == 'bilateral':
        return apply_bilateral_filter(image, **kwargs)
    else:
        raise ValueError(f"Filtro non riconosciuto: {filter_type}")
