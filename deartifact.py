import cv2

def apply_non_local_means(image, h=10, hColor=10):

    return cv2.fastNlMeansDenoisingColored(image, None, h=h, hColor=hColor, templateWindowSize=7, searchWindowSize=21)

def apply_median_filter(image, ksize=5):

    return cv2.medianBlur(image, ksize=ksize)

def apply_bilateral_filter(image, d=9, sigmaColor=75, sigmaSpace=75):

    return cv2.bilateralFilter(image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

def denoise_image(image, filter_type='non_local_means', **kwargs):

    if filter_type == 'non_local_means':
        return apply_non_local_means(image, **kwargs)
    elif filter_type == 'median':
        return apply_median_filter(image, **kwargs)
    elif filter_type == 'bilateral':
        return apply_bilateral_filter(image, **kwargs)
    else:
        raise ValueError(f"Filtro non riconosciuto: {filter_type}")
