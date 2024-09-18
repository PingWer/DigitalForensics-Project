import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def filter_frequencies(image, threshold_factor=2.0):

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = image.astype(float) / 255.0
    F = fft2(image)
    Fshift = fftshift(F)
    
    magnitude_spectrum = np.abs(Fshift)
    
    mean_magnitude = np.mean(magnitude_spectrum)
    std_magnitude = np.std(magnitude_spectrum)
    threshold = mean_magnitude + threshold_factor * std_magnitude
    
    mask = magnitude_spectrum <= threshold
    
    Fshift_filtered = Fshift * mask
    F_ishift = ifftshift(Fshift_filtered)
    img_back = ifft2(F_ishift)
    
    img_back = np.abs(img_back)
    img_back = np.clip(img_back * 255.0, 0, 255).astype(np.uint8)
    
    return img_back


def filter_frequencies_auto(image, threshold_factor=2.0):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = image.astype(float) / 255.0
    
    F = fft2(image)
    Fshift = fftshift(F)
    
    magnitude_spectrum = np.abs(Fshift)
    
    mean_magnitude = np.mean(magnitude_spectrum)
    std_magnitude = np.std(magnitude_spectrum)
    threshold = mean_magnitude + threshold_factor * std_magnitude
    
    mask = magnitude_spectrum <= threshold
    
    Fshift_filtered = Fshift * mask
    F_ishift = ifftshift(Fshift_filtered)
    img_back = ifft2(F_ishift)
    
    img_back = np.abs(img_back)
    img_back = np.clip(img_back * 255.0, 0, 255).astype(np.uint8)
    
    return img_back

def remove_periodic_noise (image, fraction):

    filtered_img = filter_frequencies(image, fraction)

    return filtered_img

def remove_auto_periodic_noise (image, fraction):

    filtered_img = filter_frequencies_auto(image, fraction)

    return filtered_img