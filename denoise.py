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


def filter_frequencies_auto(img, soglia, center_zone=None, filter_thickness=0):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    #Fourier
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)

    righe, colonne = img.shape
    centro_riga, centro_colonna = righe // 2, colonne // 2

    # Filtro geometrico in frequenza 
    filtro = np.ones((righe, colonne), np.uint8)

    if center_zone is None:
        center_zone = 10 

    # Usiamo parametri interi per semplificare i calcoli
    soglia = int(soglia)
    center_zone = int(center_zone)
    filter_thickness = int(filter_thickness)

    raggio = soglia

    # Rimozione della frequenza in verticale
    filtro[:, centro_colonna - filter_thickness:centro_colonna + filter_thickness + 1] = 0
    filtro[centro_riga - center_zone:centro_riga + center_zone + 1,
           centro_colonna - raggio:centro_colonna + raggio + 1] = 1

    # Rimozione della frequenza in verticale
    filtro[centro_riga - filter_thickness:centro_riga + filter_thickness + 1, :] = 0
    filtro[centro_riga - raggio:centro_riga + raggio + 1,
           centro_colonna - center_zone:centro_colonna + center_zone + 1] = 1

    # Applicazione del filtro notch allo spettro di Fourier
    dft_shift_filtrato = dft_shift * filtro

    # Inversa della trasformata di Fourier
    dft_inv_shift = np.fft.ifftshift(dft_shift_filtrato)
    img_filtrata = np.fft.ifft2(dft_inv_shift)
    img_filtrata = np.abs(img_filtrata)
    
    return img_filtrata

def remove_periodic_noise (image, fraction):

    filtered_img = filter_frequencies(image, fraction)

    return filtered_img

def remove_auto_periodic_noise (img, soglia, center_zone=None, filter_thickness=0):

    filtered_img = filter_frequencies_auto(img, soglia, center_zone, filter_thickness)

    return filtered_img