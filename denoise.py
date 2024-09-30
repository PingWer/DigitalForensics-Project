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
    # Verifica se l'immagine è a colori (3 canali)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converti in scala di grigi

    # Passo 1: Applicazione della trasformata di Fourier
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)

    righe, colonne = img.shape
    centro_riga, centro_colonna = righe // 2, colonne // 2

    # Creazione di un filtro che blocca le frequenze lungo le linee centrali, ma esclude il centro
    filtro = np.ones((righe, colonne), np.uint8)

    # Se center_zone non è specificato, usa un valore predefinito
    if center_zone is None:
        center_zone = 10  # Dimensione della finestra di tolleranza

    # Assicurati che tutti i parametri siano interi
    soglia = int(soglia)
    center_zone = int(center_zone)
    filter_thickness = int(filter_thickness)

    # Calcola il raggio in base alla soglia
    raggio = soglia

    # Blocco delle frequenze lungo la linea verticale, escludendo il centro
    filtro[:, centro_colonna - filter_thickness:centro_colonna + filter_thickness + 1] = 0
    filtro[centro_riga - center_zone:centro_riga + center_zone + 1,
           centro_colonna - raggio:centro_colonna + raggio + 1] = 1  # Mantiene il centro intatto

    # Blocco delle frequenze lungo la linea orizzontale, escludendo il centro
    filtro[centro_riga - filter_thickness:centro_riga + filter_thickness + 1, :] = 0
    filtro[centro_riga - raggio:centro_riga + raggio + 1,
           centro_colonna - center_zone:centro_colonna + center_zone + 1] = 1  # Mantiene il centro intatto

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