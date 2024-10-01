import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import denoise 
import logging
import os
import numpy as np

# Logging
log_file = os.path.join(os.path.dirname(__file__), "app_log.txt")
logging.basicConfig(
    filename=log_file, 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class DenoisingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Filtro di Denoising")

        self.h_var = tk.DoubleVar(value=10)
        self.hColor_var = tk.DoubleVar(value=10)
        self.ksize_var = tk.IntVar(value=5)
        self.d_var = tk.IntVar(value=9)
        self.sigmaColor_var = tk.DoubleVar(value=75)
        self.sigmaSpace_var = tk.DoubleVar(value=75)
        self.freq_threshold_var = tk.DoubleVar(value=10)
        self.center_zone_var = tk.IntVar(value=10) 
        self.filter_thickness_var = tk.IntVar(value=5)  
        
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.rect_start = None
        self.rect_end = None
        self.selection_rect = None

        # Layout
        self.create_widgets()

    def log_action(self, action):
        logging.info(action)

    def create_widgets(self):
        tk.Label(self.root, text="Scegli un filtro per denoising:").pack(pady=10)

        # Opzioni per il tipo di filtro
        self.filter_type = tk.StringVar(value='non_local_means')
        tk.Radiobutton(self.root, text="Non-Local Means", variable=self.filter_type, value='non_local_means', command=self.update_ui).pack(anchor=tk.W)
        tk.Radiobutton(self.root, text="Filtro Mediano", variable=self.filter_type, value='median', command=self.update_ui).pack(anchor=tk.W)
        tk.Radiobutton(self.root, text="Filtro Bilaterale", variable=self.filter_type, value='bilateral', command=self.update_ui).pack(anchor=tk.W)
        tk.Radiobutton(self.root, text="Rimozione Rumore Periodico", variable=self.filter_type, value='periodic_noise', command=self.update_ui).pack(anchor=tk.W)
        tk.Radiobutton(self.root, text="Rimozione Rumore Periodico Automatico", variable=self.filter_type, value='auto_periodic_noise', command=self.update_ui).pack(anchor=tk.W)

        tk.Button(self.root, text="Carica Immagine", command=self.load_image).pack(pady=10)
        tk.Button(self.root, text="Salva Immagine", command=self.save_image).pack(pady=10)
        tk.Button(self.root, text="Seleziona Porzione", command=self.select_area).pack(pady=10)
        tk.Button(self.root, text="Confronta Porzione", command=self.compare_selection).pack(pady=10)

        self.param_frame = tk.Frame(self.root)
        self.param_frame.pack(pady=10)

        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Creazione dei due canvas per le immagini
        self.original_canvas = tk.Canvas(self.image_frame, bg='gray', width=400, height=600)
        self.original_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.processed_canvas = tk.Canvas(self.image_frame, bg='gray', width=400, height=600)
        self.processed_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.update_ui()

    def load_image(self):
        self.image_path = filedialog.askopenfilename(title="Seleziona l'immagine di input", filetypes=[("Immagini", "*.tiff; *.tif; *.jpg; *.png; *.jpeg; *.webp")])
        if self.image_path:
            self.update_image()
            self.log_action(f"Immagine caricata: {self.image_path}")

    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("Salvataggio Immagine", "Nessuna immagine processata da salvare.")
            return
        
        save_path = filedialog.asksaveasfilename(title="Salva Immagine", defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            processed_pil_image = Image.fromarray(self.processed_image)
            processed_pil_image.save(save_path)
            messagebox.showinfo("Salvataggio Immagine", f"Immagine salvata come {save_path}")
            self.log_action(f"Immagine salvata: {save_path}")

    def update_ui(self):
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        filter_type = self.filter_type.get()

        if filter_type == 'non_local_means':
            tk.Label(self.param_frame, text="h:").pack(anchor=tk.W)
            tk.Scale(self.param_frame, variable=self.h_var, from_=0, to_=100, orient=tk.HORIZONTAL, command=lambda x: self.update_image()).pack(fill=tk.X)
            tk.Label(self.param_frame, text="hColor:").pack(anchor=tk.W)
            tk.Scale(self.param_frame, variable=self.hColor_var, from_=0, to_=100, orient=tk.HORIZONTAL, command=lambda x: self.update_image()).pack(fill=tk.X)

        elif filter_type == 'median':
            tk.Label(self.param_frame, text="ksize:").pack(anchor=tk.W)
            tk.Scale(self.param_frame, variable=self.ksize_var, from_=3, to_=21, orient=tk.HORIZONTAL, tickinterval=2, command=lambda x: self.update_image()).pack(fill=tk.X)

        elif filter_type == 'bilateral':
            tk.Label(self.param_frame, text="d:").pack(anchor=tk.W)
            tk.Scale(self.param_frame, variable=self.d_var, from_=1, to_=20, orient=tk.HORIZONTAL, command=lambda x: self.update_image()).pack(fill=tk.X)
            tk.Label(self.param_frame, text="sigmaColor:").pack(anchor=tk.W)
            tk.Scale(self.param_frame, variable=self.sigmaColor_var, from_=0, to_=200, orient=tk.HORIZONTAL, command=lambda x: self.update_image()).pack(fill=tk.X)
            tk.Label(self.param_frame, text="sigmaSpace:").pack(anchor=tk.W)
            tk.Scale(self.param_frame, variable=self.sigmaSpace_var, from_=0, to_=200, orient=tk.HORIZONTAL, command=lambda x: self.update_image()).pack(fill=tk.X)

        elif filter_type == 'periodic_noise':
            tk.Label(self.param_frame, text="Soglia di Frequenza:").pack(anchor=tk.W)
            tk.Scale(self.param_frame, variable=self.freq_threshold_var, from_=1000, to=0, orient=tk.HORIZONTAL, command=lambda x: self.update_image()).pack(fill=tk.X)

        elif filter_type == 'auto_periodic_noise':
            tk.Label(self.param_frame, text="Soglia di Frequenza:").pack(anchor=tk.W)
            tk.Scale(self.param_frame, variable=self.freq_threshold_var, from_=1000, to=0, orient=tk.HORIZONTAL, command=lambda x: self.update_image()).pack(fill=tk.X)
            tk.Label(self.param_frame, text="Zona Centrale da Evitare (%):").pack(anchor=tk.W)
            tk.Scale(self.param_frame, variable=self.center_zone_var, from_=0, to=50, orient=tk.HORIZONTAL, command=lambda x: self.update_image()).pack(fill=tk.X)
            tk.Label(self.param_frame, text="Spessore del Filtro:").pack(anchor=tk.W)
            tk.Scale(self.param_frame, variable=self.filter_thickness_var, from_=1, to=20, orient=tk.HORIZONTAL, command=lambda x: self.update_image()).pack(fill=tk.X)

    def update_image(self):
        if not self.image_path:
            return

        filter_type = self.filter_type.get()
        image = cv2.imread(self.image_path)
        if image is None:
            return

        self.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if filter_type == 'periodic_noise':
            self.processed_image = denoise.remove_periodic_noise(image, fraction=self.freq_threshold_var.get())
            params = f"freq_threshold: {self.freq_threshold_var.get()}"
            self.log_action(f"Rimozione rumore periodico applicata con parametri: {params}")

        elif filter_type == 'auto_periodic_noise':
            filter_thickness = self.filter_thickness_var.get()
            self.processed_image = denoise.remove_auto_periodic_noise(image, soglia=self.freq_threshold_var.get(), center_zone=self.center_zone_var.get(), filter_thickness=filter_thickness)
            params = f"freq_threshold: {self.freq_threshold_var.get()}, filter_thickness: {filter_thickness}"
            self.log_action(f"Rimozione rumore periodico automatica applicata con parametri: {params}")

        elif filter_type == 'non_local_means':
            h = self.h_var.get()
            hColor = self.hColor_var.get()
            self.processed_image = cv2.fastNlMeansDenoisingColored(image, None, h, hColor, 7, 21)
            params = f"h: {h}, hColor: {hColor}"
            self.log_action(f"Filtro Non-Local Means applicato con parametri: {params}")

        elif filter_type == 'median':
            ksize = self.ksize_var.get()
            self.processed_image = cv2.medianBlur(image, ksize)
            params = f"ksize: {ksize}"
            self.log_action(f"Filtro Mediano applicato con parametri: {params}")

        elif filter_type == 'bilateral':
            d = self.d_var.get()
            sigmaColor = self.sigmaColor_var.get()
            sigmaSpace = self.sigmaSpace_var.get()
            self.processed_image = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
            params = f"d: {d}, sigmaColor: {sigmaColor}, sigmaSpace: {sigmaSpace}"
            self.log_action(f"Filtro Bilaterale applicato con parametri: {params}")

        else:
            self.processed_image = self.original_image 
        
        self.display_images()

    def display_images(self):
        original_image_pil = Image.fromarray(self.original_image)
        processed_image_pil = Image.fromarray(self.processed_image)

        original_image_tk = ImageTk.PhotoImage(original_image_pil)
        processed_image_tk = ImageTk.PhotoImage(processed_image_pil)

        self.original_canvas.create_image(0, 0, anchor=tk.NW, image=original_image_tk)
        self.original_canvas.image = original_image_tk 

        self.processed_canvas.create_image(0, 0, anchor=tk.NW, image=processed_image_tk)
        self.processed_canvas.image = processed_image_tk 

    def select_area(self):
        self.rect_start = None
        self.rect_end = None
        self.selection_rect = None

        self.original_canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.original_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.original_canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def on_button_press(self, event):
        self.rect_start = (event.x, event.y)
        self.selection_rect = self.original_canvas.create_rectangle(self.rect_start[0], self.rect_start[1], self.rect_start[0], self.rect_start[1], outline="red")

    def on_mouse_drag(self, event):
        if self.selection_rect:
            self.original_canvas.coords(self.selection_rect, self.rect_start[0], self.rect_start[1], event.x, event.y)

    def on_button_release(self, event):
        self.rect_end = (event.x, event.y)
        self.original_canvas.unbind("<ButtonPress-1>")
        self.original_canvas.unbind("<B1-Motion>")
        self.original_canvas.unbind("<ButtonRelease-1>")

    def compare_selection(self):
        if not self.rect_start or not self.rect_end:
            messagebox.showwarning("Confronto", "Nessuna area selezionata per il confronto.")
            return
        
        x1, y1 = min(self.rect_start[0], self.rect_end[0]), min(self.rect_start[1], self.rect_end[1])
        x2, y2 = max(self.rect_start[0], self.rect_end[0]), max(self.rect_start[1], self.rect_end[1])

        selected_area = self.original_image[y1:y2, x1:x2]
        
        selected_area_pil = Image.fromarray(selected_area)
        selected_area_tk = ImageTk.PhotoImage(selected_area_pil)

        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("Area Selezionata")
        comparison_canvas = tk.Canvas(comparison_window, width=selected_area.shape[1], height=selected_area.shape[0])
        comparison_canvas.pack()
        comparison_canvas.create_image(0, 0, anchor=tk.NW, image=selected_area_tk)
        comparison_canvas.image = selected_area_tk

        self.log_action("Confronto effettuato su area selezionata.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DenoisingApp(root)
    root.mainloop()
