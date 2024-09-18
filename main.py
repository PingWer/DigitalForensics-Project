import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import deartifact
import denoise
import compare
import logging
import os


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

        # Variabili per i parametri dei filtri
        self.h_var = tk.DoubleVar(value=10)
        self.hColor_var = tk.DoubleVar(value=10)
        self.ksize_var = tk.IntVar(value=5)
        self.d_var = tk.IntVar(value=9)
        self.sigmaColor_var = tk.DoubleVar(value=75)
        self.sigmaSpace_var = tk.DoubleVar(value=75)
        self.freq_threshold_var = tk.DoubleVar(value=10)
        self.rect_start = None
        self.rect_end = None
        self.selection_rect = None

        self.image_path = None
        self.original_image = None
        self.processed_image = None

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
            tk.messagebox.showwarning("Salvataggio Immagine", "Nessuna immagine processata da salvare.")
            return
        
        save_path = filedialog.asksaveasfilename(title="Salva Immagine", defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            processed_pil_image = Image.fromarray(self.processed_image)
            processed_pil_image.save(save_path)
            tk.messagebox.showinfo("Salvataggio Immagine", f"Immagine salvata come {save_path}")
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
            tk.Scale(self.param_frame, variable=self.freq_threshold_var, from_=1000, to_=0, orient=tk.HORIZONTAL, command=lambda x: self.update_image()).pack(fill=tk.X)
            
        elif filter_type == 'auto_periodic_noise':
            tk.Label(self.param_frame, text="Soglia di Frequenza:").pack(anchor=tk.W)
            tk.Scale(self.param_frame, variable=self.freq_threshold_var, from_=1000, to_=0, orient=tk.HORIZONTAL, command=lambda x: self.update_image()).pack(fill=tk.X)

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
        elif filter_type == 'auto_periodic_noise':
            self.processed_image = denoise.remove_auto_periodic_noise(image, fraction=self.freq_threshold_var.get())
            params = f"auto_freq_threshold: {self.freq_threshold_var.get()}"
        else:
            filter_params = self.get_filter_params()
            self.processed_image = deartifact.denoise_image(image, filter_type=filter_type, **filter_params)
            params = f"Params: {filter_params}"

        self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
        self.log_action(f"Immagine processata con filtro: {filter_type} | {params}")
        self.display_images()

    def get_filter_params(self):
        filter_type = self.filter_type.get()
        if filter_type == 'non_local_means':
            return {'h': self.h_var.get(), 'hColor': self.hColor_var.get()}
        elif filter_type == 'median':
            return {'ksize': self.ksize_var.get()}
        elif filter_type == 'bilateral':
            return {'d': self.d_var.get(), 'sigmaColor': self.sigmaColor_var.get(), 'sigmaSpace': self.sigmaSpace_var.get()}
        return {}

    def display_images(self):
        self.show_image(self.original_image, self.original_canvas, 'original')
        self.show_image(self.processed_image, self.processed_canvas, 'processed')

    def show_image(self, image, canvas, canvas_type):
        image_pil = Image.fromarray(image)
        canvas.image_pil = image_pil
        tk_image = ImageTk.PhotoImage(image_pil)
        canvas.tk_image = tk_image
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

        self.center_image(canvas)
        self.bind_canvas_events(canvas, canvas_type)

    def bind_canvas_events(self, canvas, canvas_type):
        canvas.bind("<B1-Motion>", lambda event, cnv=canvas: self.pan(event, cnv, canvas_type))
        canvas.bind("<ButtonRelease-1>", self.reset_pan)
        canvas.bind("<MouseWheel>", lambda event, cnv=canvas: self.zoom(event, cnv, canvas_type))

    def zoom(self, event, canvas):
        if event.delta > 0:
            scale_factor = 1.2
        elif event.delta < 0:
            scale_factor = 0.8

        new_size = (int(canvas.image_pil.width * scale_factor),
                    int(canvas.image_pil.height * scale_factor))
        resized_image = canvas.image_pil.resize(new_size, Image.LANCZOS)
        canvas.tk_image = ImageTk.PhotoImage(resized_image)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=canvas.tk_image)
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

        self.center_image(canvas)

    def pan(self, event, canvas):
        if self.last_x is not None and self.last_y is not None:
            delta_x = event.x - self.last_x
            delta_y = event.y - self.last_y
            canvas.move("all", delta_x, delta_y)
        self.last_x = event.x
        self.last_y = event.y

    def reset_pan(self):
        self.last_x = None
        self.last_y = None

    def center_image(self, canvas):
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        image_width = canvas.tk_image.width()
        image_height = canvas.tk_image.height()

        offset_x = max((canvas_width - image_width) // 2, 0)
        offset_y = max((canvas_height - image_height) // 2, 0)
        canvas.move("all", offset_x, offset_y)

    def select_area(self):
        self.original_canvas.bind("<Button-1>", self.on_button_press)
        self.original_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.original_canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def on_button_press(self, event):
        self.rect_start = (event.x, event.y)

    def on_mouse_drag(self, event):
        self.rect_end = (event.x, event.y)
        if self.selection_rect:
            self.original_canvas.delete(self.selection_rect)
        self.selection_rect = self.original_canvas.create_rectangle(
            self.rect_start[0], self.rect_start[1], event.x, event.y, outline='red'
        )

    def on_button_release(self, event):
        self.rect_end = (event.x, event.y)
        self.original_canvas.unbind("<Button-1>")
        self.original_canvas.unbind("<B1-Motion>")
        self.original_canvas.unbind("<ButtonRelease-1>")

    def compare_selection(self):
        if not (self.rect_start and self.rect_end):
            tk.messagebox.showwarning("Selezione Porzione", "Seleziona prima una porzione dell'immagine.")
            return

        canvas_coords = self.original_canvas.bbox(tk.ALL)
        if canvas_coords:
            canvas_x1, canvas_y1, canvas_x2, canvas_y2 = canvas_coords
            canvas_width = canvas_x2 - canvas_x1
            canvas_height = canvas_y2 - canvas_y1

            zoom_scale_x = canvas_width / self.original_image.shape[1]
            zoom_scale_y = canvas_height / self.original_image.shape[0]

            x1, y1 = self.rect_start
            x2, y2 = self.rect_end
            x1 = int((x1 - canvas_x1) / zoom_scale_x)
            x2 = int((x2 - canvas_x1) / zoom_scale_x)
            y1 = int((y1 - canvas_y1) / zoom_scale_y)
            y2 = int((y2 - canvas_y1) / zoom_scale_y)

            selected_portion = self.original_image[y1:y2, x1:x2]
            selected_coords = (x1, y1, x2 - x1, y2 - y1)

            if selected_portion.size == 0:
                tk.messagebox.showwarning("Selezione Porzione", "La selezione è vuota.")
                return
            
            selected_pil_image = Image.fromarray(selected_portion)
            selected_pil_image.save("selected_portion.png")
            selected_pil_image.show()

            match_count = compare.find_matches(self.original_image, selected_portion, selected_coords, similarity_threshold=0.997)

            self.log_action(f"Immagine comparata con coordinate: {selected_coords} con threshold: 0.997. Corrispondenze trovate: {match_count}")
            tk.messagebox.showinfo("Corrispondenze trovate", f"Numero di corrispondenze: {match_count}")
        else:
            tk.messagebox.showwarning("Errore", "Non è possibile ottenere le coordinate del canvas.")


if __name__ == "__main__":
    root = tk.Tk()
    app = DenoisingApp(root)
    root.mainloop()
