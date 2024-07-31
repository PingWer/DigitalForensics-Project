import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
from denoise import denoise_image

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
        self.filter_type = tk.StringVar(value='non_local_means')
        self.image_path = None
        self.original_image = None
        self.denoised_image = None

        # Variabili per lo zoom e pan
        self.zoom_scale_original = 1.0
        self.zoom_scale_denoised = 1.0
        self.last_x = None
        self.last_y = None

        # Layout
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Scegli un filtro per denoising:").pack(pady=10)

        tk.Radiobutton(self.root, text="Non-Local Means", variable=self.filter_type, value='non_local_means', command=self.update_ui).pack(anchor=tk.W)
        tk.Radiobutton(self.root, text="Filtro Mediano", variable=self.filter_type, value='median', command=self.update_ui).pack(anchor=tk.W)
        tk.Radiobutton(self.root, text="Filtro Bilaterale", variable=self.filter_type, value='bilateral', command=self.update_ui).pack(anchor=tk.W)

        tk.Button(self.root, text="Carica Immagine", command=self.load_image).pack(pady=10)
        tk.Button(self.root, text="Salva Immagine", command=self.save_image).pack(pady=10)

        self.param_frame = tk.Frame(self.root)
        self.param_frame.pack(pady=10)

        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Creazione dei due canvas per le immagini
        self.original_canvas = tk.Canvas(self.image_frame, bg='gray', width=400, height=600)
        self.original_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.denoised_canvas = tk.Canvas(self.image_frame, bg='gray', width=400, height=600)
        self.denoised_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.update_ui()

    def load_image(self):
        self.image_path = filedialog.askopenfilename(title="Seleziona l'immagine di input", filetypes=[("Immagini", "*.jpg;*.png")])
        if self.image_path:
            self.update_image()

    def save_image(self):
        if self.denoised_image is None:
            tk.messagebox.showwarning("Salvataggio Immagine", "Nessuna immagine denoised da salvare.")
            return
        
        save_path = filedialog.asksaveasfilename(title="Salva Immagine", defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            denoised_pil_image = Image.fromarray(self.denoised_image)
            denoised_pil_image.save(save_path)
            tk.messagebox.showinfo("Salvataggio Immagine", f"Immagine salvata come {save_path}")

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

    def update_image(self):
        if not self.image_path:
            return

        params = {}
        filter_type = self.filter_type.get()

        if filter_type == 'non_local_means':
            params = {'h': self.h_var.get(), 'hColor': self.hColor_var.get()}
        elif filter_type == 'median':
            params = {'ksize': int(self.ksize_var.get())}
        elif filter_type == 'bilateral':
            params = {'d': int(self.d_var.get()), 'sigmaColor': self.sigmaColor_var.get(), 'sigmaSpace': self.sigmaSpace_var.get()}

        image = cv2.imread(self.image_path)
        if image is None:
            return

        self.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.denoised_image = denoise_image(image, filter_type=filter_type, **params)
        self.denoised_image = cv2.cvtColor(self.denoised_image, cv2.COLOR_BGR2RGB)

        self.zoom_scale_original = 1.0
        self.zoom_scale_denoised = 1.0

        self.display_images()

    def display_images(self):
        # Mostra immagine originale nel canvas di sinistra
        self.show_image(self.original_image, self.original_canvas, 'original')
        
        # Mostra immagine denoised nel canvas di destra
        self.show_image(self.denoised_image, self.denoised_canvas, 'denoised')

    def show_image(self, image, canvas, canvas_type):
        image_pil = Image.fromarray(image)
        canvas.image_pil = image_pil  # Salva l'immagine PIL per lo zoom
        tk_image = ImageTk.PhotoImage(image_pil)
        canvas.tk_image = tk_image  # Salva l'immagine Tkinter per evitare che venga eliminata dal garbage collector
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

        # Centra l'immagine
        self.center_image(canvas)

        # Associa il pan e lo zoom ai canvas
        self.bind_canvas_events(canvas, canvas_type)

    def bind_canvas_events(self, canvas, canvas_type):
        canvas.bind("<B1-Motion>", lambda event, cnv=canvas: self.pan(event, cnv, canvas_type))
        canvas.bind("<ButtonRelease-1>", self.reset_pan)
        canvas.bind("<MouseWheel>", lambda event, cnv=canvas: self.zoom(event, cnv, canvas_type))

    def zoom(self, event, canvas, canvas_type):
        # Zoom in
        if event.delta > 0:
            scale_factor = 1.2
        # Zoom out
        elif event.delta < 0:
            scale_factor = 0.8

        # Aggiorna la scala dello zoom
        if canvas_type == 'original':
            self.zoom_scale_original *= scale_factor
        elif canvas_type == 'denoised':
            self.zoom_scale_denoised *= scale_factor

        # Ridimensiona l'immagine
        new_size = (int(canvas.image_pil.width * (self.zoom_scale_original if canvas_type == 'original' else self.zoom_scale_denoised)),
                    int(canvas.image_pil.height * (self.zoom_scale_original if canvas_type == 'original' else self.zoom_scale_denoised)))
        resized_image = canvas.image_pil.resize(new_size, Image.LANCZOS)
        canvas.tk_image = ImageTk.PhotoImage(resized_image)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=canvas.tk_image)
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

        # Centra l'immagine dopo il ridimensionamento
        self.center_image(canvas)

    def pan(self, event, canvas, canvas_type):
        if self.last_x is not None and self.last_y is not None:
            delta_x = event.x - self.last_x
            delta_y = event.y - self.last_y
            canvas.move("all", delta_x, delta_y)
        self.last_x = event.x
        self.last_y = event.y

    def reset_pan(self, event):
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

if __name__ == "__main__":
    root = tk.Tk()
    app = DenoisingApp(root)
    root.mainloop()
