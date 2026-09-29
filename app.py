import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import os
from transformers import BlipProcessor, BlipForConditionalGeneration

# ──────────────────────────────────────────
# Schriftart-Konstanten (kursiv/Schreibschrift)
# ──────────────────────────────────────────
FONT_TITLE  = ("Georgia", 22, "italic")
FONT_BUTTON = ("Georgia", 18, "italic")
FONT_LABEL  = ("Georgia", 13, "italic")
FONT_SMALL  = ("Georgia", 11, "italic")
FONT_LOGO_TU = ("Arial", 10, "bold")

BG        = "white"
BORDER    = "black"
BTN_COLOR = "white"
FERTIG_BG = "#4ECDC4"   # türkis
FERTIG_FG = "white"

DATA_FILE = "fundbuero_data.json"

# ──────────────────────────────────────────
# Datenverwaltung
# ──────────────────────────────────────────
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ──────────────────────────────────────────
# KI-Modell laden (einmalig beim Start)
# ──────────────────────────────────────────
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model     = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base")
    return processor, model

def describe_image(image_path, processor, model):
    """Gibt eine automatische Bildbeschreibung zurück."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out    = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# ──────────────────────────────────────────
# Haupt-App-Klasse
# ──────────────────────────────────────────
class FundbueroApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fundbüro")
        self.geometry("390x700")
        self.resizable(False, False)
        self.configure(bg=BG)

        # Modell laden
        self.status_label = tk.Label(self, text="KI-Modell wird geladen …",
                                     font=FONT_SMALL, bg=BG)
        self.status_label.pack(pady=10)
        self.update()
        self.processor, self.model = load_model()
        self.status_label.destroy()

        self.data = load_data()
        self.current_image_path = None

        self.show_start()

    # ── Logo-Widget ──────────────────────────
    def make_logo(self, parent):
        """TU-ES Logo (oben rechts: Kompass-Symbol + TU/ES in Rot)"""
        frame = tk.Frame(parent, bg=BG)
        frame.place(relx=1.0, rely=0.0, anchor="ne", x=-8, y=8)

        # Kompass-Symbol (Unicode)
        tk.Label(frame, text="⊕", font=("Arial", 20), bg=BG,
                 fg="black").grid(row=0, column=0, rowspan=2)

        tk.Label(frame, text="TU", font=("Arial", 9, "bold"), bg=BG,
                 fg="red").grid(row=0, column=1, sticky="w")
        tk.Label(frame, text="ES", font=("Arial", 9, "bold"), bg=BG,
                 fg="red").grid(row=1, column=1, sticky="w")
        return frame

    # ── Hilfsfunktion: Box-Button ─────────────
    def box_button(self, parent, text, command, icon="", height=70):
        outer = tk.Frame(parent, bg=BORDER, bd=0)
        outer.pack(fill="x", padx=30, pady=12)

        inner = tk.Frame(outer, bg=BTN_COLOR, bd=0)
        inner.pack(padx=2, pady=2)

        row = tk.Frame(inner, bg=BTN_COLOR)
        row.pack(fill="x", padx=10, pady=0)

        tk.Label(row, text=text, font=FONT_BUTTON, bg=BTN_COLOR,
                 anchor="w").pack(side="left", expand=True, fill="x",
                                  ipady=height//4)
        if icon:
            tk.Label(row, text=icon, font=("Arial", 18), bg=BTN_COLOR
                     ).pack(side="right", padx=6)

        # Klick auf gesamten Bereich
        for widget in (outer, inner, row):
            widget.bind("<Button-1>", lambda e: command())

    # ──────────────────────────────────────────
    # STARTSEITE
    # ──────────────────────────────────────────
    def show_start(self):
        self._clear()
        container = tk.Frame(self, bg=BG)
        container.pack(fill="both", expand=True)

        self.make_logo(container)

        # Abstand oben
        tk.Label(container, text="", bg=BG).pack(pady=60)

        self.box_button(container, "Suchen",    self.show_search,   icon="🔍")
        self.box_button(container, "Hochladen", self.show_upload,   icon="⬆")

    # ──────────────────────────────────────────
    # HOCHLADESEITE
    # ──────────────────────────────────────────
    def show_upload(self):
        self._clear()
        container = tk.Frame(self, bg=BG)
        container.pack(fill="both", expand=True)

        self.make_logo(container)
        tk.Label(container, text="", bg=BG).pack(pady=20)

        # ── Foto hochladen ───────────────────
        foto_frame = tk.Frame(container, bg=BORDER)
        foto_frame.pack(fill="x", padx=30, pady=10)
        foto_inner = tk.Frame(foto_frame, bg=BTN_COLOR)
        foto_inner.pack(padx=2, pady=2)

        foto_row = tk.Frame(foto_inner, bg=BTN_COLOR)
        foto_row.pack(fill="x", padx=10)

        self.foto_label = tk.Label(foto_row, text="Foto hochladen",
                                   font=FONT_BUTTON, bg=BTN_COLOR, anchor="w")
        self.foto_label.pack(side="left", expand=True, fill="x", ipady=18)

        tk.Label(foto_row, text="⬆", font=("Arial", 18),
                 bg=BTN_COLOR).pack(side="right", padx=6)

        for w in (foto_frame, foto_inner, foto_row, self.foto_label):
            w.bind("<Button-1>", lambda e: self._choose_photo())

        # ── Name des Objekts ─────────────────
        name_frame = tk.Frame(container, bg=BORDER)
        name_frame.pack(fill="x", padx=30, pady=10)
        name_inner = tk.Frame(name_frame, bg=BTN_COLOR)
        name_inner.pack(padx=2, pady=2)

        self.name_entry = tk.Entry(name_inner, font=FONT_BUTTON,
                                   bg=BTN_COLOR, relief="flat",
                                   justify="left")
        self.name_entry.insert(0, "Name des Objekts")
        self.name_entry.pack(fill="x", padx=10, ipady=18)
        self.name_entry.bind("<FocusIn>",
            lambda e: self._clear_placeholder(self.name_entry, "Name des Objekts"))
        self.name_entry.bind("<FocusOut>",
            lambda e: self._set_placeholder(self.name_entry, "Name des Objekts"))

        # ── Beschreibung ─────────────────────
        desc_frame = tk.Frame(container, bg=BORDER)
        desc_frame.pack(fill="x", padx=30, pady=10)
        desc_inner = tk.Frame(desc_frame, bg=BTN_COLOR)
        desc_inner.pack(padx=2, pady=2)

        self.desc_entry = tk.Entry(desc_inner, font=FONT_BUTTON,
                                   bg=BTN_COLOR, relief="flat")
        self.desc_entry.insert(0, "Beschreibung")
        self.desc_entry.pack(fill="x", padx=10, ipady=18)
        self.desc_entry.bind("<FocusIn>",
            lambda e: self._clear_placeholder(self.desc_entry, "Beschreibung"))
        self.desc_entry.bind("<FocusOut>",
            lambda e: self._set_placeholder(self.desc_entry, "Beschreibung"))

        # ── Fertig-Button (türkis) ───────────
        fertig_btn = tk.Button(container, text="Fertig",
                               font=FONT_LABEL,
                               bg=FERTIG_BG, fg=FERTIG_FG,
                               relief="flat", bd=0,
                               activebackground="#3ab5ac",
                               command=self._save_item)
        fertig_btn.pack(pady=16, ipadx=30, ipady=6)

        # ── Zurück ───────────────────────────
        tk.Button(container, text="← Zurück", font=FONT_SMALL,
                  bg=BG, relief="flat", command=self.show_start
                  ).pack(pady=4)

    def _choose_photo(self):
        path = filedialog.askopenfilename(
            filetypes=[("Bilder", "*.jpg *.jpeg *.png *.webp")])
        if path:
            self.current_image_path = path
            self.foto_label.config(
                text=f"✔ {os.path.basename(path)}")

            # KI-Beschreibung automatisch einfügen
            ai_desc = describe_image(path, self.processor, self.model)
            self._clear_placeholder(self.desc_entry, "Beschreibung")
            self.desc_entry.delete(0, tk.END)
            self.desc_entry.insert(0, ai_desc)

    def _save_item(self):
        name = self.name_entry.get().strip()
        desc = self.desc_entry.get().strip()

        if name in ("", "Name des Objekts"):
            messagebox.showwarning("Fehler", "Bitte einen Namen eingeben.")
            return

        entry = {
            "name": name,
            "description": desc if desc != "Beschreibung" else "",
            "image": self.current_image_path or ""
        }
        self.data.append(entry)
        save_data(self.data)
        messagebox.showinfo("Gespeichert", f"„{name}" wurde gespeichert!")
        self.current_image_path = None
        self.show_start()

    # ──────────────────────────────────────────
    # SUCHSEITE
    # ──────────────────────────────────────────
    def show_search(self):
        self._clear()
        container = tk.Frame(self, bg=BG)
        container.pack(fill="both", expand=True)

        # ── Suchleiste ───────────────────────
        search_outer = tk.Frame(container, bg=BORDER)
        search_outer.pack(fill="x", padx=20, pady=(12, 6))
        search_inner = tk.Frame(search_outer, bg="white")
        search_inner.pack(padx=2, pady=2)

        search_row = tk.Frame(search_inner, bg="white")
        search_row.pack(fill="x", padx=6)

        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *a: self._refresh_results(
            results_frame))

        search_entry = tk.Entry(search_row, textvariable=self.search_var,
                                font=FONT_LABEL, bg="white",
                                relief="flat")
        search_entry.insert(0, "Suchen")
        search_entry.pack(side="left", fill="x", expand=True, ipady=8)
        search_entry.bind("<FocusIn>",
            lambda e: self._clear_placeholder(search_entry, "Suchen"))
        search_entry.bind("<FocusOut>",
            lambda e: self._set_placeholder(search_entry, "Suchen"))

        tk.Label(search_row, text="🔍", font=("Arial", 14),
                 bg="white").pack(side="right", padx=4)

        # ── Ergebnis-Liste (scrollbar) ───────
        canvas = tk.Canvas(container, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical",
                                  command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        results_frame = tk.Frame(canvas, bg=BG)
        canvas_window = canvas.create_window((0, 0), window=results_frame,
                                              anchor="nw")

        results_frame.bind("<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
            lambda e: canvas.itemconfig(canvas_window, width=e.width))

        self._refresh_results(results_frame)

        # ── Zurück ───────────────────────────
        tk.Button(container, text="← Zurück", font=FONT_SMALL,
                  bg=BG, relief="flat", command=self.show_start
                  ).pack(side="bottom", pady=6)

    def _refresh_results(self, frame):
        for w in frame.winfo_children():
            w.destroy()

        query = self.search_var.get().strip().lower() \
                if hasattr(self, "search_var") else ""

        for item in self.data:
            if query in ("", "suchen") or \
               query in item["name"].lower() or \
               query in item["description"].lower():
                self._make_card(frame, item)

    def _make_card(self, parent, item):
        """Eine Karte mit Bild links, Name + Beschreibung rechts."""
        card_outer = tk.Frame(parent, bg=BORDER)
        card_outer.pack(fill="x", padx=14, pady=6)
        card_inner = tk.Frame(card_outer, bg=BTN_COLOR)
        card_inner.pack(padx=2, pady=2)

        # Bild links
        img_label = tk.Label(card_inner, bg=BTN_COLOR, width=6)
        img_label.pack(side="left", padx=8, pady=8)

        if item.get("image") and os.path.exists(item["image"]):
            try:
                img = Image.open(item["image"]).resize((60, 60))
                photo = ImageTk.PhotoImage(img)
                img_label.config(image=photo, width=60, height=60)
                img_label.image = photo   # Referenz halten!
            except Exception:
                img_label.config(text="🖼", font=("Arial", 24))
        else:
            img_label.config(text="🖼", font=("Arial", 24))

        # Text rechts
        text_frame = tk.Frame(card_inner, bg=BTN_COLOR)
        text_frame.pack(side="left", fill="both", expand=True,
                        padx=4, pady=6)

        # Name unterstrichen + kursiv
        tk.Label(text_frame,
                 text=item["name"],
                 font=("Georgia", 14, "italic", "underline"),
                 bg=BTN_COLOR, anchor="w"
                 ).pack(fill="x")

        tk.Label(text_frame,
                 text="Beschreibung:",
                 font=FONT_SMALL, bg=BTN_COLOR, anchor="w"
                 ).pack(fill="x")

        for line in item["description"].split(","):
            tk.Label(text_frame,
                     text=f"- {line.strip()}",
                     font=FONT_SMALL, bg=BTN_COLOR,
                     anchor="w", wraplength=210, justify="left"
                     ).pack(fill="x")

    # ──────────────────────────────────────────
    # Hilfsfunktionen
    # ──────────────────────────────────────────
    def _clear(self):
        for w in self.winfo_children():
            w.destroy()

    def _clear_placeholder(self, entry, placeholder):
        if entry.get() == placeholder:
            entry.delete(0, tk.END)

    def _set_placeholder(self, entry, placeholder):
        if entry.get().strip() == "":
            entry.insert(0, placeholder)

# ──────────────────────────────────────────
if __name__ == "__main__":
    app = FundbueroApp()
    app.mainloop()
