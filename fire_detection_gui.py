import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO11_PATH = os.path.join(BASE_DIR, "runs", "classify", "runs", "classification", "yolo11_fire", "weights", "best.pt")
YOLOV8_PATH = os.path.join(BASE_DIR, "runs", "classify", "runs", "classification", "yolov8_fire", "weights", "best.pt")
THRESHOLD = 0.80

# ---------- Theme ----------
BG = "#EAF7FF"          # soft sky blue
CARD = "#FFFFFF"
SKY = "#87CEEB"         # sky blue
SKY_DARK = "#4EA5D9"
PINK = "#F7B6D2"        # baby pink
PINK_DARK = "#E88FB4"
TEXT = "#263238"
MUTED = "#607D8B"
SAFE_BG = "#E8F8EE"
SAFE_FG = "#207A47"
WARN_BG = "#FFF4D6"
WARN_FG = "#B26A00"
CRIT_BG = "#FFE3EC"
CRIT_FG = "#B4235A"

class FireDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Fire Detection & Emergency Alert System")
        self.root.geometry("1050x720")
        self.root.minsize(900, 650)
        self.root.configure(bg=BG)

        self.preview_photo = None
        self.yolo11 = None
        self.yolov8 = None

        self.load_models()
        self.build_ui()

    def load_models(self):
        try:
            if not os.path.exists(YOLO11_PATH):
                raise FileNotFoundError(f"YOLO11 model not found:\n{YOLO11_PATH}")
            if not os.path.exists(YOLOV8_PATH):
                raise FileNotFoundError(f"YOLOv8 model not found:\n{YOLOV8_PATH}")
            self.yolo11 = YOLO(YOLO11_PATH)
            self.yolov8 = YOLO(YOLOV8_PATH)
        except Exception as exc:
            messagebox.showerror("Model Error", str(exc))
            self.root.destroy()

    def build_ui(self):
        # Header
        header = tk.Frame(self.root, bg=PINK, height=92)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text=" AI FIRE DETECTION & EMERGENCY ALERT SYSTEM",
            bg=PINK, fg=TEXT, font=("Segoe UI", 21, "bold")
        ).pack(pady=(14, 2))

        tk.Label(
            header,
            text="YOLO11 + YOLOv8 + Explainable AI Decision Agent",
            bg=PINK, fg="#4B5563", font=("Segoe UI", 10)
        ).pack()

        # Main
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=18, pady=15)

        left = tk.Frame(main, bg=CARD, bd=1, relief="solid",
                        highlightbackground="#CFEAF7", highlightcolor="#CFEAF7", highlightthickness=1)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        right = tk.Frame(main, bg=CARD, bd=1, relief="solid",
                         highlightbackground="#F3D2E0", highlightcolor="#F3D2E0", highlightthickness=1,
                         width=420)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(left, text=" Image Preview", bg=CARD, fg=TEXT,
                 font=("Segoe UI", 15, "bold")).pack(pady=(15, 8))

        self.image_label = tk.Label(
            left, text="Select an image to begin",
            bg="#F7FCFF", fg=MUTED, font=("Segoe UI", 14),
            relief="flat"
        )
        self.image_label.pack(fill="both", expand=True, padx=18, pady=10)

        buttons = tk.Frame(left, bg=CARD)
        buttons.pack(pady=15)

        tk.Button(
            buttons, text=" Select Image", command=self.select_image,
            bg=SKY, fg=TEXT, activebackground=SKY_DARK,
            activeforeground="white", font=("Segoe UI", 11, "bold"),
            padx=18, pady=8, relief="flat", cursor="hand2"
        ).pack(side="left", padx=6)

        tk.Button(
    buttons, text=" Clear", command=self.clear,
    bg=SKY, fg=TEXT,
    activebackground=SKY_DARK,
    activeforeground="white",
    font=("Segoe UI", 11, "bold"),
    padx=18, pady=8,
    relief="flat",
    cursor="hand2"
).pack(side="left", padx=6)

        # Right panel
        tk.Label(
            right, text=" AI Analysis", bg=CARD, fg=TEXT,
            font=("Segoe UI", 16, "bold")
        ).pack(anchor="w", padx=20, pady=(18, 12))

        self.model_status = tk.StringVar(value="Models loaded")
        tk.Label(
            right, textvariable=self.model_status, bg=CARD, fg=SAFE_FG,
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", padx=20)

        result_card = tk.Frame(
            right, bg="#F8FCFF", bd=1, relief="solid",
            highlightbackground="#D8EDF7", highlightthickness=1
        )
        result_card.pack(fill="x", padx=20, pady=15)

        self.y11_result = tk.StringVar(value="YOLO11: --")
        self.y11_conf = tk.StringVar(value="Fire probability: --")
        self.v8_result = tk.StringVar(value="YOLOv8: --")
        self.v8_conf = tk.StringVar(value="Fire probability: --")

        tk.Label(result_card, textvariable=self.y11_result, bg="#F8FCFF",
                 fg=TEXT, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=15, pady=(12, 2))
        tk.Label(result_card, textvariable=self.y11_conf, bg="#F8FCFF",
                 fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", padx=15, pady=(0, 10))
        tk.Label(result_card, textvariable=self.v8_result, bg="#F8FCFF",
                 fg=TEXT, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=15, pady=(2, 2))
        tk.Label(result_card, textvariable=self.v8_conf, bg="#F8FCFF",
                 fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", padx=15, pady=(0, 12))

        tk.Label(
            right, text="AI Agent Decision", bg=CARD, fg=TEXT,
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="w", padx=20, pady=(5, 6))

        self.decision_label = tk.Label(
            right, text="WAITING", bg="#EDF2F7", fg=TEXT,
            font=("Segoe UI", 18, "bold"), height=2
        )
        self.decision_label.pack(fill="x", padx=20)

        self.action_text = tk.StringVar(value="Select an image to start detection.")
        tk.Label(
            right, textvariable=self.action_text, bg=CARD, fg="#4B5563",
            justify="left", wraplength=360, font=("Segoe UI", 10)
        ).pack(anchor="w", padx=20, pady=12)

        self.time_text = tk.StringVar(value="Detection time: --")
        tk.Label(
            right, textvariable=self.time_text, bg=CARD, fg=MUTED,
            font=("Segoe UI", 9)
        ).pack(anchor="w", padx=20, pady=(0, 10))

        threshold_card = tk.Frame(right, bg="#EEF9FF")
        threshold_card.pack(fill="x", padx=20, pady=5)
        tk.Label(
            threshold_card, text="Emergency threshold: 80%",
            bg="#EEF9FF", fg="#2C78A8",
            font=("Segoe UI", 10, "bold")
        ).pack(pady=8)

        # Footer
        footer = tk.Frame(self.root, bg=BG)
        footer.pack(fill="x", padx=18, pady=(0, 14))

        tk.Label(
            footer,
            text="YOLO11 test accuracy: 98.01%   |   YOLOv8 test accuracy: 96.69%",
            bg=BG, fg=TEXT, font=("Segoe UI", 9)
        ).pack(side="left")

        tk.Button(
            footer, text="Exit", command=self.root.destroy,
            bg=PINK, fg=TEXT, activebackground=PINK_DARK,
            font=("Segoe UI", 10, "bold"), padx=20, pady=6,
            relief="flat", cursor="hand2"
        ).pack(side="right")

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return
        try:
            image = Image.open(path).convert("RGB")
            preview = image.copy()
            preview.thumbnail((520, 430))
            self.preview_photo = ImageTk.PhotoImage(preview)
            self.image_label.configure(image=self.preview_photo, text="")
            self.run_detection(path)
        except Exception as exc:
            messagebox.showerror("Image Error", str(exc))

    @staticmethod
    def predict(model, path):
        result = model.predict(path, verbose=False)[0]
        idx = int(result.probs.top1)
        conf = float(result.probs.top1conf)
        label = str(result.names[idx]).lower()
        fire_prob = conf if label == "fire" else 1.0 - conf
        return label, fire_prob

    def run_detection(self, path):
        try:
            y11_label, y11_fire = self.predict(self.yolo11, path)
            v8_label, v8_fire = self.predict(self.yolov8, path)

            self.y11_result.set(f"YOLO11: {y11_label.upper()}")
            self.y11_conf.set(f"Fire probability: {y11_fire*100:.2f}%")
            self.v8_result.set(f"YOLOv8: {v8_label.upper()}")
            self.v8_conf.set(f"Fire probability: {v8_fire*100:.2f}%")

            if y11_fire >= THRESHOLD and v8_fire >= THRESHOLD:
                decision, bg, fg = " CRITICAL FIRE", CRIT_BG, CRIT_FG
                action = ("Both models strongly indicate fire. Generate an emergency alert "
                          "and follow the responsible emergency response procedure.")
            elif y11_fire >= THRESHOLD or v8_fire >= THRESHOLD:
                decision, bg, fg = " VERIFY FIRE", WARN_BG, WARN_FG
                action = ("Only one model crossed the emergency threshold. Treat this as a "
                          "possible fire and manually verify before escalating.")
            else:
                decision, bg, fg = " SAFE", SAFE_BG, SAFE_FG
                action = "Neither model crossed the emergency threshold. No emergency alert generated."

            self.decision_label.configure(text=decision, bg=bg, fg=fg)
            self.action_text.set(action)
            self.time_text.set("Detection time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            self.model_status.set(" Detection completed successfully")
        except Exception as exc:
            messagebox.showerror("Detection Error", str(exc))

    def clear(self):
        self.preview_photo = None
        self.image_label.configure(image="", text="Select an image to begin")
        self.y11_result.set("YOLO11: --")
        self.y11_conf.set("Fire probability: --")
        self.v8_result.set("YOLOv8: --")
        self.v8_conf.set("Fire probability: --")
        self.decision_label.configure(text="WAITING", bg="#EDF2F7", fg=TEXT)
        self.action_text.set("Select an image to start detection.")
        self.time_text.set("Detection time: --")
        self.model_status.set("Models loaded")

if __name__ == "__main__":
    root = tk.Tk()
    FireDetectionApp(root)
    root.mainloop()
