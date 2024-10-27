import cv2
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

class FingerprintMatcher:
    def __init__(self, root):
        self.root = root
        self.sample = None
        self.best_Score = 0
        self.best_filename = None
        self.best_image = None
        self.kp1, self.kp2, self.mp = None, None, None

        # Configuração da janela GUI
        self.root.title("Fingerprint Matching")
        self.root.geometry("800x550")
        self.root.resizable(False, False)
        self.root.configure(bg='gray20')

        photo = tk.PhotoImage(file='fingerprint-scan.png')
        self.root.wm_iconphoto(False, photo)

        # Componentes da GUI
        self.filename_label = tk.Label(self.root, text="No file selected", bg='gray20', fg='white')
        self.filename_label.pack(pady=(20, 10))

        # Frame para organizar os botões lado a lado
        button_frame = tk.Frame(self.root, bg='gray20')
        button_frame.pack(pady=(5, 20))

        # Botões dentro do frame
        select_button = tk.Button(button_frame, text="Select Image", command=self.select_image, width=15)
        select_button.pack(side=tk.LEFT, padx=10)

        run_button = tk.Button(button_frame, text="Run Matching", command=self.run_fingerprint_matching, width=15)
        run_button.pack(side=tk.LEFT, padx=10)

        # Texto para exibir os resultados
        self.result_text = tk.StringVar()
        self.result_text.set("Results will appear here.")
        result_label_text = tk.Label(self.root, textvariable=self.result_text, bg='gray20', fg='white')
        result_label_text.pack(pady=10)

        # Label para exibir a imagem de resultado
        self.result_label = tk.Label(self.root, bg='gray20')
        self.result_label.pack(padx=20)

    def select_image(self):
        filepath = filedialog.askopenfilename(title="Select Fingerprint Image",
                                              filetypes=[("Image Files", "*.BMP;*.JPG;*.PNG")])
        if filepath:
            pil_image = Image.open(filepath).convert("RGB")
            self.sample = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            if self.sample is None:
                messagebox.showerror("Error", "Could not load image. Please check the file format or path.")
                return
            self.filename_label.config(text=f"Selected File: {filepath}")
            print(f"Image loaded successfully from: {filepath}\n")
        else:
            messagebox.showinfo("Info", "No file selected")

    def run_fingerprint_matching(self):
        if self.sample is None:
            messagebox.showerror("Error", "Please select an image first")
            return

        if not os.path.exists("SOCOFing/Real"):
            messagebox.showerror("Error", "Directory 'SOCOFing/Real' does not exist. Please check the path.")
            return

        start_time = time.time()
        self.best_Score = 0
        self.best_filename = None
        self.best_image = None
        self.kp1, self.kp2, self.mp = None, None, None

        for file in [file for file in os.listdir("SOCOFing/Real") if file.lower().endswith(('.bmp', '.jpg', '.png'))][:1000]:
            print(f"Processing file: {file}")
            finger_print_img_path = os.path.join("SOCOFing/Real", file)
            pil_image = Image.open(finger_print_img_path).convert("RGB")
            finger_print_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            if finger_print_img is None:
                print(f"Warning: Could not load image {file}, skipping.")
                continue

            sift = cv2.SIFT_create()
            keypoints1, descriptor1 = sift.detectAndCompute(self.sample, None)
            keypoints2, descriptor2 = sift.detectAndCompute(finger_print_img, None)

            if descriptor1 is None or descriptor2 is None:
                print(f"Skipping {file}: No descriptors found.")
                continue

            flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
            matches = flann.knnMatch(descriptor1, descriptor2, k=2)
            match_points = [p for p, q in matches if p.distance < 0.7 * q.distance]

            keypoints = min(len(keypoints1), len(keypoints2))
            score = len(match_points) / keypoints * 100 if keypoints != 0 else 0

            if score > self.best_Score:
                self.best_Score = score
                self.best_filename = file
                self.best_image = finger_print_img
                self.kp1, self.kp2, self.mp = keypoints1, keypoints2, match_points

        self.display_results(start_time)

    def display_results(self, start_time):
        end_time = time.time()
        exec_time = end_time - start_time

        if self.best_image is not None:
            result = cv2.drawMatches(self.sample, self.kp1, self.best_image, self.kp2, self.mp, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            result = cv2.resize(result, None, fx=2, fy=2)

            result_img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_img = Image.fromarray(result_img)
            result_img = ImageTk.PhotoImage(result_img)

            self.result_label.config(image=result_img)
            self.result_label.image = result_img
            self.result_label.pack()

            self.result_text.set(
                f"BEST MATCH FOUND: {self.best_filename}\nSCORE: {self.best_Score:.2f}\nEXEC TIME: {exec_time:.4f} seconds")
        else:
            self.result_text.set("No match found.")
            print("No match found after processing all files.")

        print(f"\nBEST MATCH FOUND: {self.best_filename}\nSCORE: {self.best_Score:.2f}\nEXEC TIME: {exec_time:.4f} seconds")


if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintMatcher(root)
    root.mainloop()
