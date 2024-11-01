import cv2
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import threading


class FingerprintMatcher:
    def __init__(self, root):
        self.root = root
        self.sample = None
        self.best_Score = 0
        self.best_filename = None
        self.best_image = None
        self.kp1, self.kp2, self.mp = None, None, None

        # GUI Configuration
        self.setup_gui()

    def setup_gui(self):
        # Window setup
        self.root.title("Fingerprint Matching")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        self.root.configure(bg='gray20')

        # Icon and Labels
        photo = tk.PhotoImage(file='img/fingerprint-scan.png')
        self.root.wm_iconphoto(False, photo)

        self.filename_label = tk.Label(self.root, text="No file selected", bg='gray20', fg='white')
        self.filename_label.pack(pady=(20, 10))

        # Frame for buttons
        button_frame = tk.Frame(self.root, bg='gray20')
        button_frame.pack(pady=(5, 20))

        select_button = tk.Button(button_frame, text="Select Image", command=self.select_image, width=15)
        select_button.pack(side=tk.LEFT, padx=10)

        run_button = tk.Button(button_frame, text="Run Matching", command=self.run_fingerprint_matching_thread,
                               width=15)
        run_button.pack(side=tk.LEFT, padx=10)

        # Progress bar and result display text (initially hidden)
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=600, mode="determinate")
        self.progress.pack(pady=10)
        self.progress.pack_forget()  # Hide the progress bar initially

        self.result_text = tk.StringVar()
        self.result_text.set("Results will appear here.")
        result_label_text = tk.Label(self.root, textvariable=self.result_text, bg='gray20', fg='white')
        result_label_text.pack(pady=10)

        # Label to display the result image
        self.result_label = tk.Label(self.root, bg='gray20')
        self.result_label.pack(padx=20)

    def select_image(self):
        filepath = filedialog.askopenfilename(title="Select Fingerprint Image",
                                              filetypes=[("Image Files", "*.BMP;*.JPG;*.PNG")])
        if not filepath:
            messagebox.showinfo("Info", "No file selected")
            return

        # Load and display the selected image
        pil_image = Image.open(filepath).convert("RGB")
        self.sample = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        self.filename_label.config(text=f"Selected File: {filepath}")
        print(f"Image loaded successfully from: {filepath}\n")

    def run_fingerprint_matching_thread(self):
        """ Run fingerprint matching in a separate thread to avoid freezing the GUI. """
        if self.sample is None:
            messagebox.showerror("Error", "Please select an image first")
            return

        # Check if dataset path exists
        if not os.path.exists("SOCOFing/Real"):
            messagebox.showerror("Error", "Directory 'SOCOFing/Real' does not exist. Please check the path.")
            return

        # Show the progress bar when starting the process
        self.progress.pack(pady=10)

        # Start processing in a new thread
        threading.Thread(target=self.run_fingerprint_matching).start()

    def run_fingerprint_matching(self):
        self.best_Score = 0
        self.best_filename = None
        self.best_image = None
        self.kp1, self.kp2, self.mp = None, None, None

        # Retrieve image files and initialize progress
        image_files = self.get_image_files("SOCOFing/Real")[:1000]  # Limit to first 1000 images
        total_files = len(image_files)
        self.progress["maximum"] = total_files

        start_time = time.time()

        for i, file in enumerate(image_files):
            finger_print_img = self.load_image(file)
            if finger_print_img is None:
                continue  # Skip invalid images

            # Perform matching using SIFT and FLANN
            score, keypoints1, keypoints2, match_points = self.match_fingerprints(finger_print_img)

            # Update the best match if the score is higher
            if score > self.best_Score:
                self.best_Score, self.best_filename, self.best_image = score, file, finger_print_img
                self.kp1, self.kp2, self.mp = keypoints1, keypoints2, match_points

            # Update progress bar and estimate remaining time
            elapsed_time = time.time() - start_time
            remaining_time = (elapsed_time / (i + 1)) * (total_files - (i + 1))
            self.progress["value"] = i + 1
            self.result_text.set(
                f"Processing {i + 1}/{total_files} images...\nEstimated time remaining: {remaining_time:.2f} seconds")
            self.root.update_idletasks()

        # Hide the progress bar once the process is complete
        self.progress.pack_forget()

        self.display_results(start_time)

    def get_image_files(self, directory):
        """ Retrieve all valid image files from the specified directory. """
        return [os.path.join(directory, file) for file in os.listdir(directory) if
                file.lower().endswith(('.bmp', '.jpg', '.png'))]

    def load_image(self, file_path):
        """ Load an image from the specified path and return it as a BGR image. """
        try:
            pil_image = Image.open(file_path).convert("RGB")
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Warning: Could not load image {file_path}. Error: {e}")
            return None

    def match_fingerprints(self, fingerprint_image):
        """ Match the sample fingerprint with a given fingerprint image using SIFT and FLANN. """
        sift = cv2.SIFT_create()
        keypoints1, descriptor1 = sift.detectAndCompute(self.sample, None)
        keypoints2, descriptor2 = sift.detectAndCompute(fingerprint_image, None)

        if descriptor1 is None or descriptor2 is None:
            return 0, None, None, None

        # FLANN matching with ratio test
        flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
        matches = flann.knnMatch(descriptor1, descriptor2, k=2)
        match_points = [p for p, q in matches if p.distance < 0.7 * q.distance]

        # Calculate match score
        keypoints = min(len(keypoints1), len(keypoints2))
        score = (len(match_points) / keypoints) * 100 if keypoints > 0 else 0
        return score, keypoints1, keypoints2, match_points

    def display_results(self, start_time):
        """ Display the matching results on the GUI. """
        exec_time = time.time() - start_time
        if self.best_image is not None:
            result_img = self.create_result_image()
            self.show_result_image(result_img)
            self.result_text.set(
                f"BEST MATCH FOUND: {self.best_filename}\nSCORE: {self.best_Score:.2f}\nEXEC TIME: {exec_time:.4f} seconds")
        else:
            self.result_text.set("No match found.")
            print("No match found after processing all files.")

    def create_result_image(self):
        """ Draw matches between the sample and the best matching fingerprint. """
        result = cv2.drawMatches(self.sample, self.kp1, self.best_image, self.kp2, self.mp, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        result = cv2.resize(result, None, fx=2, fy=2)
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    def show_result_image(self, result_image):
        """ Display the result image on the GUI. """
        result_img_tk = ImageTk.PhotoImage(result_image)
        self.result_label.config(image=result_img_tk)
        self.result_label.image = result_img_tk  # Keep reference to prevent garbage collection


if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintMatcher(root)
    root.mainloop()
