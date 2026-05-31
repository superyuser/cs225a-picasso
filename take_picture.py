import os
import tkinter as tk
from datetime import datetime

import cv2
from PIL import Image, ImageTk


COUNTDOWN_START = 3
PHOTOS_DIR = os.path.join(os.getcwd(), "photos")


class WebcamApp:
    def __init__(self, root: tk.Tk, cap: cv2.VideoCapture) -> None:
        self.root = root
        self.cap = cap
        self.root.title("Webcam - Photo Countdown")
        self.root.configure(bg="black")

        self.canvas = tk.Canvas(
            root, bg="black", highlightthickness=0, width=960, height=720
        )
        self.canvas.pack(fill="both", expand=True)

        self.image_id = self.canvas.create_image(0, 0, anchor="nw")
        self.text_id = self.canvas.create_text(
            0, 0, text="", fill="white", font=("Helvetica", 220, "bold")
        )
        self.status_id = self.canvas.create_text(
            0, 0, text="Get ready...", fill="white", font=("Helvetica", 28, "bold")
        )

        self._photo = None
        self._countdown_value = COUNTDOWN_START
        self._captured = False
        self._latest_frame = None

        self.root.after(1000, self._tick_countdown)
        self._update_frame()

    def _update_frame(self) -> None:
        ok, frame = self.cap.read()
        if ok:
            self._latest_frame = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            self.canvas.config(width=w, height=h)
            image = Image.fromarray(rgb)
            self._photo = ImageTk.PhotoImage(image)
            self.canvas.itemconfig(self.image_id, image=self._photo)
            self.canvas.coords(self.text_id, w // 2, h // 2)
            self.canvas.coords(self.status_id, w // 2, 50)
            self.canvas.tag_raise(self.text_id)
            self.canvas.tag_raise(self.status_id)

        if not self._captured:
            self.root.after(15, self._update_frame)

    def _tick_countdown(self) -> None:
        if self._countdown_value > 0:
            self.canvas.itemconfig(self.text_id, text=str(self._countdown_value))
            self.canvas.itemconfig(self.status_id, text="Taking photo in...")
            self._countdown_value -= 1
            self.root.after(1000, self._tick_countdown)
        else:
            self.canvas.itemconfig(self.text_id, text="SNAP!")
            self.canvas.itemconfig(self.status_id, text="Captured")
            self.root.update_idletasks()
            self._capture_photo()
            self.root.after(800, self.root.destroy)

    def _capture_photo(self) -> None:
        if self._latest_frame is None:
            print("No frame available to save.")
            return
        os.makedirs(PHOTOS_DIR, exist_ok=True)
        filename = datetime.now().strftime("photo_%Y%m%d_%H%M%S.jpg")
        filepath = os.path.join(PHOTOS_DIR, filename)
        cv2.imwrite(filepath, self._latest_frame)
        self._captured = True
        print(f"Saved photo to {filepath}")


def main() -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:
        root = tk.Tk()
        app = WebcamApp(root, cap)
        root.mainloop()
    finally:
        cap.release()


if __name__ == "__main__":
    main()
