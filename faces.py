import os
import cv2
import dlib
import numpy as np
from imutils import face_utils
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk

# Initialize Dlib's face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Placeholder for selected eye points
selected_points = []
manual_mode = False
manual_frame = None

# Load all images from a folder
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):  # Add more extensions if needed
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images

# Function to detect landmarks
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) > 0:
        return predictor(gray, faces[0])  # Return landmarks for the first detected face
    return None

# Align face using detected or manual landmarks
def align_face(image, landmarks=None, manual_points=None):
    if landmarks is not None:
        landmarks = face_utils.shape_to_np(landmarks)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
    elif manual_points is not None:
        left_eye_center = np.array(manual_points[0])
        right_eye_center = np.array(manual_points[1])
    else:
        return None

    if landmarks is not None:
        left_eye_center = left_eye.mean(axis=0).astype("int")
        right_eye_center = right_eye.mean(axis=0).astype("int")

    # Align image based on eye centers
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx)) - 180
    dist = np.sqrt((dx ** 2) + (dy ** 2))
    desired_dist = 100  # Desired distance between eyes
    scale = desired_dist / dist

    eyes_center = (float((left_eye_center[0] + right_eye_center[0]) // 2),
                   float((left_eye_center[1] + right_eye_center[1]) // 2))

    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    output = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    return output

# Mouse callback to manually select eye points
def select_eyes(event, x, y, flags, param):
    global selected_points, manual_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        cv2.circle(manual_frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Manual Eye Selection", manual_frame)
        if len(selected_points) == 2:
            cv2.destroyWindow("Manual Eye Selection")

# GUI Application for reviewing and correcting eyes
class ImageReviewApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Eye Alignment Review")

        self.image_index = 0
        self.aligned_images = []
        self.manual_points = None
        self.folder_path = filedialog.askdirectory(title="Select Folder with Images")
        self.images = load_images(self.folder_path)

        if not self.images:
            messagebox.showerror("Error", "No images found in the folder!")
            self.master.quit()

        self.canvas = tk.Canvas(master, width=800, height=600)
        self.canvas.pack()

        self.next_button = tk.Button(master, text="Next", command=self.next_image)
        self.next_button.pack(side=tk.RIGHT)

        self.prev_button = tk.Button(master, text="Previous", command=self.prev_image)
        self.prev_button.pack(side=tk.RIGHT)

        self.correct_button = tk.Button(master, text="Correct Eyes", command=self.correct_eyes)
        self.correct_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(master, text="Save MP4", command=self.save_video)
        self.save_button.pack(side=tk.LEFT)

        self.show_image()

    def show_image(self):
        img = self.images[self.image_index]
        landmarks = get_landmarks(img)

        if landmarks is None:
            # Manually select eyes
            cv2.imshow("Manual Eye Selection", img)
            cv2.setMouseCallback("Manual Eye Selection", select_eyes)
            cv2.waitKey(0)
            aligned_img = align_face(img, manual_points=selected_points)
        else:
            aligned_img = align_face(img, landmarks)

        if aligned_img is not None:
            self.aligned_images.append(aligned_img)

        # Display in GUI using PIL and Tkinter
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.master.image_tk = img_tk  # Keep reference to avoid garbage collection

    def next_image(self):
        if self.image_index < len(self.images) - 1:
            self.image_index += 1
            self.show_image()

    def prev_image(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.show_image()

    def correct_eyes(self):
        global manual_frame, selected_points
        img = self.images[self.image_index]
        manual_frame = img.copy()
        selected_points = []
        cv2.imshow("Manual Eye Selection", manual_frame)
        cv2.setMouseCallback("Manual Eye Selection", select_eyes)
        cv2.waitKey(0)  # Wait until eyes are selected
        aligned_img = align_face(img, manual_points=selected_points)
        self.aligned_images[self.image_index] = aligned_img
        self.show_image()

    def save_video(self):
        duration = simpledialog.askfloat("Duration", "Enter the duration for the final video in seconds:")
        if duration:
            fps = len(self.aligned_images) / duration
            self.create_video(fps)

    def create_video(self, fps):
        height, width, _ = self.aligned_images[0].shape
        video = cv2.VideoWriter('output_animation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for img in self.aligned_images:
            video.write(img)
        
        video.release()
        messagebox.showinfo("Success", "MP4 video created successfully!")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageReviewApp(root)
    root.mainloop()
