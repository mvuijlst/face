import os
import cv2
import dlib
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps
import json
from imutils import face_utils

# Check for Pillow's Resampling module or fallback to older LANCZOS constant
try:
    from PIL import Resampling
    LANCZOS = Resampling.LANCZOS
except ImportError:
    LANCZOS = Image.LANCZOS  # Fallback for older Pillow versions

# Initialize Dlib's face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

CONFIG_FILE = "app_config.json"
EYE_POSITIONS_FILE = "eye_positions.json"

# Helper function to load configuration from JSON
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as file:
            return json.load(file)
    return {}

# Helper function to save configuration to JSON
def save_config(config):
    with open(CONFIG_FILE, 'w') as file:
        json.dump(config, file)

# Helper function to load saved eye positions from JSON
def load_eye_positions():
    if os.path.exists(EYE_POSITIONS_FILE):
        with open(EYE_POSITIONS_FILE, 'r') as file:
            return json.load(file)
    return {}

# Helper function to save eye positions to JSON
def save_eye_positions(eye_positions):
    with open(EYE_POSITIONS_FILE, 'w') as file:
        json.dump(eye_positions, file)

# Attempt to load the image using OpenCV, fallback to PIL if needed
def load_image_cv_or_pil(file_path):
    try:
        # Try loading the image using OpenCV
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"OpenCV failed to load image: {file_path}")
        return img
    except Exception as e:
        print(f"OpenCV failed with error: {e}. Attempting to load with PIL.")
        # Fallback to loading with PIL
        try:
            pil_image = Image.open(file_path)
            pil_image = ImageOps.exif_transpose(pil_image)  # Handle any EXIF orientation issues
            img = np.array(pil_image)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert back to BGR format for OpenCV compatibility
        except Exception as pil_error:
            print(f"PIL also failed to load image: {file_path}. Error: {pil_error}")
            return None

# Load all images from a folder and sort by modified date
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            file_path = os.path.join(folder, filename)
            img = load_image_cv_or_pil(file_path)  # Use the updated loader with OpenCV fallback to PIL
            if img is not None:
                # Get the modification time of the file
                mod_time = os.path.getmtime(file_path)
                images.append((filename, img, mod_time))
            else:
                print(f"Skipping file {filename} due to loading error.")
    # Sort images by modification date (mod_time)
    images.sort(key=lambda x: x[2])  # Sort by modification time
    return images

# Detect landmarks for the eyes, choosing the largest face
def detect_eyes_largest_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    
    if len(faces) == 0:
        return None
    
    # Select the largest face by area (width * height of the bounding box)
    largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
    
    # Get landmarks for the largest face
    landmarks = predictor(gray, largest_face)
    return face_utils.shape_to_np(landmarks)

def refine_eye_landmarks(image, pupil_center, box_size=150):
    """
    Refine the eye landmarks by detecting the eye in a significantly larger region around the manually clicked pupil.
    
    Parameters:
    - image: the original image (in BGR format).
    - pupil_center: the (x, y) coordinates of the manually clicked pupil center.
    - box_size: the size of the region to crop around the pupil center for detection.
    
    Returns:
    - A NumPy array with six points for the eye if detection is successful, otherwise None.
    """
    # Define the region of interest (ROI) around the pupil center
    x, y = pupil_center
    half_box = box_size // 2
    x1, y1 = max(0, x - half_box), max(0, y - half_box)
    x2, y2 = min(image.shape[1], x + half_box), min(image.shape[0], y + half_box)
    
    # Crop the region around the pupil
    roi = image[y1:y2, x1:x2]
    
    # Convert the region to grayscale for detection
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Detect the face landmarks in the cropped region
    faces = detector(gray_roi, 1)
    if len(faces) > 0:
        # Get the largest face (usually the only face in the cropped region)
        largest_face = faces[0]
        
        # Detect landmarks
        landmarks = predictor(gray_roi, largest_face)
        landmarks = face_utils.shape_to_np(landmarks)
        
        # Extract the eye landmarks from the cropped region
        left_eye_landmarks = landmarks[36:42]
        
        # Adjust the positions of the eye landmarks relative to the original image
        left_eye_landmarks[:, 0] += x1
        left_eye_landmarks[:, 1] += y1
        
        return left_eye_landmarks
    else:
        print(f"Warning: Could not detect eye in the region around {pupil_center}.")
        return None

# Load the OpenCV Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def refine_eye_with_haar(image, pupil_center, box_size=150):
    """
    Refine the eye landmarks using OpenCV's Haar Cascade for eye detection in a larger region around the manually clicked pupil.
    
    Parameters:
    - image: the original image (in BGR format).
    - pupil_center: the (x, y) coordinates of the manually clicked pupil center.
    - box_size: the size of the region to crop around the pupil center for detection.
    
    Returns:
    - A NumPy array with six points for the eye approximated from the bounding box.
    """
    # Define the region of interest (ROI) around the pupil center
    x, y = pupil_center
    half_box = box_size // 2
    x1, y1 = max(0, x - half_box), max(0, y - half_box)
    x2, y2 = min(image.shape[1], x + half_box), min(image.shape[0], y + half_box)
    
    # Crop the region around the pupil
    roi = image[y1:y2, x1:x2]
    
    # Convert the region to grayscale for detection
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Detect eyes using OpenCV's Haar Cascade
    eyes = eye_cascade.detectMultiScale(gray_roi)
    
    if len(eyes) > 0:
        # Get the first detected eye
        (ex, ey, ew, eh) = eyes[0]
        
        # Adjust the eye position to the original image coordinates
        ex += x1
        ey += y1
        
        # Approximate six landmarks from the bounding box
        eye_landmarks = np.array([
            [ex, ey + eh // 2],  # Left corner (middle)
            [ex + ew // 2, ey],  # Top center
            [ex + ew, ey + eh // 2],  # Right corner (middle)
            [ex + ew // 2, ey + eh],  # Bottom center
            [ex + ew // 4, ey + eh // 4],  # Top-left
            [ex + 3 * ew // 4, ey + eh // 4]  # Top-right
        ])
        
        return eye_landmarks
    else:
        print(f"Warning: Could not detect eye in the region around {pupil_center} using Haar Cascade.")
        return None

# GUI Application for reviewing and correcting eye detection
class EyeReviewApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Eye Detection Review")

        # Load window position and size
        config = load_config()
        if 'window_size' in config:
            self.master.geometry(config['window_size'])

        # Get the current script directory and set it as the initial directory for the file dialog
        script_dir = os.path.dirname(os.path.realpath(__file__))
        initial_dir = os.path.join(script_dir, "subfolder")  # Change 'subfolder' to your image subfolder

        self.folder_path = filedialog.askdirectory(initialdir=initial_dir, title="Select Folder with Images")
        if not self.folder_path:
            messagebox.showerror("Error", "No folder selected!")
            self.master.quit()

        self.images = load_images(self.folder_path)
        self.eye_positions = load_eye_positions()  # Load saved eye positions
        self.detection_results = [None] * len(self.images)  # Initially None, indicating not processed yet
        self.manual_corrections = [False] * len(self.images)  # Track whether manual correction was applied
        self.current_image_index = None
        self.clicks = []  # Store manual eye selection clicks

        # Left pane - Listbox for image list with detection status
        self.listbox_frame = tk.Frame(master)
        self.listbox_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        self.listbox = tk.Listbox(self.listbox_frame, width=30, height=20)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_image_select)

        # Right pane - Canvas for displaying images
        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, width=600, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)  # Bind mouse click for manual eye selection

        # Populate listbox with file names (without waiting for eye detection)
        self.populate_listbox()

        # Start background processing for eye detection
        threading.Thread(target=self.process_images_in_background, daemon=True).start()

        # Save window size and position on close
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    # Populate the listbox with file names (initially without detection status)
    def populate_listbox(self):
        for filename, _, mod_time in self.images:
            if filename in self.eye_positions:  # Use saved eye positions if available
                self.listbox.insert(tk.END, f"{filename} ⚙️")  # Mark as manually corrected
                # Convert the eye positions from the JSON format back to NumPy arrays
                left_eye, right_eye = self.eye_positions[filename]
                left_eye = np.array(left_eye)
                right_eye = np.array(right_eye)
                self.detection_results[self.images.index((filename, _, mod_time))] = np.concatenate((left_eye, right_eye), axis=0)
                print(f"Loaded eye positions for {filename}: {left_eye}, {right_eye}")  # Debug statement
            else:
                self.listbox.insert(tk.END, f"{filename} ⏳")  # ⏳ indicates waiting for detection

    # Background thread to process images for eye detection
    def process_images_in_background(self):
        for idx, (filename, image, _) in enumerate(self.images):
            if filename not in self.eye_positions:  # Skip already processed images
                landmarks = detect_eyes_largest_face(image)  # Get eyes from the largest face
                if landmarks is not None:
                    self.detection_results[idx] = landmarks  # Store landmarks
                    
                    # Save the detected landmarks to the eye_positions JSON file immediately
                    self.eye_positions[filename] = [landmarks[36:42].tolist(), landmarks[42:48].tolist()]  # Left and right eye landmarks
                    save_eye_positions(self.eye_positions)
                    print(f"Detected eye positions for {filename}: {landmarks[36:42]}, {landmarks[42:48]}")  # Debug statement
                    
                    self.update_listbox_item(idx, filename, True)  # Update to show success ✅
                else:
                    self.detection_results[idx] = False  # Indicate failure
                    self.update_listbox_item(idx, filename, False)  # Update to show failure ❌

    # Update a specific item in the listbox (success or failure, with manual correction option)
    def update_listbox_item(self, index, filename, success, manual=False):
        status_icon = "⚙️" if manual else ("✅" if success else "❌")
        new_text = f"{filename} {status_icon}"
        self.listbox.delete(index)
        self.listbox.insert(index, new_text)

    # Load and display selected image on the right pane
    def on_image_select(self, event):
        selection = event.widget.curselection()
        if not selection:
            return

        index = selection[0]
        self.current_image_index = index
        filename, image, _ = self.images[index]
        detection_status = self.detection_results[index]

        # Resize and display the image on the canvas
        self.display_image(image, detection_status)

    # Update the display_image function to use the refined eye detection if manually selected
    def display_image(self, image, detection_status):
        # Get the original image dimensions
        original_height, original_width = image.shape[:2]

        # Convert image for displaying on Canvas
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        resized_pil_image, scale_factor_w, scale_factor_h = self.resize_image(pil_image, self.canvas.winfo_width(), self.canvas.winfo_height())
        self.current_image = ImageTk.PhotoImage(resized_pil_image)

        # Clear previous drawings
        self.canvas.delete("all")

        # Display the image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)

        # Debugging statement to ensure we're calling the correct function
        print(f"Displaying image with detection status: {detection_status}")

        if isinstance(detection_status, np.ndarray):
            # Check if we have six key points (auto-detected) or just two key points (manually corrected)
            if detection_status.shape[0] == 12:  # 12 key points = 6 per eye (auto-detected)
                left_eye = detection_status[:6]  # First six points are for the left eye
                right_eye = detection_status[6:]  # Last six points are for the right eye
            elif detection_status.shape[0] == 2:  # 2 key points = manually corrected (pupil centers)
                # Try Haar cascade to refine the eye landmarks based on pupil centers
                left_eye = refine_eye_with_haar(image, detection_status[0])
                right_eye = refine_eye_with_haar(image, detection_status[1])
                
                # If we couldn't detect the landmarks, default back to the clicked centers
                if left_eye is None:
                    left_eye = np.array([detection_status[0]])
                if right_eye is None:
                    right_eye = np.array([detection_status[1]])
            else:
                print("Invalid eye position format.")
                return

            # Debugging eye positions
            print(f"Rendering eyes at positions: left_eye={left_eye}, right_eye={right_eye}")

            # Check if eyes are valid (non-empty)
            if len(left_eye) > 0 and len(right_eye) > 0:
                # Scale the landmark positions based on the resize factors
                left_eye_center = (left_eye.mean(axis=0) * [scale_factor_w, scale_factor_h]).astype("int")
                right_eye_center = (right_eye.mean(axis=0) * [scale_factor_w, scale_factor_h]).astype("int")

                # Debugging the scaled positions
                print(f"Scaled eye positions: left_eye_center={left_eye_center}, right_eye_center={right_eye_center}")

                # Draw circles on the eyes
                for point in left_eye:
                    point = (point * [scale_factor_w, scale_factor_h]).astype("int")
                    self.canvas.create_oval(point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2, outline="yellow", width=2)

                for point in right_eye:
                    point = (point * [scale_factor_w, scale_factor_h]).astype("int")
                    self.canvas.create_oval(point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2, outline="yellow", width=2)
            else:
                print("Eye positions are empty. Not rendering circles.")
        else:
            print("Invalid detection status, not rendering eyes.")

    # Resize image to fit the canvas while keeping the aspect ratio
    def resize_image(self, image, canvas_width, canvas_height):
        image_width, image_height = image.size
        ratio = min(canvas_width / image_width, canvas_height / image_height)
        new_size = (int(image_width * ratio), int(image_height * ratio))
        resized_image = image.resize(new_size, LANCZOS)  # Use the correct LANCZOS based on version
        return resized_image, ratio, ratio  # Return scale factors for width and height

    # Handle mouse click on canvas for manual eye selection
    def on_canvas_click(self, event):
        if self.current_image_index is None:
            return

        # Record the click coordinates and scale them back to the original image size
        clicked_x = event.x
        clicked_y = event.y

        # Get the scaling factors from the current displayed image
        _, scale_factor_w, scale_factor_h = self.resize_image(Image.fromarray(self.images[self.current_image_index][1]),
                                                              self.canvas.winfo_width(), self.canvas.winfo_height())

        original_x = int(clicked_x / scale_factor_w)
        original_y = int(clicked_y / scale_factor_h)

        # Store the click for manual selection
        self.clicks.append((original_x, original_y))

        # Once two points (left and right eye) are selected, update the image and mark as manually corrected
        if len(self.clicks) == 2:
            # Overwrite the detection result with manual selection
            left_eye, right_eye = self.clicks
            self.detection_results[self.current_image_index] = np.array([left_eye, right_eye])

            # Display the updated image with eye circles immediately
            self.display_image(self.images[self.current_image_index][1], self.detection_results[self.current_image_index])

            # Clear the clicks for future use
            self.clicks = []

            # Save the manually corrected eye positions
            filename, _, _ = self.images[self.current_image_index]
            self.eye_positions[filename] = [list(left_eye), list(right_eye)]  # Fixed: using list() instead of tolist()
            save_eye_positions(self.eye_positions)

            # Update the listbox to indicate manual correction
            self.update_listbox_item(self.current_image_index, filename, True, manual=True)

    # Handle window close to save size and position
    def on_close(self):
        # Save window size and position
        window_geometry = self.master.geometry()
        config = load_config()
        config['window_size'] = window_geometry
        save_config(config)

        # Exit the application
        self.master.destroy()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = EyeReviewApp(root)
    root.mainloop()
