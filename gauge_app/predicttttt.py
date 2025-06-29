import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import math
from typing import Tuple, Optional


model = YOLO("D:\sneha\BE 5 sem\Gauge_final\Gauge\Trained_models\gauge_best.pt")
print("YOLOv8 model loaded successfully!")

class PressureGaugeReader:
    def _init_(self):
        self.center = None
        self.radius = None
        self.needle_line = None
        
    def detect_circle(self, image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(min(image.shape[:2]) * 0.5),
            param1=100,
            param2=50,
            minRadius=int(min(image.shape[:2]) * 0.1),
            maxRadius=int(min(image.shape[:2]) * 0.4)
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            largest_circle = max(circles, key=lambda c: c[2])
            return (int(largest_circle[0]), int(largest_circle[1])), int(largest_circle[2])
        
        h, w = image.shape[:2]
        return (w//2, h//2), min(w, h)//3
    
    def detect_needle(self, image: np.ndarray, center: Tuple[int, int], radius: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
       
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, center, int(radius), 255, -1)
        masked = cv2.bitwise_and(gray, mask)
        edges = cv2.Canny(masked, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=int(radius * 0.5),
            minLineLength=int(radius * 0.4),
            maxLineGap=int(radius * 0.1)
        )
        
        if lines is None:
            return None
        
        best_line = None
        best_score = float('inf')
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            A = y2 - y1
            B = x1 - x2
            C = x2*y1 - x1*y2
            
            if A == 0 and B == 0:
                continue
                
            distance_to_center = abs(A*center[0] + B*center[1] + C) / np.sqrt(A*2 + B*2)
            score = distance_to_center - line_length * 0.1
            
            if score < best_score and line_length > radius * 0.3:
                best_score = score
                best_line = line[0]
        
        if best_line is not None:
            x1, y1, x2, y2 = best_line
            dist1 = np.sqrt((x1 - center[0])**2 + (y1 - center[1])**2)
            dist2 = np.sqrt((x2 - center[0])**2 + (y2 - center[1])**2)
            if dist1 < dist2:
                base, tip = (x1, y1), (x2, y2)
            else:
                base, tip = (x2, y2), (x1, y1)
            return base, tip
        return None
    
    def estimate_min_max_positions(self, center: Tuple[int, int], radius: int, 
                                  start_angle: float = 222.5, end_angle: float = -52.5) -> Tuple[Tuple[int, int], Tuple[int, int]]:
       
        if end_angle < 0:
            end_angle += 360
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)
        needle_length = radius * 0.8
        min_x = int(center[0] + needle_length * math.cos(start_rad))
        min_y = int(center[1] - needle_length * math.sin(start_rad))
        max_x = int(center[0] + needle_length * math.cos(end_rad))
        max_y = int(center[1] - needle_length * math.sin(end_rad))
        return (min_x, min_y), (max_x, max_y)
    
    def process_gauge_fallback(self, image_path: str, min_pressure: float, max_pressure: float) -> dict:
        
        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "error": "Could not load image"}
        
        center, radius = self.detect_circle(image)
        if center is None:
            return {"success": False, "error": "Could not detect gauge circle"}
        
        self.center = center
        self.radius = radius
        
        needle_result = self.detect_needle(image, center, radius)
        if needle_result is None:
            return {"success": False, "error": "Could not detect needle"}
        
        base, tip = needle_result
        self.needle_line = (base, tip)
        
        min_pos, max_pos = self.estimate_min_max_positions(center, radius)
        
        # Converting positions to YOLO-compatible format 
        return {
            "success": True,
            "base_pos": [float(base[0]), float(base[1])],
            "needle_tip": [float(tip[0]), float(tip[1])],
            "min_pos": [float(min_pos[0]), float(min_pos[1])],
            "max_pos": [float(max_pos[0]), float(max_pos[1])]
        }

def calculate_angle(base, point):
    dx = point[0] - base[0]
    dy = point[1] - base[1]
    angle = np.arctan2(dy, dx) * 180 / np.pi
    angle = angle % 360
    return angle

def resize_image(image, max_size=(300, 300)):
    img = Image.fromarray(image)
    img.thumbnail(max_size, Image.LANCZOS)
    return img

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        path_label.config(text=f"Selected Image: {file_path}")
        try:
            img = cv2.imread(file_path)
            if img is None:
                path_label.config(text="Selected Image: Failed to load image")
                for widget in original_frame.winfo_children():
                    widget.destroy()
                for widget in plot_frame.winfo_children():
                    widget.destroy()
                return
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = resize_image(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            for widget in original_frame.winfo_children():
                widget.destroy()
            original_label = tk.Label(original_frame, image=img_tk)
            original_label.image = img_tk
            original_label.pack(anchor="center")
            for widget in plot_frame.winfo_children():
                widget.destroy()
        except Exception as e:
            path_label.config(text=f"Selected Image: Error loading image ({str(e)})")
            for widget in original_frame.winfo_children():
                widget.destroy()
            for widget in plot_frame.winfo_children():
                widget.destroy()

def process_gauge():
    img_path = path_label.cget("text").replace("Selected Image: ", "")
    if not img_path or img_path == "No image selected" or img_path.startswith("Failed") or img_path.startswith("Error"):
        result_label.config(text="Please upload a valid image first.")
        return

    try:
        min_value = float(min_entry.get())
        max_value = float(max_entry.get())
        PRESET_ANGLE = 275

        # Trying YOLO detection first
        results = model(img_path)
        img = cv2.imread(img_path)
        if img is None:
            result_label.config(text="Failed to load image.")
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]

        base_pos = None
        needle_tip = None
        min_pos = None
        max_pos = None

        # YOLO detection
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls)
                x_center, y_center, width, height = box.xywh[0]
                x_pixel = x_center.item() * img_width
                y_pixel = y_center.item() * img_height

                if class_id == 0:
                    base_pos = [x_pixel, y_pixel]
                elif class_id == 3:
                    needle_tip = [x_pixel, y_pixel]
                elif class_id == 1:
                    max_pos = [x_pixel, y_pixel]
                elif class_id == 2:
                    min_pos = [x_pixel, y_pixel]

        # Checking for missing detections
        missing = []
        if base_pos is None:
            missing.append("Base")
        if needle_tip is None:
            missing.append("Needle Tip")
        if min_pos is None:
            missing.append("Min")
        if max_pos is None:
            missing.append("Max")

        # If any detections are missing, using fallback
        if missing:
            print(f"YOLO failed to detect: {', '.join(missing)}. Using fallback.")
            fallback_reader = PressureGaugeReader()
            fallback_result = fallback_reader.process_gauge_fallback(img_path, min_value, max_value)
            
            if not fallback_result["success"]:
                result_label.config(text=f"Error: {fallback_result['error']}")
                return
                
            base_pos = fallback_result["base_pos"]
            needle_tip = fallback_result["needle_tip"]
            min_pos = fallback_result["min_pos"]
            max_pos = fallback_result["max_pos"]

        # Calculating gauge value using YOLO-based approach
        min_angle = calculate_angle(base_pos, min_pos)
        max_angle = calculate_angle(base_pos, max_pos)
        needle_angle = calculate_angle(base_pos, needle_tip)

        max_angle_relative = (max_angle - min_angle) % 360
        needle_angle_relative = (needle_angle - min_angle) % 360
        angle_range = PRESET_ANGLE
        
        
        if needle_angle_relative > angle_range:
            needle_angle_relative = 0  
        
        value_range = max_value - min_value
        value_per_degree = value_range / angle_range
        needle_angle_clamped = max(0, min(needle_angle_relative, angle_range))
        gauge_value = min_value + (needle_angle_clamped * value_per_degree)


        result_label.config(text=f"Predicted Gauge Reading: {gauge_value:.2f}")

        
        plt.figure(figsize=(4, 3))
        plt.plot(base_pos[0], base_pos[1], 'go', label='Base', markersize=10)
        plt.plot(needle_tip[0], needle_tip[1], 'bo', label='Needle Tip', markersize=10)
        plt.plot(min_pos[0], min_pos[1], 'ro', label=f'Min ({min_value})', markersize=10)
        plt.plot(max_pos[0], max_pos[1], 'yo', label=f'Max ({max_value})', markersize=10)
        plt.title(f"Gauge Reading: {gauge_value:.2f}")
        plt.legend()
        plt.axis('equal')  # Ensure aspect ratio is equal for accurate point positioning
        plt.xticks([])
        plt.yticks([])

        for widget in plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(anchor="center")

        img_pil = resize_image(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        for widget in original_frame.winfo_children():
            widget.destroy()
        original_label = tk.Label(original_frame, image=img_tk)
        original_label.image = img_tk
        original_label.pack(anchor="center")

    except ValueError as e:
        result_label.config(text=f"Error: Invalid input. Please enter numeric values. ({str(e)})")
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")

# GUI Setup
root = tk.Tk()
root.title("Gauge Reading System")
root.geometry("1000x600")

tk.Button(root, text="Upload Image", command=upload_image).pack(pady=5)
path_label = tk.Label(root, text="No image selected")
path_label.pack(pady=5)

tk.Label(root, text="Minimum Value:").pack(pady=5)
min_entry = tk.Entry(root, width=10)
min_entry.insert(0, "10")
min_entry.pack(pady=5)

tk.Label(root, text="Maximum Value:").pack(pady=5)
max_entry = tk.Entry(root, width=10)
max_entry.insert(0, "1000")
max_entry.pack(pady=5)

process_button = tk.Button(root, text="Process Gauge", command=process_gauge)
process_button.pack(pady=10)

display_frame = tk.Frame(root)
display_frame.pack(pady=10, fill=tk.BOTH, expand=True)

original_frame = tk.Frame(display_frame, width=350, height=350)
original_frame.pack(side=tk.LEFT, padx=20, expand=True)
plot_frame = tk.Frame(display_frame, width=350, height=350)
plot_frame.pack(side=tk.LEFT, padx=20, expand=True)

original_frame.pack_propagate(False)
plot_frame.pack_propagate(False)

tk.Label(original_frame, text="Original Image").pack()
tk.Label(plot_frame, text="Detected Points").pack()

result_label = tk.Label(root, text="Result will appear here", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()