# Gauge Reader 📟

A computer vision-based system that reads analog gauge values automatically from images using a combination of object detection and image processing techniques.

## 🚀 Overview

The **Gauge Reader** project uses a YOLOv8 model to detect key components of an analog gauge — such as the needle tip, center, and scale range — and calculates the current reading accurately. This tool is ideal for automating the monitoring of mechanical gauges in industries where manual reading is inefficient or error-prone.

## 🧠 Features

- ⚙️ **YOLOv8-based detection** of:
  - Needle base and tip
  - Min and max scale indicators
- 📏 **Custom logic for angle-to-value conversion**
- 📊 Visualizations of results using `matplotlib`
- 🖼️ Compatible with static images of analog gauges

## 🛠️ Tech Stack

- **Python**
- **OpenCV**
- **YOLOv8 (Ultralytics)**
- **Matplotlib**
- **NumPy**

## 📁 Project Structure

```
Gauge_Reader/
│
├── models/                  # Trained YOLOv8 model weights
├── test_images/             # Sample gauge images for testing
├── src/
│   ├── detect.py            # YOLO detection script
│   ├── gauge_reader.py      # Value calculation logic
│   └── utils.py             # Utility functions
├── README.md                # Project documentation
```

## ⚙️ How It Works

1. Load the trained YOLOv8 model.
2. Detect the needle base, tip, min, and max positions.
3. Calculate the needle angle and map it to the gauge's numeric range.
4. Output the predicted gauge value.

## 🧪 Sample Result

```
Detected:
- Min Value: 0
- Max Value: 300
- Needle Angle: 120°
→ Final Reading: 150.0
```

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/WinUdupa/Gauge_Reader.git
cd Gauge_Reader
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the prediction:

```bash
python src/gauge_reader.py --image test_images/sample.jpg
```

## 🧩 Example Usage

```python
from gauge_reader import GaugeReader

reader = GaugeReader(model_path="models/best.pt")
value = reader.read("test_images/gauge1.jpg")
print("Gauge Value:", value)
```

## 📌 To Do

- [ ] Add real-time webcam support
- [ ] Deploy as a web service
- [ ] Add batch processing support

## 📜 License

This project is licensed under the MIT License.

## 🤝 Contributing

Pull requests and forks are welcome! If you'd like to improve this project or fix a bug, feel free to contribute.

---

**Created by [Vineeth Udupa](https://github.com/WinUdupa)** 🚀
