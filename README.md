# Stealth-Single-Feed
# Player Re-Identification

## 🎯 Objective

Real-time player re-identification system using a soccer broadcast video. Assigns consistent IDs to players even as they move in and out of the frame.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/player-reid.git
cd player-reid
```

### 2. Create Virtual Environment

Using **Conda**:

```bash
conda create -n player-reid python=3.9 -y
conda activate player-reid
```

Or using **venv**:

```bash
python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Project Files

* Place `best.pt` (YOLOv11 model weights) in the project root.
* Place input video (e.g., `15sec_input_720p.mp4`) in an `input/` directory, or update its path in `player_reid.py`.

### 5. Run the Code

```bash
python player_reid.py
```

> Press `q` to exit the video window.

---

## 📦 Output

* A window shows the video with bounding boxes and unique IDs on players.
* Modify the script to save output using `cv2.VideoWriter()` if required.

---

## 📁 Project Structure

```
├── player_reid.py           # Main script
├── best.pt                  # YOLOv11 model weights
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── report.md / report.pdf   # Mini project report
└── input/
    └── 15sec_input_720p.mp4 # Input video
```

---

## 🧠 Core Components

* **YOLOv11**: Custom-trained model for detecting players
* **ResNet18**: Extracts feature embeddings from cropped player images
* **Cosine Similarity**: Compares embeddings to assign consistent IDs

---

## 🛠 Dependencies

```
torch
torchvision
ultralytics
opencv-python
scikit-learn
numpy
```

---

## 📬 Contact

For help or suggestions, contact **[ayati16j2003@gmail.com](mailto:ayati16j2003@gmail.com)**
