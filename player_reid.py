# Option 2: Re-Identification in a Single Feed

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity

# Loading YOLOv11 (custom model) for player detection
model = YOLO(r"C:\Users\20214\Downloads\best.pt")

# Loading feature extractor (ResNet18)
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(r"C:\Users\20214\Downloads\15sec_input_720p.mp4")
player_db = {}  # {id: embedding}
next_id = 0

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        img_tensor = transform(crop).unsqueeze(0)
        with torch.no_grad():
            embedding = resnet(img_tensor).numpy()

        matched_id = None
        for pid, emb in player_db.items():
            sim = cosine_similarity(embedding, emb)[0][0]
            if sim > 0.75:
                matched_id = pid
                break

        if matched_id is None:
            matched_id = next_id
            player_db[matched_id] = embedding
            next_id += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {matched_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    frame_count += 1
    cv2.imshow("Re-ID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
