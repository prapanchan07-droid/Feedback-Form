import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

DISCLAIMER_TEXT = "not yet trained on real student handwriting, expect degraded accuracy until fine-tuned on real samples."

CLASSES = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]
CHAR_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CLASSES)}

class DigitCNN(nn.Module):
    def __init__(self, num_classes=36):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class DigitClassifier:
    def __init__(self, model_path="pretrained_model.pt"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DigitCNN(num_classes=len(CLASSES)).to(self.device)

        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
            except Exception:
                self._train_placeholder_model()
        else:
            self._train_placeholder_model()

    def _generate_synthetic_sample(self, char_str):
        """
        Generates a diverse synthetic sample matching ROI crop dimensions and handwriting style.
        """
        w, h = 110, 140
        img = Image.new("L", (w, h), 255)
        draw = ImageDraw.Draw(img)
        
        font_names = ["arial.ttf", "calibri.ttf", "times.ttf", "consola.ttf"]
        font_obj = None
        for fn in font_names:
            try:
                font_size = random.randint(55, 85)
                font_obj = ImageFont.truetype(fn, font_size)
                break
            except IOError:
                continue

        if font_obj is None:
            font_obj = ImageFont.load_default()

        tx = random.randint(15, 35)
        ty = random.randint(10, 30)
        draw.text((tx, ty), char_str, fill=0, font=font_obj)

        # Random rotation (-15 to +15 deg)
        angle = random.uniform(-15.0, 15.0)
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)

        # Convert to numpy array
        arr = np.array(img)

        # Random stroke dilation/erosion to simulate thin/thick pen strokes
        if random.random() > 0.5:
            kernel = np.ones((2, 2), np.uint8)
            if random.random() > 0.5:
                arr = cv2.erode(arr, kernel, iterations=1)
            else:
                arr = cv2.dilate(arr, kernel, iterations=1)

        return self.preprocess_crop_array(arr)

    def preprocess_crop_array(self, gray):
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            pad = 6
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
            thresh = thresh[y1:y2, x1:x2]

        h, w = thresh.shape
        if h == 0 or w == 0:
            return np.zeros((28, 28), dtype=np.float32)

        max_dim = max(h, w)
        square = np.zeros((max_dim, max_dim), dtype=np.uint8)
        px = (max_dim - w) // 2
        py = (max_dim - h) // 2
        square[py:py+h, px:px+w] = thresh

        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def preprocess_crop(self, crop):
        arr = self.preprocess_crop_array(crop)
        tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor

    def _train_placeholder_model(self):
        print("Training MNIST/EMNIST placeholder classifier with data augmentation...")
        samples_per_class = 400
        X_data = []
        y_data = []

        for idx, char_c in enumerate(CLASSES):
            for _ in range(samples_per_class):
                sample_28 = self._generate_synthetic_sample(char_c)
                X_data.append(sample_28)
                y_data.append(idx)

        X_np = np.array(X_data, dtype=np.float32)
        X_np = np.expand_dims(X_np, axis=1)  # (N, 1, 28, 28)
        y_np = np.array(y_data, dtype=np.int64)

        X_tensor = torch.tensor(X_np).to(self.device)
        y_tensor = torch.tensor(y_np).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.002, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(8):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/8 - Loss: {total_loss/len(loader):.4f}")

        self.model.eval()
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Placeholder model trained and saved to {self.model_path}")

    def predict(self, crop, is_digit_only=False):
        tensor = self.preprocess_crop(crop)
        with torch.no_grad():
            logits = self.model(tensor)
            
            if is_digit_only:
                logits[:, 10:] = -1e9

            probs = F.softmax(logits, dim=1).squeeze(0)
            top_probs, top_indices = torch.topk(probs, k=3)
            
            pred_idx = top_indices[0].item()
            pred_char = IDX_TO_CHAR[pred_idx]
            confidence = round(top_probs[0].item(), 4)

            top_k = [(IDX_TO_CHAR[idx.item()], round(prob.item(), 4)) for idx, prob in zip(top_indices, top_probs)]

        return {
            "predicted_char": pred_char,
            "confidence": confidence,
            "top_k": top_k,
            "disclaimer": DISCLAIMER_TEXT
        }

if __name__ == "__main__":
    if os.path.exists("pretrained_model.pt"):
        os.remove("pretrained_model.pt")
    clf = DigitClassifier()
