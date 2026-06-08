import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from collections import deque

MODEL_PATH = "best_rail_unet.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256 
THRESHOLD = 0.4 
ROI_W, ROI_H = 380, 150

MA_WINDOW_SIZE = 12
length_buffer = deque(maxlen=MA_WINDOW_SIZE)

model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, activation=None).to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"--- Đã tải Model thành công trên {DEVICE} ---")
else:
    print("LỖI: Không tìm thấy file model 'best_rail_unet.pth'"); exit()

transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

cap = cv2.VideoCapture(1) 
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

while True:
    ret, frame = cap.read()
    if not ret: break
    
    result = frame.copy()
    h_orig, w_orig = frame.shape[:2]

    x1, y1 = (w_orig - ROI_W) // 2, (h_orig - ROI_H) // 2
    x2, y2 = x1 + ROI_W, y1 + ROI_H
    roi_img = frame[y1:y2, x1:x2]

    roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    roi_enhanced = clahe.apply(roi_gray)
    roi_rgb = cv2.cvtColor(roi_enhanced, cv2.COLOR_GRAY2RGB)

    input_tensor = transform(image=roi_rgb)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        mask_roi = (probs > THRESHOLD).cpu().numpy().squeeze().astype(np.uint8)

    mask_roi_resized = cv2.resize(mask_roi, (ROI_W, ROI_H), interpolation=cv2.INTER_NEAREST)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)) 
    mask_cleaned = cv2.morphologyEx(mask_roi_resized, cv2.MORPH_CLOSE, kernel_close)
    
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros((ROI_H, ROI_W), dtype=np.uint8)
    
    K_MM_PER_PX = 0.206304
    has_crack = False
    current_frame_total_length = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        
        length_px = max(w, h)
        aspect_ratio = float(max(w, h)) / (min(w, h) + 1e-6)
        
        if area > 15 and aspect_ratio > 1.5: 
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)
            has_crack = True
            
            current_frame_total_length += (length_px * K_MM_PER_PX)
            
            box = np.intp(cv2.boxPoints(rect))
            cv2.drawContours(result[y1:y2, x1:x2], [box], 0, (0, 255, 255), 1)

    if has_crack:
        length_buffer.append(current_frame_total_length)
    else:
        length_buffer.append(0.0)

    smoothed_total_length = np.mean(length_buffer)

    if has_crack:
        roi_zone = result[y1:y2, x1:x2]
        red_layer = np.zeros_like(roi_zone)
        red_layer[:] = [0, 0, 255] 
        
        res_seg = cv2.bitwise_and(red_layer, red_layer, mask=final_mask)
        
        cv2.addWeighted(roi_zone, 1.0, res_seg, 0.3, 0, dst=roi_zone)

    color_status = (0, 0, 255) if has_crack else (0, 255, 0)
    cv2.rectangle(result, (x1, y1), (x2, y2), color_status, 2)
    
    cv2.rectangle(result, (0, 0), (240, 75), (0, 0, 0), -1)
    status_text = "DANGER: CRACK" if has_crack else "SAFE"
    cv2.putText(result, f"Status: {status_text}", (10, 30), 0, 0.6, color_status, 2)
    cv2.putText(result, f"Threshold: {THRESHOLD}", (10, 55), 0, 0.5, (200, 200, 200), 1)

    if smoothed_total_length > 1.0:
        label = f"Total Damage: {smoothed_total_length:.1f} mm"
        cv2.rectangle(result, (x1, y1 - 30), (x1 + 250, y1 - 5), (0, 0, 0), -1)
        cv2.putText(result, label, (x1 + 5, y1 - 12), 0, 0.6, (0, 255, 255), 2)

    cv2.imshow("Rail Inspection System - Optimized Total Mode", result)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
