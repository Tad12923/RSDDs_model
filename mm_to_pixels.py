import cv2
import numpy as np

# --- CẤU HÌNH ---
ROI_W, ROI_H = 350, 200
# Vật mẫu thực tế bạn đặt vào (ví dụ: 50mm)
REAL_SIZE_MM = 50.0 

cap = cv2.VideoCapture(1) # Thử 0 hoặc 1

print("--- CHẾ ĐỘ HIỆU CHUẨN (CALIBRATION MODE) ---")
print(f"1. Đặt vật mẫu dài {REAL_SIZE_MM}mm vào mặt ray.")
print(f"2. Đảm bảo khoảng cách từ cam đến vật là 9.3cm.")
print("3. Nhấn 's' để tính toán hệ số K, nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    h, w = frame.shape[:2]
    x1, y1 = (w - ROI_W) // 2, (h - ROI_H) // 2
    x2, y2 = x1 + ROI_W, y1 + ROI_H
    roi = frame[y1:y2, x1:x2]

    # Xử lý ảnh cơ bản để tìm vật mẫu
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Tìm đường biên
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pixel_length = 0
    if contours:
        # Lấy vật có diện tích lớn nhất (vật mẫu của bạn)
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 500:
            rect = cv2.minAreaRect(cnt)
            box = np.intp(cv2.boxPoints(rect))
            cv2.drawContours(roi, [box], 0, (0, 255, 0), 2)
            
            # Lấy cạnh dài nhất làm chiều dài pixel
            pixel_length = max(rect[1])
            
            cv2.putText(frame, f"Pixel Length: {pixel_length:.2f} px", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Vẽ khung ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow("Calibration", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and pixel_length > 0:
        # TÍNH TOÁN HỆ SỐ K
        k_value = REAL_SIZE_MM / pixel_length
        px_per_mm = pixel_length / REAL_SIZE_MM
        
        print("\n--- KẾT QUẢ HIỆU CHUẨN ---")
        print(f"Chieu dai vat mau: {pixel_length:.2f} pixel")
        print(f"1mm ung voi: {px_per_mm:.2f} pixel")
        print(f"Hệ số K_MM_PER_PX bạn cần nhap vao code chinh la: {k_value:.5f}")
        print("--------------------------\n")
        
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()