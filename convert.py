import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm


ROOT_DIR = r'D:\Python\Rail\NEU-DET' 

def convert_all_folders():
    folders = ['train', 'val']
    
    for folder_name in folders:
        print(f"\n[PHASE] Đang xử lý thư mục: {folder_name}")
        
        current_base = os.path.join(ROOT_DIR, folder_name)
        xml_dir = os.path.join(current_base, 'annotations')
        mask_out_dir = os.path.join(current_base, 'masks')

        if not os.path.exists(xml_dir):
            print(f"BỎ QUA: Không tìm thấy thư mục {xml_dir}")
            continue

        if not os.path.exists(mask_out_dir):
            os.makedirs(mask_out_dir)

        xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
        
        for xml_file in tqdm(xml_files, desc=f"Converting {folder_name}"):
            try:
                tree = ET.parse(os.path.join(xml_dir, xml_file))
                root = tree.getroot()

                width = int(root.find('size/width').text)
                height = int(root.find('size/height').text)

                mask = np.zeros((height, width), dtype=np.uint8)

                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    
                    cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, -1)

                mask_name = os.path.splitext(xml_file)[0] + ".png"
                cv2.imwrite(os.path.join(mask_out_dir, mask_name), mask)
                
            except Exception as e:
                print(f"Lỗi tại {xml_file}: {e}")

    print(f"\n--- HOÀN TẤT CẢ TRAIN VÀ VAL! ---")

if __name__ == '__main__':
    convert_all_folders()
