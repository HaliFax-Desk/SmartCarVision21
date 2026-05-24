import cv2
import os
import numpy as np
import time

INPUT_DIR = '../bmp'
OUTPUT_DIR = '../output'
IMAGE_WIDTH = 180
IMAGE_HEIGHT = 122

global_img = None
global_edges = None

def Load_Image(filename):
    global global_img, global_edges
    
    input_path = os.path.join(INPUT_DIR, filename)
    
    global_img = cv2.imread(input_path)
    
    if global_img is None:
        return False
    
    global_img = cv2.resize(global_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    
    gray = cv2.cvtColor(global_img, cv2.COLOR_BGR2GRAY)
    
    global_edges = cv2.Canny(gray, 100, 200)
    
    return True

def process_images():
    start_time = time.time()
    
    if not os.path.exists(INPUT_DIR):
        print(f"\033[91mError: {INPUT_DIR} directory not found\033[0m")
        return
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    bmp_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.bmp')]
    
    if not bmp_files:
        print(f"\033[91mWarning: No .bmp files found in {INPUT_DIR}\033[0m")
        return
    
    print(f"Found {len(bmp_files)} BMP files")
    
    for filename in bmp_files:
        file_start_time = time.time()
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        print(f"Processing: {filename}")
        
        if not Load_Image(filename):
            print(f"\033[91mError: Cannot read {filename}\033[0m")
            continue
        
        result = global_img.copy()
        
        result[global_edges > 0] = [0, 0, 255]
        
        cv2.imwrite(output_path, result)
        file_end_time = time.time()
        file_duration = (file_end_time - file_start_time) * 1000
        print(f"Saved: {output_path}")
        print(f"Time: {file_duration:.0f}ms")
    
    end_time = time.time()
    total_duration = (end_time - start_time) * 1000
    print(f"Done! Total time: {total_duration:.0f}ms")

if __name__ == '__main__':
    process_images()
