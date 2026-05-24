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
        
        center_x = IMAGE_WIDTH // 2
        
        filtered_edges = np.zeros_like(global_edges)
        
        left_boundary_points = []
        right_boundary_points = []
        center_line_points = []
        
        for row in range(IMAGE_HEIGHT - 1, -1, -1):
            left_found = False
            right_found = False
            left_x = center_x
            right_x = center_x
            
            for offset in range(0, max(center_x, IMAGE_WIDTH - center_x)):
                if not left_found and center_x - offset >= 0:
                    if global_edges[row, center_x - offset] > 0:
                        left_x = center_x - offset
                        left_found = True
                        filtered_edges[row, left_x] = 255
                        left_boundary_points.append((left_x, row))
                
                if not right_found and center_x + offset < IMAGE_WIDTH:
                    if global_edges[row, center_x + offset] > 0:
                        right_x = center_x + offset
                        right_found = True
                        filtered_edges[row, right_x] = 255
                        right_boundary_points.append((right_x, row))
                
                if left_found and right_found:
                    break
            
            if left_found and right_found:
                center_x_avg = (left_x + right_x) // 2
                center_line_points.append((center_x_avg, row))
        
        result = global_img.copy()
        
        result[filtered_edges > 0] = [0, 0, 255]
        
        for x, y in center_line_points:
            cv2.circle(result, (x, y), 1, (0, 255, 0), -1)
        
        cv2.imwrite(output_path, result)
        file_end_time = time.time()
        file_duration = (file_end_time - file_start_time) * 1000
        print(f"Saved: {output_path}")
        print(f"Left boundary points: {len(left_boundary_points)}")
        print(f"Right boundary points: {len(right_boundary_points)}")
        print(f"Center line points: {len(center_line_points)}")
        print(f"Time: {file_duration:.0f}ms")
    
    end_time = time.time()
    total_duration = (end_time - start_time) * 1000
    print(f"Done! Total time: {total_duration:.0f}ms")

if __name__ == '__main__':
    process_images()
