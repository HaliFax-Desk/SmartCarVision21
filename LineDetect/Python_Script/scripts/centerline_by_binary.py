import cv2
import os
import numpy as np
import time

INPUT_DIR = '../bmp'
OUTPUT_DIR = '../output'
BINARY_THRESHOLD = 70
IMAGE_WIDTH = 180
IMAGE_HEIGHT = 122

global_img = None
global_binary = None
global_reference_col = 0
global_left_boundary_count = 0
global_right_boundary_count = 0

def Load_Image(filename):
    global global_img, global_binary, global_reference_col
    global global_left_boundary_count, global_right_boundary_count
    
    input_path = os.path.join(INPUT_DIR, filename)
    
    global_img = cv2.imread(input_path)
    
    if global_img is None:
        return False
    
    global_img = cv2.resize(global_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    
    gray = cv2.cvtColor(global_img, cv2.COLOR_BGR2GRAY)
    
    _, global_binary = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    binary_output_path = os.path.join(OUTPUT_DIR, f"binary_{filename}")
    cv2.imwrite(binary_output_path, global_binary)
    
    column_white_counts = np.sum(global_binary > 0, axis=0)
    global_reference_col = np.argmax(column_white_counts)
    
    global_left_boundary_count = 0
    global_right_boundary_count = 0
    
    return True

def process_images():
    start_time = time.time()
    
    bmp_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.bmp')]
    
    for filename in bmp_files:
        file_start_time = time.time()
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        print(f"Processing: {filename}")
        
        filtered_edges = np.zeros_like(global_binary)
        
        left_boundary_points = []
        right_boundary_points = []
        center_line_points = []
        
        for row in range(IMAGE_HEIGHT - 1, -1, -1):
            left_found = False
            right_found = False
            left_x = global_reference_col
            right_x = global_reference_col
            
            for offset in range(0, max(global_reference_col, IMAGE_WIDTH - global_reference_col)):
                if not left_found and global_reference_col - offset >= 0:
                    if global_binary[row, global_reference_col - offset] > 0:
                        left_x = global_reference_col - offset
                        left_found = True
                        filtered_edges[row, left_x] = 255
                        left_boundary_points.append((left_x, row))
                        global_left_boundary_count += 1
                
                if not right_found and global_reference_col + offset < IMAGE_WIDTH:
                    if global_binary[row, global_reference_col + offset] > 0:
                        right_x = global_reference_col + offset
                        right_found = True
                        filtered_edges[row, right_x] = 255
                        right_boundary_points.append((right_x, row))
                        global_right_boundary_count += 1
                
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
