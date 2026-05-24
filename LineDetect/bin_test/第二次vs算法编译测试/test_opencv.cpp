#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include "../core_lib/image.h"
#include <time.h>

using namespace cv;
using namespace std;

extern uint8 mt9v03x_image[image_h][image_w];

void capture_from_camera() {
    VideoCapture cap(0);
    
    if (!cap.isOpened()) {
        printf("无法打开摄像头\n");
        return;
    }
    
    printf("摄像头已打开，按 'ESC' 退出\n");
    
    Mat frame;
    Mat gray_frame;
    
    namedWindow("Line Detection", WINDOW_NORMAL);
    
    int frame_count = 0;
    clock_t start_time = clock();
    double fps = 0;
    
    while (true) {
        cap >> frame;
        
        if (frame.empty()) {
            printf("无法读取帧\n");
            break;
        }
        
        resize(frame, frame, Size(image_w, image_h));
        cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
        
        for (int i = 0; i < image_h; i++) {
            for (int j = 0; j < image_w; j++) {
                mt9v03x_image[i][j] = gray_frame.at<uchar>(i, j);
            }
        }
        
        image_process();
        
        Mat display_frame(image_h, image_w, CV_8UC3);
        
        for (int i = 0; i < image_h; i++) {
            for (int j = 0; j < image_w; j++) {
                uchar pixel = bin_image[i][j] ? 255 : 0;
                display_frame.at<Vec3b>(i, j)[0] = pixel;
                display_frame.at<Vec3b>(i, j)[1] = pixel;
                display_frame.at<Vec3b>(i, j)[2] = pixel;
            }
        }
        
        for (int i = 0; i < data_stastics_l; i++) {
            circle(display_frame, Point(points_l[i][0]+2, points_l[i][1]), 1, Scalar(255, 0, 0), 1);
        }
        
        for (int i = 0; i < data_stastics_r; i++) {
            circle(display_frame, Point(points_r[i][0]-2, points_r[i][1]), 1, Scalar(0, 0, 255), 1);
        }
        
        for (int i = hightest; i < image_h-1; i++) {
            if (center_line[i] > 0 && center_line[i] < image_w) {
                circle(display_frame, Point(center_line[i], i), 1, Scalar(0, 255, 0), 2);
            }
            if (l_border[i] > 0 && l_border[i] < image_w) {
                circle(display_frame, Point(l_border[i], i), 1, Scalar(255, 0, 0), 1);
            }
            if (r_border[i] > 0 && r_border[i] < image_w) {
                circle(display_frame, Point(r_border[i], i), 1, Scalar(0, 0, 255), 1);
            }
        }
        
        frame_count++;
        double current_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        if (current_time > 0) {
            fps = frame_count / current_time;
        }
        
        char text[50];
        sprintf_s(text, "FPS: %.2f", fps);
        putText(display_frame, text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        
        sprintf_s(text, "Time: %.2fs", current_time);
        putText(display_frame, text, Point(10, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        
        imshow("Line Detection", display_frame);
        
        if (waitKey(1) == 27) {
            break;
        }
    }
    
    cap.release();
    destroyAllWindows();
}

int main() {
    printf("SJTU 8-Area Line Detection Test (OpenCV)\n");
    printf("==========================================\n\n");
    
    printf("使用摄像头作为视频源\n");
    printf("按 'ESC' 键退出程序\n\n");
    
    capture_from_camera();
    
    return 0;
}
