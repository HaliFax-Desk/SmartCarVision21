#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>

// ==================== 类型定义 ====================
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;

// ==================== 全局常量 ====================
// 处理图像尺寸（resize后的尺寸）
const int UVC_WIDTH = 160;
const int UVC_HEIGHT = 120;

// 摄像头原始尺寸
const int CAMERA_WIDTH = 640;
const int CAMERA_HEIGHT = 480;

// 边界限制常量
const int BORDER_MAX = UVC_WIDTH - 2;
const int BORDER_MIN = 1;

// 边界点数组大小（保守估计为图像高度的3倍）
const int USE_NUM = UVC_HEIGHT * 3;

// 角度抑制参数
const int ANGLE_NMS_KERNEL = 11;  // 角度抑制核大小（必须为奇数）

// ==================== 全局变量 ====================
// 图像缓冲区
cv::Mat mt9v03x_image;
cv::Mat original_image;
cv::Mat bin_image;

// 阈值和边界线
int image_threshold = 0;
int l_border[UVC_HEIGHT];  // 左边界数组
int r_border[UVC_HEIGHT];  // 右边界数组
int center_line[UVC_HEIGHT];  // 中心线数组

// 中线偏移检测参数
const int MID_OFFSET_LINE = 40;    // 中线偏移检测行
const int OFFSET_DETECT_LINE = 5;  // 偏移检测行

// 中线偏移结果
int mid_offset = 40;
int detect_offset = 5;

// 边界点统计
int data_stastics_l = 0;
int data_stastics_r = 0;

// 边界点坐标数组
cv::Point points_l[USE_NUM];
cv::Point points_r[USE_NUM];

// 最高点位置
int hightest = 0;

// 起始点坐标
cv::Point start_point_l;
cv::Point start_point_r;

// 边界点搜索方向数组
int dir_r[USE_NUM];
int dir_l[USE_NUM];

// ==================== 系统函数替代 ====================
uint32 system_get_ms() {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
    return static_cast<uint32>(duration.count());
}

void system_delay_ms(uint32 ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// ==================== 图像二值化函数 ====================
void turn_to_bin() {
    cv::threshold(original_image, bin_image, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
}

// ==================== 搜线起始行函数 ====================
int get_start_point(int start_row) {
    int l_found = 0;
    int r_found = 0;

    for (int i = UVC_WIDTH / 2; i > BORDER_MIN; i--) {
        start_point_l.x = i;
        start_point_l.y = start_row;
        if (bin_image.at<uchar>(start_row, i) == 255 && bin_image.at<uchar>(start_row, i - 1) == 0) {
            l_found = 1;
            break;
        }
    }
    
    for (int i = UVC_WIDTH / 2; i < BORDER_MAX; i++) {
        start_point_r.x = i;
        start_point_r.y = start_row;
        if (bin_image.at<uchar>(start_row, i) == 255 && bin_image.at<uchar>(start_row, i + 1) == 0) {
            r_found = 1;
            break;
        }
    }
    
    return (l_found && r_found) ? 1 : 0;
}

// ==================== 补线函数 ====================
void add_line(int x1, int y1, int x2, int y2, int* line_array, bool is_left = true) {
    x1 = std::max(0, std::min(UVC_WIDTH - 1, x1));
    y1 = std::max(0, std::min(UVC_HEIGHT - 1, y1));
    x2 = std::max(0, std::min(UVC_WIDTH - 1, x2));
    y2 = std::max(0, std::min(UVC_HEIGHT - 1, y2));
    
    int a1 = y1, a2 = y2;
    if (a1 > a2) {
        std::swap(a1, a2);
        std::swap(x1, x2);
    }
    
    for (int i = a1; i <= a2; i++) {
        if (y2 == y1) {
            int hx = x1;
            line_array[i] = hx;
        } else {
            int hx = (i - y1) * (x2 - x1) / (y2 - y1) + x1;
            hx = std::max(0, std::min(UVC_WIDTH - 1, hx));
            line_array[i] = hx;
        }
    }
}

// ==================== 八领域巡线法 ====================
std::pair<int, int> Line8Area(int break_flag, const cv::Mat& image, int l_stastic, int r_stastic, int l_start_x, int l_start_y, int r_start_x, int r_start_y, int& hightest_ref) {
    for (int i = 0; i < UVC_HEIGHT; i++) {
        l_border[i] = BORDER_MIN;
        r_border[i] = BORDER_MAX;
    }
    
    int l_data_statics = l_stastic;
    int r_data_statics = r_stastic;
    
    cv::Point center_point_l(l_start_x, l_start_y);
    cv::Point center_point_r(r_start_x, r_start_y);
    
    cv::Point seeds_l[] = { {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1} };
    cv::Point seeds_r[] = { {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1} };
    
    while (break_flag > 0) {
        break_flag--;
        
        cv::Point search_filds_l[8];
        for (int i = 0; i < 8; i++) {
            int x = center_point_l.x + seeds_l[i].x;
            int y = center_point_l.y + seeds_l[i].y;
            x = std::max(0, std::min(UVC_WIDTH - 1, x));
            y = std::max(0, std::min(UVC_HEIGHT - 1, y));
            search_filds_l[i] = cv::Point(x, y);
        }
        
        points_l[l_data_statics] = center_point_l;
        l_data_statics++;
        
        cv::Point search_filds_r[8];
        for (int i = 0; i < 8; i++) {
            int x = center_point_r.x + seeds_r[i].x;
            int y = center_point_r.y + seeds_r[i].y;
            x = std::max(0, std::min(UVC_WIDTH - 1, x));
            y = std::max(0, std::min(UVC_HEIGHT - 1, y));
            search_filds_r[i] = cv::Point(x, y);
        }
        
        points_r[r_data_statics] = center_point_r;
        
        cv::Point temp_l[8];
        int index_l = 0;
        
        for (int i = 0; i < 8; i++) {
            if (image.at<uchar>(search_filds_l[i].y, search_filds_l[i].x) == 0 &&
                image.at<uchar>(search_filds_l[(i + 1) & 7].y, search_filds_l[(i + 1) & 7].x) == 255) {
                temp_l[index_l] = search_filds_l[i];
                index_l++;
                dir_l[l_data_statics - 1] = i;
            }
        }
        
        if (index_l > 0) {
            center_point_l = temp_l[0];
            for (int j = 0; j < index_l; j++) {
                if (center_point_l.y > temp_l[j].y) {
                    center_point_l = temp_l[j];
                }
            }
        }
        
        if ((r_data_statics >= 2 && 
             points_r[r_data_statics] == points_r[r_data_statics - 1] && 
             points_r[r_data_statics] == points_r[r_data_statics - 2]) ||
            (l_data_statics >= 3 &&
             points_l[l_data_statics - 1] == points_l[l_data_statics - 2] &&
             points_l[l_data_statics - 1] == points_l[l_data_statics - 3])) {
            break;
        }
        
        if (abs(points_r[r_data_statics].x - points_l[l_data_statics - 1].x) < 2 &&
            abs(points_r[r_data_statics].y - points_l[l_data_statics - 1].y) < 2) {
            hightest_ref = (points_r[r_data_statics].y + points_l[l_data_statics - 1].y) / 2;
            break;
        }
        
        if (points_r[r_data_statics].y < points_l[l_data_statics - 1].y) {
            continue;
        }
        
        if (dir_l[l_data_statics - 1] == 7 &&
            points_r[r_data_statics].y > points_l[l_data_statics - 1].y) {
            center_point_l = points_l[l_data_statics - 1];
            l_data_statics--;
        }
        
        r_data_statics++;
        
        cv::Point temp_r[8];
        int index_r = 0;
        
        for (int i = 0; i < 8; i++) {
            if (image.at<uchar>(search_filds_r[i].y, search_filds_r[i].x) == 0 &&
                image.at<uchar>(search_filds_r[(i + 1) & 7].y, search_filds_r[(i + 1) & 7].x) == 255) {
                temp_r[index_r] = search_filds_r[i];
                index_r++;
                dir_r[r_data_statics - 1] = i;
            }
        }
        
        if (index_r > 0) {
            center_point_r = temp_r[0];
            for (int j = 0; j < index_r; j++) {
                if (center_point_r.y > temp_r[j].y) {
                    center_point_r = temp_r[j];
                }
            }
        }
    }
    
    if (l_data_statics >= 2) {
        int h = UVC_HEIGHT - 2;
        cv::Point prev_point = cv::Point(-1, -1);
        
        for (int j = 0; j < l_data_statics; j++) {
            if (points_l[j].y == h) {
                l_border[h] = points_l[j].x + 1;
                
                if (prev_point.x != -1) {
                    int x1 = prev_point.x;
                    int y1 = prev_point.y;
                    int x2 = points_l[j].x + 1;
                    int y2 = h;
                    if (abs(y1 - y2) > 1) {
                        add_line(x1, y1, x2, y2, l_border, true);
                    }
                }
                
                prev_point = cv::Point(points_l[j].x + 1, h);
            }
            h--;
            if (h == 0) {
                break;
            }
        }
        
        if (prev_point.x != -1 && h > 0) {
            int x1 = prev_point.x;
            int y1 = prev_point.y;
            add_line(x1, y1, l_border[h + 1], h + 1, l_border, true);
        }
    }
    
    if (r_data_statics >= 2) {
        int h = UVC_HEIGHT - 2;
        cv::Point prev_point = cv::Point(-1, -1);
        
        for (int j = 0; j < r_data_statics; j++) {
            if (points_r[j].y == h) {
                r_border[h] = points_r[j].x - 1;
                
                if (prev_point.x != -1) {
                    int x1 = prev_point.x;
                    int y1 = prev_point.y;
                    int x2 = points_r[j].x - 1;
                    int y2 = h;
                    if (abs(y1 - y2) > 1) {
                        add_line(x1, y1, x2, y2, r_border, false);
                    }
                }
                
                prev_point = cv::Point(points_r[j].x - 1, h);
            }
            h--;
            if (h == 0) {
                break;
            }
        }
        
        if (prev_point.x != -1 && h > 0) {
            int x1 = prev_point.x;
            int y1 = prev_point.y;
            add_line(x1, y1, r_border[h + 1], h + 1, r_border, false);
        }
    }
    
    return {l_data_statics, r_data_statics};
}

// ==================== 中线偏移检测函数 ====================
void get_mid_offset() {
    int screen_mid = UVC_WIDTH / 2;
    
    int mid_line = UVC_HEIGHT - MID_OFFSET_LINE;
    int detect_line = UVC_HEIGHT - OFFSET_DETECT_LINE;
    
    mid_line = std::max(0, std::min(UVC_HEIGHT - 1, mid_line));
    detect_line = std::max(0, std::min(UVC_HEIGHT - 1, detect_line));
    
    mid_offset = center_line[mid_line] - screen_mid;
    detect_offset = center_line[detect_line] - screen_mid;
}

// ==================== 计算中心线函数 ====================
void calculate_center_line() {
    for (int i = hightest; i < UVC_HEIGHT - 1; i++) {
        center_line[i] = (l_border[i] + r_border[i]) / 2;
    }
}

// ==================== 显示巡线结果函数 ====================
void display_line_result(cv::Mat& color_image) {
    for (int i = 0; i < UVC_HEIGHT - 1; i++) {
        if (l_border[i] > 0 && l_border[i] < UVC_WIDTH - 1) {
            color_image.at<cv::Vec3b>(i, l_border[i]) = cv::Vec3b(0, 255, 0);
        }
        if (r_border[i] > 0 && r_border[i] < UVC_WIDTH - 1) {
            color_image.at<cv::Vec3b>(i, r_border[i]) = cv::Vec3b(0, 255, 0);
        }
    }
    
    for (int i = 0; i < UVC_HEIGHT - 1; i++) {
        if (center_line[i] > 0 && center_line[i] < UVC_WIDTH - 1) {
            color_image.at<cv::Vec3b>(i, center_line[i]) = cv::Vec3b(0, 0, 255);
        }
    }
    
    int screen_mid = UVC_WIDTH / 2;
    for (int i = 0; i < UVC_HEIGHT - 1; i++) {
        color_image.at<cv::Vec3b>(i, screen_mid) = cv::Vec3b(0, 255, 255);
    }
    
    int mid_line = UVC_HEIGHT - MID_OFFSET_LINE;
    int detect_line = UVC_HEIGHT - OFFSET_DETECT_LINE;
    
    mid_line = std::max(0, std::min(UVC_HEIGHT - 1, mid_line));
    detect_line = std::max(0, std::min(UVC_HEIGHT - 1, detect_line));
    
    if (center_line[mid_line] > 0 && center_line[mid_line] < UVC_WIDTH - 1) {
        for (int x = std::min(screen_mid, center_line[mid_line]); x <= std::max(screen_mid, center_line[mid_line]); x++) {
            color_image.at<cv::Vec3b>(mid_line, x) = cv::Vec3b(0, 255, 255);
        }
    }
}

// ==================== 主函数 ====================
int main() {
    cv::VideoCapture cap(0);
    
    if (!cap.isOpened()) {
        printf("无法打开摄像头!\n");
        return -1;
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);
    
    uint32 frame_count = 0;
    uint32 start_time = system_get_ms();
    char fps_str[32];
    char version_str[] = "LabVersion1.1";
    
    mt9v03x_image = cv::Mat(UVC_HEIGHT, UVC_WIDTH, CV_8UC1);
    original_image = cv::Mat(UVC_HEIGHT, UVC_WIDTH, CV_8UC1);
    bin_image = cv::Mat(UVC_HEIGHT, UVC_WIDTH, CV_8UC1);
    
    while (true) {
        cv::Mat frame;
        cap >> frame;
        
        if (frame.empty()) {
            printf("无法获取帧!\n");
            break;
        }
        
        cv::Mat gray_frame;
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        
        cv::Mat resized_gray;
        cv::resize(gray_frame, resized_gray, cv::Size(UVC_WIDTH, UVC_HEIGHT));
        
        original_image = resized_gray.clone();
        
        turn_to_bin();
        
        data_stastics_l = 0;
        data_stastics_r = 0;
        hightest = 0;
        
        if (get_start_point(UVC_HEIGHT - 2)) {
            auto result = Line8Area(USE_NUM, bin_image, data_stastics_l, data_stastics_r,
                       start_point_l.x, start_point_l.y, start_point_r.x, start_point_r.y, hightest);
            
            data_stastics_l = result.first;
            data_stastics_r = result.second;
            
            calculate_center_line();
            
            get_mid_offset();
        }
        
        cv::Mat color_image;
        cv::cvtColor(bin_image, color_image, cv::COLOR_GRAY2BGR);
        
        display_line_result(color_image);
        
        cv::putText(color_image, version_str, cv::Point(0, 174), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        
        frame_count++;
        uint32 current_time = system_get_ms();
        if (current_time - start_time >= 1000) {
            float fps = frame_count * 1000.0f / (current_time - start_time);
            sprintf(fps_str, "FPS: %.1f", fps);
            frame_count = 0;
            start_time = current_time;
        }
        cv::putText(color_image, fps_str, cv::Point(0, 194), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        
        cv::imshow("Line Detection", color_image);
        
        int key = cv::waitKey(1);
        if (key == 27) {
            break;
        }
        
        system_delay_ms(10);
    }
    
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}
