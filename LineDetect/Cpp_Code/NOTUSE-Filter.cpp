#include "zf_common_headfile.hpp"

// ==================== 全局常量 ====================
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

// 设备对象
zf_device_ips200 ips200;
zf_device_uvc uvc_dev;

// ==================== 图像二值化函数 ====================
void turn_to_bin() {
    /*使用OpenCV的Otsu算法将original_image二值化到bin_image*/
    // 使用OpenCV的Otsu阈值分割，自动计算最优阈值并进行二值化
    cv::threshold(original_image, bin_image, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
}

// ==================== 搜线起始行函数 ====================
int get_start_point(int start_row) {
    /*
    在指定行寻找左右边界的起始点
    
    参数:
        start_row: 起始行
    
    返回:
        成功找到返回1，否则返回0
    */
    int l_found = 0;
    int r_found = 0;

    // 从中间向左寻找左边界起始点（从255变为0的位置）
    for (int i = UVC_WIDTH / 2; i > BORDER_MIN; i--) {
        start_point_l.x = i;
        start_point_l.y = start_row;
        if (bin_image.at<uchar>(start_row, i) == 255 && bin_image.at<uchar>(start_row, i - 1) == 0) {
            l_found = 1;
            break;
        }
    }
    
    // 从中间向右寻找右边界起始点（从255变为0的位置）
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
    /*
    在两点之间插值填充边界线
    
    参数:
        x1, y1: 起点坐标
        x2, y2: 终点坐标
        line_array: 边界数组
        is_left: 是否为左边界（True=左边界，False=右边界）
    */
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
    /*
    使用8邻域边界搜索算法搜索左右边界，并转换为每行边界值数组
    
    参数:
        break_flag: 搜索停止条件
        image: 二值图像
        l_stastic: 左边界点计数
        r_stastic: 右边界点计数
        l_start_x, l_start_y: 左边界起始点坐标
        r_start_x, r_start_y: 右边界起始点坐标
        hightest_ref: 最高点引用
    
    返回:
        左右边界点计数
    */
    // 初始化边界数组
    for (int i = 0; i < UVC_HEIGHT; i++) {
        l_border[i] = BORDER_MIN;
        r_border[i] = BORDER_MAX;
    }
    
    int l_data_statics = l_stastic;
    int r_data_statics = r_stastic;
    
    cv::Point center_point_l(l_start_x, l_start_y);
    cv::Point center_point_r(r_start_x, r_start_y);
    
    // 8邻域搜索方向（左边界：逆时针，右边界：顺时针）
    cv::Point seeds_l[] = { {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1} };
    cv::Point seeds_r[] = { {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1} };
    
    // 搜索边界点
    while (break_flag > 0) {
        break_flag--;
        
        // 计算左边界8邻域位置
        cv::Point search_filds_l[8];
        for (int i = 0; i < 8; i++) {
            int x = center_point_l.x + seeds_l[i].x;
            int y = center_point_l.y + seeds_l[i].y;
            x = std::max(0, std::min(UVC_WIDTH - 1, x));
            y = std::max(0, std::min(UVC_HEIGHT - 1, y));
            search_filds_l[i] = cv::Point(x, y);
        }
        
        // 记录左边界点
        points_l[l_data_statics] = center_point_l;
        l_data_statics++;
        
        // 计算右边界8邻域位置
        cv::Point search_filds_r[8];
        for (int i = 0; i < 8; i++) {
            int x = center_point_r.x + seeds_r[i].x;
            int y = center_point_r.y + seeds_r[i].y;
            x = std::max(0, std::min(UVC_WIDTH - 1, x));
            y = std::max(0, std::min(UVC_HEIGHT - 1, y));
            search_filds_r[i] = cv::Point(x, y);
        }
        
        // 记录右边界点
        points_r[r_data_statics] = center_point_r;
        
        // 寻找左边界下一个点（边界点：当前为0，右侧为255）
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
        
        // 更新左边界中心点
        if (index_l > 0) {
            center_point_l = temp_l[0];
            for (int j = 0; j < index_l; j++) {
                if (center_point_l.y > temp_l[j].y) {
                    center_point_l = temp_l[j];
                }
            }
        }
        
        // 检查是否重复（停止条件）
        if ((r_data_statics >= 2 && 
             points_r[r_data_statics] == points_r[r_data_statics - 1] && 
             points_r[r_data_statics] == points_r[r_data_statics - 2]) ||
            (l_data_statics >= 3 &&
             points_l[l_data_statics - 1] == points_l[l_data_statics - 2] &&
             points_l[l_data_statics - 1] == points_l[l_data_statics - 3])) {
            break;
        }
        
        // 检查是否相遇（计算最高点）
        if (abs(points_r[r_data_statics].x - points_l[l_data_statics - 1].x) < 2 &&
            abs(points_r[r_data_statics].y - points_l[l_data_statics - 1].y) < 2) {
            hightest_ref = (points_r[r_data_statics].y + points_l[l_data_statics - 1].y) / 2;
            break;
        }
        
        // 如果右边界点在左边界点上方，继续搜索
        if (points_r[r_data_statics].y < points_l[l_data_statics - 1].y) {
            continue;
        }
        
        // 检查是否需要回退（特殊处理）
        if (dir_l[l_data_statics - 1] == 7 &&
            points_r[r_data_statics].y > points_l[l_data_statics - 1].y) {
            center_point_l = points_l[l_data_statics - 1];
            l_data_statics--;
        }
        
        r_data_statics++;
        
        // 寻找右边界下一个点
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
        
        // 更新右边界中心点
        if (index_r > 0) {
            center_point_r = temp_r[0];
            for (int j = 0; j < index_r; j++) {
                if (center_point_r.y > temp_r[j].y) {
                    center_point_r = temp_r[j];
                }
            }
        }
    }
    
    // 将边界点转换为每行边界值
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
    /**
     * @brief 检测特定行的中线与屏幕中线的差值
     *        并将结果写入全局变量mid_offset和detect_offset
     */
    // 1. 计算屏幕中线
    int screen_mid = UVC_WIDTH / 2;
    
    // 2. 计算目标行号
    int mid_line = UVC_HEIGHT - MID_OFFSET_LINE;
    int detect_line = UVC_HEIGHT - OFFSET_DETECT_LINE;
    
    // 3. 确保行号在有效范围内
    mid_line = std::max(0, std::min(UVC_HEIGHT - 1, mid_line));
    detect_line = std::max(0, std::min(UVC_HEIGHT - 1, detect_line));
    
    // 4. 计算偏移值并写入全局变量
    mid_offset = center_line[mid_line] - screen_mid;
    detect_offset = center_line[detect_line] - screen_mid;
}

// ==================== 计算中心线函数 ====================
void calculate_center_line() {
    /*计算中心线函数*/
    // 从最高点到图像底部，计算每行的中心线
    // 中心线 = (左边界 + 右边界) / 2
    for (int i = hightest; i < UVC_HEIGHT - 1; i++) {
        center_line[i] = (l_border[i] + r_border[i]) / 2;
    }
}

// ==================== 显示巡线结果函数 ====================
void display_line_result() {
    /*显示巡线结果函数*/
    // 1. 创建彩色图像
    cv::Mat color_image;
    cv::cvtColor(bin_image, color_image, cv::COLOR_GRAY2BGR);
    
    // 2. 画左右边线（绿色）
    for (int i = 0; i < UVC_HEIGHT - 1; i++) {
        if (l_border[i] > 0 && l_border[i] < UVC_WIDTH - 1) {
            color_image.at<cv::Vec3b>(i, l_border[i]) = cv::Vec3b(0, 255, 0);
        }
        if (r_border[i] > 0 && r_border[i] < UVC_WIDTH - 1) {
            color_image.at<cv::Vec3b>(i, r_border[i]) = cv::Vec3b(0, 255, 0);
        }
    }
    
    // 3. 画中心线（红色）
    for (int i = 0; i < UVC_HEIGHT - 1; i++) {
        if (center_line[i] > 0 && center_line[i] < UVC_WIDTH - 1) {
            color_image.at<cv::Vec3b>(i, center_line[i]) = cv::Vec3b(0, 0, 255);
        }
    }
    
    // 4. 画画面中心线（黄色）
    int screen_mid = UVC_WIDTH / 2;
    for (int i = 0; i < UVC_HEIGHT - 1; i++) {
        color_image.at<cv::Vec3b>(i, screen_mid) = cv::Vec3b(0, 255, 255);
    }
    
    // 5. 画偏移值（黄色）
    int mid_line = UVC_HEIGHT - MID_OFFSET_LINE;
    int detect_line = UVC_HEIGHT - OFFSET_DETECT_LINE;
    
    // 确保行号在有效范围内
    mid_line = std::max(0, std::min(UVC_HEIGHT - 1, mid_line));
    detect_line = std::max(0, std::min(UVC_HEIGHT - 1, detect_line));
    
    // 画偏移线
    if (center_line[mid_line] > 0 && center_line[mid_line] < UVC_WIDTH - 1) {
        for (int x = std::min(screen_mid, center_line[mid_line]); x <= std::max(screen_mid, center_line[mid_line]); x++) {
            color_image.at<cv::Vec3b>(mid_line, x) = cv::Vec3b(0, 255, 255);
        }
    }

    // 6. 转换为RGB565格式并显示
    cv::Mat rgb565_image(UVC_HEIGHT, UVC_WIDTH, CV_16UC1);
    
    for (int i = 0; i < UVC_HEIGHT; i++) {
        for (int j = 0; j < UVC_WIDTH; j++) {
            cv::Vec3b bgr = color_image.at<cv::Vec3b>(i, j);
            uint8 r = bgr[2];
            uint8 g = bgr[1];
            uint8 b = bgr[0];
            
            // 转换为RGB565格式
            uint16 rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
            rgb565_image.at<uint16>(i, j) = rgb565;
        }
    }
    
    // 传入转换完成的RGB565图像数据，严格匹配函数参数类型
    ips200.displayimage_rgb565(reinterpret_cast<const uint16_t*>(rgb565_image.data), 
                          static_cast<uint16_t>(UVC_WIDTH), 
                          static_cast<uint16_t>(UVC_HEIGHT));
}

// ==================== 主函数 ====================
int main() {
    // 初始化设备
    ips200.init(FB_PATH);
    
    // 显示版本信息
    ips200.show_string(0,174, "LabVersion1.1");

    // 帧率统计变量
    uint32 frame_count = 0;
    uint32 start_time = system_get_ms();
    char fps_str[32];

    if (uvc_dev.init(UVC_PATH) < 0) {
        return -1;
    }
    
    // 初始化图像缓冲区
    mt9v03x_image = cv::Mat(UVC_HEIGHT, UVC_WIDTH, CV_8UC1);
    original_image = cv::Mat(UVC_HEIGHT, UVC_WIDTH, CV_8UC1);
    bin_image = cv::Mat(UVC_HEIGHT, UVC_WIDTH, CV_8UC1);
    
    // 主循环
    while (1) {
        if (uvc_dev.wait_image_refresh() == 0) {
            // 获取灰度图像
            uint8* gray_image = uvc_dev.get_gray_image_ptr();
            
            if (gray_image != NULL) {
                // 将灰度图像转换为cv::Mat
                original_image = cv::Mat(UVC_HEIGHT, UVC_WIDTH, CV_8UC1, gray_image);
                
                // 二值化处理
                turn_to_bin();
                
                // 1. 初始化边界点统计计数器
                data_stastics_l = 0;  // 左边界点计数器清零
                data_stastics_r = 0;  // 右边界点计数器清零
                hightest = 0;         // 最高点计数器清零
                
                // 2. 在图像底部寻找左右边界的起始点
                //    从倒数第二行开始搜索（UVC_HEIGHT - 2）
                if (get_start_point(UVC_HEIGHT - 2)) {
                    // 3. 使用八领域巡线算法搜索左右边界
                    //    参数说明：
                    //    USE_NUM - 最大搜索次数
                    //    bin_image - 二值化图像
                    //    data_stastics_l, data_stastics_r - 边界点计数器
                    //    start_point_l, start_point_r - 起始点坐标
                    //    hightest - 最高点引用（返回值）
                    auto result = Line8Area(USE_NUM, bin_image, data_stastics_l, data_stastics_r,
                               start_point_l.x, start_point_l.y, start_point_r.x, start_point_r.y, hightest);
                    
                    // 4. 更新边界点计数器
                    data_stastics_l = result.first;  // 左边界点数量
                    data_stastics_r = result.second; // 右边界点数量
                    
                    // 5. 计算中心线
                    calculate_center_line();
                    
                    // 6. 更新中线偏移值
                    get_mid_offset();
                }
                
                // 显示巡线结果
                display_line_result();
                
                // 统计帧率
                frame_count++;
                uint32 current_time = system_get_ms();
                if (current_time - start_time >= 1000) {
                    // 计算帧率
                    float fps = frame_count * 1000.0f / (current_time - start_time);
                    sprintf(fps_str, "FPS: %.1f", fps);
                    
                    // 显示帧率
                    ips200.show_string(0, 194, fps_str);
                    
                    // 重置统计
                    frame_count = 0;
                    start_time = current_time;
                }
            }
        }
        
        // 延时
        system_delay_ms(10);
    }
    
    return 0;
}