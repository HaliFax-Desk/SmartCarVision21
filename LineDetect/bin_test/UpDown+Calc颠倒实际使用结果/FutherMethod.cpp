/* ============================================================================
 *  文件名: FutherMethod.cpp
 *  功  能: 基于八邻域算法的实时巡线系统（Windows 版本）
 *         算法层：OTSU + 形态学滤波 + 八邻域追踪
 *         驱动层：OpenCV VideoCapture + imshow
 *
 *  整体流程（一帧）:
 *    摄像头采集(resize 160x120) → 灰度转换 → OTSU二值化 →
 *    形态学滤波 → 搜索起点 → 八邻域追踪 → 提取逐行边界 →
 *    计算中线 → 偏移检测 → imshow 显示
 *
 *  依赖: OpenCV 4.x
 *  平台: Windows
 * ============================================================================ */

#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstdlib>
#include <algorithm>

// 自定义数据类型
#ifndef _INT8_DEFINED_
#define _INT8_DEFINED_
typedef   signed          char int8;
typedef   signed short     int int16;
typedef   signed           int int32;
typedef unsigned          char uint8;
typedef unsigned short     int uint16;
typedef unsigned           int uint32;
#endif

// 图像尺寸
#define UVC_WIDTH   160
#define UVC_HEIGHT  120

// 算法参数
#define white_pixel  255             // 二值图中白色的灰度值
#define black_pixel  0               // 二值图中黑色的灰度值

#define BORDER_MAX  (UVC_WIDTH - 2)  // 边界搜索最大值（预留 2 像素黑框）
#define BORDER_MIN  1                // 边界搜索最小值（预留黑框）

#define USE_NUM  (UVC_HEIGHT * 3)    // 八邻域追踪点数组容量（360 点，保守估算）

#define THRESHOLD_MAX  255 * 5       // 形态学膨胀阈值：周围 8 点像素和 >= 此值时填充为白
#define THRESHOLD_MIN  255 * 2       // 形态学腐蚀阈值：周围 8 点像素和 <= 此值时置为黑

// 中线偏移检测参数
const int MID_OFFSET_LINE    = 40;    // 远行（40 行处）偏移检测
const int OFFSET_DETECT_LINE = 10;    // 近行（10 行处）偏移检测
const int SEARCH_MAX_LINES   = 10;    // 起始搜索最多向上查找行数

// 全局图像数据（裸二维数组，算法层统一使用，步长 = UVC_WIDTH）
uint8 original_image[UVC_HEIGHT][UVC_WIDTH];  // 原始灰度图像（uvc 采集后拷贝到此）
uint8 bin_image[UVC_HEIGHT][UVC_WIDTH];       // 二值化后的图像（0 = 黑/赛道外，255 = 白/赛道内）
uint8 image_thereshold;                       // OTSU 算法自动计算出的二值化分割阈值

// 赛道边界与中线数据（逐行存储，索引为行号 y，0 ≤ y < UVC_HEIGHT）
int l_border[UVC_HEIGHT];     // 左边界 x 坐标数组
int r_border[UVC_HEIGHT];     // 右边界 x 坐标数组
int center_line[UVC_HEIGHT];  // 中线 x 坐标数组，center_line[y] = (l + r) / 2

// 中线偏移检测结果
int mid_offset    = 40;    // 远行中线与屏幕中心的偏差
int detect_offset = 5;     // 近行中线与屏幕中心的偏差

// 八邻域边界追踪数据
int data_stastics_l = 0;   // 八邻域追踪到的左边界点数
int data_stastics_r = 0;   // 八邻域追踪到的右边界点数
int hightest = 0;          // 八邻域追踪达到的最高行号（y 最小值，越往上 y 越小）

uint16 points_l[USE_NUM][2];  // 左边界追踪点坐标集 [序号][0=x, 1=y]
uint16 points_r[USE_NUM][2];  // 右边界追踪点坐标集 [序号][0=x, 1=y]

int    dir_r[USE_NUM];        // 右边生长方向记录（0~7 对应 8 个邻域方向）
int    dir_l[USE_NUM];        // 左边生长方向记录

// 边界起始搜索点
uint8 start_point_l[2] = { 0 };  // 左边起点 [0]=x 坐标, [1]=y 坐标
uint8 start_point_r[2] = { 0 };  // 右边起点 [0]=x 坐标, [1]=y 坐标
uint8 actual_start_row  = 0;     // 实际搜索到的起始行号

/*
 * 函数名称: turn_to_bin
 * 功能说明: 使用 OpenCV 的 OTSU 阈值对图像进行二值化
 *           像素值 > 阈值 → 白色(255, 赛道内部)
 *           像素值 ≤ 阈值 → 黑色(0,   赛道外部)
 *           操作全局数组 original_image → bin_image
 */
void turn_to_bin(void)
{
    // 步骤 1: 将裸数组 original_image 包装为 cv::Mat
    cv::Mat gray_mat(UVC_HEIGHT, UVC_WIDTH, CV_8UC1, original_image);
    // 步骤 2: 使用 OpenCV 的 OTSU 二值化
    cv::Mat bin_mat;
    image_thereshold = (uint8)cv::threshold(gray_mat, bin_mat, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    // 步骤 3: 将结果写回全局数组 bin_image
    memcpy(bin_image, bin_mat.data, UVC_WIDTH * UVC_HEIGHT);
}

// 图像后处理

/*
 * 函数名称: image_filter
 * 功能说明: 形态学滤波（膨胀 + 腐蚀的简化实现）
 *           遍历每个像素，统计其 8 邻域像素值总和 num:
 *             num >= 1275(THRESHOLD_MAX) 且当前为黑 → 膨胀为白（消除内部空洞）
 *             num <= 510(THRESHOLD_MIN)  且当前为白 → 腐蚀为黑（消除孤立噪点）
 * 参数说明: data  - 二值图像数据（一维指针，按行存储）
 *           width - 图像宽度（列数，用于计算一维索引）
 */
void image_filter(uint8* data, int width)
{
    uint16 i, j;
    uint32 num = 0;

    for (i = 1; i < UVC_HEIGHT - 1; i++)          // 跳过最外圈像素
    {
        for (j = 1; j < (uint16)(width - 1); j++)
        {
            // 统计八个方向的像素值总和
            num =
                data[(i - 1) * width + (j - 1)] + data[(i - 1) * width + j] + data[(i - 1) * width + (j + 1)]
              + data[ i      * width + (j - 1)]                           + data[ i      * width + (j + 1)]
              + data[(i + 1) * width + (j - 1)] + data[(i + 1) * width + j] + data[(i + 1) * width + (j + 1)];

            // 膨胀: 周围白点足够多，填充当前黑点
            if (num >= THRESHOLD_MAX && data[i * width + j] == 0)
            {
                data[i * width + j] = 255;
            }
            // 腐蚀: 周围白点太少，清除当前白点
            if (num <= THRESHOLD_MIN && data[i * width + j] == 255)
            {
                data[i * width + j] = 0;
            }
        }
    }
}

// 八邻域边界追踪

/*
 * 函数名称: get_start_point
 * 功能说明: 从图像底部向上最多 SEARCH_MAX_LINES 行搜索赛道的左右边界起点
 *           左起点: 从 UVC_WIDTH/2 向左扫，找"白→黑"的跳变点（赛道左边缘）
 *           右起点: 从 UVC_WIDTH/2 向右扫，找"白→黑"的跳变点（赛道右边缘）
 *           判断条件: 当前点为白(255) 且 相邻外侧点为黑(0)
 * 函数返回: 1 = 左右起点都找到, 0 = 未找到
 * 全局变量: actual_start_row - 存储实际搜索到的起始行号
 */
uint8 get_start_point(void)
{
    uint8 i = 0, l_found = 0, r_found = 0;
    int current_row = UVC_HEIGHT - 2; // 从底部倒数第 2 行开始
    int search_end = UVC_HEIGHT - 2 - SEARCH_MAX_LINES;
    if (search_end < 0) search_end = 0; // 防止越界

    // 向上循环搜索最多 SEARCH_MAX_LINES 行
    for (; current_row >= search_end; current_row--)
    {
        // 重置找到标志
        l_found = 0;
        r_found = 0;
        start_point_l[0] = 0;
        start_point_l[1] = 0;
        start_point_r[0] = 0;
        start_point_r[1] = 0;

        // 从图像中线往左搜索左边界起点
        for (i = UVC_WIDTH / 2; i > BORDER_MIN; i--)
        {
            start_point_l[0] = i;
            start_point_l[1] = current_row;
            if (bin_image[current_row][i] == 255 && bin_image[current_row][i - 1] == 0)
            {
                l_found = 1;
                break;
            }
        }

        // 从图像中线往右搜索右边界起点
        for (i = UVC_WIDTH / 2; i < BORDER_MAX; i++)
        {
            start_point_r[0] = i;
            start_point_r[1] = current_row;
            if (bin_image[current_row][i] == 255 && bin_image[current_row][i + 1] == 0)
            {
                r_found = 1;
                break;
            }
        }

        // 如果在这一行找到了左右起点
        if (l_found && r_found)
        {
            actual_start_row = current_row; // 记录实际起始行
            return 1;
        }
    }

    // 所有行都搜索完了，没找到
    return 0;
}

/*
 * 函数名称: search_l_r
 * 功能说明: 八邻域边界追踪核心算法
 *
 *           从给定的左右起点出发，沿赛道边界"生长"向上追踪。
 *           左边顺时针搜索，右边逆时针搜索。
 *           每步在当前中心点的 8 邻域中寻找"黑→白"的跳变点作为下一步中心点。
 *
 *           八邻域方向编码:
 *               0:正下   1:左下   2:左     3:左上
 *               4:正上   5:右上   6:右     7:右下
 *
 *           三条退出条件:
 *           (1) 同一点重复 3 次 → 死循环保护
 *           (2) 左右两点相遇（xy 差值均 < 2）→ 到达赛道顶部
 *           (3) break_flag 耗尽 → 最多循环上限保护
 *
 *           左右同步机制:
 *           - 左边比右边高了(y 更小)，左边等待右边
 *           - 左边向下生长(dir=7)而右边未追上，左边回退等待
 *
 * 参数说明:
 *   break_flag  - 最大循环次数
 *   data        - 二值图像数据（一维指针，按行存储）
 *   width       - 图像宽度（列数）
 *   l_stastic   - [输入/输出] 左边已找到的点数
 *   r_stastic   - [输入/输出] 右边已找到的点数
 *   l_start_x/y - 左边起点坐标
 *   r_start_x/y - 右边起点坐标
 *   hightest    - [输出] 追踪到达的最高行号
 */
void search_l_r(uint16 break_flag, uint8* data, int width,
                uint16* l_stastic, uint16* r_stastic,
                uint8 l_start_x, uint8 l_start_y,
                uint8 r_start_x, uint8 r_start_y,
                uint8* hightest)
{
    uint8 i = 0, j = 0;

    // 左边追踪变量
    uint8 search_filds_l[8][2] = { { 0 } };    // 中心点 8 邻域坐标
    uint8 index_l = 0;                          // 候选新中心点个数
    uint8 temp_l[8][2] = { { 0 } };            // 候选新中心点暂存区
    uint8 center_point_l[2] = { 0 };            // 当前中心点 [0]=x, [1]=y
    uint16 l_data_statics;                      // 左边已找到点数
    // 左边八邻域方向（顺时针搜索）
    static int8 seeds_l[8][2] = { {0,1},{-1,1},{-1,0},{-1,-1},{0,-1},{1,-1},{1,0},{1,1} };

    // 右边追踪变量
    uint8 search_filds_r[8][2] = { { 0 } };
    uint8 center_point_r[2] = { 0 };
    uint8 index_r = 0;
    uint8 temp_r[8][2] = { { 0 } };
    uint16 r_data_statics;
    // 右边八邻域方向（逆时针搜索）
    static int8 seeds_r[8][2] = { {0,1},{1,1},{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1} };

    l_data_statics = *l_stastic;
    r_data_statics = *r_stastic;

    // 初始化中心点为传入的起点
    center_point_l[0] = l_start_x;
    center_point_l[1] = l_start_y;
    center_point_r[0] = r_start_x;
    center_point_r[1] = r_start_y;

    while (break_flag--)  // 主循环，每次迭代左右各走一步
    {
        // ===== 计算左边当前中心点 8 邻域（带边界裁剪）=====
        for (i = 0; i < 8; i++)
        {
            int16 nx = (int16)center_point_l[0] + seeds_l[i][0];
            int16 ny = (int16)center_point_l[1] + seeds_l[i][1];
            if (nx < 0) nx = 0;
            if (nx >= width) nx = (uint16)(width - 1);
            if (ny < 0) ny = 0;
            if (ny >= UVC_HEIGHT) ny = UVC_HEIGHT - 1;
            search_filds_l[i][0] = (uint8)nx;
            search_filds_l[i][1] = (uint8)ny;
        }
        // 记录当前中心点到轨迹
        points_l[l_data_statics][0] = center_point_l[0];
        points_l[l_data_statics][1] = center_point_l[1];
        l_data_statics++;

        // ===== 计算右边当前中心点 8 邻域（带边界裁剪）=====
        for (i = 0; i < 8; i++)
        {
            int16 nx = (int16)center_point_r[0] + seeds_r[i][0];
            int16 ny = (int16)center_point_r[1] + seeds_r[i][1];
            if (nx < 0) nx = 0;
            if (nx >= width) nx = (uint16)(width - 1);
            if (ny < 0) ny = 0;
            if (ny >= UVC_HEIGHT) ny = UVC_HEIGHT - 1;
            search_filds_r[i][0] = (uint8)nx;
            search_filds_r[i][1] = (uint8)ny;
        }
        points_r[r_data_statics][0] = center_point_r[0];
        points_r[r_data_statics][1] = center_point_r[1];

        // ===== 左边: 在 8 邻域中找下一个中心点 =====
        // 寻找"当前邻域点为黑 且 下一邻域点为白"的跳变（跨过赛道边界的位置）
        index_l = 0;
        for (i = 0; i < 8; i++)
        {
            temp_l[i][0] = 0;
            temp_l[i][1] = 0;
        }

        for (i = 0; i < 8; i++)
        {
            if (data[search_filds_l[i][1] * width + search_filds_l[i][0]] == 0
                && data[search_filds_l[(i + 1) & 7][1] * width + search_filds_l[(i + 1) & 7][0]] == 255)
            {
                temp_l[index_l][0] = search_filds_l[(i)][0];
                temp_l[index_l][1] = search_filds_l[(i)][1];
                index_l++;
                dir_l[l_data_statics - 1] = (i);  // 记录生长方向
            }

            if (index_l)
            {
                // 有多个候选点时，选 y 最小的（越往上 y 越小，即最接近赛道顶部）
                center_point_l[0] = temp_l[0][0];
                center_point_l[1] = temp_l[0][1];
                for (j = 0; j < index_l; j++)
                {
                    if (center_point_l[1] > temp_l[j][1])
                    {
                        center_point_l[0] = temp_l[j][0];
                        center_point_l[1] = temp_l[j][1];
                    }
                }
            }
        }

        // ===== 退出条件 1: 同一点连续出现 3 次（须有足够历史点才检查）=====
        if ((r_data_statics >= 2
             && points_r[r_data_statics][0] == points_r[r_data_statics - 1][0]
             && points_r[r_data_statics][0] == points_r[r_data_statics - 2][0]
             && points_r[r_data_statics][1] == points_r[r_data_statics - 1][1]
             && points_r[r_data_statics][1] == points_r[r_data_statics - 2][1])
            || (l_data_statics >= 3
                && points_l[l_data_statics - 1][0] == points_l[l_data_statics - 2][0]
                && points_l[l_data_statics - 1][0] == points_l[l_data_statics - 3][0]
                && points_l[l_data_statics - 1][1] == points_l[l_data_statics - 2][1]
                && points_l[l_data_statics - 1][1] == points_l[l_data_statics - 3][1]))
        {
            break;
        }

        // ===== 退出条件 2: 左右两点相遇（赛道顶部收拢）=====
        if (std::abs(points_r[r_data_statics][0] - points_l[l_data_statics - 1][0]) < 2
            && std::abs(points_r[r_data_statics][1] - points_l[l_data_statics - 1][1]) < 2)
        {
            *hightest = (points_r[r_data_statics][1] + points_l[l_data_statics - 1][1]) >> 1;
            break;
        }

        // ===== 左右同步: 左边比右边高，左边等待 =====
        if ((points_r[r_data_statics][1] < points_l[l_data_statics - 1][1]))
        {
            continue;
        }

        // ===== 左右同步: 左边向下生长而右边未追上，左边回退 =====
        if (dir_l[l_data_statics - 1] == 7
            && (points_r[r_data_statics][1] > points_l[l_data_statics - 1][1]))
        {
            center_point_l[0] = points_l[l_data_statics - 1][0];
            center_point_l[1] = points_l[l_data_statics - 1][1];
            l_data_statics--;
        }
        r_data_statics++;

        // ===== 右边: 在 8 邻域中找下一个中心点 =====
        index_r = 0;
        for (i = 0; i < 8; i++)
        {
            temp_r[i][0] = 0;
            temp_r[i][1] = 0;
        }

        for (i = 0; i < 8; i++)
        {
            if (data[search_filds_r[i][1] * width + search_filds_r[i][0]] == 0
                && data[search_filds_r[(i + 1) & 7][1] * width + search_filds_r[(i + 1) & 7][0]] == 255)
            {
                temp_r[index_r][0] = search_filds_r[(i)][0];
                temp_r[index_r][1] = search_filds_r[(i)][1];
                index_r++;
                dir_r[r_data_statics - 1] = (i);
            }
            if (index_r)
            {
                center_point_r[0] = temp_r[0][0];
                center_point_r[1] = temp_r[0][1];
                for (j = 0; j < index_r; j++)
                {
                    if (center_point_r[1] > temp_r[j][1])
                    {
                        center_point_r[0] = temp_r[j][0];
                        center_point_r[1] = temp_r[j][1];
                    }
                }
            }
        }
    }

    // 将局部统计写回外部变量
    *l_stastic = l_data_statics;
    *r_stastic = r_data_statics;
}

// 从八邻域追踪点中提取逐行边界线

/*
 * 函数名称: get_left
 * 功能说明: 从八邻域追踪点中提取每行的左边界 x 坐标
 *           从下往上逐行遍历，每行取第一个匹配的追踪点
 *           左边界 = 追踪点 x + 1（偏移到真实边界位置）
 * 参数说明: total_L - 八邻域追踪到的左边点总数
 */
void get_left(uint16 total_L)
{
    uint8 i = 0;
    uint16 j = 0;
    int h = 0;
    for (i = 0; i < UVC_HEIGHT; i++)
    {
        l_border[i] = BORDER_MIN;  // 默认初始化为最左
    }
    h = actual_start_row;  // 从实际搜索到的起始行开始
    for (j = 0; j < total_L; j++)
    {
        if (points_l[j][1] == (uint16)h)
        {
            l_border[h] = points_l[j][0] + 1;
        }
        else continue;
        h--;  // 上一行
        if (h == 0) break;
    }
}

/*
 * 函数名称: get_right
 * 功能说明: 从八邻域追踪点中提取每行的右边界 x 坐标
 *           从下往上逐行遍历，每行取第一个匹配的追踪点
 *           右边界 = 追踪点 x - 1（偏移到真实边界位置）
 * 参数说明: total_R - 八邻域追踪到的右边点总数
 */
void get_right(uint16 total_R)
{
    uint8 i = 0;
    uint16 j = 0;
    int h = 0;
    for (i = 0; i < UVC_HEIGHT; i++)
    {
        r_border[i] = BORDER_MAX;  // 默认初始化为最右
    }
    h = actual_start_row;
    for (j = 0; j < total_R; j++)
    {
        if (points_r[j][1] == (uint16)h)
        {
            r_border[h] = points_r[j][0] - 1;
        }
        else continue;
        h--;
        if (h == 0) break;
    }
}

// 中线计算与偏移检测

/*
 * 函数名称: calculate_center_line
 * 功能说明: 计算每行的中线 x 坐标
 *           center_line[y] = (l_border[y] + r_border[y]) / 2
 *           范围: hightest ~ UVC_HEIGHT-2（从最高追踪点到图像底部）
 */
void calculate_center_line(void)
{
    for (int i = hightest; i < UVC_HEIGHT - 1; i++)
    {
        center_line[i] = (l_border[i] + r_border[i]) / 2;
    }
}

/*
 * 函数名称: get_mid_offset
 * 功能说明: 检测特定行的中线与屏幕中线的偏差值
 *           mid_offset    = center_line[UVC_HEIGHT - 40] - screen_mid  （远行，用于前瞻）
 *           detect_offset = center_line[UVC_HEIGHT - 5]  - screen_mid  （近行，用于即时修正）
 *           结果写入全局变量 mid_offset 和 detect_offset
 */
void get_mid_offset(void)
{
    // 计算屏幕中线
    int screen_mid = UVC_WIDTH / 2;

    // 计算目标行号（从底部向上数）
    int mid_line    = UVC_HEIGHT - MID_OFFSET_LINE;    // 远行
    int detect_line = UVC_HEIGHT - OFFSET_DETECT_LINE; // 近行

    // 确保行号在有效范围内
    mid_line    = std::max(0, std::min(UVC_HEIGHT - 1, mid_line));
    detect_line = std::max(0, std::min(UVC_HEIGHT - 1, detect_line));

    // 计算偏移值并写入全局变量
    mid_offset    = center_line[mid_line] - screen_mid;
    detect_offset = center_line[detect_line] - screen_mid;
}

// 显示巡线结果

/*
 * 函数名称: display_line_result
 * 功能说明: 将巡线结果可视化，构建显示图像（存入全局 g_display_image）
 *           显示内容:
 *           - 二值化图像背景（黑白）
 *           - 左右边界线（绿色）
 *           - 中心线（红色）
 *           - 画面中线（黄色竖线）
 *           - 偏移检测线（黄色横线）
 *           内部流程:
 *           1. 将裸数组 bin_image 包装为 cv::Mat
 *           2. 转为 BGR 彩色图
 *           3. 绘制边界线、中线、偏移线
 *           4. 放大 4 倍到 640×480 存入 g_display_image
 */
cv::Mat g_display_image;

void display_line_result(void)
{
    cv::Mat bin_mat(UVC_HEIGHT, UVC_WIDTH, CV_8UC1, (void*)bin_image);

    cv::Mat color_image;
    cv::cvtColor(bin_mat, color_image, cv::COLOR_GRAY2BGR);

    for (int i = 0; i < UVC_HEIGHT - 1; i++)
    {
        if (l_border[i] > 0 && l_border[i] < UVC_WIDTH - 1)
            color_image.at<cv::Vec3b>(i, l_border[i]) = cv::Vec3b(0, 255, 0);
        if (r_border[i] > 0 && r_border[i] < UVC_WIDTH - 1)
            color_image.at<cv::Vec3b>(i, r_border[i]) = cv::Vec3b(0, 255, 0);
    }

    for (int i = 0; i < UVC_HEIGHT - 1; i++)
    {
        if (center_line[i] > 0 && center_line[i] < UVC_WIDTH - 1)
            color_image.at<cv::Vec3b>(i, center_line[i]) = cv::Vec3b(0, 0, 255);
    }

    int screen_mid = UVC_WIDTH / 2;
    for (int i = 0; i < UVC_HEIGHT - 1; i++)
        color_image.at<cv::Vec3b>(i, screen_mid) = cv::Vec3b(0, 255, 255);

    int mid_line    = UVC_HEIGHT - MID_OFFSET_LINE;
    int detect_line = UVC_HEIGHT - OFFSET_DETECT_LINE;
    mid_line    = std::max(0, std::min(UVC_HEIGHT - 1, mid_line));
    detect_line = std::max(0, std::min(UVC_HEIGHT - 1, detect_line));

    if (center_line[mid_line] > 0 && center_line[mid_line] < UVC_WIDTH - 1)
    {
        for (int x = std::min(screen_mid, center_line[mid_line]);
             x <= std::max(screen_mid, center_line[mid_line]); x++)
            color_image.at<cv::Vec3b>(mid_line, x) = cv::Vec3b(0, 255, 255);
    }
    if (center_line[detect_line] > 0 && center_line[detect_line] < UVC_WIDTH - 1)
    {
        for (int x = std::min(screen_mid, center_line[detect_line]);
             x <= std::max(screen_mid, center_line[detect_line]); x++)
            color_image.at<cv::Vec3b>(detect_line, x) = cv::Vec3b(0, 255, 255);
    }

    cv::resize(color_image, g_display_image, cv::Size(640, 480), 0, 0, cv::INTER_NEAREST);
}

int main(void)
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    int frame_count = 0;
    auto last_update = std::chrono::steady_clock::now();
    char fps_str[32] = "FPS: --";
    char offset_str[32] = "Mid:--  Det:--";
    char start_row_str[32] = "StartRow:--";

    cv::namedWindow("LineDetect", cv::WINDOW_AUTOSIZE);

    while (1)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        cv::Mat gray_160x120;
        cv::resize(frame, gray_160x120, cv::Size(UVC_WIDTH, UVC_HEIGHT));
        cv::cvtColor(gray_160x120, gray_160x120, cv::COLOR_BGR2GRAY);
        memcpy(original_image, gray_160x120.data, UVC_WIDTH * UVC_HEIGHT);

        turn_to_bin();

        image_filter((uint8*)bin_image, UVC_WIDTH);

        data_stastics_l = 0;
        data_stastics_r = 0;
        hightest = 0;

        if (get_start_point())
        {
            search_l_r((uint16)USE_NUM, (uint8*)bin_image, UVC_WIDTH,
                       (uint16*)&data_stastics_l, (uint16*)&data_stastics_r,
                       start_point_l[0], start_point_l[1],
                       start_point_r[0], start_point_r[1],
                       (uint8*)&hightest);

            get_left(data_stastics_l);
            get_right(data_stastics_r);

            calculate_center_line();
            get_mid_offset();
        }

        display_line_result();

        sprintf(offset_str, "Mid:%d  Det:%d", mid_offset, detect_offset);
        sprintf(start_row_str, "StartRow:%d", actual_start_row);

        frame_count++;

        auto now = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count();

        if (elapsed_ms >= 500)
        {
            float fps = (float)frame_count * 1000.0f / (float)elapsed_ms;
            sprintf(fps_str, "FPS: %.1f", fps);
            frame_count = 0;
            last_update = now;
        }

        cv::putText(g_display_image, "USEINGI_Ver1.1", cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
        cv::putText(g_display_image, fps_str,             cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
        cv::putText(g_display_image, offset_str,          cv::Point(5, 45), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
        cv::putText(g_display_image, start_row_str,       cv::Point(5, 60), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);

        cv::imshow("LineDetect", g_display_image);

        if (cv::waitKey(1) == 27)
            break;
    }

    cv::destroyAllWindows();
    return 0;
}
