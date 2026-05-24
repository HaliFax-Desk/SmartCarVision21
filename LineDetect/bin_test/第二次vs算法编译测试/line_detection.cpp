#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <time.h>

using namespace cv;
using namespace std;

typedef   signed          char int8;
typedef   signed short     int int16;
typedef   signed           int int32;
typedef unsigned          char uint8;
typedef unsigned short     int uint16;
typedef unsigned           int uint32;

#define uesr_RED     0XF800
#define uesr_GREEN   0X07E0
#define uesr_BLUE    0X001F

#define image_h  120
#define image_w  188

#define white_pixel  255
#define black_pixel  0

#define bin_jump_num  1
#define border_max  image_w-2
#define border_min  1

#define USE_num  image_h*3

#define threshold_max  255*5
#define threshold_min  255*2

uint8 mt9v03x_image[image_h][image_w];
uint8 original_image[image_h][image_w];
uint8 bin_image[image_h][image_w];
uint8 image_thereshold;
uint8 l_border[image_h];
uint8 r_border[image_h];
uint8 center_line[image_h];
uint16 data_stastics_l = 0;
uint16 data_stastics_r = 0;
uint16 points_l[(uint16)(image_h * 3)][2] = { { 0 } };
uint16 points_r[(uint16)(image_h * 3)][2] = { { 0 } };
uint16 dir_r[(uint16)USE_num] = { 0 };
uint16 dir_l[(uint16)USE_num] = { 0 };
uint8 hightest = 0;
uint8 start_point_l[2] = { 0 };
uint8 start_point_r[2] = { 0 };

int my_abs(int value)
{
    if (value >= 0) return value;
    else return -value;
}

int16 limit_a_b(int16 x, int a, int b)
{
    if (x < a) x = a;
    if (x > b) x = b;
    return x;
}

int16 limit1(int16 x, int16 y)
{
    if (x > y)       return y;
    else if (x < -y) return -y;
    else             return x;
}

void Get_image(uint8(*mt9v03x_image)[image_w])
{
#define use_num  1
    uint8 i = 0, j = 0, row = 0, line = 0;
    for (i = 0; i < image_h; i += use_num)
    {
        for (j = 0; j < image_w; j += use_num)
        {
            original_image[row][line] = mt9v03x_image[i][j];
            line++;
        }
        line = 0;
        row++;
    }
}

uint8 otsuThreshold(uint8* image, uint16 col, uint16 row)
{
#define GrayScale 256
    uint16 Image_Width = col;
    uint16 Image_Height = row;
    int X; uint16 Y;
    uint8* data = image;
    int HistGram[GrayScale] = { 0 };

    uint32 Amount = 0;
    uint32 PixelBack = 0;
    uint32 PixelIntegralBack = 0;
    uint32 PixelIntegral = 0;
    int32 PixelIntegralFore = 0;
    int32 PixelFore = 0;
    double OmegaBack = 0, OmegaFore = 0, MicroBack = 0, MicroFore = 0, SigmaB = 0, Sigma = 0;
    uint8 MinValue = 0, MaxValue = 0;
    uint8 Threshold = 0;

    for (Y = 0; Y < Image_Height; Y++)
    {
        for (X = 0; X < Image_Width; X++)
        {
            HistGram[(int)data[Y * Image_Width + X]]++;
        }
    }

    for (MinValue = 0; MinValue < 256 && HistGram[MinValue] == 0; MinValue++);
    for (MaxValue = 255; MaxValue > MinValue && HistGram[MinValue] == 0; MaxValue--);

    if (MaxValue == MinValue)
    {
        return MaxValue;
    }
    if (MinValue + 1 == MaxValue)
    {
        return MinValue;
    }

    for (Y = MinValue; Y <= MaxValue; Y++)
    {
        Amount += HistGram[Y];
    }

    PixelIntegral = 0;
    for (Y = MinValue; Y <= MaxValue; Y++)
    {
        PixelIntegral += HistGram[Y] * Y;
    }
    SigmaB = -1;
    for (Y = MinValue; Y < MaxValue; Y++)
    {
        PixelBack = PixelBack + HistGram[Y];
        PixelFore = Amount - PixelBack;
        OmegaBack = (double)PixelBack / Amount;
        OmegaFore = (double)PixelFore / Amount;
        PixelIntegralBack += HistGram[Y] * Y;
        PixelIntegralFore = PixelIntegral - PixelIntegralBack;
        MicroBack = (double)PixelIntegralBack / PixelBack;
        MicroFore = (double)PixelIntegralFore / PixelFore;
        Sigma = OmegaBack * OmegaFore * (MicroBack - MicroFore) * (MicroBack - MicroFore);
        if (Sigma > SigmaB)
        {
            SigmaB = Sigma;
            Threshold = (uint8)Y;
        }
    }
    return Threshold;
}

void turn_to_bin(void)
{
    uint8 i, j;
    image_thereshold = otsuThreshold(original_image[0], image_w, image_h);
    for (i = 0; i < image_h; i++)
    {
        for (j = 0; j < image_w; j++)
        {
            if (original_image[i][j] > image_thereshold) bin_image[i][j] = white_pixel;
            else bin_image[i][j] = black_pixel;
        }
    }
}

uint8 get_start_point(uint8 start_row)
{
    uint8 i = 0, l_found = 0, r_found = 0;
    start_point_l[0] = 0;
    start_point_l[1] = 0;
    start_point_r[0] = 0;
    start_point_r[1] = 0;

    for (i = image_w / 2; i > border_min; i--)
    {
        start_point_l[0] = i;
        start_point_l[1] = start_row;
        if (bin_image[start_row][i] == 255 && bin_image[start_row][i - 1] == 0)
        {
            l_found = 1;
            break;
        }
    }

    for (i = image_w / 2; i < border_max; i++)
    {
        start_point_r[0] = i;
        start_point_r[1] = start_row;
        if (bin_image[start_row][i] == 255 && bin_image[start_row][i + 1] == 0)
        {
            r_found = 1;
            break;
        }
    }

    if (l_found && r_found) return 1;
    else return 0;
}

void search_l_r(uint16 break_flag, uint8(*image)[image_w], uint16* l_stastic, uint16* r_stastic,
    uint8 l_start_x, uint8 l_start_y, uint8 r_start_x, uint8 r_start_y, uint8* hightest)
{
    uint8 i = 0, j = 0;

    uint8 search_filds_l[8][2] = { { 0 } };
    uint8 index_l = 0;
    uint8 temp_l[8][2] = { { 0 } };
    uint8 center_point_l[2] = { 0 };
    uint16 l_data_statics;
    static int8 seeds_l[8][2] = { {0,1},{-1,1},{-1,0},{-1,-1},{0,-1},{1,-1},{1,0},{1,1} };

    uint8 search_filds_r[8][2] = { { 0 } };
    uint8 center_point_r[2] = { 0 };
    uint8 index_r = 0;
    uint8 temp_r[8][2] = { { 0 } };
    uint16 r_data_statics;
    static int8 seeds_r[8][2] = { {0,1},{1,1},{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1} };

    l_data_statics = *l_stastic;
    r_data_statics = *r_stastic;

    center_point_l[0] = l_start_x;
    center_point_l[1] = l_start_y;
    center_point_r[0] = r_start_x;
    center_point_r[1] = r_start_y;

    while (break_flag--)
    {
        for (i = 0; i < 8; i++)
        {
            search_filds_l[i][0] = center_point_l[0] + seeds_l[i][0];
            search_filds_l[i][1] = center_point_l[1] + seeds_l[i][1];
        }
        points_l[l_data_statics][0] = center_point_l[0];
        points_l[l_data_statics][1] = center_point_l[1];
        l_data_statics++;

        for (i = 0; i < 8; i++)
        {
            search_filds_r[i][0] = center_point_r[0] + seeds_r[i][0];
            search_filds_r[i][1] = center_point_r[1] + seeds_r[i][1];
        }
        points_r[r_data_statics][0] = center_point_r[0];
        points_r[r_data_statics][1] = center_point_r[1];

        index_l = 0;
        for (i = 0; i < 8; i++)
        {
            temp_l[i][0] = 0;
            temp_l[i][1] = 0;
        }

        for (i = 0; i < 8; i++)
        {
            if (image[search_filds_l[i][1]][search_filds_l[i][0]] == 0
                && image[search_filds_l[(i + 1) & 7][1]][search_filds_l[(i + 1) & 7][0]] == 255)
            {
                temp_l[index_l][0] = search_filds_l[(i)][0];
                temp_l[index_l][1] = search_filds_l[(i)][1];
                index_l++;
                dir_l[l_data_statics - 1] = (i);
            }

            if (index_l)
            {
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
        if ((points_r[r_data_statics][0] == points_r[r_data_statics - 1][0] && points_r[r_data_statics][0] == points_r[r_data_statics - 2][0]
            && points_r[r_data_statics][1] == points_r[r_data_statics - 1][1] && points_r[r_data_statics][1] == points_r[r_data_statics - 2][1])
            || (points_l[l_data_statics - 1][0] == points_l[l_data_statics - 2][0] && points_l[l_data_statics - 1][0] == points_l[l_data_statics - 3][0]
                && points_l[l_data_statics - 1][1] == points_l[l_data_statics - 2][1] && points_l[l_data_statics - 1][1] == points_l[l_data_statics - 3][1]))
        {
            break;
        }
        if (my_abs(points_r[r_data_statics][0] - points_l[l_data_statics - 1][0]) < 2
            && my_abs(points_r[r_data_statics][1] - points_l[l_data_statics - 1][1] < 2)
            )
        {
            *hightest = (points_r[r_data_statics][1] + points_l[l_data_statics - 1][1]) >> 1;
            break;
        }
        if ((points_r[r_data_statics][1] < points_l[l_data_statics - 1][1]))
        {
            printf("\n如果左边比右边高了，左边等待右边\n");
            continue;
        }
        if (dir_l[l_data_statics - 1] == 7
            && (points_r[r_data_statics][1] > points_l[l_data_statics - 1][1]))
        {
            center_point_l[0] = points_l[l_data_statics - 1][0];
            center_point_l[1] = points_l[l_data_statics - 1][1];
            l_data_statics--;
        }
        r_data_statics++;

        index_r = 0;
        for (i = 0; i < 8; i++)
        {
            temp_r[i][0] = 0;
            temp_r[i][1] = 0;
        }

        for (i = 0; i < 8; i++)
        {
            if (image[search_filds_r[i][1]][search_filds_r[i][0]] == 0
                && image[search_filds_r[(i + 1) & 7][1]][search_filds_r[(i + 1) & 7][0]] == 255)
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

    *l_stastic = l_data_statics;
    *r_stastic = r_data_statics;
}

void get_left(uint16 total_L)
{
    uint8 i = 0;
    uint16 j = 0;
    uint8 h = 0;
    for (i = 0; i < image_h; i++)
    {
        l_border[i] = border_min;
    }
    h = image_h - 2;
    for (j = 0; j < total_L; j++)
    {
        if (points_l[j][1] == h)
        {
            l_border[h] = points_l[j][0] + 1;
        }
        else continue;
        h--;
        if (h == 0)
        {
            break;
        }
    }
}

void get_right(uint16 total_R)
{
    uint8 i = 0;
    uint16 j = 0;
    uint8 h = 0;
    for (i = 0; i < image_h; i++)
    {
        r_border[i] = border_max;
    }
    h = image_h - 2;
    for (j = 0; j < total_R; j++)
    {
        if (points_r[j][1] == h)
        {
            r_border[h] = points_r[j][0] - 1;
        }
        else continue;
        h--;
        if (h == 0) break;
    }
}

void image_filter(uint8(*bin_image)[image_w])
{
    uint16 i, j;
    uint32 num = 0;

    for (i = 1; i < image_h - 1; i++)
    {
        for (j = 1; j < (image_w - 1); j++)
        {
            num =
                bin_image[i - 1][j - 1] + bin_image[i - 1][j] + bin_image[i - 1][j + 1]
                + bin_image[i][j - 1] + bin_image[i][j + 1]
                + bin_image[i + 1][j - 1] + bin_image[i + 1][j] + bin_image[i + 1][j + 1];

            if (num >= threshold_max && bin_image[i][j] == 0)
            {
                bin_image[i][j] = 255;
            }
            if (num <= threshold_min && bin_image[i][j] == 255)
            {
                bin_image[i][j] = 0;
            }
        }
    }
}

void image_draw_rectan(uint8(*image)[image_w])
{
    uint8 i = 0;
    for (i = 0; i < image_h; i++)
    {
        image[i][0] = 0;
        image[i][1] = 0;
        image[i][image_w - 1] = 0;
        image[i][image_w - 2] = 0;
    }
    for (i = 0; i < image_w; i++)
    {
        image[0][i] = 0;
        image[1][i] = 0;
    }
}

void image_process(void)
{
    uint16 i;
    uint8 hightest = 0;

    Get_image(mt9v03x_image);
    turn_to_bin();
    image_filter(bin_image);
    image_draw_rectan(bin_image);

    data_stastics_l = 0;
    data_stastics_r = 0;
    if (get_start_point(image_h - 2))
    {
        search_l_r((uint16)USE_num, bin_image, &data_stastics_l, &data_stastics_r,
            start_point_l[0], start_point_l[1], start_point_r[0], start_point_r[1], &hightest);
        get_left(data_stastics_l);
        get_right(data_stastics_r);

        for (i = hightest; i < image_h - 1; i++)
        {
            center_line[i] = (l_border[i] + r_border[i]) >> 1;
        }
    }
}

void capture_from_camera()
{
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
            circle(display_frame, Point(points_l[i][0] + 2, points_l[i][1]), 1, Scalar(255, 0, 0), 1);
        }

        for (int i = 0; i < data_stastics_r; i++) {
            circle(display_frame, Point(points_r[i][0] - 2, points_r[i][1]), 1, Scalar(0, 0, 255), 1);
        }

        for (int i = hightest; i < image_h - 1; i++) {
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

int main()
{
    printf("SJTU 8-Area Line Detection Test (OpenCV)\n");
    printf("==========================================\n\n");

    printf("使用摄像头作为视频源\n");
    printf("按 'ESC' 键退出程序\n\n");

    capture_from_camera();

    return 0;
}
