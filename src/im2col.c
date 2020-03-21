#include "im2col.h"
#include <stdio.h>
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
/*
将channels*height*width的输入图像与channels*ksize*ksize的卷基核进行步长为stride，零填充为pad的卷基运算转换成矩阵运算，可以调用gemm
(height + 2*pad - ksize) / stride + 1）* （(width + 2*pad - ksize) / stride + 1）* （channels * ksize * ksize）
|-------------输出图像的高-------------| * |-------------输出图像的宽--------------|
|------------------------------------矩阵的列-----------------------------------| * |--------矩阵的行----------|
*/
void im2col_cpu(float* data_im/*输入图像*/,
     int channels,  int height,  int width,
     int ksize/*滤波器尺寸*/,  int stride/*步长*/, int pad/*零填充*/, float* data_col/*输出矩阵*/) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                /*根据计算出来的坐标去原图data_im取像素值，注意此时的坐标是加上了零填充pad后的坐标*/
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

