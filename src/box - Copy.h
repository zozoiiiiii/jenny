/************************************************************************/
/*
@author:  junliang
@brief:   
@time:    2019/03/20
*/
/************************************************************************/
#pragma once




typedef struct {
    int h;
    int w;
    int c;
    float *data;
} image;

// image.c
void rgbgr_image(image im);

// image.c
image make_empty_image(int w, int h, int c);

// image.c
void free_image(image m);

// image.c
void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b);

// image.c
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);

// image.c
image make_image(int w, int h, int c);

// image.c
float get_pixel(image m, int x, int y, int c);

// image.c
void set_pixel(image m, int x, int y, int c, float val);

// image.c
image resize_image(image im, int w, int h);

// image.c
image load_image(char *filename, int w, int h, int c);

// image.c
image load_image_stb(char *filename, int channels);


void save_image_png(image im, const char *name);