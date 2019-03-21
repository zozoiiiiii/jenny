/************************************************************************/
/*
@author:  junliang
@brief:   
@time:    2019/03/20
*/
/************************************************************************/
#pragma once


#include <iostream>


namespace JJ {

    // image.c

    struct ImageInfo
    {
        int h;
        int w;
        int c;
        float *data;
    };

    class ImageUtil
    {
    public:

        static void rgbgr_image(ImageInfo im);
        static ImageInfo make_empty_image(int w, int h, int c);
        static void free_image(ImageInfo m);
        static void draw_box(ImageInfo a, int x1, int y1, int x2, int y2, float r, float g, float b);
        static void draw_box_width(ImageInfo a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
        static ImageInfo make_image(int w, int h, int c);
        static float get_pixel(ImageInfo m, int x, int y, int c);
        static void set_pixel(ImageInfo m, int x, int y, int c, float val);
        static ImageInfo resize_image(ImageInfo im, int w, int h);
        static ImageInfo load_image(char *filename, int w, int h, int c);
        static ImageInfo load_image_stb(char *filename, int channels);
        static void save_image_png(ImageInfo im, const char *name);
        static void show_image(ImageInfo p, const char *name);

        static float get_color(int c, int x, int max);

        static void add_pixel(ImageInfo m, int x, int y, int c, float val);
        static ImageInfo copy_image(ImageInfo p);
        static void constrain_image(ImageInfo im);
    };

}