#include "image_util.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include "../stb_image_write.h"
#include "../stb_image.h"

// namespace
#ifndef NS_JJ_BEGIN
#define NS_JJ_BEGIN                     namespace JJ {
#define NS_JJ_END                       }
#define USING_NS_JJ                     using namespace JJ;
#endif

NS_JJ_BEGIN

void ImageUtil::rgbgr_image(ImageInfo im)
{
    int i;
    for (i = 0; i < im.w*im.h; ++i)
    {
        float swap = im.data[i];
        im.data[i] = im.data[i + im.w*im.h * 2];
        im.data[i + im.w*im.h * 2] = swap;
    }
}

ImageInfo ImageUtil::make_empty_image(int w, int h, int c)
{
    ImageInfo out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

void ImageUtil::free_image(ImageInfo m)
{
    if (m.data) {
        free(m.data);
    }
}

// image.c
void ImageUtil::draw_box(ImageInfo a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    //normalize_image(a);
    int i;
    if (x1 < 0) x1 = 0;
    if (x1 >= a.w) x1 = a.w - 1;
    if (x2 < 0) x2 = 0;
    if (x2 >= a.w) x2 = a.w - 1;

    if (y1 < 0) y1 = 0;
    if (y1 >= a.h) y1 = a.h - 1;
    if (y2 < 0) y2 = 0;
    if (y2 >= a.h) y2 = a.h - 1;

    for (i = x1; i <= x2; ++i) {
        a.data[i + y1 * a.w + 0 * a.w*a.h] = r;
        a.data[i + y2 * a.w + 0 * a.w*a.h] = r;

        a.data[i + y1 * a.w + 1 * a.w*a.h] = g;
        a.data[i + y2 * a.w + 1 * a.w*a.h] = g;

        a.data[i + y1 * a.w + 2 * a.w*a.h] = b;
        a.data[i + y2 * a.w + 2 * a.w*a.h] = b;
    }
    for (i = y1; i <= y2; ++i) {
        a.data[x1 + i * a.w + 0 * a.w*a.h] = r;
        a.data[x2 + i * a.w + 0 * a.w*a.h] = r;

        a.data[x1 + i * a.w + 1 * a.w*a.h] = g;
        a.data[x2 + i * a.w + 1 * a.w*a.h] = g;

        a.data[x1 + i * a.w + 2 * a.w*a.h] = b;
        a.data[x2 + i * a.w + 2 * a.w*a.h] = b;
    }
}

// image.c
void ImageUtil::draw_box_width(ImageInfo a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
    int i;
    for (i = 0; i < w; ++i) {
        draw_box(a, x1 + i, y1 + i, x2 - i, y2 - i, r, g, b);
    }
}

// image.c
ImageInfo ImageUtil::make_image(int w, int h, int c)
{
    ImageInfo out = make_empty_image(w, h, c);
    out.data = (float*)calloc(h*w*c, sizeof(float));
    return out;
}

// image.c
float ImageUtil::get_pixel(ImageInfo m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y * m.w + x];
}

// image.c
void ImageUtil::set_pixel(ImageInfo m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y * m.w + x] = val;
}

// image.c
void ImageUtil::add_pixel(ImageInfo m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y * m.w + x] += val;
}

// image.c
ImageInfo ImageUtil::resize_image(ImageInfo im, int w, int h)
{
    ImageInfo resized = make_image(w, h, im.c);
    ImageInfo part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < im.h; ++r) {
            for (c = 0; c < w; ++c) {
                float val = 0;
                if (c == w - 1 || im.w == 1) {
                    val = get_pixel(im, im.w - 1, r, k);
                }
                else {
                    float sx = c * w_scale;
                    int ix = (int)sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix + 1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < h; ++r) {
            float sy = r * h_scale;
            int iy = (int)sy;
            float dy = sy - iy;
            for (c = 0; c < w; ++c) {
                float val = (1 - dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if (r == h - 1 || im.h == 1) continue;
            for (c = 0; c < w; ++c) {
                float val = dy * get_pixel(part, c, iy + 1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}

// image.c
ImageInfo ImageUtil::load_image(char *filename, int w, int h, int c)
{
#ifdef OPENCV
    ImageInfo out = load_image_cv(filename, c);
#else
    ImageInfo out = load_image_stb(filename, c);
#endif

    if ((h && w) && (h != out.h || w != out.w)) {
        ImageInfo resized = resize_image(out, w, h);
        free_image(out);
        out = resized;
    }
    return out;
}

// image.c
ImageInfo ImageUtil::load_image_stb(char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load ImageInfo \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
    if (channels) c = channels;
    int i, j, k;
    ImageInfo im = make_image(w, h, c);
    for (k = 0; k < c; ++k) {
        for (j = 0; j < h; ++j) {
            for (i = 0; i < w; ++i) {
                int dst_index = i + w * j + w * h*k;
                int src_index = k + c * i + c * w*j;
                im.data[dst_index] = (float)data[src_index] / 255.;
            }
        }
    }
    free(data);
    return im;
}


// image.c
ImageInfo ImageUtil::copy_image(ImageInfo p)
{
    ImageInfo copy = p;
    copy.data = (float*)calloc(p.h*p.w*p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h*p.w*p.c * sizeof(float));
    return copy;
}

// image.c
void ImageUtil::constrain_image(ImageInfo im)
{
    int i;
    for (i = 0; i < im.w*im.h*im.c; ++i) {
        if (im.data[i] < 0) im.data[i] = 0;
        if (im.data[i] > 1) im.data[i] = 1;
    }
}

// image.c
void ImageUtil::save_image_png(ImageInfo im, const char *name)
{
    char buff[256];
    sprintf(buff, "%s.png", name);
    unsigned char *data = (unsigned char*)calloc(im.w*im.h*im.c, sizeof(char));
    int i, k;
    for (k = 0; k < im.c; ++k) {
        for (i = 0; i < im.w*im.h; ++i) {
            data[i*im.c + k] = (unsigned char)(255 * im.data[i + k * im.w*im.h]);
        }
    }
    int success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
    free(data);
    if (!success) fprintf(stderr, "Failed to write ImageInfo %s\n", buff);
}


// image.c
void ImageUtil::show_image(ImageInfo p, const char *name)
{
#ifdef OPENCV
    show_image_cv(p, name);
#else
    fprintf(stderr, "Not compiled with OpenCV, saving to %s.png instead\n", name);
    save_image_png(p, name);
#endif
}

// image.c
float ImageUtil::get_color(int c, int x, int max)
{
    static float colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
    float ratio = ((float)x / max) * 5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
    //printf("%f\n", r);
    return r;
}
NS_JJ_END