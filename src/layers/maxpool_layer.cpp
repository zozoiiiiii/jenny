#include "maxpool_layer.h"

NS_JJ_BEGIN
layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    layer l = { 0 };
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size) / stride + 1;
    l.out_h = (h + padding - size) / stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h * w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes.assign(output_size, 0);
    l.output.assign(output_size, 0.0f);
    l.output_int8.assign(output_size, 0);
    //l.delta = calloc(output_size, sizeof(float));
    // commented only for this custom version of Yolo v2
    //l.forward = forward_maxpool_layer;
    //l.backward = backward_maxpool_layer;
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

bool MaxpoolLayer::load(const IniParser* pParser, int section, size_params params)
{
    int stride = pParser->ReadInteger(section, "stride", 1);
    int size = pParser->ReadInteger(section, "size", stride);
    int padding = pParser->ReadInteger(section, "padding", size - 1);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c))
        return false;// error("Layer before maxpool layer must output image.");

    m_layerInfo = make_maxpool_layer(batch, h, w, c, size, stride, padding);
    return true;
}

NS_JJ_END