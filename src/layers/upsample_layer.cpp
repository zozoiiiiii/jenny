#include "upsample_layer.h"

NS_JJ_BEGIN
layer make_upsample_layer(int batch, int w, int h, int c, int stride)
{
    layer l = { 0 };
    l.type = UPSAMPLE;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w * stride;
    l.out_h = h * stride;
    l.out_c = c;
    if (stride < 0) {
        stride = -stride;
        l.reverse = 1;
        l.out_w = w / stride;
        l.out_h = h / stride;
    }
    l.stride = stride;
    l.outputs = l.out_w*l.out_h*l.out_c;
    l.inputs = l.w*l.h*l.c;
    //l.delta = calloc(l.outputs*batch, sizeof(float));
    l.output.assign(l.outputs*batch, 0.0f);;

    //l.forward = forward_upsample_layer;
    if (l.reverse)
        fprintf(stderr, "downsample         %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    else
        fprintf(stderr, "upsample           %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

bool UpsampleLayer::load(const IniParser* pParser, int section, size_params params)
{
    int stride = pParser->ReadInteger(section, "stride", 2);
    layer l = make_upsample_layer(params.batch, params.w, params.h, params.c, stride);
    l.scale = pParser->ReadFloat(section, "scale", 1);
    //return layer;
    return true;
}


void UpsampleLayer::resize_upsample_layer(layer *l, int w, int h)
{
  
}

void UpsampleLayer::forward_upsample_layer(const layer l, network net)
{
}

void UpsampleLayer::backward_upsample_layer(const layer l, network net)
{
   
}


NS_JJ_END