/************************************************************************/
/*
@author:  junliang
@brief:   convolution layer
@time:    2019/02/22
*/
/************************************************************************/
#pragma once

#include <string>
#include <vector>
#include <map>
#include "layer.h"
#include "network.h"


NS_JJ_BEGIN



class UpsampleLayer : public ILayer
{
public:
    virtual bool load(const IniParser* pParser, int section, size_params params);
    virtual void forward_layer_cpu(network_state state);


    void forward_upsample_layer(const layer l, network net);
    void backward_upsample_layer(const layer l, network net);
    void resize_upsample_layer(layer *l, int w, int h);
private:
    float m_scale;    // unsample
};

NS_JJ_END