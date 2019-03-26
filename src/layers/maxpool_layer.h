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



class MaxpoolLayer : public ILayer
{
public:
    virtual bool load(const IniParser* pParser, int section, size_params params);
    virtual void forward_layer_cpu(network_state state);
private:
    layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);

private:
    int* m_indexes; // maxpoll layer
};

NS_JJ_END