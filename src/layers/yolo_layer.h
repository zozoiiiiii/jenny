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
#include "../box.h"

NS_JJ_BEGIN



typedef layer convolutional_layer;
class YoloLayer : public ILayer
{
public:
    virtual bool load(const IniParser* pParser, int section, size_params params);
    virtual void forward_layer_cpu(network_state state);

    int get_yolo_detections(int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter);
private:
};

NS_JJ_END