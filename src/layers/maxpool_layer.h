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

private:
};

NS_JJ_END