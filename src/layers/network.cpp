#include "network.h"

NS_JJ_BEGIN

int NetWork::get_network_output_size(network* pNet)
{
    int i;
    for (i = pNet->n - 1; i > 0; --i)
    {
        ILayer* pLayer = pNet->jjLayers[i];
        if (pLayer->getLayer()->type != COST)
            break;
    }

    return pNet->jjLayers[i]->getLayer()->outputs;
}

std::vector<float> NetWork::get_network_output(network* pNet)
{
    int i;
    for (i = pNet->n - 1; i > 0; --i)
    {
        ILayer* pLayer = pNet->jjLayers[i];
        if (pLayer->getLayer()->type != COST)
            break;
    }

    return pNet->jjLayers[i]->getLayer()->output;
}



NS_JJ_END