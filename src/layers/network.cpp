#include "network.h"

NS_JJ_BEGIN

int NetWork::get_network_output_size(network* pNet)
{
    int i;
    for (i = pNet->n - 1; i > 0; --i)
    if (pNet->layers[i].type != COST)
        break;

    return pNet->layers[i].outputs;
}

std::vector<float> NetWork::get_network_output(network* pNet)
{
    int i;
    for (i = pNet->n - 1; i > 0; --i)
    if (pNet->layers[i].type != COST)
        break;

    return pNet->layers[i].output;
}



NS_JJ_END