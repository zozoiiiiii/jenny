#include "layer.h"

NS_JJ_BEGIN

int ILayer::entry_index(int batch, int location, int entry)
{
    int n = location / (m_layerInfo.w*m_layerInfo.h);
    int loc = location % (m_layerInfo.w*m_layerInfo.h);
    return batch * m_layerInfo.outputs + n * m_layerInfo.w*m_layerInfo.h*(4 + m_layerInfo.classes + 1) + entry * m_layerInfo.w*m_layerInfo.h + loc;
}


NS_JJ_END