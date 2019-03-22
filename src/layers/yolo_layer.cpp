#include "yolo_layer.h"

NS_JJ_BEGIN

layer make_yolo_layer(int batch, int w, int h, int n, int total, std::vector<int> mask, int classes, int max_boxes)
{
    int i;
    layer l = { 0 };
    l.type = YOLO;

    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n * (classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost.assign(1, 0.0f);
    l.biases.assign(total * 2, 0.0f);
    if (!mask.empty())
        l.mask = mask;
    else
    {
        l.mask.assign(n, 0);
        for (i = 0; i < n; ++i)
        {
            l.mask[i] = i;
        }
    }
    l.outputs = h * w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    l.max_boxes = max_boxes;
    l.truths = l.max_boxes*(4 + 1);    // 90*(4 + 1);
    l.output.assign(batch*l.outputs, 0.0f);
    for (i = 0; i < total * 2; ++i)
    {
        l.biases[i] = .5;
    }


    fprintf(stderr, "yolo\n");
    srand(0);

    return l;
}

int *parse_yolo_mask(char *a, int *num)
{
    int *mask = 0;
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == ',') ++n;
        }
        mask = (int*)calloc(n, sizeof(int));
        for (i = 0; i < n; ++i) {
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',') + 1;
        }
        *num = n;
    }
    return mask;
}

bool YoloLayer::load(const IniParser* pParser, int section, size_params params)
{
    int classes = pParser->ReadInteger(section, "classes", 20);
    int total = pParser->ReadInteger(section, "num", 1);
    int num = total;

    std::string a = pParser->ReadString(section, "mask");
    std::vector<int> mask;
    StringUtil::splitInt(mask, a, ",");
    num = mask.size();

    int max_boxes = pParser->ReadInteger(section, "max", 90);
    m_layerInfo = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes, max_boxes);
    if (m_layerInfo.outputs != params.inputs)
    {
        printf("Error: layer.outputs == params.inputs \n");
        printf("filters= in the [convolutional]-layer doesn't correspond to classes= or mask= in [yolo]-layer \n");
        return false;
    }

    //assert(m_layerInfo.outputs == params.inputs);
    std::string map_file = pParser->ReadString(section, "map");
    if (!map_file.empty())
    {
      //  m_layerInfo.map = read_map(map_file);
    }

    m_layerInfo.jitter = pParser->ReadFloat(section, "jitter", .2);
    m_layerInfo.focal_loss = pParser->ReadInteger(section, "focal_loss", 0);

    m_layerInfo.ignore_thresh = pParser->ReadFloat(section, "ignore_thresh", .5);
    m_layerInfo.truth_thresh = pParser->ReadFloat(section, "truth_thresh", 1);
    m_layerInfo.random = pParser->ReadInteger(section, "random", 0);

    a = pParser->ReadString(section, "anchors");
    std::vector<float> bias;
    StringUtil::splitFloat(bias, a, ",");
    m_layerInfo.biases = bias;

    //return l;
    return true;
}

NS_JJ_END