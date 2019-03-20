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


bool YoloLayer::load(const IniParser* pParser, int section, size_params params)
{
    int classes = pParser->ReadInteger(section, "classes", 20);
    int total = pParser->ReadInteger(section, "num", 1);
    int num = total;

    std::string a = pParser->ReadString(section, "mask", 0);
    std::vector<int> mask;
    StringUtil::splitInt(mask, a, ",");
    int max_boxes = pParser->ReadInteger(section, "max", 90);
    layer l = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes, max_boxes);
    if (l.outputs != params.inputs)
    {
        printf("Error: l.outputs == params.inputs \n");
        printf("filters= in the [convolutional]-layer doesn't correspond to classes= or mask= in [yolo]-layer \n");
        return false;
    }

    //assert(l.outputs == params.inputs);
    std::string map_file = pParser->ReadString(section, "map", 0);
    if (!map_file.empty())
    {
      //  l.map = read_map(map_file);
    }

    l.jitter = pParser->ReadFloat(section, "jitter", .2);
    l.focal_loss = pParser->ReadInteger(section, "focal_loss", 0);

    l.ignore_thresh = pParser->ReadFloat(section, "ignore_thresh", .5);
    l.truth_thresh = pParser->ReadFloat(section, "truth_thresh", 1);
    l.random = pParser->ReadInteger(section, "random", 0);

    a = pParser->ReadString(section, "anchors", 0);
    std::vector<float> bias;
    StringUtil::splitFloat(bias, a, ",");
    l.biases = bias;

    //return l;
    return true;
}

NS_JJ_END