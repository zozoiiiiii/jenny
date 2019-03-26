#include "yolo_layer.h"
#include "convolutional_layer.h"


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
    //l.max_boxes = max_boxes;
   // l.truths = l.max_boxes*(4 + 1);    // 90*(4 + 1);
    l.output = (float*)calloc(batch*l.outputs, sizeof(float));
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








void YoloLayer::forward_layer_cpu(network_state state)
{
    layer& l = m_layerInfo;

    int b, n;
    memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b)
    {
        for (n = 0; n < l.n; ++n)
        {
            int index = entry_index(b, n*l.w*l.h, 0);
            ConvolutionLayer::activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);
            index = entry_index(b, n*l.w*l.h, 4);
            ConvolutionLayer::activate_array(l.output + index, (1 + l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif

}



void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{
    int i;
    int new_w = 0;
    int new_h = 0;
    if (letter) {
        if (((float)netw / w) < ((float)neth / h)) {
            new_w = netw;
            new_h = (h * netw) / w;
        }
        else {
            new_h = neth;
            new_w = (w * neth) / h;
        }
    }
    else {
        new_w = netw;
        new_h = neth;
    }
    for (i = 0; i < n; ++i) {
        box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
        b.w *= (float)netw / new_w;
        b.h *= (float)neth / new_h;
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

// yolo_layer.c
box get_yolo_box(float *x, const std::vector<float>& biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0 * stride]) / lw;
    b.y = (j + x[index + 1 * stride]) / lh;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
}

int YoloLayer::get_yolo_detections(int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter)
{
    layer& l = m_layerInfo;

    int i, j, n;
    float *predictions = l.output;
    //if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i) {
        int row = i / l.w;
        int col = i % l.w;
        for (n = 0; n < l.n; ++n) {
            int obj_index = entry_index(0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            //if (objectness <= thresh) continue;   // incorrect behavior for Nan values
            if (objectness > thresh) {
                int box_index = entry_index(0, n*l.w*l.h + i, 0);
                dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
                dets[count].objectness = objectness;
                dets[count].classes = l.classes;
                for (j = 0; j < l.classes; ++j)
                {
                    int class_index = entry_index(0, n*l.w*l.h + i, 4 + 1 + j);
                    float prob = objectness * predictions[class_index];
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
                ++count;
            }
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
    return count;
}



NS_JJ_END