#include "detector.h"
#include "parser/ini_parser.h"
#include "layers/convolutional_layer.h"


NS_JJ_BEGIN
void Detector::loadData(const char* cfgfile, const char* weightfile)
{
    JJ::network* net = readConfigFile(cfgfile, 1, 0);
    if (weightfile)
    {
        // 2. read weight file, init the layer information in the network. cutoff == net.n, means do not cut off any layer
        //load_weights_upto_cpu(&net, weightfile, net.n);    // parser.c
    }
}

void Detector::detectImage(char **names, char *filename, float thresh, int quantized, int dont_show)
{

}


JJ::network* Detector::readConfigFile(const char* filename, int batch, int quantized)
{
    IniParser parser(filename);
    
    JJ::network* pNetWork = new JJ::network;
    pNetWork->n = parser.GetSectionCount() - 1; //  layer count
    pNetWork->quantized = quantized;
    pNetWork->do_input_calibration = 0;
    

    size_params params;
    params.quantized = quantized;

    if (!parseNetOptions(&parser, pNetWork))
        return nullptr;

    params.h = pNetWork->h;
    params.w = pNetWork->w;
    params.c = pNetWork->c;
    params.inputs = pNetWork->inputs;
    if (batch > 0)
        pNetWork->batch = batch;
    params.batch = pNetWork->batch;
    params.time_steps = pNetWork->time_steps;
    params.net = pNetWork;



    size_t workspace_size = 0;    

    // read convolutional, maxpool
    for(int i=0; i<pNetWork->n; i++)
    {
        int sectionIndex = i + 1;
        params.index = i;
        fprintf(stderr, "%5d ", i);
        std::string sectionName = parser.GetSectionByIndex(sectionIndex);
        int itemCnt = parser.GetSectionItemCount(sectionIndex);

        //s = (section *)n->val;
        //options = s->options;

        // parse data into this layer and save to net
        JJ::layer l;
        JJ::LAYER_TYPE lt = string_to_layer_type(sectionName.c_str());

        // [convolutional]
        if (lt == JJ::CONVOLUTIONAL)
        {
            //l = parse_convolutional(options, params);
            ConvolutionLayer* pLayer = new ConvolutionLayer;
            pLayer->load(&parser, sectionIndex, params);
            l = pLayer->m_layerInfo;
        }
        //else if (lt == REGION) {
            //l = parse_region(options, params);
        //}
        else if (lt == YOLO) {
            //l = parse_yolo(options, params);
        }
        //else if (lt == SOFTMAX) {
            //l = parse_softmax(options, params);
            //net.hierarchy = l.softmax_tree;
        //}
        else if (lt == MAXPOOL) {
            //l = parse_maxpool(options, params);
        }
        //else if (lt == REORG) {
            //l = parse_reorg(options, params);
        //}
        else if (lt == ROUTE) {
            //l = parse_route(options, params, net);
        }
        else if (lt == UPSAMPLE) {
            //l = parse_upsample(options, params, net);
        }
        //else if (lt == SHORTCUT) {
            //l = parse_shortcut(options, params, net);
        //}
        else {
            fprintf(stderr, "Type not recognized: %s\n", sectionName.c_str());
        }

        l.dontload = parser.ReadInteger(sectionIndex, "dontload", 0);
        l.dontloadscales = parser.ReadInteger(sectionIndex, "dontloadscales", 0);
        //option_unused(options);

        // save this layer
        pNetWork->layers.push_back(l);
        if (l.workspace_size > workspace_size)
            workspace_size = l.workspace_size;

        //free_section(s);
        //n = n->next;
        //++count;
        //if (n)
        {
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }


    //free_list(sections);
    pNetWork->outputs = NetWork::get_network_output_size(pNetWork);
    pNetWork->output = NetWork::get_network_output(pNetWork);
    if (workspace_size)
    {
        //printf("%ld\n", workspace_size);
        pNetWork->workspace.assign(workspace_size, 0.0f);
    }
    return pNetWork;
}



// convolutional_layer Detector::parse_convolutional(const IniParser* pParser, int section, size_params params)
// {
// }

/*
// parser.c
layer Detector::parse_region(list *options, size_params params)
{
    int coords = pParser->ReadInteger(section, "coords", 4);
    int classes = pParser->ReadInteger(section, "classes", 20);
    int num = pParser->ReadInteger(section, "num", 1);

    layer l = make_region_layer(params.batch, params.w, params.h, num, classes, coords);
    assert(l.outputs == params.inputs);

    l.log = pParser->ReadInteger(section, "log", 0);
    l.sqrt = pParser->ReadInteger(section, "sqrt", 0);

    l.softmax = pParser->ReadInteger(section, "softmax", 0);
    l.max_boxes = pParser->ReadInteger(section, "max", 30);
    l.jitter = option_find_float(options, "jitter", .2);
    l.rescore = pParser->ReadInteger(section, "rescore", 0);

    l.thresh = option_find_float(options, "thresh", .5);
    l.classfix = pParser->ReadInteger(section, "classfix", 0);
    l.absolute = pParser->ReadInteger(section, "absolute", 0);
    l.random = pParser->ReadInteger(section, "random", 0);

    l.coord_scale = option_find_float(options, "coord_scale", 1);
    l.object_scale = option_find_float(options, "object_scale", 1);
    l.noobject_scale = option_find_float(options, "noobject_scale", 1);
    l.class_scale = option_find_float(options, "class_scale", 1);
    l.bias_match = pParser->ReadInteger(section, "bias_match", 0);

    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) l.softmax_tree = read_tree(tree_file);
    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    char *a = option_find_str(options, "anchors", 0);
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == ',') ++n;
        }
        for (i = 0; i < n; ++i) {
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',') + 1;
        }
    }
    return l;
}

// parser.c
int * Detector::parse_yolo_mask(char *a, int *num)
{
    int *mask = 0;
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == ',') ++n;
        }
        mask = calloc(n, sizeof(int));
        for (i = 0; i < n; ++i) {
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',') + 1;
        }
        *num = n;
    }
    return mask;
}

// parser.c
layer Detector::parse_yolo(list *options, size_params params)
{
    int classes = pParser->ReadInteger(section, "classes", 20);
    int total = pParser->ReadInteger(section, "num", 1);
    int num = total;

    char *a = option_find_str(options, "mask", 0);
    int *mask = parse_yolo_mask(a, &num);
    int max_boxes = pParser->ReadInteger(section, "max", 90);
    layer l = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes, max_boxes);
    if (l.outputs != params.inputs) {
        printf("Error: l.outputs == params.inputs \n");
        printf("filters= in the [convolutional]-layer doesn't correspond to classes= or mask= in [yolo]-layer \n");
        exit(EXIT_FAILURE);
    }
    //assert(l.outputs == params.inputs);
    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    l.jitter = option_find_float(options, "jitter", .2);
    l.focal_loss = pParser->ReadInteger(section, "focal_loss", 0);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
    l.random = pParser->ReadInteger(section, "random", 0);

    a = option_find_str(options, "anchors", 0);
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == ',') ++n;
        }
        for (i = 0; i < n && i < total * 2; ++i) {
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',') + 1;
        }
    }
    return l;
}

// parser.c
softmax_layer Detector::parse_softmax(list *options, size_params params)
{
    int groups = pParser->ReadInteger(section, "groups", 1);
    softmax_layer layer = make_softmax_layer(params.batch, params.inputs, groups);
    layer.temperature = option_find_float_quiet(options, "temperature", 1);
    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) layer.softmax_tree = read_tree(tree_file);
    return layer;
}

// parser.c
maxpool_layer Detector::parse_maxpool(list *options, size_params params)
{
    int stride = pParser->ReadInteger(section, "stride", 1);
    int size = pParser->ReadInteger(section, "size", stride);
    int padding = pParser->ReadInteger(section, "padding", size - 1);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c)) error("Layer before maxpool layer must output image.");

    maxpool_layer layer = make_maxpool_layer(batch, h, w, c, size, stride, padding);
    return layer;
}

// parser.c
layer Detector::parse_reorg(list *options, size_params params)
{
    int stride = pParser->ReadInteger(section, "stride", 1);
    int reverse = pParser->ReadInteger(section, "reverse", 0);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c)) error("Layer before reorg layer must output image.");

    layer layer = make_reorg_layer(batch, w, h, c, stride, reverse);
    return layer;
}

// parser.c
layer Detector::parse_upsample(list *options, size_params params, network net)
{

    int stride = pParser->ReadInteger(section, "stride", 2);
    layer l = make_upsample_layer(params.batch, params.w, params.h, params.c, stride);
    l.scale = option_find_float_quiet(options, "scale", 1);
    return l;
}

// parser.c
layer Detector::parse_shortcut(list *options, size_params params, network net)
{
    char *l = option_find(options, "from");
    int index = atoi(l);
    if (index < 0) index = params.index + index;

    int batch = params.batch;
    layer from = net.layers[index];

    layer s = make_shortcut_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);

    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    s.activation = activation;
    return s;
}

// parser.c
route_layer Detector::parse_route(list *options, size_params params, network net)
{
    char *l = option_find(options, "layers");
    int len = strlen(l);
    if (!l) error("Route Layer must specify input layers");
    int n = 1;
    int i;
    for (i = 0; i < len; ++i) {
        if (l[i] == ',') ++n;
    }

    int *layers = calloc(n, sizeof(int));
    int *sizes = calloc(n, sizeof(int));
    for (i = 0; i < n; ++i) {
        int index = atoi(l);
        l = strchr(l, ',') + 1;
        if (index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = net.layers[index].outputs;
    }
    int batch = params.batch;

    route_layer layer = make_route_layer(batch, n, layers, sizes);

    convolutional_layer first = net.layers[layers[0]];
    layer.out_w = first.out_w;
    layer.out_h = first.out_h;
    layer.out_c = first.out_c;
    for (i = 1; i < n; ++i) {
        int index = layers[i];
        convolutional_layer next = net.layers[index];
        if (next.out_w == first.out_w && next.out_h == first.out_h) {
            layer.out_c += next.out_c;
        }
        else {
            layer.out_h = layer.out_w = layer.out_c = 0;
        }
    }

    return layer;
}
*/

JJ::learning_rate_policy Detector::get_policy(const char *s)
{
    if (strcmp(s, "random") == 0) return JJ::RANDOM;
    if (strcmp(s, "poly") == 0) return JJ::POLY;
    if (strcmp(s, "constant") == 0) return JJ::CONSTANT;
    if (strcmp(s, "step") == 0) return JJ::STEP;
    if (strcmp(s, "exp") == 0) return JJ::EXP;
    if (strcmp(s, "sigmoid") == 0) return JJ::SIG;
    if (strcmp(s, "steps") == 0) return JJ::STEPS;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return JJ::CONSTANT;
}

JJ::LAYER_TYPE Detector::string_to_layer_type(const std::string& type)
{
    if (type == "yolo") return JJ::YOLO;
    if (type == "region") return JJ::REGION;
    if (type ==  "conv" || type == "convolutional") return JJ::CONVOLUTIONAL;
    if (type == "net"  || type == "network") return JJ::NETWORK;
    if (type, "max"|| type == "maxpool") return JJ::MAXPOOL;
    if (type, "reorg") return JJ::REORG;
    if (type, "upsample") return JJ::UPSAMPLE;
    if (type, "shortcut") return JJ::SHORTCUT;
    if (type, "soft" || type == "softmax") return JJ::SOFTMAX;
    if (type, "route") return JJ::ROUTE;
    return JJ::BLANK;
}

bool Detector::parseNetOptions(const IniParser* pIniParser, JJ::network* net)
{
    int netSectionIndex = 0;
    net->batch = pIniParser->ReadInteger(netSectionIndex, "batch", 1);
    net->learning_rate = pIniParser->ReadFloat(netSectionIndex, "learning_rate", .001);
    net->momentum = pIniParser->ReadFloat(netSectionIndex, "momentum", .9);
    net->decay = pIniParser->ReadFloat(netSectionIndex, "decay", .0001);
    int subdivs = pIniParser->ReadInteger(netSectionIndex, "subdivisions", 1);
    net->time_steps = pIniParser->ReadInteger(netSectionIndex, "time_steps", 1);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;

    std::string input_calibration= pIniParser->ReadString(netSectionIndex, "input_calibration", 0);
    if (!input_calibration.empty())
    {
        splitFloat(net->input_calibration, input_calibration, ",");
    }

    net->adam = pIniParser->ReadInteger(netSectionIndex, "adam", 0);
    if (net->adam)
    {
        net->B1 = pIniParser->ReadFloat(netSectionIndex, "B1", .9);
        net->B2 = pIniParser->ReadFloat(netSectionIndex, "B2", .999);
        net->eps = pIniParser->ReadFloat(netSectionIndex, "eps", .000001);
    }

    net->h = pIniParser->ReadInteger(netSectionIndex, "height", 0);
    net->w = pIniParser->ReadInteger(netSectionIndex, "width", 0);
    net->c = pIniParser->ReadInteger(netSectionIndex, "channels", 0);
    net->inputs = pIniParser->ReadInteger(netSectionIndex, "inputs", net->h * net->w * net->c);
    net->max_crop = pIniParser->ReadInteger(netSectionIndex, "max_crop", net->w * 2);
    net->min_crop = pIniParser->ReadInteger(netSectionIndex, "min_crop", net->w);

    net->angle = pIniParser->ReadFloat(netSectionIndex, "angle", 0);
    net->aspect = pIniParser->ReadFloat(netSectionIndex, "aspect", 1);
    net->saturation = pIniParser->ReadFloat(netSectionIndex, "saturation", 1);
    net->exposure = pIniParser->ReadFloat(netSectionIndex, "exposure", 1);
    net->hue = pIniParser->ReadFloat(netSectionIndex, "hue", 0);

    if (!net->inputs && !(net->h && net->w && net->c))
        return false;
        //error("No input parameters supplied");

    std::string policy_s = pIniParser->ReadString(netSectionIndex, "policy", "constant");
    net->policy = get_policy(policy_s.c_str());
    net->burn_in = pIniParser->ReadInteger(netSectionIndex, "burn_in", 0);
    if (net->policy == JJ::STEP)
    {
        net->step = pIniParser->ReadInteger(netSectionIndex, "step", 1);
        net->scale = pIniParser->ReadFloat(netSectionIndex, "scale", 1);
    }
    else if (net->policy == JJ::STEPS)
    {
        std::string l = pIniParser->ReadString(netSectionIndex, "steps");
        std::string p = pIniParser->ReadString(netSectionIndex, "scales");
        if (l.empty() || p.empty())
            return false;// ("STEPS policy must have steps and scales in cfg file");

        splitInt(net->steps, l, ",");
        splitFloat(net->scales, p, ",");

        //net->num_steps = net->steps.size();
    }
    else if (net->policy == JJ::EXP)
    {
        net->gamma = pIniParser->ReadFloat(netSectionIndex, "gamma", 1);
    }
    else if (net->policy == JJ::SIG)
    {
        net->gamma = pIniParser->ReadFloat(netSectionIndex, "gamma", 1);
        net->step = pIniParser->ReadInteger(netSectionIndex, "step", 1);
    }
    else if (net->policy == JJ::POLY || net->policy == JJ::RANDOM)
    {
        net->power = pIniParser->ReadFloat(netSectionIndex, "power", 1);
    }
    net->max_batches = pIniParser->ReadInteger(netSectionIndex, "max_batches", 0);
    return true;
}



void Detector::splitString(std::vector<std::string>& result, const std::string & str, const std::string& delims)
{
    if (0 == str.compare(""))
    {
        return;
    }

    size_t start, pos;
    start = 0;
    do
    {
        pos = str.find(delims, start);
        if (pos == std::string::npos)
        {
            std::string tmp_str = str.substr(start);
            tmp_str = Trim(tmp_str, 10); // \n
            tmp_str = Trim(tmp_str, 13); // \r
            tmp_str = Trim(tmp_str, 32); // space
            result.push_back(tmp_str);
            break;
        }
        else
        {
            std::string tmp_str = str.substr(start, pos - start);
            tmp_str = Trim(tmp_str, 10); // 
            tmp_str = Trim(tmp_str, 13); // 
            tmp_str = Trim(tmp_str, 32); // 
            result.push_back(tmp_str);
            start = pos + delims.length();
        }

    } while (pos != std::string::npos);
    return;
}


void Detector::splitInt(std::vector<int>& result, const std::string & str, const std::string& delims)
{
    if (0 == str.compare(""))
    {
        return;
    }

    size_t start, pos;
    start = 0;
    do
    {
        pos = str.find(delims, start);
        if (pos == std::string::npos)
        {
            std::string tmp_str = str.substr(start);
            tmp_str = Trim(tmp_str, 10); // \n
            tmp_str = Trim(tmp_str, 13); // \r
            tmp_str = Trim(tmp_str, 32); // space
            result.push_back(atoi(tmp_str.c_str()));
            break;
        }
        else
        {
            std::string tmp_str = str.substr(start, pos - start);
            tmp_str = Trim(tmp_str, 10); // 
            tmp_str = Trim(tmp_str, 13); // 
            tmp_str = Trim(tmp_str, 32); // 
            result.push_back(atoi(tmp_str.c_str()));
            start = pos + delims.length();
        }

    } while (pos != std::string::npos);
    return;
}


void Detector::splitFloat(std::vector<float>& result, const std::string & str, const std::string& delims)
{
    if (0 == str.compare(""))
    {
        return;
    }

    size_t start, pos;
    start = 0;
    do
    {
        pos = str.find(delims, start);
        if (pos == std::string::npos)
        {
            std::string tmp_str = str.substr(start);
            tmp_str = Trim(tmp_str, 10); // \n
            tmp_str = Trim(tmp_str, 13); // \r
            tmp_str = Trim(tmp_str, 32); // space
            result.push_back(atof(tmp_str.c_str()));
            break;
        }
        else
        {
            std::string tmp_str = str.substr(start, pos - start);
            tmp_str = Trim(tmp_str, 10); // 
            tmp_str = Trim(tmp_str, 13); // 
            tmp_str = Trim(tmp_str, 32); // 
            result.push_back(atof(tmp_str.c_str()));
            start = pos + delims.length();
        }

    } while (pos != std::string::npos);
    return;
}

std::string Detector::Trim(const std::string& str, const char ch)
{
    if (str.empty())
    {
        return "";
    }

    if (str[0] != ch && str[str.size() - 1] != ch)
    {
        return str;
    }

    size_t pos_begin = str.find_first_not_of(ch, 0);
    size_t pos_end = str.find_last_not_of(ch, str.size());

    if (pos_begin == std::string::npos || pos_end == std::string::npos)
    {
        return "";
    }

    return str.substr(pos_begin, pos_end - pos_begin + 1);
}


NS_JJ_END