#include "convolutional_layer.h"

NS_JJ_BEGIN
static inline ACTIVATION get_activation(const std::string& s)
{
    if (s == "logistic") return LOGISTIC;
    if (s == "loggy") return LOGGY;
    if (s == "relu") return RELU;
    if (s ==  "elu") return ELU;
    if (s == "relie") return RELIE;
    if (s == "plse") return PLSE;
    if (s ==  "hardtan") return HARDTAN;
    if (s == "lhtan") return LHTAN;
    if (s == "linear") return LINEAR;
    if (s ==  "ramp") return RAMP;
    if (s ==  "leaky") return LEAKY;
    if (s ==  "tanh") return TANH;
    if (s == "stair") return STAIR;
    //fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

bool ConvolutionLayer::load(const IniParser* pParser, int section, size_params params)
{
    int n = pParser->ReadInteger(section, "filters", 1);
    int size = pParser->ReadInteger(section, "size", 1);
    int stride = pParser->ReadInteger(section, "stride", 1);
    int pad = pParser->ReadInteger(section, "pad", 0);
    int padding = pParser->ReadInteger(section, "padding", 0);
    if (pad) padding = size / 2;

    std::string activation_s = pParser->ReadString(section, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c))
        return false;// ("Layer before convolutional layer must output image.");

    int batch_normalize = pParser->ReadInteger(section, "batch_normalize", 0);
    int binary = pParser->ReadInteger(section, "binary", 0);
    int xnor = pParser->ReadInteger(section, "xnor", 0);
    int use_bin_output = pParser->ReadInteger(section, "bin_output", 0);

    int quantized = params.quantized;
    if (params.index == 0 || activation == LINEAR || (params.index > 1 && stride > 1) || size == 1)
        quantized = 0; // disable Quantized for 1st and last layers

    convolutional_layer layer = make_convolutional_layer(batch, h, w, c, n, size,
        stride, padding, activation, batch_normalize, binary, xnor, params.net->adam, quantized, use_bin_output);

    layer.flipped = pParser->ReadInteger(section, "flipped", 0);
    layer.dot = pParser->ReadFloat(section, "dot", 0);
    if (params.net->adam)
    {
        layer.B1 = params.net->B1;
        layer.B2 = params.net->B2;
        layer.eps = params.net->eps;
    }

    //return layer;
    return true;
}


float rand_uniform(float min, float max)
{
    if (max < min) {
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand() / RAND_MAX * (max - min)) + min;
}

// utils.c
float rand_scale(float s)
{
    float scale = rand_uniform(1, s);
    if (rand() % 2) return scale;
    return 1. / scale;
}

// utils.c
int rand_int(int min, int max)
{
    if (max < min) {
        int s = min;
        min = max;
        max = s;
    }
    int r = (rand() % (max - min + 1)) + min;
    return r;
}

// utils.c
int constrain_int(int a, int min, int max)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2 * l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2 * l.pad - l.size) / l.stride + 1;
}


size_t get_workspace_size(convolutional_layer l)
{
    if (l.xnor)
        return (size_t)l.bit_align*l.size*l.size*l.c * sizeof(float);

    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c * sizeof(float);
}

layer ConvolutionLayer::make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding,
    ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int quantized, int use_bin_output)
{
    int i;
    
    m_layerInfo.type = CONVOLUTIONAL;
    m_layerInfo.quantized = quantized;

    m_layerInfo.h = h;
    m_layerInfo.w = w;
    m_layerInfo.c = c;
    m_layerInfo.n = n;
    m_layerInfo.binary = binary;
    m_layerInfo.xnor = xnor;
    m_layerInfo.use_bin_output = use_bin_output;
    m_layerInfo.batch = batch;
    m_layerInfo.stride = stride;
    m_layerInfo.size = size;
    m_layerInfo.pad = padding;
    m_layerInfo.batch_normalize = batch_normalize;

    m_layerInfo.weights.assign( c*n*size*size, 0.0f);
    m_layerInfo.weights_int8.assign( c*n*size*size, 0);

    m_layerInfo.biases.assign(n, 0.0f);
    m_layerInfo.biases_quant.assign(n, 0.0f);

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2. / (size*size*c));
    for (i = 0; i < c*n*size*size; ++i)
        m_layerInfo.weights[i] = scale * rand_uniform(-1, 1) ;

    int out_h = convolutional_out_height(m_layerInfo);
    int out_w = convolutional_out_width(m_layerInfo);
    m_layerInfo.out_h = out_h;
    m_layerInfo.out_w = out_w;
    m_layerInfo.out_c = n;
    m_layerInfo.outputs = m_layerInfo.out_h * m_layerInfo.out_w * m_layerInfo.out_c;
    m_layerInfo.inputs = m_layerInfo.w * m_layerInfo.h * m_layerInfo.c;

    m_layerInfo.output.assign(m_layerInfo.batch * m_layerInfo.outputs, 0.0f);
    m_layerInfo.output_int8.assign(m_layerInfo.batch * m_layerInfo.outputs, 0);

    if (binary)
    {
        m_layerInfo.binary_weights.assign (c*n*size*size, 0.0f);
        m_layerInfo.cweights.assign( c*n*size*size, 0);
        m_layerInfo.scales.assign(n, 0.0f);
    }

    if (xnor)
    {
        m_layerInfo.binary_weights.assign(c*n*size*size, 0.0f);
        m_layerInfo.binary_input.assign(m_layerInfo.inputs*m_layerInfo.batch, 0.0f);

        int align = 32;// 8;
        int src_align = m_layerInfo.out_h*m_layerInfo.out_w;
        m_layerInfo.bit_align = src_align + (align - src_align % align);

        m_layerInfo.mean_arr.assign(m_layerInfo.n, 0.0f);
    }

    if (batch_normalize)
    {
        m_layerInfo.scales.assign(n, 0.0f);
        //m_layerInfo.scale_updates = calloc(n, sizeof(float));
        for (i = 0; i < n; ++i)
        {
            m_layerInfo.scales[i] = 1;
        }

        m_layerInfo.mean.assign(n, 0.0f);
        m_layerInfo.variance.assign(n, 0.0f);

        //m_layerInfo.mean_delta = calloc(n, sizeof(float));
        //m_layerInfo.variance_delta = calloc(n, sizeof(float));

        m_layerInfo.rolling_mean.assign(n, 0.0f);
        m_layerInfo.rolling_variance.assign(n, 0.0f);
        m_layerInfo.x.assign(m_layerInfo.batch*m_layerInfo.outputs, 0.0f);
        m_layerInfo.x_norm.assign(m_layerInfo.batch*m_layerInfo.outputs, 0.0f);
    }
    if (adam)
    {
        m_layerInfo.adam = 1;
        m_layerInfo.m.assign( c*n*size*size, 0.0f);
        m_layerInfo.v.assign(c*n*size*size, 0.0f);
    }


    m_layerInfo.workspace_size = get_workspace_size(m_layerInfo);
    m_layerInfo.activation = activation;

    m_layerInfo.bflops = (2.0 * m_layerInfo.n * m_layerInfo.size*m_layerInfo.size*m_layerInfo.c * m_layerInfo.out_h*m_layerInfo.out_w) / 1000000000.;
    if (m_layerInfo.xnor && m_layerInfo.use_bin_output) fprintf(stderr, "convXB");
    else if (m_layerInfo.xnor) fprintf(stderr, "convX ");
    else fprintf(stderr, "conv  ");
    fprintf(stderr, "%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d %5.3f BF\n", n, size, size, stride, w, h, c, m_layerInfo.out_w, m_layerInfo.out_h, m_layerInfo.out_c, m_layerInfo.bflops);

    return m_layerInfo;
}


NS_JJ_END