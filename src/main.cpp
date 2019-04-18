#include <stdio.h>
#include <stdlib.h>

#include "box.h"
#include "pthread.h"

#include "detector.h"
#include "utils/args_util.h"


//yolo_cpu.exe detector [train/test/valid] [object names] [cfg] [weights(optional)] [filename(optional)]
void run_detector(int argc, char **argv)
{
    float thresh = ArgsUtil::find_float_arg(argc, argv, "-thresh", .25);
    if (argc < 4) {
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    int clear = 0;                // find_arg(argc, argv, "-clear");

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6] : 0;


    if (0 == strcmp(argv[2], "test"))
    {
        // xxx.exe detector test coco.names yolov3-tiny.cfg yolov3-tiny.weights -thresh 0.2 dog.jpg
        JJ::Detector::instance()->test(datacfg, cfg, weights, filename, thresh);
    }
    else if(0 == strcmp(argv[2], "train"))
    {
        // https://github.com/PowerOfDream/yolo-transfer-demo
        // 利用迁移学习来训练
        // xxx.exe detector train coco.data yolov3-tiny.cfg darknet53.conv.74 -dont_show -mjpeg_port 8090 -map ? not begin yet.
        JJ::Detector::instance()->train(datacfg, cfg, weights);
    }
}


int main(int argc, char **argv)
{
    run_detector(argc, argv);
    return 0;
}
