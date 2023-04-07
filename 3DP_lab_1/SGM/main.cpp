#include "sgm.h"



int main(int argc, char** argv) {

    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " <right image> <left image> <gt depth map> <output image file> <disparity range> " << endl;
        return -1;
    }

    char *firstFileName = argv[1];
    char *secondFileName = argv[2];
    char *gtFileName = argv[3];
    char *outputFileName = argv[4];
    cv::Mat firstImage;
    cv::Mat secondImage;
    cv::Mat gt;

    firstImage = cv::imread(firstFileName, IMREAD_GRAYSCALE);
    secondImage = cv::imread(secondFileName, IMREAD_GRAYSCALE);
    gt = cv::imread(gtFileName, IMREAD_GRAYSCALE);

    if(!firstImage.data || !secondImage.data) {
        cerr <<  "Could not open or find one of the images!" << endl;
        return -1;
    }

    unsigned int disparityRange = atoi(argv[5]);


    sgm::SGM sgm(disparityRange);
    sgm.set(firstImage, secondImage);
    sgm.compute_disparity();
    sgm.save_disparity(outputFileName);

    std::cerr<<"Right Image MSE error: "<<sgm.compute_mse(gt)<<std::endl;

    return 0;
}
