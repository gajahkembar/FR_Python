#ifndef FACE_ALIGNER_H
#define FACE_ALIGNER_H

#include <opencv2/opencv.hpp>
#include <vector>

class FaceAligner {
public:
    // Align wajah ke ukuran 112x112 dari landmark
    static cv::Mat align(const cv::Mat& img, const std::vector<cv::Point2f>& landmark);
};

#endif