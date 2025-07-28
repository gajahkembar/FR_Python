#include "face_aligner.h"

cv::Mat FaceAligner::align(const cv::Mat& img, const std::vector<cv::Point2f>& landmark) {
    std::vector<cv::Point2f> standard = {
        {38.2946f, 51.6963f},
        {73.5318f, 51.5014f},
        {56.0252f, 71.7366f},
        {41.5493f, 92.3655f},
        {70.7299f, 92.2041f}
    };

    cv::Mat transform = cv::estimateAffinePartial2D(landmark, standard);
    cv::Mat aligned;
    cv::warpAffine(img, aligned, transform, cv::Size(112, 112), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(128,128,128));
    return aligned;
}