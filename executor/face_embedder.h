#ifndef FACE_EMBEDDER_H
#define FACE_EMBEDDER_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class FaceEmbedder {
public:
    FaceEmbedder(const std::string& model_path);
    std::vector<float> getEmbedding(const cv::Mat& aligned_face);

private:
    Ort::Env env;
    Ort::Session session;
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;
};

#endif