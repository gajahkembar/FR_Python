#include "face_embedder.h"

FaceEmbedder::FaceEmbedder(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "embedder"),
      session(nullptr),
      session_options() {
    
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = Ort::Session(env, model_path.c_str(), session_options);
}

std::vector<float> FaceEmbedder::getEmbedding(const cv::Mat& aligned_face) {
    // 1. Convert to float32 and normalize to [-1, 1]
    cv::Mat input;
    aligned_face.convertTo(input, CV_32FC3, 1.0 / 128.0, -127.5 / 128.0);  // (pixel - 127.5) / 128.0

    // 2. Change HWC to CHW
    std::vector<cv::Mat> chw(3);
    for (int i = 0; i < 3; ++i) {
        chw[i] = cv::Mat(112, 112, CV_32FC1);
    }
    cv::split(input, chw);

    std::vector<float> input_tensor_values(3 * 112 * 112);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(input_tensor_values.data() + c * 112 * 112, chw[c].data, 112 * 112 * sizeof(float));
    }

    // 3. Define tensor shape
    std::array<int64_t, 4> input_shape{1, 3, 112, 112};

    // 4. Create tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size()
    );

    // 5. Run inference
    const char* input_names[] = {"input.1"};
    const char* output_names[] = {"683"};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
    );

    // 6. Get output embedding
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    std::vector<float> embedding(output_data, output_data + 512);
    return embedding;
}