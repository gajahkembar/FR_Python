#pragma once

#include <grpcpp/grpcpp.h>
#include "../proto/driver.pb.h"
#include "../proto/driver.grpc.pb.h"
#include "../proto/executor.pb.h"
#include "../proto/executor.grpc.pb.h"
#include "../executor/face_aligner.h"
#include "../executor/face_embedder.h"
#include "../model/mtcnn/mtcnn/detector.h"
#include <vector>
#include <memory>
#include <mutex>

using Executor::ExecutorService;

class DriverServiceImpl final : public Driver::DriverService::Service {
public:
    explicit DriverServiceImpl(int port);

    grpc::Status RouteRegisterFace(grpc::ServerContext* context,
                                const Driver::RegisterRequest* request,
                                Driver::RegisterResponse* response) override;

    grpc::Status RouteIdentifyFace(grpc::ServerContext* context,
                                const Driver::ImageQuery* request,
                                Driver::IdentifyResult* response) override;

    grpc::Status RouteVerifyFace(grpc::ServerContext* context,
                                const Driver::VerifyImagePair* request,
                                Driver::VerifyResult* response) override;

private:
    std::vector<std::string> executor_addresses;
    std::vector<std::unique_ptr<Executor::ExecutorService::Stub>> executor_stubs;
    Executor::ExecutorService::Stub* getNextExecutorStub();
    size_t executor_index;
    std::mutex executor_mutex;
    int port;
    FaceAligner face_aligner;
    FaceEmbedder face_embedder{"/mnt/d/Kantor/fr_cpp_/model/w600k_r50.onnx"};
    void log_to_file(const std::string& message) const;
    std::unique_ptr<MTCNNDetector> detector;
};