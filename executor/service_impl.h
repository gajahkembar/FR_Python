#pragma once

#include <memory>
#include <string>
#include <grpcpp/grpcpp.h>
#include "../proto/executor.grpc.pb.h"
#include "face_embedder.h"
#include "../model/mtcnn/mtcnn/detector.h"
#include "src/storage/redis_client.h"
#include "src/storage/postgres_client.h"

class ExecutorServiceImpl final : public Executor::ExecutorService::Service {
public:
    explicit ExecutorServiceImpl(int port);

    grpc::Status RegisterFace(grpc::ServerContext* context,
                              const Executor::RegisterRequest* request,
                              Executor::RegisterReply* reply) override;

    grpc::Status IdentifyFace(grpc::ServerContext* context,
                              const Executor::IdentifyRequest* request,
                              Executor::IdentifyReply* reply) override;

    grpc::Status VerifyFace(grpc::ServerContext* context,
                              const Executor::VerifyRequest* request,
                              Executor::VerifyReply* reply) override;

private:
    int port;
    std::unique_ptr<MTCNNDetector> detector;
    FaceEmbedder embedder;
    void log_to_file(const std::string& message) const;
    std::unique_ptr<RedisClient> redis;
    std::unique_ptr<PostgresClient> postgres;
};