#include "service_impl.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include "../executor/face_aligner.h"
#include "../executor/face_embedder.h"
#include "../proto/driver.pb.h"
#include "../proto/driver.grpc.pb.h"
#include "../proto/executor.pb.h"
#include "../proto/executor.grpc.pb.h"
#include "../executor/base64.h"
#include "../model/mtcnn/mtcnn/detector.h"
using Executor::ExecutorService;

DriverServiceImpl::DriverServiceImpl(int port_) : port(port_),
    face_embedder("/Users/rahmandaafridiansyah/Downloads/fr_cpp_/model/w600k_r50.onnx") {
    face_aligner = FaceAligner();

    ProposalNetwork::Config pConfig{ "../model/mtcnn/models/det1.prototxt", "../model/mtcnn/models/det1.caffemodel", 0.6f, 0.7f };
    RefineNetwork::Config rConfig{ "../model/mtcnn/models/det2.prototxt", "../model/mtcnn/models/det2.caffemodel", 0.7f };
    OutputNetwork::Config oConfig{ "../model/mtcnn/models/det3.prototxt", "../model/mtcnn/models/det3.caffemodel", 0.7f };
    detector = std::make_unique<MTCNNDetector>(pConfig, rConfig, oConfig);

    for (int port = 6001; port <= 6006; ++port) {
        std::string addr = "localhost:" + std::to_string(port);
        executor_addresses.push_back(addr);
        executor_stubs.push_back(
            Executor::ExecutorService::NewStub(
                grpc::CreateChannel(addr, grpc::InsecureChannelCredentials())
            )
        );
    }
    std::srand(std::time(nullptr) + port);  
    executor_index = std::rand() % executor_stubs.size();
}

void DriverServiceImpl::log_to_file(const std::string& message) const {
    std::ofstream log_file("driver.log", std::ios::app);
    if (log_file.is_open()) {
        auto now = std::chrono::system_clock::now();
        auto t_c = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&t_c), "%Y-%m-%d %H:%M:%S");
        log_file << "[" << ss.str() << "] [port " << port << "] " << message << std::endl;
    }
}

Executor::ExecutorService::Stub* DriverServiceImpl::getNextExecutorStub() {
    std::lock_guard<std::mutex> lock(executor_mutex);
    auto stub = executor_stubs[executor_index].get();
    executor_index = (executor_index + 1) % executor_stubs.size();
    return stub;
}

grpc::Status DriverServiceImpl::RouteRegisterFace(
    grpc::ServerContext* context,
    const Driver::RegisterRequest* request,
    Driver::RegisterResponse* response
) {
    try {
        log_to_file("⚙️  Called RouteRegisterFace: " + request->user_id());

        size_t index_now;
        Executor::ExecutorService::Stub* stub;
        {
            std::lock_guard<std::mutex> lock(executor_mutex);
            index_now = executor_index;
            stub = executor_stubs[executor_index].get();
            executor_index = (executor_index + 1) % executor_stubs.size();
        }
        std::string executor_addr = executor_addresses[index_now];
        log_to_file("🔁 Routing to Executor at " + executor_addr);

        Executor::RegisterRequest executor_req;
        executor_req.set_user_id(request->user_id());
        executor_req.set_name(request->name());
        executor_req.set_origin(request->origin());
        executor_req.set_image_data(request->image_data());

        Executor::RegisterReply executor_resp;
        grpc::ClientContext exec_context;
        grpc::Status status = stub->RegisterFace(&exec_context, executor_req, &executor_resp);

        if (!status.ok()) {
            log_to_file("❌ Register failed: " + status.error_message());
            response->set_message("FAILED: " + status.error_message());
            return status;
        }

        log_to_file("✅ Registered: " + executor_resp.user_id() + " | name=" + request->name() + ", origin=" + request->origin());
        response->set_user_id(executor_resp.user_id());
        response->set_message(executor_resp.message());
        return grpc::Status::OK;

    } catch (const std::exception& e) {
        log_to_file(std::string("❌ Exception in RouteRegisterFace: ") + e.what());
        response->set_message("EXCEPTION");
        return grpc::Status(grpc::StatusCode::INTERNAL, "Exception occurred");
    }
}

grpc::Status DriverServiceImpl::RouteIdentifyFace(
    grpc::ServerContext* context,
    const Driver::ImageQuery* request,
    Driver::IdentifyResult* response
) {
    try {
        log_to_file("⚙️  Called RouteIdentifyFace");
        
        size_t index_now;
        Executor::ExecutorService::Stub* stub;
        {
            std::lock_guard<std::mutex> lock(executor_mutex);
            index_now = executor_index;
            stub = executor_stubs[executor_index].get();
            executor_index = (executor_index + 1) % executor_stubs.size();
        }
        std::string executor_addr = executor_addresses[index_now];
        log_to_file("🔁 Routing to Executor at " + executor_addr);
        
        Executor::IdentifyRequest exec_req;
        exec_req.set_image_data(request->image_data());
        
        Executor::IdentifyReply exec_resp;
        grpc::ClientContext exec_context;
        grpc::Status status = stub->IdentifyFace(&exec_context, exec_req, &exec_resp);
        
        if (!status.ok()) {
            log_to_file("❌ Identify failed: " + status.error_message());
            return status;
        }
        
        for (const auto& face_result : exec_resp.results()) {
            Executor::FaceResult* dr_result = response->add_results();
            dr_result->set_face_index(face_result.face_index());
            dr_result->set_crop_image(face_result.crop_image());
            
            log_to_file("🔍 Face-" + std::to_string(face_result.face_index()) + ":");
            
            for (const auto& candidate : face_result.top_matches()) {
                auto* dr_cand = dr_result->add_top_matches();
                dr_cand->set_user_id(candidate.user_id());
                dr_cand->set_confidence(candidate.confidence());
                
                log_to_file("   - match " + candidate.user_id() + " (sim=" + std::to_string(candidate.confidence()) + ")");
            }
        }
        
        log_to_file("✅ Identify selesai, wajah terdeteksi: " + std::to_string(exec_resp.results_size()));
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        log_to_file("❌ Exception in RouteIdentifyFace: " + std::string(e.what()));
        return grpc::Status(grpc::StatusCode::INTERNAL, "Exception occurred");
    }
}

grpc::Status DriverServiceImpl::RouteVerifyFace(
    grpc::ServerContext* context,
    const Driver::VerifyImagePair* request,
    Driver::VerifyResult* response
) {
    try {
        log_to_file("⚙️  Called RouteVerifyFace");

        Executor::VerifyRequest exec_req;
        exec_req.set_image1(request->image1());
        exec_req.set_image2(request->image2());

        Executor::VerifyReply exec_reply;
        grpc::ClientContext exec_ctx;

        auto* stub = getNextExecutorStub();
        grpc::Status status = stub->VerifyFace(&exec_ctx, exec_req, &exec_reply);

        if (!status.ok()) {
            log_to_file("❌ VerifyFace failed: " + status.error_message());
            return status;
        }

        response->set_similarity(exec_reply.similarity());
        response->set_result(exec_reply.result());

        log_to_file("✅ RouteVerifyFace done: sim=" +
                    std::to_string(exec_reply.similarity()) +
                    " | result=" + exec_reply.result());

        return grpc::Status::OK;

    } catch (const std::exception& e) {
        log_to_file("❌ Exception in RouteVerifyFace: " + std::string(e.what()));
        return grpc::Status(grpc::StatusCode::INTERNAL, "Exception occurred");
    }
}