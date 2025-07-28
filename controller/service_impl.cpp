#include "service_impl.h"
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

ControllerServiceImpl::ControllerServiceImpl(int port_) : port(port_), driver_index(0) {
    for (int port = 1967; port <= 1969; ++port) {
        std::string addr = "localhost:" + std::to_string(port);
        driver_addresses.push_back(addr);
        driver_stubs.push_back(
            Driver::DriverService::NewStub(
                grpc::CreateChannel(addr, grpc::InsecureChannelCredentials())
            )
        );
    }
    std::srand(std::time(nullptr) + port_);
    driver_index = std::rand() % driver_stubs.size();
}

void ControllerServiceImpl::log_to_file(const std::string& message) const {
    std::ofstream log_file("controller.log", std::ios::app);
    if (log_file.is_open()) {
        auto now = std::chrono::system_clock::now();
        auto t_c = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&t_c), "%Y-%m-%d %H:%M:%S");
        log_file << "[" << ss.str() << "] [port " << port << "] " << message << std::endl;
    }
}

Driver::DriverService::Stub* ControllerServiceImpl::getNextDriverStub() {
    std::lock_guard<std::mutex> lock(driver_mutex);
    Driver::DriverService::Stub* stub = driver_stubs[driver_index].get();
    driver_index = (driver_index + 1) % driver_stubs.size();
    return stub;
}

grpc::Status ControllerServiceImpl::RouteRegisterFace(grpc::ServerContext*,
    const Controller::RegisterRequest* request,
    Controller::RegisterReply* reply) {

    log_to_file("⚙️  Called RouteRegisterFace");

    Driver::RegisterRequest dr_req;
    dr_req.set_user_id(request->user_id());
    dr_req.set_name(request->name());
    dr_req.set_origin(request->origin());
    dr_req.set_image_data(request->image_data());

    Driver::RegisterResponse dr_resp;
    grpc::ClientContext ctx;

    auto* stub = getNextDriverStub();
    grpc::Status status = stub->RouteRegisterFace(&ctx, dr_req, &dr_resp);

    if (!status.ok()) {
        log_to_file("❌ RouteRegisterFace failed: " + status.error_message());
        return status;
    }

    reply->set_user_id(dr_resp.user_id());
    reply->set_message(dr_resp.message());

    log_to_file("✅ Registered via driver, user_id=" + dr_resp.user_id());
    return grpc::Status::OK;
}

grpc::Status ControllerServiceImpl::RouteIdentifyFace(grpc::ServerContext*,
    const Controller::IdentifyRequest* request,
    Controller::IdentifyReply* reply) {

    log_to_file("⚙️  Called RouteIdentifyFace");

    Driver::ImageQuery dr_req;
    dr_req.set_image_data(request->image_data());

    Driver::IdentifyResult dr_resp;
    grpc::ClientContext ctx;

    auto* stub = getNextDriverStub();
    grpc::Status status = stub->RouteIdentifyFace(&ctx, dr_req, &dr_resp);

    if (!status.ok()) {
        log_to_file("❌ RouteIdentifyFace failed: " + status.error_message());
        return status;
    }

    for (const auto& face : dr_resp.results()) {
        auto* face_result = reply->add_results();
        face_result->set_face_index(face.face_index());
        face_result->set_crop_image(face.crop_image());

        // 🔍 Log untuk face index
        log_to_file("🔍 Face-" + std::to_string(face.face_index()) + ":");

        for (const auto& match : face.top_matches()) {
            auto* m = face_result->add_top_matches();
            m->set_user_id(match.user_id());
            m->set_confidence(match.confidence());

            // 📝 Log setiap match candidate
            log_to_file("   - match " + match.user_id() +
                        " (sim=" + std::to_string(match.confidence()) + ")");
        }
    }

    log_to_file("✅ IdentifyFace done. Total result = " + std::to_string(dr_resp.results_size()));
    return grpc::Status::OK;
}

grpc::Status ControllerServiceImpl::RouteVerifyFace(grpc::ServerContext*,
    const Controller::VerifyRequest* request,
    Controller::VerifyReply* reply) {

    log_to_file("⚙️  Called RouteVerifyFace");

    Driver::VerifyImagePair dr_req;
    dr_req.set_image1(request->image1());
    dr_req.set_image2(request->image2());

    Driver::VerifyResult dr_resp;
    grpc::ClientContext ctx;

    auto* stub = getNextDriverStub();
    grpc::Status status = stub->RouteVerifyFace(&ctx, dr_req, &dr_resp);

    if (!status.ok()) {
        log_to_file("❌ RouteVerifyFace failed: " + status.error_message());
        return status;
    }

    reply->set_similarity(dr_resp.similarity());
    reply->set_result(dr_resp.result());

    log_to_file("✅ VerifyFace done. sim=" + std::to_string(dr_resp.similarity()) + " result=" + dr_resp.result());
    return grpc::Status::OK;
}