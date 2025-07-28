#pragma once

#include "../proto/controller.grpc.pb.h"
#include "../proto/driver.grpc.pb.h"
#include <vector>
#include <memory>
#include <mutex>

class ControllerServiceImpl final : public Controller::ControllerService::Service {
public:
    ControllerServiceImpl(int port_);

    grpc::Status RouteRegisterFace(grpc::ServerContext* context, const Controller::RegisterRequest* request, Controller::RegisterReply* reply) override;
    grpc::Status RouteIdentifyFace(grpc::ServerContext* context, const Controller::IdentifyRequest* request, Controller::IdentifyReply* reply) override;
    grpc::Status RouteVerifyFace(grpc::ServerContext* context, const Controller::VerifyRequest* request, Controller::VerifyReply* reply) override;

private:
    std::vector<std::string> driver_addresses;
    std::vector<std::unique_ptr<Driver::DriverService::Stub>> driver_stubs;
    size_t driver_index;
    std::mutex driver_mutex;
    int port;
    Driver::DriverService::Stub* getNextDriverStub();
    void log_to_file(const std::string& message) const;
};