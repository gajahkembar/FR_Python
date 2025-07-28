#include <iostream>
#include <grpcpp/grpcpp.h>
#include "controller.grpc.pb.h"
#include "service_impl.h"
#include <fstream>

void log_to_file(const std::string& message) {
    std::ofstream log_file("controller.log", std::ios::app);
    if (log_file.is_open()) {
        log_file << message << std::endl;
    }
}

int main() {
    int port = 2003;
    std::string address = "0.0.0.0:" + std::to_string(port);
    ControllerServiceImpl service(port); // ✅ inject port ke constructor

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    log_to_file("🚀 Controller aktif di " + std::to_string(port));

    server->Wait();
    return 0;
}