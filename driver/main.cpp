#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <grpcpp/grpcpp.h>
#include "driver.grpc.pb.h"
#include "service_impl.h"

using grpc::Server;
using grpc::ServerBuilder;

// Fungsi log ke driver.log
void log_to_file(const std::string& message) {
    std::ofstream log_file("driver.log", std::ios::app);
    if (log_file.is_open()) {
        log_file << message << std::endl;
    }
}

void RunServer(int port) {
    std::string server_address = "0.0.0.0:" + std::to_string(port);
    auto service = std::make_unique<DriverServiceImpl>(port);  // inject port

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(service.get());

    std::unique_ptr<Server> server(builder.BuildAndStart());
    log_to_file("🚀 Driver aktif di port " + std::to_string(port));

    server->Wait();
}

int main() {
    std::vector<std::thread> threads;
    for (int port = 1967; port <= 1969; ++port) {
        threads.emplace_back(RunServer, port);
    }

    for (auto& t : threads) {
        t.join();
    }

    return 0;
}