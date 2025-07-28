#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>
#include "executor.grpc.pb.h"
#include "service_impl.h"

void log_to_file(const std::string& message) {
    std::ofstream log_file("executor.log", std::ios::app);
    if (log_file.is_open()) {
        log_file << message << std::endl;
    }
}

void run_executor_server(int port) {
    std::string address = "0.0.0.0:" + std::to_string(port);
    ExecutorServiceImpl* service = new ExecutorServiceImpl(port);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());

    log_to_file("🚀 Executor aktif di port " + std::to_string(port));

    server->Wait();
}

int main() {
    const int base_port = 6001;
    const int total_instances = 6;

    std::vector<std::thread> threads;

    for (int i = 0; i < total_instances; ++i) {
        int port = base_port + i;
        threads.emplace_back(std::thread(run_executor_server, port));
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    return 0;
}