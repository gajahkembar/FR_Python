#pragma once
#include <chrono>
#include <string>
#include <sstream>
#include <fstream>

#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)
#define UNIQUE_VAR_NAME(base) CONCATENATE(base, __COUNTER__)

#define LOG_DURATION(label) \
    Timer UNIQUE_VAR_NAME(timer_instance)(label, port);

struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    std::string label;
    int port;

    Timer(const std::string& lbl, int p)
        : start(std::chrono::high_resolution_clock::now()), label(lbl), port(p) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::ofstream log_file("executor.log", std::ios::app);
        if (log_file.is_open()) {
            auto t_c = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::stringstream ss;
            ss << std::put_time(std::localtime(&t_c), "%Y-%m-%d %H:%M:%S");
            log_file << "[" << ss.str() << "] [port " << port << "] ⏱️ " << label << ": " << dur << "ms" << std::endl;
        }
    }
};