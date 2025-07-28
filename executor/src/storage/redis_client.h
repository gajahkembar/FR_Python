#pragma once
#include <hiredis/hiredis.h>
#include <string>
#include <vector>
#include <unordered_map>

class RedisClient {
public:
    RedisClient(const std::string& host, int port);
    ~RedisClient();
    bool setEmbedding(const std::string& uuid, const std::vector<float>& embedding);
    bool deleteEmbedding(const std::string& uuid);
    std::unordered_map<std::string, std::vector<float>> loadAllEmbeddings();

private:
    redisContext* context;
};