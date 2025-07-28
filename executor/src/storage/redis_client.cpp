#include "redis_client.h"
#include <iostream>
#include <cstring>
#include <unordered_map>

RedisClient::RedisClient(const std::string& host, int port) {
    context = redisConnect(host.c_str(), port);
    if (!context || context->err) {
        std::cerr << "❌ Redis connection error: " << (context ? context->errstr : "null context") << std::endl;
        context = nullptr;
    }
}

RedisClient::~RedisClient() {
    if (context) redisFree(context);
}

bool RedisClient::setEmbedding(const std::string& uuid, const std::vector<float>& embedding) {
    if (!context) return false;

    std::string key = "face:" + uuid;
    const char* raw_data = reinterpret_cast<const char*>(embedding.data());
    size_t byte_len = embedding.size() * sizeof(float);

    redisReply* reply = (redisReply*)redisCommand(context, "SET %b %b", key.c_str(), key.size(), raw_data, byte_len);
    bool success = reply && reply->type == REDIS_REPLY_STATUS && std::string(reply->str) == "OK";

    if (reply) freeReplyObject(reply);
    return success;
}

std::unordered_map<std::string, std::vector<float>> RedisClient::loadAllEmbeddings() {
    std::unordered_map<std::string, std::vector<float>> embeddings;
    if (!context) return embeddings;

    redisReply* keys_reply = (redisReply*)redisCommand(context, "KEYS face:*");
    if (!keys_reply || keys_reply->type != REDIS_REPLY_ARRAY) return embeddings;

    for (size_t i = 0; i < keys_reply->elements; ++i) {
        std::string key(keys_reply->element[i]->str);
        std::string uuid = key.substr(5); // remove "face:"

        redisReply* val_reply = (redisReply*)redisCommand(context, "GET %s", key.c_str());
        if (val_reply && val_reply->type == REDIS_REPLY_STRING) {
            const char* raw = val_reply->str;
            size_t len = val_reply->len;

            if (len % sizeof(float) == 0) {
                std::vector<float> vec(len / sizeof(float));
                std::memcpy(vec.data(), raw, len);
                embeddings[uuid] = std::move(vec);
            }
            freeReplyObject(val_reply);
        }
    }

    freeReplyObject(keys_reply);
    return embeddings;
}

bool RedisClient::deleteEmbedding(const std::string& uuid) {
    if (!context) return false;

    std::string key = "face:" + uuid;
    redisReply* reply = (redisReply*)redisCommand(context, "DEL %s", key.c_str());
    bool success = reply && reply->type == REDIS_REPLY_INTEGER && reply->integer > 0;

    if (reply) freeReplyObject(reply);
    return success;
}