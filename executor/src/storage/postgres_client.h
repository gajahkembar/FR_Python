#pragma once
#include <pqxx/pqxx>
#include <string>
#include <vector>
#include <unordered_map>

class PostgresClient {
public:
    PostgresClient(const std::string& conn_str);
    bool insertEmbedding(const std::string& uuid, const std::vector<float>& embedding);
    std::unordered_map<std::string, std::vector<float>> loadAllEmbeddings();

private:
    pqxx::connection conn;
};