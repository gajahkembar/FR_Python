#include "postgres_client.h"
#include <cstring>  // for memcpy
#include <iostream>
#include <cstddef>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <pqxx/pqxx>
#pragma GCC diagnostic pop

PostgresClient::PostgresClient(const std::string& conn_str) : conn(conn_str) {}

bool PostgresClient::insertEmbedding(const std::string &uuid, const std::vector<float> &embedding)
{
    try
    {
        pqxx::work txn(conn);

        const std::byte *byte_ptr = reinterpret_cast<const std::byte *>(embedding.data());
        size_t byte_size = embedding.size() * sizeof(float);
        pqxx::bytes_view bin_view{byte_ptr, byte_size};

        txn.exec(
            "INSERT INTO face_embedding (uuid, embedding) VALUES (" +
            txn.quote(uuid) + ", " + txn.quote(bin_view) + "::bytea) " +
            "ON CONFLICT (uuid) DO UPDATE SET embedding = EXCLUDED.embedding");

        txn.commit();
        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "❌ PostgreSQL insert error: " << e.what() << std::endl;
        return false;
    }
}

std::unordered_map<std::string, std::vector<float>> PostgresClient::loadAllEmbeddings() {
    std::unordered_map<std::string, std::vector<float>> result;
    try {
        pqxx::work txn(conn);
        pqxx::result rows = txn.exec("SELECT uuid, embedding FROM face_embedding");

        for (const auto& row : rows) {
            std::string uuid = row["uuid"].as<std::string>();
            const auto& field = row["embedding"];
            const char* raw_data = field.c_str();
            size_t byte_len = field.size();  // total bytes

            if (byte_len % sizeof(float) == 0) {
                const float* float_data = reinterpret_cast<const float*>(raw_data);
                size_t len = byte_len / sizeof(float);
                std::vector<float> vec(float_data, float_data + len);
                result[uuid] = std::move(vec);
            }
        }

        return result;
    } catch (...) {
        return {};
    }
}