#include "service_impl.h"
#include "../model/mtcnn/mtcnn/detector.h"
#include "face_aligner.h"
#include "face_embedder.h"
#include <uuid/uuid.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include "base64.h"
#include "src/storage/redis_client.h"
#include "src/storage/postgres_client.h"
#include "../model/mtcnn/mtcnn/detector.h"
#include "timer_macro.h"

namespace fs = std::filesystem;

ExecutorServiceImpl::ExecutorServiceImpl(int port)
    : embedder("/Users/rahmandaafridiansyah/Downloads/fr_cpp_/model/w600k_r50.onnx"), port(port) {

    LOG_DURATION("Load MTCNN model");
    
    // Init MTCNN
    ProposalNetwork::Config pConfig{ "../model/mtcnn/models/det1.prototxt", "../model/mtcnn/models/det1.caffemodel", 0.6f, 0.7f };
    RefineNetwork::Config rConfig{ "../model/mtcnn/models/det2.prototxt", "../model/mtcnn/models/det2.caffemodel", 0.7f };
    OutputNetwork::Config oConfig{ "../model/mtcnn/models/det3.prototxt", "../model/mtcnn/models/det3.caffemodel", 0.7f };
    detector = std::make_unique<MTCNNDetector>(pConfig, rConfig, oConfig);

    redis = std::make_unique<RedisClient>("127.0.0.1", 6379);
    postgres = std::make_unique<PostgresClient>("host=localhost port=5432 user=postgres password=Rahmanda07 dbname=engine_face");
}

void ExecutorServiceImpl::log_to_file(const std::string& message) const {
    std::ofstream log_file("executor.log", std::ios::app);
    if (log_file.is_open()) {
        auto now = std::chrono::system_clock::now();
        auto t_c = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&t_c), "%Y-%m-%d %H:%M:%S");
        log_file << "[" << ss.str() << "] [port " << port << "] " << message << std::endl;
    }
}

grpc::Status ExecutorServiceImpl::RegisterFace(grpc::ServerContext* context,
                                               const Executor::RegisterRequest* request,
                                               Executor::RegisterReply* reply) {
    try {
        LOG_DURATION("RegisterFace total");

        log_to_file("⚙️  Called RegisterFace: " + request->user_id());

        LOG_DURATION("Decode image");
        // std::string decoded = base64_decode(request->image_data());
        // std::vector<uchar> data(decoded.begin(), decoded.end());
        const std::string& binary_data = request->image_data();
        std::vector<uchar> data(binary_data.begin(), binary_data.end());
        cv::Mat img = cv::imdecode(data, cv::IMREAD_COLOR);
        if (img.empty()) {
            log_to_file("❌ Gagal decode image");
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid image");
        }

        if (img.cols > 640) {
            float scale = 640.0f / img.cols;
            cv::resize(img, img, cv::Size(), scale, scale);
        }

        LOG_DURATION("Face detection");
        auto faces = detector->detect(img, 40.0f, 0.709f);
        if (faces.empty()) return grpc::Status(grpc::StatusCode::NOT_FOUND, "No face detected");

        LOG_DURATION("Face alignment");
        auto face = faces[0];
        std::vector<cv::Point2f> pts;
        for (int i = 0; i < NUM_PTS; ++i)
            pts.emplace_back(face.ptsCoords[i * 2], face.ptsCoords[i * 2 + 1]);
        cv::Mat aligned = FaceAligner::align(img, pts);

        LOG_DURATION("Generate embedding");

        uuid_t uuid;
        uuid_generate(uuid);
        char uuid_str[37];
        uuid_unparse(uuid, uuid_str);
        std::string uuid_str_s(uuid_str);
        reply->set_user_id(uuid_str_s);

        // Ambil nama dan asal dari request
        std::string name = request->name();
        std::string origin = request->origin();

        // Buat path folder seperti Python: data/<nama>_<asal>_<uuid>/
        std::string folder_name = "data/" + name + "_" + origin + "_" + uuid_str_s;
        fs::create_directories(folder_name);

        // Simpan crop ke path lengkap
        std::string file_path = folder_name + "/" + name + "_" + origin + "_" + uuid_str_s + ".jpg";
        cv::imwrite(file_path, aligned);

        std::vector<float> embedding = embedder.getEmbedding(aligned);
        reply->set_message("success");

        LOG_DURATION("Save to Redis & PostgreSQL");
        bool redis_ok = redis->setEmbedding(uuid_str_s, embedding);
        bool postgres_ok = postgres->insertEmbedding(uuid_str_s, embedding);

        if (!redis_ok || !postgres_ok) {
            log_to_file("❌ Gagal simpan embedding untuk UUID: " + uuid_str_s);
            return grpc::Status(grpc::StatusCode::INTERNAL, "Failed to store embedding");
        }

        log_to_file("✅ Registered: " + uuid_str_s + " | name=" + name + ", origin=" + origin);
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        log_to_file("❌ Exception: " + std::string(e.what()));
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
}

grpc::Status ExecutorServiceImpl::IdentifyFace(grpc::ServerContext* context,
                                               const Executor::IdentifyRequest* request,
                                               Executor::IdentifyReply* reply) {
    try {
        LOG_DURATION("IdentifyFace total");

        log_to_file("⚙️  Called IdentifyFace");

        LOG_DURATION("Decode image");
        // std::string decoded = base64_decode(request->image_data());
        // std::vector<uchar> data(decoded.begin(), decoded.end());
        const std::string& binary_data = request->image_data();
        std::vector<uchar> data(binary_data.begin(), binary_data.end());
        cv::Mat img = cv::imdecode(data, cv::IMREAD_COLOR);
        if (img.empty()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid image");
        }

        if (img.cols > 640) {
            float scale = 640.0f / img.cols;
            cv::resize(img, img, cv::Size(), scale, scale);
        }

        LOG_DURATION("Face detection");
        auto faces = detector->detect(img, 40.0f, 0.709f);
        if (faces.empty()) return grpc::Status(grpc::StatusCode::NOT_FOUND, "No face detected");

        LOG_DURATION("Load embeddings from Redis/PostgreSQL");
        // Ambil dari Redis
        auto gallery = redis->loadAllEmbeddings();
        // Fallback jika Redis kosong
        if (gallery.empty()) {
            log_to_file("⚠️ Redis kosong, fallback ke PostgreSQL");
            gallery = postgres->loadAllEmbeddings();
            // Cache ke Redis
            for (const auto& [uuid, emb] : gallery) {
                redis->setEmbedding(uuid, emb);
            }
        }

        if (gallery.empty()) {
            return grpc::Status(grpc::StatusCode::NOT_FOUND, "No embeddings available");
        }

        auto cosine = [](const std::vector<float>& a, const std::vector<float>& b) -> float {
            float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
            for (size_t i = 0; i < a.size(); ++i) {
                dot += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }
            return (norm_a > 0 && norm_b > 0) ? (dot / (std::sqrt(norm_a) * std::sqrt(norm_b))) : 0.0f;
        };

        for (size_t idx = 0; idx < faces.size(); ++idx) {
            LOG_DURATION("Align & Embed face-" + std::to_string(idx));
            const auto& face = faces[idx];
            std::vector<cv::Point2f> pts;
            for (int i = 0; i < NUM_PTS; ++i)
                pts.emplace_back(face.ptsCoords[i * 2], face.ptsCoords[i * 2 + 1]);
            cv::Mat aligned = FaceAligner::align(img, pts);
            std::vector<float> query_emb = embedder.getEmbedding(aligned);

            // Cari best match
            std::string best_uuid;
            float best_score = -1.0f;

            LOG_DURATION("Find best match face-" + std::to_string(idx));
            for (const auto& [uuid, emb] : gallery) {
                float score = cosine(query_emb, emb);
                if (score > best_score) {
                    best_score = score;
                    best_uuid = uuid;
                }
            }

            // Bangun response
            Executor::FaceResult* face_result = reply->add_results();
            face_result->set_face_index(idx);

            if (!best_uuid.empty()) {
                Executor::Candidate* candidate = face_result->add_top_matches();
                candidate->set_user_id(best_uuid);
                candidate->set_confidence(best_score);
            }

            std::vector<uchar> buf;
            cv::imencode(".jpg", aligned, buf);
            std::string crop_base64 = base64_encode(buf.data(), buf.size());
            face_result->set_crop_image(crop_base64);

            log_to_file("🔍 Face-" + std::to_string(idx) + ": " + best_uuid + " (sim=" + std::to_string(best_score) + ")");
        }

        return grpc::Status::OK;

    } catch (const std::exception& e) {
        log_to_file("❌ Exception: " + std::string(e.what()));
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
}

grpc::Status ExecutorServiceImpl::VerifyFace(grpc::ServerContext* context,
                                              const Executor::VerifyRequest* request,
                                              Executor::VerifyReply* reply) {
    try {
        LOG_DURATION("VerifyFace total");

        // Decode image
        cv::Mat img1 = cv::imdecode(std::vector<uchar>(request->image1().begin(), request->image1().end()), cv::IMREAD_COLOR);
        cv::Mat img2 = cv::imdecode(std::vector<uchar>(request->image2().begin(), request->image2().end()), cv::IMREAD_COLOR);
        if (img1.empty() || img2.empty()) {
            reply->set_result("FAILED");
            reply->set_similarity(0.0f);
            return grpc::Status::OK;
        }

        // Resize
        const int max_width = 640;
        if (img1.cols > max_width) cv::resize(img1, img1, cv::Size(), 640.0f / img1.cols, 640.0f / img1.cols);
        if (img2.cols > max_width) cv::resize(img2, img2, cv::Size(), 640.0f / img2.cols, 640.0f / img2.cols);

        // Detect
        auto faces1 = detector->detect(img1, 40.0f, 0.8f);
        auto faces2 = detector->detect(img2, 40.0f, 0.8f);
        if (faces1.empty() || faces2.empty()) {
            reply->set_result("FAILED");
            reply->set_similarity(0.0f);
            return grpc::Status::OK;
        }

        // Align
        std::vector<cv::Point2f> pts1, pts2;
        for (int i = 0; i < 5; ++i) {
            pts1.emplace_back(faces1[0].ptsCoords[i * 2], faces1[0].ptsCoords[i * 2 + 1]);
            pts2.emplace_back(faces2[0].ptsCoords[i * 2], faces2[0].ptsCoords[i * 2 + 1]);
        }
        cv::Mat aligned1 = FaceAligner::align(img1, pts1);
        cv::Mat aligned2 = FaceAligner::align(img2, pts2);

        std::vector<float> emb1 = embedder.getEmbedding(aligned1);
        std::vector<float> emb2 = embedder.getEmbedding(aligned2);

        // Cosine
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < emb1.size(); ++i) {
            dot += emb1[i] * emb2[i];
            norm_a += emb1[i] * emb1[i];
            norm_b += emb2[i] * emb2[i];
        }
        float sim = (norm_a > 0 && norm_b > 0) ? (dot / (std::sqrt(norm_a) * std::sqrt(norm_b))) : 0.0f;

        std::string result = sim >= 0.5f ? "MATCH" : (sim < 0.3f ? "NOT_MATCH" : "REVIEW");
        reply->set_similarity(sim);
        reply->set_result(result);

        log_to_file("✅ VerifyFaces done: sim=" + std::to_string(sim) + " | result=" + result);
        return grpc::Status::OK;
    }
    catch (const std::exception& e) {
        log_to_file("❌ Exception in VerifyFaces: " + std::string(e.what()));
        reply->set_result("FAILED");
        reply->set_similarity(0.0f);
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
}