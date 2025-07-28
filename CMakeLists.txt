cmake_minimum_required(VERSION 3.10)
project(fr_cpp)

set(CMAKE_CXX_STANDARD 17)

# Include dan Link Directories untuk hiredis dan libpqxx
include_directories(/opt/homebrew/opt/hiredis/include /opt/homebrew/opt/libpqxx/include)
link_directories(/opt/homebrew/opt/hiredis/lib /opt/homebrew/opt/libpqxx/lib)
include_directories(/opt/homebrew/opt/ossp-uuid/include)
link_directories(/opt/homebrew/opt/ossp-uuid/lib)

# === ONNX Runtime ===
set(ONNXRUNTIME_DIR "/Users/rahmandaafridiansyah/onnxruntime-osx-arm64-1.18.0")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")

# === OpenCV ===
find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/model/mtcnn/mtcnn
    ${CMAKE_CURRENT_SOURCE_DIR}/proto
)

# # === Protobuf & gRPC ===
# find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)

# === Executor ===
add_executable(executor
    executor/main.cpp
    executor/face_aligner.cpp
    executor/face_embedder.cpp
    executor/service_impl.cpp
    executor/base64.cpp
    executor/src/storage/redis_client.cpp
    executor/src/storage/postgres_client.cpp
    model/mtcnn/mtcnn/detector.cpp
    model/mtcnn/mtcnn/pnet.cpp
    model/mtcnn/mtcnn/rnet.cpp
    model/mtcnn/mtcnn/onet.cpp
    proto/executor.pb.cc
    proto/executor.grpc.pb.cc
)

target_link_libraries(executor
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    protobuf::libprotobuf
    gRPC::grpc++
    hiredis pqxx uuid
)

# === Driver ===
add_executable(driver
    driver/main.cpp
    driver/service_impl.cpp
    executor/face_aligner.cpp
    executor/face_embedder.cpp
    executor/base64.cpp
    model/mtcnn/mtcnn/detector.cpp
    model/mtcnn/mtcnn/pnet.cpp
    model/mtcnn/mtcnn/rnet.cpp
    model/mtcnn/mtcnn/onet.cpp
    proto/driver.pb.cc
    proto/driver.grpc.pb.cc
    proto/executor.pb.cc
    proto/executor.grpc.pb.cc
)

target_link_libraries(driver
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    protobuf::libprotobuf
    gRPC::grpc++
    hiredis pqxx uuid
)

# === Controller ===
add_executable(controller
    controller/main.cpp
    controller/service_impl.cpp
    proto/controller.pb.cc
    proto/controller.grpc.pb.cc
    proto/driver.pb.cc
    proto/driver.grpc.pb.cc
    proto/executor.pb.cc       
    proto/executor.grpc.pb.cc  
)

target_link_libraries(controller
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    protobuf::libprotobuf
    gRPC::grpc++
    hiredis pqxx uuid
)