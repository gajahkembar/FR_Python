#include "base64.h"
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <vector>
#include <iostream>
#include <string>

static const std::string base64_chars = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(const unsigned char* data, size_t len) {
    std::string ret;
    int i = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (len--) {
        char_array_3[i++] = *(data++);
        if (i == 3) {
            char_array_4[0] =  (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] =   char_array_3[2] & 0x3f;
            for(i = 0; i < 4 ; i++) ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for(int j = i; j < 3; j++) char_array_3[j] = '\0';
        char_array_4[0] =  (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] =   char_array_3[2] & 0x3f;

        for (int j = 0; j < i + 1; j++) ret += base64_chars[char_array_4[j]];
        while((i++ < 3)) ret += '=';
    }

    return ret;
}

std::string base64_decode(const std::string &encoded) {
    BIO *bio, *b64;
    int decodeLen = encoded.length() * 3 / 4;
    std::vector<char> buffer(decodeLen + 1);
    buffer[decodeLen] = '\0';

    bio = BIO_new_mem_buf(encoded.data(), static_cast<int>(encoded.length()));
    b64 = BIO_new(BIO_f_base64());
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL); // Jangan newline
    bio = BIO_push(b64, bio);

    int decodedLen = BIO_read(bio, buffer.data(), static_cast<int>(buffer.size()));
    BIO_free_all(bio);

    if (decodedLen <= 0) {
        return "";
    }

    return std::string(buffer.data(), decodedLen);
}

cv::Mat base64_to_mat(const std::string& base64_data) {
    std::string clean_base64;
    for (char c : base64_data) {
        if (isalnum(c) || c == '+' || c == '/' || c == '=') {
            clean_base64 += c;
        }
    }

    while (clean_base64.length() % 4 != 0) {
        clean_base64 += '=';
    }

    // Decode pakai OpenSSL
    BIO* bio = BIO_new_mem_buf(clean_base64.data(), static_cast<int>(clean_base64.length()));
    BIO* b64 = BIO_new(BIO_f_base64());
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
    bio = BIO_push(b64, bio);

    std::vector<unsigned char> decoded(clean_base64.length());
    int decoded_len = BIO_read(bio, decoded.data(), static_cast<int>(decoded.size()));
    BIO_free_all(bio);

    if (decoded_len <= 0) {
        throw std::runtime_error("❌ base64 decode failed");
    }

    std::vector<uchar> buf(decoded.begin(), decoded.begin() + decoded_len);
    cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("❌ imdecode gagal, hasil gambar kosong");
    }

    return img;
}

std::string mat_to_base64(const cv::Mat& img) {
    std::vector<uchar> buf;
    cv::imencode(".jpg", img, buf);  // bisa diganti ".png" kalau butuh transparansi
    return base64_encode(buf.data(), buf.size());
}