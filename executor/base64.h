#ifndef BASE64_H
#define BASE64_H

#include <string>
#include <opencv2/opencv.hpp>

cv::Mat base64_to_mat(const std::string& base64_str);
std::string mat_to_base64(const cv::Mat& img);
std::string base64_encode(const unsigned char* data, size_t len);
std::string base64_decode(const std::string& encoded_string);

#endif // BASE64_H