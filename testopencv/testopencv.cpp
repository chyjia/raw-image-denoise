#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>

int main()
{
    const std::string filename =
        "../Video_20260531164125231_w1920_h1200_pMono12_f16_10frame.gray16";

    const std::string outputFilename =
        "all_frames_roi_values.txt";

    const std::string histogramFilename =
        "all_frames_roi_histogram.png";

    const int width = 1920;
    const int height = 1200;
    const int totalFrames = 10;

    const size_t pixelCount = static_cast<size_t>(width) * height;
    const size_t bytesPerPixel = 2;
    const size_t frameBytes = pixelCount * bytesPerPixel;

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open input file: " << filename << std::endl;
        return 1;
    }

    std::ofstream outputFile(outputFilename);
    if (!outputFile) {
        std::cerr << "Failed to open output file: " << outputFilename << std::endl;
        return 1;
    }

    // ROI 区域
    const int left = 916;
    const int top = 461;
    const int right = 1309;
    const int bottom = 763;

    if (left < 0 || right >= width || top < 0 || bottom >= height ||
        left > right || top > bottom) {
        std::cerr << "Invalid ROI." << std::endl;
        return 1;
    }

    const int roiWidth = right - left + 1;
    const int roiHeight = bottom - top + 1;

    const size_t roiPixelCountPerFrame =
        static_cast<size_t>(roiWidth) * static_cast<size_t>(roiHeight);

    std::vector<unsigned short> roiValues;
    roiValues.reserve(roiPixelCountPerFrame * totalFrames);

    // ==============================
    // 读取 10 帧，并收集 ROI 数据
    // ==============================
    static int v1828 = 0;
    static int v1829 = 0;
    static int v1830 = 0;
    for (int frameIndex = 0; frameIndex < totalFrames; ++frameIndex) {
        cv::Mat image(height, width, CV_16UC1);

        file.read(reinterpret_cast<char*>(image.data), frameBytes);

        if (file.gcount() != static_cast<std::streamsize>(frameBytes)) {
            std::cerr << "Failed to read frame: " << frameIndex << std::endl;
            break;
        }

        image /= 4*4.0f;
        // 不再乘以 16
        // image.convertTo(image, CV_16UC1, 16.0);

        std::cout << "Read frame: " << frameIndex << std::endl;

        outputFile << "Frame " << frameIndex << std::endl;

        for (int y = top; y <= bottom; ++y) {
            const unsigned short* rowPtr = image.ptr<unsigned short>(y);

            for (int x = left; x <= right; ++x) {
                unsigned short value = rowPtr[x];

                roiValues.push_back(value);

                if (value == 1828)
                {
                    v1828++;
                }
                if (value == 1829)
                {
                    v1829++;
                }
                if (value == 1830)
                {
                    v1830++;
                }

                if (x > left) {
                    outputFile << ",";
                }

                outputFile << value;
            }

            outputFile << std::endl;
        }

        outputFile << std::endl;
    }

    outputFile.close();
    file.close();

    if (roiValues.empty()) {
        std::cerr << "No ROI values collected." << std::endl;
        return 1;
    }

    std::cout << "Total ROI values collected: "
        << roiValues.size() << std::endl;

    std::cout << "ROI values written to: "
        << outputFilename << std::endl;

    // ==============================
    // 根据 ROI 实际最大最小值统计直方图
    // x轴：ROI 实际像素值范围
    // y轴：频率
    // ==============================

    auto minMaxPair = std::minmax_element(roiValues.begin(), roiValues.end());

    int minPixelValue = static_cast<int>(*minMaxPair.first);
    int maxPixelValue = static_cast<int>(*minMaxPair.second);

    std::cout << "ROI min pixel value: " << minPixelValue << std::endl;
    std::cout << "ROI max pixel value: " << maxPixelValue << std::endl;

    // 防止所有像素值相同导致除以 0
    if (maxPixelValue == minPixelValue) {
        maxPixelValue = minPixelValue + 1;
    }

    int pixelRange = maxPixelValue - minPixelValue + 1;

    // 根据实际范围自动设置 binCount
    // 如果范围小于 256，则每个像素值一个 bin
    // 如果范围大于等于 256，则用 256 个 bin
    int binCount = pixelRange;

    std::cout << "Histogram pixel range: " << pixelRange << std::endl;
    std::cout << "Histogram bin count: " << binCount << std::endl;

    std::vector<int> histogram(binCount, 0);

    for (unsigned short value : roiValues) {
        int v = static_cast<int>(value);
        if (v == 1829)
            v = v;
        int binIndex = v - minPixelValue;
            

        if (binIndex < 0) {
            binIndex = 0;
        }

        if (binIndex >= binCount) {
            binIndex = binCount - 1;
        }

        histogram[binIndex]++;
    }

    int maxCount = *std::max_element(histogram.begin(), histogram.end());

    if (maxCount == 0) {
        std::cerr << "Histogram is empty." << std::endl;
        return 1;
    }

    double maxFrequency =
        static_cast<double>(maxCount) / static_cast<double>(roiValues.size());

    std::cout << "Max count in one bin: " << maxCount << std::endl;
    std::cout << "Max frequency in one bin: " << maxFrequency << std::endl;

    // ==============================
    // 绘制直方图
    // x轴：ROI 实际像素值范围
    // y轴：频率
    // ==============================

    const int histWidth = 1200;
    const int histHeight = 700;

    const int marginLeft = 90;
    const int marginRight = 40;
    const int marginTop = 60;
    const int marginBottom = 90;

    const int plotWidth = histWidth - marginLeft - marginRight;
    const int plotHeight = histHeight - marginTop - marginBottom;

    cv::Mat histImage(
        histHeight,
        histWidth,
        CV_8UC3,
        cv::Scalar(255, 255, 255)
    );

    // 绘图区背景
    cv::rectangle(
        histImage,
        cv::Point(marginLeft, marginTop),
        cv::Point(marginLeft + plotWidth, marginTop + plotHeight),
        cv::Scalar(245, 245, 245),
        -1
    );

    // y轴
    cv::line(
        histImage,
        cv::Point(marginLeft, marginTop),
        cv::Point(marginLeft, marginTop + plotHeight),
        cv::Scalar(0, 0, 0),
        2
    );

    // x轴
    cv::line(
        histImage,
        cv::Point(marginLeft, marginTop + plotHeight),
        cv::Point(marginLeft + plotWidth, marginTop + plotHeight),
        cv::Scalar(0, 0, 0),
        2
    );

    // 横向网格线
    for (int i = 1; i <= 4; ++i) {
        int y = marginTop + plotHeight - i * plotHeight / 4;

        cv::line(
            histImage,
            cv::Point(marginLeft, y),
            cv::Point(marginLeft + plotWidth, y),
            cv::Scalar(220, 220, 220),
            1
        );
    }

    double binDrawWidth =
        static_cast<double>(plotWidth) / static_cast<double>(binCount);

    // 绘制柱状图
    for (int i = 0; i < binCount; ++i) {
        double frequency =
            static_cast<double>(histogram[i]) /
            static_cast<double>(roiValues.size());

        int barHeight = static_cast<int>(
            frequency / maxFrequency * plotHeight
            );

        int x1 = marginLeft + static_cast<int>(i * binDrawWidth);
        int x2 = marginLeft + static_cast<int>((i + 1) * binDrawWidth);

        if (x2 <= x1) {
            x2 = x1 + 1;
        }

        int y1 = marginTop + plotHeight - barHeight;
        int y2 = marginTop + plotHeight;

        cv::rectangle(
            histImage,
            cv::Point(x1, y1),
            cv::Point(x2, y2),
            cv::Scalar(30, 90, 220),
            -1
        );
    }

    // 标题
    cv::putText(
        histImage,
        "All 10 Frames ROI Pixel Distribution",
        cv::Point(260, 35),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(0, 0, 0),
        2
    );

    // x轴标签
    cv::putText(
        histImage,
        "Pixel Value",
        cv::Point(marginLeft + plotWidth / 2 - 70, histHeight - 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(0, 0, 0),
        2
    );

    // y轴标签
    cv::putText(
        histImage,
        "Frequency",
        cv::Point(10, marginTop + plotHeight / 2),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(0, 0, 0),
        2
    );

    // ==============================
    // x轴刻度：根据实际 min/max 显示
    // ==============================

    for (int i = 0; i <= 5; ++i) {
        int pixelValue =
            minPixelValue +
            static_cast<int>(
                static_cast<double>(i) / 5.0 *
                static_cast<double>(maxPixelValue - minPixelValue)
                );

        int x = marginLeft + i * plotWidth / 5;
        int y = marginTop + plotHeight;

        cv::line(
            histImage,
            cv::Point(x, y),
            cv::Point(x, y + 6),
            cv::Scalar(0, 0, 0),
            1
        );

        cv::putText(
            histImage,
            std::to_string(pixelValue),
            cv::Point(x - 25, y + 30),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 0),
            1
        );
    }

    // ==============================
    // y轴刻度：显示频率
    // ==============================

    for (int i = 0; i <= 4; ++i) {
        double frequency =
            static_cast<double>(i) / 4.0 * maxFrequency;

        int x = marginLeft;
        int y = marginTop + plotHeight - i * plotHeight / 4;

        cv::line(
            histImage,
            cv::Point(x - 6, y),
            cv::Point(x, y),
            cv::Scalar(0, 0, 0),
            1
        );

        char text[64];
        std::snprintf(text, sizeof(text), "%.6f", frequency);

        cv::putText(
            histImage,
            text,
            cv::Point(5, y + 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            cv::Scalar(0, 0, 0),
            1
        );
    }

    // 在图上显示实际范围
    std::string rangeText =
        "Range: " + std::to_string(minPixelValue) +
        " - " + std::to_string(maxPixelValue);

    cv::putText(
        histImage,
        rangeText,
        cv::Point(marginLeft + 10, marginTop + 25),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(0, 0, 0),
        1
    );

    // 保存直方图
    bool saveOk = cv::imwrite(histogramFilename, histImage);

    if (!saveOk) {
        std::cerr << "Failed to save histogram image: "
            << histogramFilename << std::endl;
        return 1;
    }

    std::cout << "Histogram image written to: "
        << histogramFilename << std::endl;

    cv::imshow("All 10 Frames ROI Histogram", histImage);
    cv::waitKey(0);

    cv::destroyAllWindows();

    return 0;
}