#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <limits>

static cv::Mat cropWhiteBorders(const cv::Mat& image, int threshold = 248)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    int top = gray.rows;
    int bottom = 0;
    int left = gray.cols;
    int right = 0;

    for (int y = 0; y < gray.rows; ++y) {
        for (int x = 0; x < gray.cols; ++x) {
            if (gray.at<unsigned char>(y, x) < threshold) {
                top = std::min(top, y);
                bottom = std::max(bottom, y);
                left = std::min(left, x);
                right = std::max(right, x);
            }
        }
    }

    if (top > bottom || left > right) {
        return image.clone();
    }

    const int pad = 8;
    top = std::max(0, top - pad);
    left = std::max(0, left - pad);
    bottom = std::min(gray.rows - 1, bottom + pad);
    right = std::min(gray.cols - 1, right + pad);

    return image(cv::Rect(left, top, right - left + 1, bottom - top + 1)).clone();
}

struct TemporalPointStats
{
    cv::Point point;
    int minValue = std::numeric_limits<int>::max();
    int maxValue = std::numeric_limits<int>::min();
    double mean = 0.0;
    double variance = 0.0;
};

static bool processTemporalPointStats(
    const std::string& filename,
    const std::string& prefix,
    const std::string& title,
    int width,
    int height,
    int left,
    int top,
    int right,
    int bottom)
{
    const size_t bytesPerPixel = 1;
    const size_t frameBytes = static_cast<size_t>(width) * height * bytesPerPixel;

    const std::string statsFilename = prefix + "_temporal_16_points.csv";
    const std::string plotFilename = prefix + "_temporal_mean_variance.png";

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open input file: " << filename << std::endl;
        return false;
    }

    const std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (fileSize < static_cast<std::streamsize>(frameBytes)) {
        std::cerr << "File too small for one frame: " << filename << std::endl;
        return false;
    }

    const int totalFrames = static_cast<int>(fileSize / frameBytes);
    constexpr int gridSize = 4;
    constexpr int pointCount = gridSize * gridSize;
    const int roiWidth = right - left + 1;
    const int roiHeight = bottom - top + 1;

    std::vector<TemporalPointStats> stats;
    stats.reserve(pointCount);
    for (int gy = 0; gy < gridSize; ++gy) {
        for (int gx = 0; gx < gridSize; ++gx) {
            const int x = std::min(
                right,
                left + (2 * gx + 1) * roiWidth / (2 * gridSize));
            const int y = std::min(
                bottom,
                top + (2 * gy + 1) * roiHeight / (2 * gridSize));
            TemporalPointStats pointStats;
            pointStats.point = cv::Point(x, y);
            stats.push_back(pointStats);
        }
    }

    std::vector<int> counts(pointCount, 0);
    std::vector<long double> runningMeans(pointCount, 0.0L);
    std::vector<long double> runningM2(pointCount, 0.0L);

    std::cout << "\n=== " << title << " temporal 16-point stats ===" << std::endl;
    std::cout << "File: " << filename << std::endl;
    std::cout << "Total frames: " << totalFrames << std::endl;

    for (int frameIndex = 0; frameIndex < totalFrames; ++frameIndex) {
        file.seekg(static_cast<std::streamoff>(frameIndex) *
                       static_cast<std::streamoff>(frameBytes),
                   std::ios::beg);

        cv::Mat image(height, width, CV_8UC1);
        file.read(reinterpret_cast<char*>(image.data),
                  static_cast<std::streamsize>(frameBytes));

        if (file.gcount() != static_cast<std::streamsize>(frameBytes)) {
            std::cerr << "Failed to read frame: " << frameIndex << std::endl;
            return false;
        }

        for (int i = 0; i < pointCount; ++i) {
            const int pixelValue =
                image.at<unsigned char>(stats[i].point.y, stats[i].point.x);
            const long double sampleValue = static_cast<long double>(pixelValue);

            counts[i]++;
            const long double delta = sampleValue - runningMeans[i];
            runningMeans[i] += delta / static_cast<long double>(counts[i]);
            const long double deltaAfterMean = sampleValue - runningMeans[i];
            runningM2[i] += delta * deltaAfterMean;

            stats[i].minValue = std::min(stats[i].minValue, pixelValue);
            stats[i].maxValue = std::max(stats[i].maxValue, pixelValue);
        }
    }

    std::ofstream statsFile(statsFilename);
    if (!statsFile) {
        std::cerr << "Failed to open output file: " << statsFilename << std::endl;
        return false;
    }

    statsFile << "index,x,y,frames,min,max,mean_dn,variance_dn,variance_over_mean"
              << std::endl;

    double minMean = std::numeric_limits<double>::max();
    double maxMean = std::numeric_limits<double>::lowest();
    double maxVariance = 0.0;
    for (int i = 0; i < pointCount; ++i) {
        stats[i].mean = static_cast<double>(runningMeans[i]);
        stats[i].variance = static_cast<double>(
            runningM2[i] / static_cast<long double>(counts[i]));
        stats[i].variance = std::max(0.0, stats[i].variance);

        const double varianceOverMean =
            stats[i].mean > 0.0 ? stats[i].variance / stats[i].mean : 0.0;

        minMean = std::min(minMean, stats[i].mean);
        maxMean = std::max(maxMean, stats[i].mean);
        maxVariance = std::max(maxVariance, stats[i].variance);

        statsFile << i << ","
                  << stats[i].point.x << ","
                  << stats[i].point.y << ","
                  << totalFrames << ","
                  << stats[i].minValue << ","
                  << stats[i].maxValue << ","
                  << stats[i].mean << ","
                  << stats[i].variance << ","
                  << varianceOverMean << std::endl;

        std::cout << "point " << i
                  << " (" << stats[i].point.x << "," << stats[i].point.y << ")"
                  << ": mean_dn=" << stats[i].mean
                  << ", variance_dn=" << stats[i].variance
                  << ", variance/mean=" << varianceOverMean
                  << ", range=" << stats[i].minValue << "-" << stats[i].maxValue
                  << std::endl;
    }
    statsFile.close();
    std::cout << "Temporal stats written to: " << statsFilename << std::endl;

    const int imageWidth = 900;
    const int imageHeight = 700;
    const int marginLeft = 110;
    const int marginRight = 60;
    const int marginTop = 70;
    const int marginBottom = 90;
    const int plotWidth = imageWidth - marginLeft - marginRight;
    const int plotHeight = imageHeight - marginTop - marginBottom;

    const double meanPadding = std::max((maxMean - minMean) * 0.08, 0.5);
    const double xMin = std::max(0.0, minMean - meanPadding);
    const double xMax = maxMean + meanPadding;
    const double yMax = std::max(maxVariance * 1.20, 1.0);

    cv::Mat plotImage(
        imageHeight,
        imageWidth,
        CV_8UC3,
        cv::Scalar(255, 255, 255)
    );

    cv::rectangle(
        plotImage,
        cv::Point(marginLeft, marginTop),
        cv::Point(marginLeft + plotWidth, marginTop + plotHeight),
        cv::Scalar(245, 245, 245),
        -1
    );

    cv::line(
        plotImage,
        cv::Point(marginLeft, marginTop),
        cv::Point(marginLeft, marginTop + plotHeight),
        cv::Scalar(0, 0, 0),
        2
    );
    cv::line(
        plotImage,
        cv::Point(marginLeft, marginTop + plotHeight),
        cv::Point(marginLeft + plotWidth, marginTop + plotHeight),
        cv::Scalar(0, 0, 0),
        2
    );

    for (int i = 1; i <= 4; ++i) {
        const int x = marginLeft + i * plotWidth / 4;
        const int y = marginTop + plotHeight - i * plotHeight / 4;
        cv::line(
            plotImage,
            cv::Point(x, marginTop),
            cv::Point(x, marginTop + plotHeight),
            cv::Scalar(225, 225, 225),
            1
        );
        cv::line(
            plotImage,
            cv::Point(marginLeft, y),
            cv::Point(marginLeft + plotWidth, y),
            cv::Scalar(225, 225, 225),
            1
        );
    }

    auto mapX = [&](double mean) {
        return marginLeft +
            static_cast<int>((mean - xMin) / (xMax - xMin) * plotWidth);
    };
    auto mapY = [&](double variance) {
        return marginTop + plotHeight -
            static_cast<int>(variance / yMax * plotHeight);
    };

    for (int i = 0; i < pointCount; ++i) {
        const int x = mapX(stats[i].mean);
        const int y = mapY(stats[i].variance);
        cv::circle(plotImage, cv::Point(x, y), 6, cv::Scalar(30, 90, 220), -1);
        cv::putText(
            plotImage,
            std::to_string(i),
            cv::Point(x + 8, y - 8),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            cv::Scalar(0, 0, 0),
            1
        );
    }

    cv::putText(
        plotImage,
        title + " temporal mean vs variance",
        cv::Point(35, 35),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(0, 0, 0),
        2
    );
    cv::putText(
        plotImage,
        "Mean DN across frames",
        cv::Point(marginLeft + plotWidth / 2 - 100, imageHeight - 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(0, 0, 0),
        2
    );
    cv::putText(
        plotImage,
        "Variance DN",
        cv::Point(15, marginTop + plotHeight / 2),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(0, 0, 0),
        2
    );

    for (int i = 0; i <= 4; ++i) {
        const double xTick = xMin + (xMax - xMin) * i / 4.0;
        const double yTick = yMax * i / 4.0;
        char xText[32];
        char yText[32];
        std::snprintf(xText, sizeof(xText), "%.2f", xTick);
        std::snprintf(yText, sizeof(yText), "%.2f", yTick);

        cv::putText(
            plotImage,
            xText,
            cv::Point(marginLeft + i * plotWidth / 4 - 18,
                      marginTop + plotHeight + 28),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            cv::Scalar(0, 0, 0),
            1
        );
        cv::putText(
            plotImage,
            yText,
            cv::Point(20, marginTop + plotHeight - i * plotHeight / 4 + 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            cv::Scalar(0, 0, 0),
            1
        );
    }

    if (!cv::imwrite(plotFilename, plotImage)) {
        std::cerr << "Failed to save temporal plot image: "
                  << plotFilename << std::endl;
        return false;
    }

    std::cout << "Temporal mean-variance plot written to: "
              << plotFilename << std::endl;

    cv::imshow(prefix + " Temporal Mean vs Variance", plotImage);
    return true;
}

static bool processRoiHistogram(
    const std::string& filename,
    const std::string& prefix,
    const std::string& title,
    int width,
    int height,
    int left,
    int top,
    int right,
    int bottom)
{
    const size_t bytesPerPixel = 1;
    const size_t frameBytes = static_cast<size_t>(width) * height * bytesPerPixel;

    const std::string histogramFilename = prefix + "_roi_histogram.png";
    const std::string proportionFilename = prefix + "_roi_proportions.txt";

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open input file: " << filename << std::endl;
        return false;
    }

    const std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (fileSize < static_cast<std::streamsize>(frameBytes)) {
        std::cerr << "File too small for one frame: " << filename << std::endl;
        return false;
    }

    const int totalFrames = static_cast<int>(fileSize / frameBytes);
    const size_t roiPixelCountPerFrame =
        static_cast<size_t>(right - left + 1) * static_cast<size_t>(bottom - top + 1);

    std::vector<unsigned char> roiValues;
    roiValues.reserve(roiPixelCountPerFrame * totalFrames);

    std::cout << "\n=== " << title << " ===" << std::endl;
    std::cout << "File: " << filename << std::endl;
    std::cout << "Total frames: " << totalFrames << std::endl;

    for (int frameIndex = 0; frameIndex < totalFrames; ++frameIndex) {
        file.seekg(static_cast<std::streamoff>(frameIndex) *
                       static_cast<std::streamoff>(frameBytes),
                   std::ios::beg);

        cv::Mat image(height, width, CV_8UC1);
        file.read(reinterpret_cast<char*>(image.data),
                  static_cast<std::streamsize>(frameBytes));

        if (file.gcount() != static_cast<std::streamsize>(frameBytes)) {
            std::cerr << "Failed to read frame: " << frameIndex << std::endl;
            break;
        }

        for (int y = top; y <= bottom; ++y) {
            const unsigned char* rowPtr = image.ptr<unsigned char>(y);
            for (int x = left; x <= right; ++x) {
                roiValues.push_back(rowPtr[x]);
            }
        }
    }

    if (roiValues.empty()) {
        std::cerr << "No ROI values collected." << std::endl;
        return false;
    }

    const size_t totalPixels = roiValues.size();
    std::cout << "Total ROI pixels: " << totalPixels << std::endl;

    constexpr int binCount = 256;
    std::vector<int> histogram(binCount, 0);
    for (unsigned char value : roiValues) {
        histogram[static_cast<int>(value)]++;
    }

    auto minMaxPair = std::minmax_element(roiValues.begin(), roiValues.end());
    const int minPixelValue = static_cast<int>(*minMaxPair.first);
    const int maxPixelValue = static_cast<int>(*minMaxPair.second);

    long double sum = 0.0L;
    long double sumSquares = 0.0L;
    for (unsigned char value : roiValues) {
        const long double pixelValue = static_cast<long double>(value);
        sum += pixelValue;
        sumSquares += pixelValue * pixelValue;
    }

    const double mean =
        static_cast<double>(sum / static_cast<long double>(totalPixels));
    double variance = static_cast<double>(
        sumSquares / static_cast<long double>(totalPixels) -
        static_cast<long double>(mean) * static_cast<long double>(mean));
    variance = std::max(0.0, variance);

    std::cout << "ROI min pixel value: " << minPixelValue << std::endl;
    std::cout << "ROI max pixel value: " << maxPixelValue << std::endl;
    std::cout << "ROI mean pixel value: " << mean << std::endl;
    std::cout << "ROI variance: " << variance << std::endl;

    std::ofstream proportionFile(proportionFilename);
    if (!proportionFile) {
        std::cerr << "Failed to open output file: " << proportionFilename
                  << std::endl;
        return false;
    }

    proportionFile << "pixel_value,count,proportion" << std::endl;
    for (int v = 0; v < binCount; ++v) {
        const double proportion =
            static_cast<double>(histogram[v]) / static_cast<double>(totalPixels);
        proportionFile << v << "," << histogram[v] << "," << proportion
                       << std::endl;
        if (histogram[v] > 0) {
            std::cout << "pixel " << v << ": count=" << histogram[v]
                      << ", proportion=" << proportion << std::endl;
        }
    }
    proportionFile.close();
    std::cout << "Proportions written to: " << proportionFilename << std::endl;

    const int valueCount = maxPixelValue - minPixelValue + 1;
    double maxProportion = 0.0;
    for (int v = minPixelValue; v <= maxPixelValue; ++v) {
        const double p =
            static_cast<double>(histogram[v]) / static_cast<double>(totalPixels);
        maxProportion = std::max(maxProportion, p);
    }

    const double yMax = std::max(maxProportion * 1.12, maxProportion + 1e-9);

    constexpr int pixelsPerBin = 48;
    const int plotWidth = valueCount * pixelsPerBin;
    const int histWidth = plotWidth + 120 + 40;
    const int histHeight = 700;
    const int marginLeft = 120;
    const int marginRight = 40;
    const int marginTop = 60;
    const int marginBottom = 90;
    const int plotHeight = histHeight - marginTop - marginBottom;

    cv::Mat histImage(
        histHeight,
        histWidth,
        CV_8UC3,
        cv::Scalar(255, 255, 255)
    );

    cv::rectangle(
        histImage,
        cv::Point(marginLeft, marginTop),
        cv::Point(marginLeft + plotWidth, marginTop + plotHeight),
        cv::Scalar(245, 245, 245),
        -1
    );

    cv::line(
        histImage,
        cv::Point(marginLeft, marginTop),
        cv::Point(marginLeft, marginTop + plotHeight),
        cv::Scalar(0, 0, 0),
        2
    );

    cv::line(
        histImage,
        cv::Point(marginLeft, marginTop + plotHeight),
        cv::Point(marginLeft + plotWidth, marginTop + plotHeight),
        cv::Scalar(0, 0, 0),
        2
    );

    for (int i = 1; i <= 4; ++i) {
        const int y = marginTop + plotHeight - i * plotHeight / 4;
        cv::line(
            histImage,
            cv::Point(marginLeft, y),
            cv::Point(marginLeft + plotWidth, y),
            cv::Scalar(220, 220, 220),
            1
        );
    }

    for (int v = minPixelValue; v <= maxPixelValue; ++v) {
        const int binIndex = v - minPixelValue;
        const double proportion =
            static_cast<double>(histogram[v]) / static_cast<double>(totalPixels);

        const int barHeight =
            static_cast<int>(proportion / yMax * plotHeight);

        const int x1 = marginLeft + binIndex * pixelsPerBin;
        const int x2 = x1 + pixelsPerBin - 1;
        const int y1 = marginTop + plotHeight - barHeight;
        const int y2 = marginTop + plotHeight;

        if (histogram[v] > 0) {
            cv::rectangle(
                histImage,
                cv::Point(x1, y1),
                cv::Point(x2, y2),
                cv::Scalar(30, 90, 220),
                -1
            );

            char label[32];
            std::snprintf(label, sizeof(label), "%.6f", proportion);
            cv::putText(
                histImage,
                label,
                cv::Point(x1, std::max(marginTop + 12, y1 - 4)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.35,
                cv::Scalar(200, 0, 0),
                1
            );
        }

        cv::putText(
            histImage,
            std::to_string(v),
            cv::Point(x1 + 8, marginTop + plotHeight + 28),
            cv::FONT_HERSHEY_SIMPLEX,
            0.4,
            cv::Scalar(0, 0, 0),
            1
        );
    }

    const int meanX = std::max(
        marginLeft,
        std::min(
            marginLeft + plotWidth,
            static_cast<int>(
                marginLeft + (mean - minPixelValue + 0.5) * pixelsPerBin)));

    cv::line(
        histImage,
        cv::Point(meanX, marginTop),
        cv::Point(meanX, marginTop + plotHeight),
        cv::Scalar(0, 0, 255),
        2
    );

    cv::putText(
        histImage,
        "Mean",
        cv::Point(std::max(marginLeft + 2, meanX - 24), marginTop + 18),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        cv::Scalar(0, 0, 255),
        1
    );

    cv::putText(
        histImage,
        title,
        cv::Point(40, 35),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(0, 0, 0),
        2
    );

    cv::putText(
        histImage,
        "Pixel Value",
        cv::Point(marginLeft + plotWidth / 2 - 60, histHeight - 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(0, 0, 0),
        2
    );

    cv::putText(
        histImage,
        "Proportion",
        cv::Point(15, marginTop + plotHeight / 2),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(0, 0, 0),
        2
    );

    for (int i = 0; i <= 4; ++i) {
        const double tickProportion =
            static_cast<double>(i) / 4.0 * yMax;

        const int x = marginLeft;
        const int y = marginTop + plotHeight - i * plotHeight / 4;

        cv::line(
            histImage,
            cv::Point(x - 6, y),
            cv::Point(x, y),
            cv::Scalar(0, 0, 0),
            1
        );

        char text[32];
        std::snprintf(text, sizeof(text), "%.4f", tickProportion);

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

    char rangeText[128];
    std::snprintf(
        rangeText,
        sizeof(rangeText),
        "Range: %d-%d  Y: 0-%.4f (true proportion on bars)",
        minPixelValue,
        maxPixelValue,
        yMax);

    cv::putText(
        histImage,
        rangeText,
        cv::Point(marginLeft + 10, marginTop + 25),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        cv::Scalar(0, 0, 0),
        1
    );

    char statsText[128];
    std::snprintf(
        statsText,
        sizeof(statsText),
        "Mean: %.4f  Variance: %.4f",
        mean,
        variance);

    cv::putText(
        histImage,
        statsText,
        cv::Point(marginLeft + 10, marginTop + 50),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        cv::Scalar(0, 0, 255),
        1
    );

    cv::Mat cropped = cropWhiteBorders(histImage);

    if (!cv::imwrite(histogramFilename, cropped)) {
        std::cerr << "Failed to save histogram image: "
                  << histogramFilename << std::endl;
        return false;
    }

    std::cout << "Histogram image written to: " << histogramFilename
              << " (" << cropped.cols << " x " << cropped.rows
              << ", cropped from " << histImage.cols << " x " << histImage.rows
              << ")" << std::endl;

    cv::imshow(prefix + " ROI Histogram", cropped);
    return true;
}

int main()
{
    const int width = 1920;
    const int height = 1200;

    const int left = 1156;
    const int top = 384;
    const int right = 1919;
    const int bottom = 970;

    if (left < 0 || right >= width || top < 0 || bottom >= height ||
        left > right || top > bottom) {
        std::cerr << "Invalid ROI." << std::endl;
        return 1;
    }

    std::cout << "ROI: (" << left << ", " << top << ") - ("
              << right << ", " << bottom << ")" << std::endl;

    const bool brightOk = processRoiHistogram(
        "../bright_w1920_h1200_pMono8_f10.raw",
        "bright",
        "bright ROI (1156,384)-(1919,970) zoomed histogram",
        width,
        height,
        left,
        top,
        right,
        bottom);

    const bool brightTemporalOk = processTemporalPointStats(
        "../bright_w1920_h1200_pMono8_f10.raw",
        "bright",
        "bright ROI (1156,384)-(1919,970)",
        width,
        height,
        left,
        top,
        right,
        bottom);

    const bool darkOk = processRoiHistogram(
        "../dark_w1920_h1200_pMono8_f2.raw",
        "dark",
        "dark ROI (1156,384)-(1919,970) zoomed histogram",
        width,
        height,
        left,
        top,
        right,
        bottom);

    const bool darkTemporalOk = processTemporalPointStats(
        "../dark_w1920_h1200_pMono8_f2.raw",
        "dark",
        "dark ROI (1156,384)-(1919,970)",
        width,
        height,
        left,
        top,
        right,
        bottom);

    if (!brightOk || !brightTemporalOk || !darkOk || !darkTemporalOk) {
        return 1;
    }

    std::cout << "\nPress any key to close histogram windows..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
