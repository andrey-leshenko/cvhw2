#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>

#include <stdio.h>
#include <stdint.h>

#include <opencv2/opencv.hpp>

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

using std::vector;
using std::string;

using cv::Mat;
using cv::Scalar;
using cv::InputArray;
using cv::OutputArray;
using cv::Point2d;
using cv::Point2f;
using cv::Point2i;
using cv::Point3d;
using cv::Point3f;
using cv::Point3i;
using cv::Vec2d;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using cv::Vec4i;
using cv::Size;
using cv::Size2i;
using cv::Range;
using cv::Affine3d;
using cv::Affine3f;
using cv::Rect2i;
using cv::String;

using cv::Mat1i;
using cv::Mat1f;
using cv::Mat2f;
using cv::Mat1b;
using cv::Mat3b;
using cv::Mat3f;

using cv::Vec3b;

using cv::Point;
using cv::Rect;
using cv::CascadeClassifier;

#define QQQ do {std::cerr << "QQQ " << __FUNCTION__ << " " << __LINE__ << std::endl;} while(0)

void printTimeSinceLastCall(const char* message)
{
	static s64 freq = static_cast<int>(cv::getTickFrequency());
	static s64 last = cv::getTickCount();

	s64 curr = cv::getTickCount();
	s64 delta = curr - last;
	double deltaMs = (double)delta / freq * 1000;
	printf("%s: %.4f\n", message, deltaMs);
	fflush(stdout);

	last = curr;
}

inline void ltrim(std::string &s)
{
	s.erase(s.begin(), std::find_if(s.begin(), s.end(),
				std::not1(std::ptr_fun<int, int>(std::isspace))));
}

inline void rtrim(std::string &s)
{
	s.erase(std::find_if(s.rbegin(), s.rend(),
				std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
}

inline void trim(std::string &s)
{
	ltrim(s);
	rtrim(s);
}

void readDataset(const char* path, vector<Mat> &images, vector<int> &labels, vector<string> &labelNames)
{
	images.resize(0);
	labels.resize(0);
	labelNames.resize(0);

	int currentLabel = -1;

	std::size_t lastSlash = string{path}.find_last_of('/');
	if (lastSlash == std::string::npos) {
		lastSlash = 0;
	}
	else {
		lastSlash = lastSlash + 1;
	}
	string directoryPath{path, path + lastSlash};

	std::cout << directoryPath << std::endl;

	std::ifstream f{path};

	if (!f.is_open()) {
		std::cout << "ERROR: Couldn't open dataset '" << path << "'" << std::endl;
	}

	std::cout << ">>> DATASET BEGIN" << std::endl;

	std::string line;
	while (std::getline(f, line))
	{
		trim(line);

		if (line == "XXX") {
			// EOF mark
			break;
		}
		else if (line.size() > 0 && line[0] == '#') {
			string label = line.substr(1);
			trim(label);

			labelNames.push_back(label);
			currentLabel++;

			std::cout << std::endl;
			std::cout << "# label '" << label << "'" << std::endl;
		}
		else if (line.size() > 0) {
			CV_Assert(currentLabel >= 0);
			Mat image = cv::imread(directoryPath + line, cv::IMREAD_GRAYSCALE);

			if (!image.empty()) {
				images.push_back(image);
				labels.push_back(currentLabel);

				std::cout << "loaded '" << line << "'" << std::endl;
			}
			else {
				std::cout << "ERROR: Couldn't load image '" << line << "'" << std::endl;
			}
		}
	}

	std::cout << std::endl << "<<< DATASET END" << std::endl;
}

void normalizeFace(
		InputArray _src,
		OutputArray _dst,
		CascadeClassifier &faceClassifier,
		CascadeClassifier &eyeClassifier,
		int outputSize = 128)
{
	CV_Assert(_src.type() == CV_8U);

	Mat image = _src.getMat();

	std::vector<Rect> faces;
	faceClassifier.detectMultiScale(image, faces);

	std::vector<Rect> eyes;
	eyeClassifier.detectMultiScale(image, eyes);

	if (faces.size() == 1) {
		_dst.create(outputSize, outputSize, CV_8U);
		Mat dst = _dst.getMat();
		dst.setTo(128);

		Rect faceRect = faces[0];
		Mat face = image(faceRect);

		Mat faceResized;

		int maxSize = std::max(faceRect.width, faceRect.height);
		cv::resize(
				face,
				dst,
				Size{outputSize, outputSize},
				0,
				0,
				maxSize > outputSize ? cv::INTER_AREA : cv::INTER_CUBIC);

		// NOTE(Andery): 
		cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);

		cv::imshow("w", dst);
		cv::waitKey(0);
	}
	else {
		Mat display;
		cv::cvtColor(image, display, cv::COLOR_GRAY2BGR);

		for (Rect rect : faces) {
			cv::rectangle(display, rect, Scalar{0, 0, 255}, 2);
		}

		for (Rect rect : eyes) {
			cv::rectangle(display, rect, Scalar{0, 255, 0}, 2);
			cv::circle(display, (rect.tl() + rect.br()) / 2, 2, Scalar{255, 0, 0}, -1);
		}

		cv::imshow("w", display);
		cv::waitKey(0);
	}
}

int main(int argc, char* argv[])
{
	CascadeClassifier faceClassifier{"../images/haarcascade_frontalface_default.xml"};
	CascadeClassifier eyeClassifier{"../images/haarcascade_eye.xml"};

	if (faceClassifier.empty() || eyeClassifier.empty()) {
		std::cout << "ERROR: Couldn't load classifier" << std::endl;
		return 0;
	}

	vector<Mat> trainImages;
	vector<int> trainLabels;
	vector<string> labelNames;

	readDataset("../images/dataset_train.txt", trainImages, trainLabels, labelNames);

	vector<Mat> testImages;
	vector<int> testLabels;
	vector<string> testLabelNames;

	readDataset("../images/dataset_test.txt", testImages, testLabels, labelNames);

	for (Mat im : trainImages) {
		Mat face;
		normalizeFace(im, face, faceClassifier, eyeClassifier);
	}

	// TODO(Andrey): Check if all faces could be normalized

	//cv::Ptr<cv::BasicFaceRecognizer> model = cv::createEigenFaceRecognizer();
	//model->train(images, labels);

	return 0;
}
