#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>

#include <stdio.h>
#include <stdint.h>

#include <opencv2/opencv.hpp>

#ifdef __linux__

#include <dirent.h>
#include <errno.h>

#elif _WIN32
#endif

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
using cv::PCA;

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

struct Dataset
{
	vector<Mat> images;
	vector<int> labels;
	vector<string> labelNames;
};

int listDirectory(const char* path, vector<string> &files)
{
#ifdef __linux__
	DIR *dp;
	struct dirent *dirp;

	files.resize(0);

	dp = opendir(path);

	if (!dp) {
		std::cout << errno << std::endl;
		return errno;
	}

	while ((dirp = readdir(dp))) {
		files.push_back(dirp->d_name);
	}

	return 0;
#elif _WIN32
	NOT IMPLEMENTED
#else
	NOT IMPLEMENTED
#endif
}

bool readDataset(const string &datasetPath, vector<Mat> &images, vector<int> &labels, vector<string> &labelNames)
{
	vector<string> directories;

	if (listDirectory(datasetPath.c_str(), directories) != 0) {
		return false;
	}

	std::sort(directories.begin(), directories.end());

	int currentLabel = 0;
	images.resize(0);
	labels.resize(0);
	labelNames.resize(0);

	for (const string &dir : directories) {
		if (dir == "." || dir == "..")
			continue;

		string directoryPath = datasetPath + "/" + dir;

		vector<string> imageNames;
		if (listDirectory(directoryPath.c_str(), imageNames) != 0)
			continue;

		for (const string &imageFile : imageNames) {
			if (imageFile == "." || imageFile == "..")
				continue;

			string filePath = directoryPath + "/" + imageFile;
			Mat image = cv::imread(filePath);

			if (!image.empty()) {
				images.push_back(image);
				labels.push_back(currentLabel);

				std::cout << "loaded '" << filePath << "'" << std::endl;
			}
			else {
				std::cout << "ERROR: Couldn't load image '" << filePath << "'" << std::endl;
			}
		}

		currentLabel++;
		labelNames.push_back(dir);
	}

	return true;
}

bool readDataset(const char *path, Dataset &dataset)
{
	return readDataset(path, dataset.images, dataset.labels, dataset.labelNames);
}

void addToDataset(Dataset &a, const Dataset &other)
{
	for (Mat m : other.images) {
		a.images.push_back(m.clone());
	}
	a.labels.insert(a.labels.begin(), other.labels.begin(), other.labels.end());
	a.labelNames.insert(a.labelNames.begin(), other.labelNames.begin(), other.labelNames.end());
}

void normalizeFace(
		InputArray _src,
		OutputArray _dst,
		CascadeClassifier &faceClassifier,
		CascadeClassifier &eyeClassifier,
		int outputSize = 128)
{
	CV_Assert(_src.type() == CV_8U || _src.type() == CV_8UC3);

	Mat image;

	if (_src.type() == CV_8U) {
		Mat image = _src.getMat();
	}
	else if (_src.type() == CV_8UC3) {
		cv::cvtColor(_src.getMat(), image, cv::COLOR_BGR2GRAY);
	}
	else {
		CV_Assert(0);
	}

	std::vector<Rect> faces;
	faceClassifier.detectMultiScale(image, faces);

	std::vector<Rect> eyes;
	eyeClassifier.detectMultiScale(image, eyes);

	if (faces.size() == 1) {
		_dst.create(outputSize, outputSize, CV_8U);
		Mat dst = _dst.getMat();
		dst.setTo(128);

		Rect faceRect = faces[0];
		Rect faceRectClipped = faceRect & Rect{0, 0, image.cols, image.rows};

		Mat face{faceRect.size(), CV_8U};
		face.setTo(0);
		image(faceRectClipped).copyTo(face(Rect{faceRectClipped.x - faceRect.x, faceRectClipped.y - faceRect.y, faceRectClipped.width, faceRectClipped.height}));

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

		dst.colRange(0, dst.cols * 2 / 10).setTo(0);
		dst.colRange(dst.cols - dst.cols * 2 / 10, dst.cols).setTo(0);

		cv::imshow("w", dst);
		cv::waitKey(1);
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

void normalizeFaceDataset(
		Dataset &dataset,
		CascadeClassifier &faceClassifier,
		CascadeClassifier &eyeClassifier,
		int outputSize = 128)
{
	for (Mat &m : dataset.images) {
		normalizeFace(
				m,
				m,
				faceClassifier,
				eyeClassifier,
				outputSize);
	}
}

struct EigenFacesModel
{
	int dimensions;
	cv::PCA pca;

	vector<Mat> projections;

	vector<Mat> faces;
	vector<int> labels;
	vector<string> labelNames;
};

void createEigenFacesModel(EigenFacesModel &model, const Dataset &dataset, int dimensions)
{
	CV_Assert(dataset.images.size() > 0);
	CV_Assert(dataset.images.size() == dataset.labels.size());

	int n = dataset.images.size();
	int k = dataset.images[0].total();

	if (dimensions <= 0 || dimensions > n) {
		dimensions = n;
	}

	Mat data{n, k, CV_32F};

	for (int i = 0; i < n; i++) {
		Mat currRow = dataset.images[i].reshape(0, 1);
		currRow.convertTo(currRow, data.type());
		currRow.copyTo(data.row(i));
	}

	model.pca = cv::PCA{data, cv::noArray(), cv::PCA::DATA_AS_ROW, dimensions};

	{
		Mat mean = model.pca.mean.reshape(1, dataset.images[0].rows);
		cv::imshow("w", mean / 255);
		cv::waitKey(0);

		Mat basis = model.pca.eigenvectors.reshape(1, dataset.images[0].rows * dimensions).clone();
		cv::normalize(basis, basis, 0, 1, cv::NORM_MINMAX, CV_32F);
		cv::imshow("w", basis);
		cv::waitKey(0);
	}

	for (int i = 0; i < n; i++) {
		if (false) {
			cv::imshow("w", data.row(i).reshape(1, dataset.images[0].rows) / 255);
			cv::waitKey(0);
		}

		Mat projection = model.pca.project(data.row(i));
		model.projections.push_back(projection);

		if (false) {
			cv::imshow("w", model.pca.backProject(projection).reshape(1, dataset.images[0].rows) / 255);
			cv::waitKey(0);
		}
	}

	// TODO(Andrey): Deep copy?

	model.faces = dataset.images;
	model.labels = dataset.labels;
	model.labelNames = dataset.labelNames;

}

void predict(EigenFacesModel &model, Mat face)
{
	CV_Assert(face.total() == model.pca.mean.total());
	Mat testVector = face.reshape(0, 1);
	Mat projection = model.pca.project(testVector);

	cv::imshow("w", face);
	cv::waitKey(0);

	std::cout << "============================" << std::endl;

	float minDist = FLT_MAX;
	int minLabel = -1;

	for (int i = 0; i < (int)model.projections.size(); i++) {
		float dist = cv::norm(projection, model.projections[i], cv::NORM_L2);
		std::cout << dist << " " << model.labelNames[model.labels[i]] << std::endl;

		if (dist < minDist) {
			minDist = dist;
			minLabel = model.labels[i];
		}
	}

	std::cout << "Closest to: " << model.labelNames[minLabel] << std::endl;

	cv::imshow("w", model.pca.backProject(projection).reshape(1, face.rows) / 255);
	cv::waitKey(0);
}

void predict(EigenFacesModel &model, Dataset &testDataset)
{
	for (int i = 0; i < (int)testDataset.images.size(); i++) {
		predict(model, testDataset.images[i]);
		std::cout << "Correct: " << testDataset.labelNames[testDataset.labels[i]] << std::endl;
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

	Dataset train;
	if (!readDataset("../images/dataset1", train)) {
		std::cout << "ERROR: Clouldn't load dataset" << std::endl;
	}
	normalizeFaceDataset(train, faceClassifier, eyeClassifier);

	EigenFacesModel model;
	createEigenFacesModel(model, train, 30);

	Dataset test;
	if (!readDataset("../images/dataset2", test)) {
		std::cout << "ERROR: Clouldn't load dataset" << std::endl;
	}
	normalizeFaceDataset(test, faceClassifier, eyeClassifier);

	predict(model, test);

	return 0;
}
