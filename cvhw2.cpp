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
			if (imageFile.substr(0, 1) == ".")
				continue;

			string filePath = directoryPath + "/" + imageFile;
			std::cout << filePath << std::endl;
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
	faceClassifier.detectMultiScale(image, faces, 1.05, 12);

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

		cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::equalizeHist(dst, dst);

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
		cv::waitKey(1);
	}
}

void normalizeFaceDataset(
		Dataset &dataset,
		CascadeClassifier &faceClassifier,
		CascadeClassifier &eyeClassifier,
		int outputSize = 128)
{
	vector<Mat> newImages;
	vector<int> newLabels;

	for (int i = 0; i < (int)dataset.images.size(); i++) {
		Mat normalized;
		normalizeFace(
				dataset.images[i],
				normalized,
				faceClassifier,
				eyeClassifier,
				outputSize);

		if (!normalized.empty()) {
			newImages.push_back(normalized);
			newLabels.push_back(dataset.labels[i]);
		}
	}

	dataset.images = newImages;
	dataset.labels = newLabels;
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

	std::cout << "BEGIN PCA" << std::endl;

	model.pca = cv::PCA{data, cv::noArray(), cv::PCA::DATA_AS_ROW, dimensions};

	std::cout << "END PCA" << std::endl;

	{
		Mat mean = model.pca.mean.reshape(1, dataset.images[0].rows);
		if (true) {
			cv::imshow("w", mean / 255);
			cv::waitKey(0);
		}

		Mat basis = model.pca.eigenvectors.reshape(1, dataset.images[0].rows * dimensions).clone();
		cv::normalize(basis, basis, 0, 1, cv::NORM_MINMAX, CV_32F);
		if (true) {
			cv::imshow("w", basis);
			cv::waitKey(0);
		}
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

void visualizeEigenSpaceDots(
		const vector<Mat> &trainVectors,
		const vector<int> &trainLabels,
		const vector<Mat> &testVectors,
		const vector<int> &testLabels)
{
	int xAxis = 0;
	int yAxis = 0;

	int axeCount = 0;

	if (trainVectors.size() > 0) {
		axeCount = trainVectors[0].total();
	}
	else if (testVectors.size() > 0) {
		axeCount = testVectors[0].total();
	}

	float xMin = FLT_MAX;
	float xMax = FLT_MIN;

	float yMin = FLT_MAX;
	float yMax = FLT_MIN;

	Mat canvas{800, 800, CV_8UC3};

	vector<Scalar> colorPalette = {
		{180, 119, 31},
		{14, 127, 255},
		{44, 160, 44},
		{40, 39, 214},
		{189, 103, 148},
		{75, 86, 140},
		{194, 119, 227},
		{127, 127, 127},
		{34, 189, 188},
		{207, 190, 23},
	};

	int pressedKey = 0;

	while (pressedKey != 'q') {
		auto updateAxisMinMax = [](const vector<Mat> &vectors, int axis, float &axisMin, float &axisMax) {
			for (const Mat &m : vectors) {
				CV_Assert(m.type() == CV_32F);

				if (axis < (int)m.total()) {
					float val = m.at<float>(axis);

					axisMin = std::min(axisMin, val);
					axisMax = std::max(axisMax, val);
				}
			}
		};

		updateAxisMinMax(trainVectors, xAxis, xMin, xMax);
		updateAxisMinMax(testVectors, xAxis, xMin, xMax);

		updateAxisMinMax(trainVectors, yAxis, yMin, yMax);
		updateAxisMinMax(testVectors, yAxis, yMin, yMax);

		canvas.setTo(Scalar{52, 44, 40});

		for (int i = 0; i < (int)trainVectors.size(); i++) {
			const Mat &v = trainVectors[i];
			int label = trainLabels[i];

			Point2f p;

			p.x = (v.at<float>(xAxis) - xMin) / (xMax - xMin) * canvas.cols;
			p.y = (v.at<float>(yAxis) - yMin) / (yMax - yMin) * canvas.rows;

			Scalar drawingColor = label < 0 ?
				Scalar{0} : colorPalette[label % colorPalette.size()];

			cv::circle(canvas, p, 7, drawingColor, -1);
		}

		for (int i = 0; i < (int)testVectors.size(); i++) {
			const Mat &v = testVectors[i];
			int label = trainLabels[i];

			Point2f p;

			p.x = (v.at<float>(xAxis) - xMin) / (xMax - xMin) * canvas.cols;
			p.y = (v.at<float>(yAxis) - yMin) / (yMax - yMin) * canvas.rows;

			int rectSize = 10;

			Rect r{(int)(p.x - rectSize / 2), (int)(p.y - rectSize / 2), rectSize, rectSize};

			Scalar drawingColor = label < 0 ?
				Scalar{0} : colorPalette[label % colorPalette.size()];

			cv::rectangle(canvas, r, drawingColor, -1);
		}

		cv::imshow("Eigenspace", canvas);
		pressedKey = cv::waitKey(0);

		switch (pressedKey) {
			case 'j':
				if (yAxis < axeCount - 1)
					yAxis++;
				else
					std::cout << '\a';
				break;
			case 'k':
				if (yAxis > 0)
					yAxis--;
				else
					std::cout << '\a';
				break;
			case 'h':
				if (xAxis > 0)
					xAxis--;
				else
					std::cout << '\a';
				break;
			case 'l':
				if (xAxis < axeCount - 1)
					xAxis++;
				else
					std::cout << '\a';
				break;
			case 'm':
				if (yAxis < axeCount - 1) {
					yAxis++;
				}
				else if (xAxis < axeCount - 1) {
					yAxis = 0;
					xAxis++;
				}
				else {
					std::cout << '\a';
				}
				break;
			case 'n':
				if (yAxis > 0) {
					yAxis--;
				}
				else if (xAxis > 0) {
					yAxis = axeCount - 1;
					xAxis--;
				}
				else {
					std::cout << '\a';
				}
				break;
		}

		std::cout << xAxis << " : " << yAxis << std::endl;
	}
}

void visualizeEigenSpaceLines(
		const vector<Mat> &trainVectors,
		const vector<int> &trainLabels,
		const vector<Mat> &testVectors,
		const vector<int> &testLabels)
{
	int axeCount = 0;

	if (trainVectors.size() > 0) {
		axeCount = trainVectors[0].total();
	}
	else if (testVectors.size() > 0) {
		axeCount = testVectors[0].total();
	}

	vector<float> axisMin(axeCount, FLT_MAX);
	vector<float> axisMax(axeCount, FLT_MIN);

	auto updateAxisMinMax = [axeCount](const vector<Mat> &vectors, vector<float> &axisMin, vector<float> &axisMax) {
		for (int i = 0; i < (int)vectors.size(); i++) {
			Mat m = vectors[i];

			CV_Assert(vectors[i].type() == CV_32F);
			CV_Assert(axeCount == (int)m.total());

			for (int axis = 0; axis < (int)m.total(); axis++) {
				float val = m.at<float>(axis);

				axisMin[axis] = std::min(axisMin[axis], val);
				axisMax[axis] = std::max(axisMax[axis], val);
			}
		}
	};

	updateAxisMinMax(trainVectors, axisMin, axisMax);
	updateAxisMinMax(testVectors, axisMin, axisMax);

	float globalMin = *std::min_element(axisMin.begin(), axisMin.end());
	float globalMax = *std::max_element(axisMax.begin(), axisMax.end());

	Mat canvas{1024 / 4, 1024, CV_8UC3};

	vector<Scalar> colorPalette = {
		{180, 119, 31},
		{14, 127, 255},
		{44, 160, 44},
		{40, 39, 214},
		{189, 103, 148},
		{75, 86, 140},
		{194, 119, 227},
		{127, 127, 127},
		{34, 189, 188},
		{207, 190, 23},
	};

	int shownLabel = -1;
	bool uniformScaling = false;

	int pressedKey = 0;

	while (pressedKey != 'q') {

		canvas.setTo(Scalar{52, 44, 40});

		for (int i = 0; i < axeCount; i++) {
			int x = canvas.cols * i / (axeCount - 1);
			Point2i top{x, 0};
			Point2i bot{x, canvas.rows};
			cv::line(canvas, top, bot, Scalar{0}, 1);
		}

		vector<Point2i> featureLine(axeCount);

		for (int i = 0; i < (int)trainVectors.size(); i++) {
			const Mat &v = trainVectors[i];
			int label = trainLabels[i];

			if (shownLabel >= 0 && shownLabel != label)
				continue;

			Scalar drawingColor = label < 0 ?
				Scalar{0} : colorPalette[label % colorPalette.size()];

			for (int axis = 0; axis < axeCount; axis++) {
				Point2i p;

				float yMin = uniformScaling ? globalMin : axisMin[axis];
				float yMax = uniformScaling ? globalMax : axisMax[axis];

				p.x = canvas.cols * (axis) / (axeCount - 1);
				p.y = (v.at<float>(axis) - yMin) / (yMax - yMin) * canvas.rows;

				featureLine[axis] = p;
			}

			cv::polylines(canvas, {featureLine}, false, drawingColor, 1);
		}

		for (int i = 0; i < (int)testVectors.size(); i++) {
			const Mat &v = testVectors[i];
			int label = testLabels[i];

			if (shownLabel >= 0 && shownLabel != label)
				continue;

			Scalar drawingColor = label < 0 ?
				Scalar{0} : colorPalette[label % colorPalette.size()];

			for (int axis = 0; axis < axeCount; axis++) {
				Point2i p;

				float yMin = uniformScaling ? globalMin : axisMin[axis];
				float yMax = uniformScaling ? globalMax : axisMax[axis];

				p.x = canvas.cols * (axis) / (axeCount - 1);
				p.y = (v.at<float>(axis) - yMin) / (yMax - yMin) * canvas.rows;

				featureLine[axis] = p;
			}

			cv::polylines(canvas, {featureLine}, false, drawingColor, 2);
		}

		cv::imshow("Eigenspace", canvas);
		pressedKey = cv::waitKey(0);

		switch (pressedKey) {
			case 'j':
				shownLabel++;
				break;
			case 'k':
				if (shownLabel > -1)
					shownLabel--;
				else
					std::cout << '\a';
				break;
			case 'u':
				uniformScaling = !uniformScaling;
				break;
		}
	}
}

void predict(EigenFacesModel &model, Dataset &testDataset)
{
	vector<Mat> testProjections;

	for (int i = 0; i < (int)testDataset.images.size(); i++) {
		Mat face = testDataset.images[i];
		CV_Assert(face.total() == model.pca.mean.total());

		std::cout << "============================" << std::endl;

		Mat testVector = face.reshape(0, 1);
		Mat projection = model.pca.project(testVector);

		testProjections.push_back(projection);

		float minDist = FLT_MAX;
		int minLabel = -1;

		for (int i = 0; i < (int)model.projections.size(); i++) {
			float dist = cv::norm(projection, model.projections[i], cv::NORM_L2);
			//float dist = cv::norm(projection, model.projections[i], cv::NORM_L1);

			if (false) {
				std::cout << dist << " " << model.labelNames[model.labels[i]] << std::endl;
			}

			if (dist < minDist) {
				minDist = dist;
				minLabel = model.labels[i];
			}
		}

		std::cout << "Closest to: " << model.labelNames[minLabel] << std::endl;
		std::cout << "Correct: " << testDataset.labelNames[testDataset.labels[i]] << std::endl;

		if (false) {
			cv::imshow("w", face);
			cv::waitKey(0);
			cv::imshow("w", model.pca.backProject(projection).reshape(1, face.rows) / 255);
			cv::waitKey(0);
		}
	}

	visualizeEigenSpaceDots(model.projections, model.labels, testProjections, testDataset.labels);
	visualizeEigenSpaceLines(model.projections, model.labels, testProjections, testDataset.labels);
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
	if (!readDataset("../images/faces96_small", train)) {
		std::cout << "ERROR: Clouldn't load dataset" << std::endl;
		return 0;
	}

	Dataset test;
	if (!readDataset("../images/dataset2/test", test)) {
		std::cout << "ERROR: Clouldn't load dataset" << std::endl;
	}

	normalizeFaceDataset(train, faceClassifier, eyeClassifier);
	normalizeFaceDataset(test, faceClassifier, eyeClassifier);

	EigenFacesModel model;
	createEigenFacesModel(model, train, 8);

	predict(model, test);

	return 0;
}
