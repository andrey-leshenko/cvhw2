#include "image_db.hpp"

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <string>
#include <fstream>
#include <utility>
#include <thread>

#include <opencv2/opencv.hpp>

//#define USE_DLIB

#ifdef USE_DLIB
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#endif

using std::vector;
using std::string;
using std::map;
using std::pair;

using cv::Mat;
using cv::Mat1i;
using cv::Mat1f;

using cv::CascadeClassifier;
using cv::PCA;

using cv::Scalar;
using cv::InputArray;
using cv::OutputArray;
using cv::Point2f;
using cv::Point2i;
using cv::Rect;
using cv::Size;
using cv::String;

#define QQQ do {std::cerr << "QQQ " << __FUNCTION__ << " " << __LINE__ << std::endl;} while(0)

void printTimeSinceLastCall(const char* message)
{
	static int64 freq = static_cast<int>(cv::getTickFrequency());
	static int64 last = cv::getTickCount();

	int64 curr = cv::getTickCount();
	int64 delta = curr - last;
	double deltaMs = (double)delta / freq * 1000;
	printf("%s: %.4f\n", message, deltaMs);
	fflush(stdout);

	last = curr;
}

Mat asRowMatrix(cv::InputArrayOfArrays src, int rtype, double alpha=1, double beta=0) {
	CV_Assert(src.kind() == cv::_InputArray::STD_VECTOR_MAT || src.kind() == cv::_InputArray::STD_VECTOR_VECTOR);

	size_t rows = src.total();

	if(rows == 0)
		return Mat();

	size_t cols = src.getMat(0).total();

	Mat data((int)rows, (int)cols, rtype);

	for(unsigned int i = 0; i < rows; i++) {
		Mat currentMat = src.getMat(i);

		// make sure data can be reshaped
		if(currentMat.total() != cols) {
			String error_message = cv::format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, cols, currentMat.total());
			CV_Error(cv::Error::StsBadArg, error_message);
		}

		// Reshape only workes with continuous matrices
		if (!currentMat.isContinuous()) {
			currentMat = currentMat.clone();
		}

		currentMat.reshape(1, 1).convertTo(data.row(i), rtype, alpha, beta);
	}
	return data;
}

void normalizeFace(
		InputArray _src,
		OutputArray _dst,
		CascadeClassifier &faceClassifier,
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

		cv::imshow("w", display);
		cv::waitKey(1);
	}
}

vector<int> normalizeFacesVJ(
		ImageDB &idb,
		const vector<int> &ids,
		CascadeClassifier &faceClassifier,
		int outputSize = 128)
{
	vector<int> resultIds;

	for (auto id : ids) {
		const image_item m = idb.items[id];

		string newPath = m.path + "#NORMALIZED_VJ";
		int normalizedId = idb.getImageId(newPath);

		if (normalizedId < 0) {
			Mat normalized;

			normalizeFace(
					m.image,
					normalized,
					faceClassifier,
					outputSize);

			image_item newItem;

			// NOTE(Andrey): We add empty images to the DB too,
			// to avoid processing them each time

			newItem.image = normalized;
			newItem.label = m.label;
			newItem.path = newPath;
			normalizedId = idb.addImage(newItem);
		}

		if (!idb.items[normalizedId].image.empty()) {
			resultIds.push_back(normalizedId);
		}
	}

	return resultIds;
}

#ifdef USE_DLIB

vector<Point2f> convertDlibShapeToOpenCV(dlib::full_object_detection objectDet, Rect& outputRect)
{
	dlib::rectangle dlibRect = objectDet.get_rect();
	outputRect = Rect(dlibRect.left(), dlibRect.top(), dlibRect.width(), dlibRect.height());

	vector<Point2f> parts;

	for(unsigned long i = 0; i < objectDet.num_parts(); i++)
	{
		dlib::point p = objectDet.part(i);
		Point2f cvPoint{(float)p.x(), (float)p.y()};
		parts.push_back(cvPoint);
	}

	return parts;
}

vector<Mat> alignImageFaces(Mat image, dlib::frontal_face_detector detector, dlib::shape_predictor pose_model)
{
	try
	{
		// Turn OpenCV's Mat into something dlib can deal with. Note that this just
		// wraps the Mat object, it doesn't copy anything. So cimg is only valid as
		// long as temp is valid. Also don't do anything to temp that would cause it
		// to reallocate the memory which stores the image as that will make cimg
		// contain dangling pointers. This basically means you shouldn't modify temp
		// while using cimg.
		cv::Mat temp = image.clone();
		dlib::cv_image<dlib::bgr_pixel> cimg(temp);

		// Detect faces
		vector<dlib::rectangle> faces = detector(cimg);

		// Find the pose of each face.
		vector<dlib::full_object_detection> shapes;
		for (unsigned long i = 0; i < faces.size(); i++)
			shapes.push_back(pose_model(cimg, faces[i]));

		// Convert the faces detected by dlib to something OpenCV can deal with.
		vector<vector<Point2f>> facialLandmarks(shapes.size());

		for(unsigned long i = 0; i < shapes.size(); i++)
		{
			Rect faceRect;
			facialLandmarks[i] = convertDlibShapeToOpenCV(shapes[i], faceRect);
		}

		// for(int i = 0; i < facialLandmarks[0].size(); i++)
		// {
		//	 circle(myImage, facialLandmarks[0][i], 3, Scalar(0, 0, 255));
		//	 string objectTitle = std::to_string(i);
		//	 cv::putText(myImage, objectTitle, facialLandmarks[0][i], cv::FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 255, 0), 0.5);
		// }

		vector<Mat> alignedFaces;
		alignedFaces.reserve(facialLandmarks.size());

		for(const vector<Point2f> &face : facialLandmarks)
		{
			auto centroid = [](const vector<Point2f> &points, int begin, int end) {
				Point2f sum = std::accumulate(
						points.begin() + begin,
						points.begin() + end,
						Point2f{0.0f, 0.0f});

				return sum / (end - begin);
			};

			// The locations of the facial landmarks visually presented:
			// https://github.com/cmusatyalab/openface/blob/master/images/dlib-landmark-mean.png

			Point2f leftEye = centroid(face, 36, 42);
			Point2f rightEye = centroid(face, 42, 48);
			Point2f mouth = centroid(face, 48, 68);

			vector<Point2f> srcPoints = {leftEye, mouth, rightEye};
			vector<Point2f> dstPoints = {Point2f(50, 60), Point2f(75, 120), Point2f(100, 60)};
			Mat affineTransform = cv::getAffineTransform(srcPoints, dstPoints);

			Mat transformedFace;
			cv::warpAffine(image, transformedFace, affineTransform, Size(150, 175));

			alignedFaces.push_back(transformedFace);
		}

		if (alignedFaces.size() == 0)
			std::cerr << "No faces Detected! returning empty array" << std::endl;

		return alignedFaces;
	}
	catch(dlib::serialization_error& e)
	{
		std::cout << "You need dlib's default face landmarking model file to run this example." << std::endl
			<< "You can get it from the following URL: " << std::endl
			<< "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << std::endl
			<< std::endl << e.what() << std::endl;
	}
	catch(std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	return vector<Mat>();
}

vector<int> normalizeFacesDlib(
		ImageDB &idb,
		const vector<int> &ids,
		dlib::frontal_face_detector detector,
		dlib::shape_predictor pose_model)
{
	vector<int> resultIds;

	for (auto id : ids) {
		const image_item m = idb.items[id];

		string newPath = m.path + "#NORMALIZED_DLIB";
		int normalizedId = idb.getImageId(newPath);

		if (normalizedId < 0) {
			vector<Mat> faces = alignImageFaces(m.image, detector, pose_model);

			if (faces.size() >= 1) {
				Mat normalized = faces[0];
				cv::cvtColor(normalized, normalized, cv::COLOR_BGR2GRAY);
				cv::normalize(normalized, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
				cv::equalizeHist(normalized, normalized);
				normalized.colRange(0, normalized.cols * 2 / 10).setTo(0);
				normalized.colRange(normalized.cols - normalized.cols * 2 / 10, normalized.cols).setTo(0);

				image_item newItem;

				// NOTE(Andrey): We add empty images to the DB too,
				// to avoid processing them each time

				newItem.image = normalized;
				newItem.label = m.label;
				newItem.path = newPath;
				normalizedId = idb.addImage(newItem);

				cv::imshow("w", normalized);
				cv::waitKey(1);
			}
		}

		if (!idb.items[normalizedId].image.empty()) {
			resultIds.push_back(normalizedId);
		}
	}

	return resultIds;
}

#endif // USE_DLIB

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
		const Mat &trainProj,
		const vector<int> &trainLabels,
		const Mat &testProj,
		const vector<int> &testChosenLabels,
		const vector<int> &testCorrectLabels)
{
	int axeCount = std::max(trainProj.cols, testProj.cols);

	double globalMin;
	double globalMax;

	{
		double trainMin;
		double trainMax;

		double testMin;
		double testMax;

		cv::minMaxIdx(trainProj, &trainMin, &trainMax);
		cv::minMaxIdx(testProj, &testMin, &testMax);

		globalMin = std::min(trainMin, testMin);
		globalMax = std::max(trainMax, testMax);
	}

	vector<int> nonEmptyLabels{trainLabels};
	nonEmptyLabels.insert(nonEmptyLabels.end(), testChosenLabels.begin(), testChosenLabels.end());
	nonEmptyLabels.insert(nonEmptyLabels.end(), testCorrectLabels.begin(), testCorrectLabels.end());
	std::sort(nonEmptyLabels.begin(), nonEmptyLabels.end());
	nonEmptyLabels.erase(std::unique(nonEmptyLabels.begin(), nonEmptyLabels.end()), nonEmptyLabels.end());

	Mat canvas{1024 / 2, 1024, CV_8UC3};

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

	int shownLabelIndex = -1;
	int pressedKey = 0;

	while (pressedKey != 'q' && pressedKey != '\r') {
		int shownLabel = shownLabelIndex < 0 ? -1 : nonEmptyLabels[shownLabelIndex];
		canvas.setTo(Scalar{52, 44, 40});

		for (int i = 0; i < axeCount; i++) {
			int x = canvas.cols * i / (axeCount - 1);
			Point2i top{x, 0};
			Point2i bot{x, canvas.rows};
			cv::line(canvas, top, bot, Scalar{0}, 1);
		}

		vector<Point2i> featureLine(axeCount);

		auto drawLine = [&canvas, &featureLine, globalMin, globalMax, axeCount](const Mat& yCoords, Scalar color, int thickness = 1) {
			for (int axis = 0; axis < axeCount; axis++) {
				featureLine[axis] = Point2i {
					canvas.cols * (axis) / (axeCount - 1),
					(int)((yCoords.at<float>(axis) - globalMin) / (globalMax - globalMin) * canvas.rows)
				};
			}

			cv::polylines(canvas, {featureLine}, false, color, thickness);
		};

		for (int i = 0; i < trainProj.rows; i++) {
			int label = trainLabels[i];

			Scalar color;
			int thickness = 1;

			if (shownLabel < 0) {
				color = (label >= 0) ? colorPalette[label % colorPalette.size()] : Scalar{0};
				thickness = 1;
			}
			else if (shownLabel == label) {
				color = Scalar{200, 200, 200};
				thickness = 2;
			}
			else {
				thickness = 0;
			}

			if (thickness > 0) {
				drawLine(trainProj.row(i), color, thickness);
			}
		}

		for (int i = 0; i < testProj.rows; i++) {
			int correctLabel = testChosenLabels[i];
			int chosenLabel = testCorrectLabels[i];

			Scalar color;
			int thickness = 1;

			if (shownLabel < 0) {
				color = (correctLabel >= 0) ? colorPalette[correctLabel % colorPalette.size()] : Scalar{0};
				thickness = 2;
			}
			else if (shownLabel == correctLabel) {
				if (chosenLabel == correctLabel) {
					color = Scalar{0, 200, 0};
				}
				else {
					color = Scalar{0, 0, 180};
				}
				thickness = 2;
			}
			else if (chosenLabel == shownLabel) {
				color = Scalar{0};
				thickness = 2;
			}
			else {
				thickness = 0;
			}

			if (thickness > 0) {
				drawLine(testProj.row(i), color, thickness);
			}
		}

		cv::imshow("Eigenspace", canvas);
		pressedKey = cv::waitKey(0);

		switch (pressedKey) {
			case 'j':
				if (shownLabelIndex < (int)nonEmptyLabels.size() - 1)
					shownLabelIndex++;
				break;
			case 'k':
				if (shownLabelIndex > -1)
					shownLabelIndex--;
				break;
		}
	}
}

vector<int> classifyKNN(Mat trainProj, const vector<int> &trainLabels, int labelCount, Mat testProj, int k = 1)
{
	vector<std::pair<float, int>> distancesAndLabels;
	distancesAndLabels.reserve(trainProj.rows);

	vector<int> labelVotes(labelCount);
	vector<int> chosenLabels;

	for (int i = 0; i < testProj.rows; i++) {
		distancesAndLabels.resize(0);

		for (int j = 0; j < trainProj.rows; j++) {
			float dist = cv::norm(testProj.row(i), trainProj.row(j), cv::NORM_L2);

			distancesAndLabels.push_back({dist, trainLabels[j]});
		}

		std::sort(distancesAndLabels.begin(), distancesAndLabels.end());
		if ((int)distancesAndLabels.size() > k)
			distancesAndLabels.resize(k);

		std::fill(labelVotes.begin(), labelVotes.end(), 0);

		for (auto p : distancesAndLabels) {
			labelVotes[p.second]++;
		}

		int mostVoted = std::max_element(labelVotes.begin(), labelVotes.end()) - labelVotes.begin();
		chosenLabels.push_back(mostVoted);
	}

	return chosenLabels;
}

void labelMeanAndSD(Mat data, const vector<int> &labels, int labelCount, Mat &outMean, Mat &outSD)
{
	//
	// You can see that the mean and variance calculations are very similar.
	// They are actually part of a more general idea called "Moments":
	// the mean is the first raw moment, and the variance is the second central
	// moment (moment about the mean). You can read about them here:
	//
	// https://en.wikipedia.org/wiki/Moment_(mathematics)
	//

	Mat1f mean{labelCount, data.cols};
	mean.setTo(0);

	{
		Mat1i count{1, labelCount, 0};

		for (int i = 0; i < data.rows; i++) {
			mean.row(labels[i]) += data.row(i);
			count(labels[i])++;
		}

		for (int i = 0; i < labelCount; i++) {
			if (count(i) != 0) {
				mean.row(i) /= count(i);
			}
		}
	}

	Mat1f variance{labelCount, data.cols};
	variance.setTo(0);

	{
		Mat1i count{1, labelCount, 0};

		for (int i = 0; i < data.rows; i++) {
			Mat1f diff = data.row(i) - mean.row(labels[i]);
			variance.row(labels[i]) += diff.mul(diff);
			count(labels[i])++;
		}

		for (int i = 0; i < labelCount; i++) {
			if (count(i) != 0) {
				variance.row(i) /= count(i);
			}
			else {
				variance.row(i).setTo(-1);
			}
		}
	}

	Mat1f standardDeviation{labelCount, data.cols};

	{
		for (int i = 0; i < labelCount; i++) {
			if (cv::countNonZero(variance.row(i) < 0) == 0) {
				cv::sqrt(variance.row(i), standardDeviation.row(i));
			}
			else {
				standardDeviation.row(i).setTo(-1);
			}
		}
	}

	outMean = mean;
	outSD = standardDeviation;
}

vector<int> classifyMahalanobis(Mat trainProj, const vector<int> &trainLabels, int labelCount, Mat testProj)
{
	Mat1f mean;
	Mat1f standardDeviation;

	labelMeanAndSD(trainProj, trainLabels, labelCount,
			mean, standardDeviation);

	vector<int> chosenLabels;

	for (int i = 0; i < testProj.rows; i++) {
		Mat testVector = testProj.row(i);

		int closestLabel = -1;
		float minDistance = FLT_MAX;

		for (int k = 0; k < labelCount; k++) {
			Mat1f labelMean = mean.row(k);
			Mat1f labelSD = standardDeviation.row(k);

			if (cv::countNonZero(labelSD < 0) != 0) {
				continue;
			}

			float distance = cv::norm((testVector - labelMean).mul(1 / labelSD), cv::NORM_L2);

			if (distance < minDistance) {
				minDistance = distance;
				closestLabel = k;
			}
		}

		chosenLabels.push_back(closestLabel);
	}

	return chosenLabels;
}

struct FaceRecSystem
{
	enum class DistanceType {
		KNN,
		Mahalanobis,
	};

	CascadeClassifier faceClassifier;

#ifdef USE_DLIB
	dlib::frontal_face_detector detector;
	dlib::shape_predictor pose_model;
#endif

	ImageDB idb;

	vector<int> trainIds;
	vector<int> testIds;

	Mat trainData;
	Mat testData;

	int maxDimensions = 12;
	cv::PCA pca;

	Mat trainProj;
	Mat testProj;

	DistanceType distanceType = DistanceType::KNN;
	bool useDlib = false;

	std::thread savingThread;

	void loadImageDB()
	{
		idb.load("idb.bin");
	}

	void saveImageDB()
	{
		ImageDB idbCopy = idb;

		// NOTE: The new thread will first wait for the old thread

		savingThread = std::thread{[idbCopy](std::thread oldSavingThread) {
			if (oldSavingThread.joinable()) {
				oldSavingThread.join();
			}

			idbCopy.save("idb.bin");
		}, std::move(savingThread)};
	}

	FaceRecSystem()
	{
		faceClassifier = CascadeClassifier{"../data/haarcascade_frontalface_default.xml"};

		if (faceClassifier.empty()) {
			std::cout << "ERROR: Couldn't load classifier" << std::endl;
			std::getchar();
			std::exit(-1);
		}

#ifdef USE_DLIB
		detector = dlib::get_frontal_face_detector();
		dlib::deserialize("../data/shape_predictor_68_face_landmarks.dat") >> pose_model;
#endif
		loadImageDB();
	}

	~FaceRecSystem()
	{
		if (savingThread.joinable()) {
			savingThread.join();
		}
	}

	void filterMod2(vector<int> &v, int mod2)
	{
		vector<int> newValues;

		for (size_t i = mod2; i < v.size(); i += 2)
			newValues.push_back(v[i]);

		v = newValues;
	}

	vector<int> loadFaceDataset(const vector<string> &files)
	{
		vector<int> allIds;

		for (int i = 0; i < (int)files.size(); i++) {
			string name = files[i];
			string path = string{name.begin(), std::find(name.begin(), name.end(), '#')};
			string tag = string{std::find(name.begin(), name.end(), '#'), name.end()};

			vector<int> ids = idb.addDataset("../images/" + path);

			if (tag == "#even") {
				filterMod2(ids, 0);
			}
			else if (tag == "#odd") {
				filterMod2(ids, 1);
			}

			allIds.insert(allIds.end(), ids.begin(), ids.end());
		}

		return allIds;
	}

	Mat ids2RowMatrix(vector<int> ids, int type)
	{
		vector<Mat> images;
		images.reserve(ids.size());

		for (int i : ids) {
			images.push_back(idb.items[i].image);
		}

		return asRowMatrix(images, type);
	}

	vector<int> normalizeFaces(const vector<int> ids)
	{
		if (useDlib) {
#ifdef USE_DLIB
			return normalizeFacesDlib(idb, ids, detector, pose_model);
#else
			CV_Assert(0);
			return {};
#endif
		}
		else {
			return normalizeFacesVJ(idb, ids, faceClassifier);
		}
	}

	void train(const vector<string> &files)
	{
		trainIds.resize(0);
		trainMore(files);
	}

	void trainMore(const vector<string> &files)
	{
		vector<int> newIds = loadFaceDataset(files);
		newIds = normalizeFaces(newIds);

		saveImageDB();

		trainIds.insert(trainIds.end(), newIds.begin(), newIds.end());
		trainData = ids2RowMatrix(trainIds, CV_32F);
		testIds.resize(0);

		if (trainIds.size() == 0) {
			std::cout << "No training data - can't test." << std::endl;
		}
		else {
			//
			// We use the ImageDB caching mechanism to avoid redoing PCA.
			// The path of a PCA is made from the combined paths of all
			// train images, plus the dimension of the PCA.
			//

			string pcaPath = "";

			if (trainIds.size() > 0) {
				pcaPath.reserve((idb.items[trainIds[0]].path.size() + 1) * trainIds.size() + 10);
			}

			for (int i : trainIds) {
				pcaPath += idb.items[i].path;
				pcaPath += ";";
			}

			pcaPath += "##PCA#DIMS=" + std::to_string(maxDimensions) + "#EIGENVECTORS";

			int eigenvectorsIndex = idb.getImageId(pcaPath);

			if (eigenvectorsIndex >= 0) {
				Mat mean;
				int reduceToSingleRow = 0;
				cv::reduce(trainData, mean, reduceToSingleRow, cv::REDUCE_AVG);

				Mat eigenvectors = idb.items[eigenvectorsIndex].image;

				pca.mean = mean;
				pca.eigenvectors = eigenvectors;

				std::cout << "PCA loaded" << std::endl;
			}
			else {
				std::cout << "BEGIN PCA" << std::endl;
				pca = cv::PCA{trainData, cv::noArray(), cv::PCA::DATA_AS_ROW, maxDimensions};
				std::cout << "END PCA" << std::endl;

				image_item savedPCA;
				savedPCA.image = pca.eigenvectors;
				savedPCA.path = pcaPath;
				savedPCA.label = -1;

				idb.addImage(savedPCA);
			}

			trainProj = pca.project(trainData);
		}
	}

	void predict()
	{
		// TODO: Remove the list
		vector<Mat> trainProjectionsList;
		vector<int> trainLabels;

		for (int i = 0; i < trainProj.rows; i++) {
			trainProjectionsList.push_back(trainProj.row(i).clone());
			trainLabels.push_back(idb.items[trainIds[i]].label);
		}

		vector<Mat> testProjectionsList;
		vector<int> testLabels;

		for (int i = 0; i < testProj.rows; i++) {
			testProjectionsList.push_back(testProj.row(i).clone());
			testLabels.push_back(idb.items[testIds[i]].label);
		}

		vector<int> chosenLabels;

		switch (distanceType) {
			case DistanceType::KNN:
				chosenLabels = classifyKNN(trainProj, trainLabels, idb.labelNames.size(), testProj, 1);
				break;
			case DistanceType::Mahalanobis:
				chosenLabels = classifyMahalanobis(trainProj, trainLabels, idb.labelNames.size(), testProj);
				break;
		};

		int correct = 0;

		for (int i = 0; i < (int)chosenLabels.size(); i++) {
			int chosen = chosenLabels[i];
			int realLabel = idb.items[testIds[i]].label;
			std::cout << idb.labelNames[realLabel] << " -> " << idb.labelNames[chosen] << ((chosen == realLabel) ? "" : " X") << std::endl;
			if (idb.items[testIds[i]].label == chosenLabels[i]) {
				correct++;
			}
		}

		printf("%2.01f%% (%d/%d) correct\n", correct * 100.0f / chosenLabels.size(), correct, (int)chosenLabels.size());
		fflush(stdout);

		visualizeEigenSpaceLines(trainProj, trainLabels, testProj, chosenLabels, testLabels);
	}

	void test(const vector<string> &files)
	{
		testIds = loadFaceDataset(files);
		testIds = normalizeFaces(testIds);

		saveImageDB();

		testData = ids2RowMatrix(testIds, CV_32F);

		if (trainIds.size() > 0 && testIds.size() > 0) {
			testProj = pca.project(testData);
			predict();
		}
	}

	void showEigenfaces()
	{
		if (trainIds.size() == 0) {
			std::cout << "No train data." << std::endl;
			return;
		}

		int imageRows = idb.items[trainIds[0]].image.rows;

		Mat mean = pca.mean.reshape(1, imageRows).clone();
		mean /= 255;

		Mat basis = pca.eigenvectors.reshape(1, imageRows * pca.eigenvectors.rows).clone();
		cv::normalize(basis, basis, 0, 1, cv::NORM_MINMAX, CV_32F);

		cv::vconcat(mean, basis, basis);

		cv::imshow("w", basis);
		cv::waitKey(0);
	}

	void showProjetions(const Mat &data, int imageRows)
	{
		for (int i = 0; i < data.rows; i++) {
			Mat image = data.row(i).reshape(1, imageRows) / 255;
			Mat projection = pca.project(data.row(i));
			Mat backProjection = pca.backProject(projection).reshape(1, imageRows) / 255;

			cv::hconcat(image, backProjection, image);
			cv::imshow("w", image);

			int pressedKey = cv::waitKey(0);

			if (pressedKey == 'q') {
				break;
			}
		}
	}

	void showTrainProjections()
	{
		if (trainIds.size() == 0) {
			std::cout << "No train data." << std::endl;
		}
		else {
			showProjetions(trainData, idb.items[trainIds[0]].image.rows);
		}
	}

	void showTestProjections()
	{
		if (trainIds.size() == 0 || testIds.size() == 0) {
			std::cout << "Not enough data." << std::endl;
		}
		else {
			showProjetions(testData, idb.items[testIds[0]].image.rows);
		}
	}
};

int main(int argc, char *argv[])
{
	FaceRecSystem facerec;

	bool commandlineMode = (argc == 1);

	std::istringstream argStream;

	if (!commandlineMode) {
		string argCommands{argv[1]};
		std::replace(argCommands.begin(), argCommands.end(), ';', '\n');
		argStream = std::istringstream{argCommands};
	}

	std::istream &inStream = commandlineMode ? std::cin : argStream;
	string line;

	while (true) {
		if (commandlineMode) {
			std::cout << "?> ";
		}

		if (!std::getline(inStream, line)) {
			break;
		}

		vector<string> words;

		{
			std::istringstream lineStream{line};
			string word;

			while (lineStream >> word || !lineStream.eof()) {
				words.push_back(word);
			}
		}

		if (words.size() == 0) {
			continue;
		}
		else if (words[0] == "exit" || words[0] == "x") {
			return 0;
		}
		else if (words[0] == "train" || words[0] == "tr") {
			facerec.train(vector<string>{words.begin() + 1, words.end()});
		}
		else if (words[0] == "train+" || words[0] == "tr+") {
			facerec.trainMore(vector<string>{words.begin() + 1, words.end()});
		}
		else if (words[0] == "test" || words[0] == "tst") {
			facerec.test(vector<string>{words.begin() + 1, words.end()});
		}
		else if (words[0] == "show_eigenfaces" || words[0] == "eig") {
			facerec.showEigenfaces();
		}
		else if (words[0] == "show_train" || words[0] == "shtr") {
			facerec.showTrainProjections();
		}
		else if (words[0] == "show_test" || words[0] == "shtst") {
			facerec.showTestProjections();
		}
		else if (words[0] == "dlib") {
			if (words.size() > 1) {
				if (words[1] == "0") {
					facerec.useDlib = false;
				}
				else if (words[1] == "1") {
#ifdef USE_DLIB
					facerec.useDlib = true;
#else
					std::cout << "No Dlib support. Compile with USE_DLIB to enable." << std::endl;
#endif
				}
				else {
					std::cout << "0 or 1 expected." << std::endl;
				}
			}

			std::cout << (facerec.useDlib ? "1" : "0") << std::endl;
		}
		else if (words[0] == "dist_type") {
			if (words.size() > 1) {
				if (words[1] == "knn" || words[1] == "KNN" || words[1] == "k") {
					facerec.distanceType = FaceRecSystem::DistanceType::KNN;
				}
				else if (words[1] == "mahalanobis" || words[1] == "Mahalanobis" || words[1] == "m") {
					facerec.distanceType = FaceRecSystem::DistanceType::Mahalanobis;
				}
				else {
					std::cout << "Unkown distance " << words[1] << std::endl;
				}
			}

			if (facerec.distanceType == FaceRecSystem::DistanceType::KNN) {
				std::cout << "knn " << std::endl;
			}
			else if (facerec.distanceType == FaceRecSystem::DistanceType::Mahalanobis) {
				std::cout << "mahalanobis" << std::endl;
			}
		}
		else if (words[0] == "pca_n") {
			if (words.size() > 1) {
				try {
					int n = std::stoi(words[1]);

					if (n >= 1) {
						facerec.maxDimensions = n;
					}
					else {
						std::cout << "The number of dimensions can't be less than 1." <<std::endl;
					}
				}
				catch(...) {
					std::cout << "Integer expected" << std::endl;
				}
			}

			std::cout << facerec.maxDimensions << std::endl;
		}
		else {
			std::cout << "Unkown command '" << words[0] << "'. Skipping to next line." << std::endl;
		}

		cv::destroyAllWindows();
	}
}
