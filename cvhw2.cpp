#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <string>
#include <fstream>
#include <utility>

#include <stdio.h>
#include <stdint.h>

#include <opencv2/opencv.hpp>

/*
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
*/

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
using std::map;
using std::pair;

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
using cv::FileStorage;

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

struct Dataset
{
	vector<Mat> images;
	vector<string> fileNames;

	//int labelCount;
	vector<int> labels;
	vector<string> labelNames;
};

bool readDataset(
		const string &datasetPath,
		vector<Mat> &images,
		vector<string> &fileNames,
		vector<int> &labels,
		vector<string> &labelNames)
{
	vector<string> directories;

	if (listDirectory(datasetPath.c_str(), directories) != 0) {
		return false;
	}

	std::sort(directories.begin(), directories.end());

	int currentLabel = 0;
	images.resize(0);
	fileNames.resize(0);
	labels.resize(0);
	labelNames.resize(0);

	for (const string &dir : directories) {
		if (dir == "." || dir == "..")
			continue;

		string directoryPath = datasetPath + "/" + dir;

		vector<string> imageNames;
		if (listDirectory(directoryPath.c_str(), imageNames) != 0)
			continue;

		std::sort(imageNames.begin(), imageNames.end());

		for (const string &imageFile : imageNames) {
			if (imageFile.substr(0, 1) == ".")
				continue;

			string filePath = directoryPath + "/" + imageFile;
			Mat image = cv::imread(filePath);

			if (!image.empty()) {
				images.push_back(image);
				labels.push_back(currentLabel);
				fileNames.push_back(filePath);

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

bool readDataset(const string &path, Dataset &dataset)
{
	return readDataset(path, dataset.images, dataset.fileNames, dataset.labels, dataset.labelNames);
}

void addToDataset(Dataset &a, const Dataset &other)
{
	for (Mat m : other.images) {
		a.images.push_back(m.clone());
	}
	a.labels.insert(a.labels.begin(), other.labels.begin(), other.labels.end());
	a.fileNames.insert(a.fileNames.begin(), other.fileNames.begin(), other.fileNames.end());
	a.labelNames.insert(a.labelNames.begin(), other.labelNames.begin(), other.labelNames.end());
}

void clearDataset(Dataset &d)
{
	d.images.resize(0);
	d.fileNames.resize(0);
	d.labels.resize(0);
	d.labelNames.resize(0);
}

void keepImages(Dataset &d, int modulu2)
{
	vector<Mat> newImages;
	vector<string> newFileNames;
	vector<int> newLabels;

	for (int i = modulu2; i < (int)d.images.size(); i += 2) {
		newImages.push_back(d.images[i]);
		newFileNames.push_back(d.fileNames[i]);
		newLabels.push_back(d.labels[i]);
	}

	d.images = newImages;
	d.fileNames = newFileNames;
	d.labels = newLabels;
}

void keepEvenImages(Dataset &d)
{
	keepImages(d, 0);
}

void keepOddImages(Dataset &d)
{
	keepImages(d, 1);
}

void loadFaceCache(const string &path, map<string, Mat> &faceCache)
{
	faceCache.clear();

	Mat dbMat = cv::imread(path + "_images.tiff", cv::IMREAD_UNCHANGED);

	if (dbMat.empty())
		return;

	FileStorage fs{path + "_paths.yaml", FileStorage::READ};

	if (!fs.isOpened())
		return;

	Size imageSize;
	cv::read(fs["image_size"], imageSize, Size{-1, -1});

	vector<cv::String> paths;
	cv::read(fs["paths"], paths);

	if ((int)paths.size() != dbMat.rows ||
			imageSize.width * imageSize.height != dbMat.cols)
		return;

	for (int i = 0; i < dbMat.rows; i++) {
		faceCache[paths[i]] = dbMat.row(i).reshape(0, imageSize.height);
	}
}

void saveFaceCache(const string &path, const map<string, Mat> &faceCache)
{
	int n = faceCache.size();

	if (n == 0)
		return;

	Size size = faceCache.begin()->second.size();
	int type = faceCache.begin()->second.type();

	vector<string> paths;
	paths.reserve(n);
	vector<Mat> images;
	images.reserve(n);

	for (pair<string, Mat> p : faceCache) {
		paths.push_back(p.first);
		images.push_back(p.second);
	}

	Mat dbMat = asRowMatrix(images, type);

	cv::imwrite(path + "_images.tiff", dbMat);

	FileStorage fs{path + "_paths.yaml", FileStorage::WRITE};

	if (!fs.isOpened())
		return;

	fs << "image_size" << size;
	fs << "paths" << paths;
}

struct image_item
{
	Mat image;
	int label;
	string path;
};

struct image_db_header
{
	size_t itemCount;
	size_t labelCount;
};

struct image_item_header
{
	int label;

	int imageType;
	int imageRows;
	int imageCols;
	size_t imageSize;

	size_t pathSize;
};

struct string_header
{
	size_t size;
};

struct ImageDB
{
	vector<image_item> items;
	vector<string> labelNames;
	map<string, int> path2item;

	void clear()
	{
		items.resize(0);
		labelNames.resize(0);
		path2item.clear();
	}

	void load(const string &path)
	{
		clear();

		FILE *f = fopen(path.c_str(), "rb");

		if (!f) {
			std::cout << "No existing ImageDB file found. New one will be created." << std::endl;
			return;
		}

		vector<char> textBuffer(128);

		image_db_header dbHeader;

		if (fread(&dbHeader, sizeof(dbHeader), 1, f) != 1)
			goto error;

		for (size_t i = 0; i < dbHeader.itemCount; i++) {
			image_item_header itemHeader;

			if (fread(&itemHeader, sizeof(itemHeader), 1, f) != 1)
				goto error;

			items.push_back(image_item{});
			items.back().label = itemHeader.label;
			items.back().image = Mat{itemHeader.imageRows, itemHeader.imageCols, itemHeader.imageType};

			if (fread(items.back().image.data, itemHeader.imageSize, 1, f) != 1) {
				goto error;
			}

			textBuffer.resize(itemHeader.pathSize);

			if (fread(&textBuffer[0], itemHeader.pathSize, 1, f) != 1)
				goto error;

			items.back().path = string{&textBuffer[0]};
			path2item[items.back().path] = items.size() - 1;
		}

		for (size_t i = 0; i < dbHeader.labelCount; i++) {
			string_header header;

			if (fread(&header, sizeof(header), 1, f) != 1)
				goto error;

			textBuffer.resize(header.size);

			if (fread(&textBuffer[0], header.size, 1, f) != 1)
				goto error;

			labelNames.push_back(string{&textBuffer[0]});
		}

		fclose(f);
		return;

error:
		std::cout << "Reading error" << std::endl;
		fclose(f);
	}

	void save(const string &path)
	{
		FILE *f = fopen(path.c_str(), "wb");

		if (!f) {
			std::cout << "Unable to open file for saving image db." << std::endl;
			return;
		}

		image_db_header dbHeader = {
			.itemCount = items.size(),
			.labelCount = labelNames.size(),
		};

		if (fwrite(&dbHeader, sizeof(dbHeader), 1, f) != 1)
			goto error;

		for (image_item &m : items) {
			const char *c_path = m.path.c_str();

			image_item_header itemHeader = {
				.label = m.label,
				.imageType = m.image.type(),
				.imageRows = m.image.rows,
				.imageCols = m.image.cols,
				.imageSize = m.image.total() * m.image.elemSize(),
				.pathSize = strlen(c_path) + 1,
			};

			if (fwrite(&itemHeader, sizeof(itemHeader), 1, f) != 1)
				goto error;
			if (fwrite(m.image.data, itemHeader.imageSize, 1, f) != 1)
				goto error;
			if (fwrite(c_path, itemHeader.pathSize, 1, f) != 1)
				goto error;
		}

		for (string &label : labelNames) {
			const char *c_label = label.c_str();

			string_header header = {
				.size = strlen(c_label) + 1,
			};

			if (fwrite(&header, sizeof(header), 1, f) != 1)
				goto error;
			if (fwrite(c_label, header.size, 1, f) != 1)
				goto error;
		}

		fclose(f);
		return;

error:
		std::cout << "Writing error" << std::endl;
		fclose(f);
	}

	int getLabelId(const string &label)
	{
		auto it = std::find(labelNames.begin(), labelNames.end(), label);

		if (it == labelNames.end()) {
			labelNames.push_back(label);
			return labelNames.size() - 1;
		}
		else {
			return it - labelNames.begin();
		}
	}

	int getImageId(const string &path, int label = -1)
	{
		auto it = path2item.find(path);

		if (it == path2item.end()) {
			Mat image = cv::imread(path);

			if (image.empty()) {
				std::cout << "ERROR: Couldn't load image '" << path << "'" << std::endl;
				return -1;
			}
			else {
				std::cout << "loaded '" << path << "'" << std::endl;

				items.push_back(image_item{});
				items.back().image = image;
				items.back().path = path;
				items.back().label = label;

				path2item[path] = items.size() - 1;
				return items.size() - 1;

			}
		}
		else {
			return it->second;
		}
	}

	vector<int> addDataset(const string &path)
	{
		vector<int> addedIds;

		vector<string> directories;

		if (listDirectory(path.c_str(), directories) != 0) {
			std::cout << "ERROR: Couldn't load dataset '" << path << "'" << std::endl;
			return {};
		}

		std::sort(directories.begin(), directories.end());

		for (const string &dir : directories) {
			if (dir == "." || dir == "..")
				continue;

			string directoryPath = path + "/" + dir;
			int directoryLabel = getLabelId(dir);

			vector<string> imageNames;
			if (listDirectory(directoryPath.c_str(), imageNames) != 0)
				continue;

			std::sort(imageNames.begin(), imageNames.end());

			for (const string &imageFile : imageNames) {
				if (imageFile.substr(0, 1) == ".")
					continue;

				string filePath = directoryPath + "/" + imageFile;
				int imageId = getImageId(filePath, directoryLabel);

				if (imageId >= 0)
					addedIds.push_back(imageId);
			}
		}

		return addedIds;
	}
};

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
		map<string, Mat> &faceCache,
		int outputSize = 128)
{
	vector<Mat> newImages;
	vector<int> newLabels;

	for (int i = 0; i < (int)dataset.images.size(); i++) {
		Mat normalized;

		auto it = faceCache.find(dataset.fileNames[i]);

		if (it != faceCache.end() && it->second.size() == Size{outputSize, outputSize}) {
			normalized = it->second.clone();
		}
		else {
			normalizeFace(
					dataset.images[i],
					normalized,
					faceClassifier,
					eyeClassifier,
					outputSize);

			if (!normalized.empty()) {
				faceCache[dataset.fileNames[i]] = normalized;
			}
		}

		if (!normalized.empty()) {
			newImages.push_back(normalized);
			newLabels.push_back(dataset.labels[i]);
		}
	}

	dataset.images = newImages;
	dataset.labels = newLabels;
}

/*
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

void normalizeFaceDatasetDlib(
		Dataset &dataset,
		dlib::frontal_face_detector detector,
		dlib::shape_predictor pose_model)
{
	vector<Mat> newImages;
	vector<int> newLabels;

	for (int i = 0; i < (int)dataset.images.size(); i++) {
		vector<Mat> faces = alignImageFaces(dataset.images[i], detector, pose_model);

		if (faces.size() == 1) {
			Mat dst = faces[0];
			cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
			cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			cv::equalizeHist(dst, dst);
			dst.colRange(0, dst.cols * 2 / 10).setTo(0);
			dst.colRange(dst.cols - dst.cols * 2 / 10, dst.cols).setTo(0);
			cv::imshow("w", dst);

			cv::waitKey(1);
			newImages.push_back(dst);
			newLabels.push_back(dataset.labels[i]);
		}
	}

	dataset.images = newImages;
	dataset.labels = newLabels;
}
*/

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

	if (dimensions <= 0 || dimensions > n) {
		dimensions = n;
	}

	Mat data = asRowMatrix(dataset.images, CV_32F);

	std::cout << "BEGIN PCA" << std::endl;

	model.pca = cv::PCA{data, cv::noArray(), cv::PCA::DATA_AS_ROW, dimensions};

	std::cout << "END PCA" << std::endl;

	if (false) {
		Mat mean = model.pca.mean.reshape(1, dataset.images[0].rows);
		mean /= 255;

		Mat basis = model.pca.eigenvectors.reshape(1, dataset.images[0].rows * dimensions).clone();
		cv::normalize(basis, basis, 0, 1, cv::NORM_MINMAX, CV_32F);

		cv::vconcat(mean, basis, basis);

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

void visualizePrediction(
		EigenFacesModel model,
		Dataset &testDataset,
		Mat testProjections,
		vector<int> testPredictions)
{
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

void predict(EigenFacesModel &model, Dataset &testDataset)
{
	vector<Mat> testProjectionsList;

	Mat trainProjections = asRowMatrix(model.projections, CV_32F);
	Mat testProjections = model.pca.project(asRowMatrix(testDataset.images, CV_32F));

	for (int i = 0; i < testProjections.rows; i++) {
		testProjectionsList.push_back(testProjections.row(i).clone());
	}

	vector<int> chosenLabels = classifyKNN(trainProjections, model.labels, model.labelNames.size(), testProjections, 5);

	int correct = 0;

	for (int i = 0; i < (int)chosenLabels.size(); i++) {
		std::cout << testDataset.labels[i] << " -> " << chosenLabels[i] << std::endl;
		if (model.labelNames[chosenLabels[i]] == testDataset.labelNames[testDataset.labels[i]]) {
			correct++;
		}
	}

	printf("%2.01f%% (%d/%d) correct\n", correct * 100.0f / chosenLabels.size(), correct, (int)chosenLabels.size());
	fflush(stdout);

	visualizeEigenSpaceLines(model.projections, model.labels, testProjectionsList, testDataset.labels);
	visualizeEigenSpaceDots(model.projections, model.labels, testProjectionsList, testDataset.labels);
}

struct FaceRecSystem
{
	CascadeClassifier faceClassifier;
	CascadeClassifier eyeClassifier;

	//dlib::frontal_face_detector detector;
	//dlib::shape_predictor pose_model;

	map<string, Mat> faceCache;

	ImageDB idb;

	Dataset trainDS;
	Dataset testDS;

	EigenFacesModel model;

	bool useDlib = false;

	FaceRecSystem()
	{
		faceClassifier = CascadeClassifier{"../images/haarcascade_frontalface_default.xml"};
		eyeClassifier = CascadeClassifier{"../images/haarcascade_eye.xml"};

		if (faceClassifier.empty() || eyeClassifier.empty()) {
			std::cout << "ERROR: Couldn't load classifier" << std::endl;
			std::exit(-1);
		}

		//detector = dlib::get_frontal_face_detector();
		//dlib::deserialize("../shape_predictor_68_face_landmarks.dat") >> pose_model;
		loadFaceCache("face_cache", faceCache);

		idb.load("idb.bin");
	}

	void train(const vector<string> &files)
	{
		Dataset newDataset;
		Dataset tmpDataset;

		for (int i = 0; i < (int)files.size(); i++) {
			string name = files[i];
			string path = string{name.begin(), std::find(name.begin(), name.end(), '#')};
			string tag = string{std::find(name.begin(), name.end(), '#'), name.end()};

			if (!readDataset("../images/" + path, tmpDataset)) {
				std::cout << "ERROR: Clouldn't load dataset" << std::endl;
			}

			idb.addDataset("../images/" + path);
			idb.save("idb.bin");

			if (tag == "#even") {
				keepEvenImages(tmpDataset);
			}
			else if (tag == "#odd") {
				keepOddImages(tmpDataset);
			}

			addToDataset(newDataset, tmpDataset);
		}

		if (!useDlib)
			normalizeFaceDataset(newDataset, faceClassifier, eyeClassifier, faceCache);
		else
			/*normalizeFaceDatasetDlib(newDataset, detector, pose_model);*/;
		saveFaceCache("face_cache", faceCache);

		clearDataset(trainDS);
		addToDataset(trainDS, newDataset);

		if (trainDS.images.size() > 0) {
			createEigenFacesModel(model, trainDS, 16);
		}
	}

	void trainMore(const vector<string> &files)
	{
		std::cout << "Coming soon :)" << std::endl;
	}

	void visualize(const vector<string> &files)
	{
		std::cout << "Coming soon :)" << std::endl;
	}

	void test(const vector<string> &files)
	{
		Dataset newDataset;
		Dataset tmpDataset;

		for (int i = 0; i < (int)files.size(); i++) {
			string name = files[i];
			string path = string{name.begin(), std::find(name.begin(), name.end(), '#')};
			string tag = string{std::find(name.begin(), name.end(), '#'), name.end()};

			if (!readDataset("../images/" + path, tmpDataset)) {
				std::cout << "ERROR: Clouldn't load dataset" << std::endl;
			}

			idb.addDataset("../images/" + path);
			idb.save("idb.bin");

			if (tag == "#even") {
				keepEvenImages(tmpDataset);
			}
			else if (tag == "#odd") {
				keepOddImages(tmpDataset);
			}

			addToDataset(newDataset, tmpDataset);
		}

		if (!useDlib)
			normalizeFaceDataset(newDataset, faceClassifier, eyeClassifier, faceCache);
		else
			/*normalizeFaceDatasetDlib(newDataset, detector, pose_model);*/;
		saveFaceCache("face_cache", faceCache);

		clearDataset(testDS);
		addToDataset(testDS, newDataset);

		if (trainDS.images.size() > 0 && testDS.images.size() > 0) {
			predict(model, testDS);
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
		else if (words[0] == "visualize" || words[0] == "v") {
			facerec.visualize(vector<string>{words.begin() + 1, words.end()});
		}
		else if (words[0] == "test" || words[0] == "tst") {
			facerec.test(vector<string>{words.begin() + 1, words.end()});
		}
		else {
			std::cout << "Unkown command '" << words[0] << "'. Skipping to next line." << std::endl;
		}
	}
}
