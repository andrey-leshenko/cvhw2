#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <numeric> // std::iota

using std::vector;
using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::to_string;

using cv::Mat;
using cv::Point2f;
using cv::imshow;
using cv::waitKey;
using cv::Scalar;
using cv::Rect;
using cv::Size;
using cv::imread;
using cv::waitKey;

#ifdef __linux__

#include <dirent.h>
#include <errno.h>

#elif _WIN32
#endif

#define PrintTime if (false)

typedef int64_t s64;

struct Dataset
{
	vector<Mat> images;
	vector<int> labels;
	vector<string> labelNames;
};

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

enum faceRegionEnum {
    leftEye,
    rightEye,
    mouth
};

vector<Point2f> pushActualPointsFromRegion(vector<Point2f> points, vector<int> toChoose)
{
    vector<Point2f> chosenPoints;

    for(int num : toChoose)
    {
        chosenPoints.push_back(points[num]);
    }

    return chosenPoints;
}

Point2f centroidOfRegion(vector<Point2f> facialLandmark, faceRegionEnum faceRegion )
{
    int regionSize;
    int regionStartPoint;

    switch(faceRegion) {
        case faceRegionEnum::leftEye:   regionSize = 6;
                                        regionStartPoint = 36;
                                        break;
        case faceRegionEnum::rightEye:  regionSize = 6;
                                        regionStartPoint = 42;
                                        break;
        case faceRegionEnum::mouth:      regionSize = 19;
                                        regionStartPoint = 48;
                                        break;
        default: // The chin area
                                        regionSize = 17;
                                        regionStartPoint = 0;
    }

    vector<int> regionPoints(regionSize);
    std::iota(regionPoints.begin(), regionPoints.end(), regionStartPoint);

    vector<Point2f> actualPoints = pushActualPointsFromRegion(facialLandmark, regionPoints);

    Point2f sum  = std::accumulate(actualPoints.begin(), actualPoints.end(), Point2f(0.0f, 0.0f) );

    return Point2f(sum.x / regionSize, sum.y / regionSize);
}

vector<Point2f> convertDlibShapeToOpenCV(dlib::full_object_detection objectDet, Rect& outputRect)
{
    vector<Point2f> cvParts;
    dlib::rectangle dlibRect = objectDet.get_rect();

    for(int i = 0; i < 68; i++)
    {
        dlib::point p = objectDet.part(i);
        Point2f cvPoint{ (float)p.x(), (float)p.y() };
        cvParts.push_back(cvPoint);
    }
    
    outputRect = Rect(dlibRect.left(), dlibRect.top(), dlibRect.width(), dlibRect.height());

    return cvParts;
}

vector<Mat> alignImageFaces(Mat image, dlib::frontal_face_detector detector, dlib::shape_predictor pose_model)
{
    try
    {
        PrintTime
            printTimeSinceLastCall("Start Align function");

        cv::Mat temp = image.clone();
        // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
        // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
        // long as temp is valid.  Also don't do anything to temp that would cause it
        // to reallocate the memory which stores the image as that will make cimg
        // contain dangling pointers.  This basically means you shouldn't modify temp
        // while using cimg.

        PrintTime
            printTimeSinceLastCall("Create temp image");

        dlib::cv_image<dlib::bgr_pixel> cimg(temp);

        PrintTime
            printTimeSinceLastCall("Convert to dlib image");

        // Detect faces 
        vector<dlib::rectangle> faces = detector(cimg);

        PrintTime
            printTimeSinceLastCall("Detect Face");

        // Find the pose of each face.
        vector<dlib::full_object_detection> shapes;
        for (unsigned long i = 0; i < faces.size(); ++i)
            shapes.push_back(pose_model(cimg, faces[i]));

        PrintTime
            printTimeSinceLastCall("Find face Poses");

        // Convert the faces detected by dlib to something OpenCV can deal with.
        vector<vector<Point2f>> facialLandmarks(shapes.size());

        for(int i = 0; i < shapes.size(); i++)
        {
            Rect dummyRect;
            facialLandmarks[i] = convertDlibShapeToOpenCV(shapes[i], dummyRect);
        }

        PrintTime
            printTimeSinceLastCall("Convert to Open CV points");

        // The locations of the facial landmarks visually presented:
        // https://github.com/cmusatyalab/openface/blob/master/images/dlib-landmark-mean.png

        vector<Mat> alignedFaces;

        if(facialLandmarks.size() > 0)
        {
            // for(int i = 0; i < facialLandmarks[0].size(); i++)
            // {
            //     circle(myImage, facialLandmarks[0][i], 3, Scalar(0, 0, 255));
            //     string objectTitle = std::to_string(i);
            //     cv::putText(myImage, objectTitle, facialLandmarks[0][i], cv::FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 255, 0), 0.5);
            // }

            for(vector<Point2f> face : facialLandmarks)
            {
                // circle(myImage, centroidOfRegion(face, faceRegionEnum::leftEye), 3, Scalar(0, 255, 0));
                // circle(myImage, centroidOfRegion(face, faceRegionEnum::rightEye), 3, Scalar(0, 255, 0));
                // circle(myImage, centroidOfRegion(face, faceRegionEnum::mouth), 3, Scalar(0, 255, 0));

                vector<Point2f> dstPoints = {Point2f(50, 60), Point2f(75, 120), Point2f(100, 60)};
                vector<Point2f> srcPoints = {centroidOfRegion(face, faceRegionEnum::leftEye)
                                                , centroidOfRegion(face, faceRegionEnum::mouth)
                                                , centroidOfRegion(face, faceRegionEnum::rightEye)};

                Mat affineTransformation = getAffineTransform(srcPoints, dstPoints);
                
                // cout << affineTrans << endl;
                Mat transformedFace;
                warpAffine(image, transformedFace, affineTransformation, Size(150, 175));

                PrintTime
                    printTimeSinceLastCall("Perform Transform");

                alignedFaces.push_back(transformedFace);

                PrintTime
                    printTimeSinceLastCall("Align PushBack");
            }

            return alignedFaces;
        }
        else
        {
            cerr << "No faces Detected! returning empty array" << endl;
            return vector<Mat>();
        }
    }
    catch(dlib::serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(std::exception& e)
    {
        cout << e.what() << endl;
    }

    return vector<Mat>();
}

int main()
{
    Dataset AllFaces;

    int i = 0;
    int notDetected = 0;

    if (!readDataset("../test_images_hw2", AllFaces)) {
		std::cout << "ERROR: Clouldn't load dataset" << std::endl;
	}

    int total = AllFaces.images.size();
    vector<Mat> faces;
    faces.reserve(total);

    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor pose_model;
    dlib::deserialize("../shape_predictor_68_face_landmarks.dat") >> pose_model;

    for (Mat image : AllFaces.images) {
        printTimeSinceLastCall("Start Align call");
        vector<Mat> currentFaces = alignImageFaces(image, detector, pose_model);
        printTimeSinceLastCall("End Align call");
        for(Mat im : currentFaces)
        {
            faces.push_back( im );
            // imshow("w", image);
            // waitKey(0);
        }

        if(currentFaces.size() == 0)
        {
            notDetected++;
            // imshow("ww", image);
            // waitKey(0);
        }
        
        printTimeSinceLastCall("Push Back");
            
        cout << i++ << "/" << total << endl;
	}

    cout << notDetected << " faces were not detected." << endl;
    cout << "Percent of unsuccessful detection: " << ((double)notDetected / i) * 100 << "%" << endl;

    cout << "Started showing faces: " << endl;

    i = 0;

    for(Mat face : faces)
    {
        imshow("w", face);
        // string fileדName = std::to_string(i++) + ".png";
		// cv::imwrite(fileName, face);
        waitKey(0);
    }

    cout << "Finished successfully" << endl;

    return 0;
}