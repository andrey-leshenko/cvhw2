# pragma once

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <string>
#include <fstream>
#include <utility>

#include <opencv2/opencv.hpp>

using std::vector;
using std::string;
using std::map;
using std::pair;

using cv::Mat;

struct image_item
{
	Mat image;
	int label;
	string path;
};

struct ImageDB
{
	vector<image_item> items;
	vector<string> labelNames;
	map<string, int> path2item;

	void load(const string &path);
	void save(const string &path) const;
	void clear();

	vector<int> addDataset(const string &path);
	int addImage(const image_item &m);

	int getLabelId(const string &label);
	int getImageId(const string &path);
	int getOrLoadImage(const string &path, int label = -1);
};
