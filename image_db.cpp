#include "image_db.hpp"
#include "list_directory.hpp"

#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <stdint.h>

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

void ImageDB::clear()
{
	items.resize(0);
	labelNames.resize(0);
	path2item.clear();
}

void ImageDB::load(const string &path)
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

		if (fread(items.back().image.data, itemHeader.imageSize, 1, f) != 1 * !!itemHeader.imageSize)
			goto error;

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
	std::cout << "ImageDB loaded." << std::endl;

	return;

error:
	std::cout << "Reading error" << std::endl;
	fclose(f);
	clear();
}

void ImageDB::save(const string &path) const
{
	FILE *f = fopen(path.c_str(), "wb");

	if (!f) {
		std::cout << "Unable to open file for saving image db." << std::endl;
		return;
	}

	image_db_header dbHeader;
	dbHeader.itemCount = items.size();
	dbHeader.labelCount = labelNames.size();

	if (fwrite(&dbHeader, sizeof(dbHeader), 1, f) != 1)
		goto error;

	for (const image_item &m : items) {
		const char *c_path = m.path.c_str();

		image_item_header itemHeader;
		itemHeader.label = m.label;
		itemHeader.imageType = m.image.type();
		itemHeader.imageRows = m.image.rows;
		itemHeader.imageCols = m.image.cols;
		itemHeader.imageSize = m.image.total() * m.image.elemSize();
		itemHeader.pathSize = strlen(c_path) + 1;

		if (fwrite(&itemHeader, sizeof(itemHeader), 1, f) != 1)
			goto error;
		if (fwrite(m.image.data, itemHeader.imageSize, 1, f) != 1 * !!itemHeader.imageSize)
			goto error;
		if (fwrite(c_path, itemHeader.pathSize, 1, f) != 1)
			goto error;
	}

	for (const string &label : labelNames) {
		const char *c_label = label.c_str();

		string_header header;
		header.size = strlen(c_label) + 1;

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

int ImageDB::getLabelId(const string &label)
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

int ImageDB::getImageId(const string &path)
{
	auto it = path2item.find(path);

	if (it == path2item.end()) {
		return -1;
	}
	else {
		return it->second;
	}
}

int ImageDB::getOrLoadImage(const string &path, int label)
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

int ImageDB::addImage(const image_item &m)
{
	auto it = path2item.find(m.path);

	if (it == path2item.end()) {
		items.push_back(m);
		path2item[m.path] = items.size() - 1;
		return items.size() - 1;
	}
	else {
		int id = it->second;
		items[id] = m;
		return id;
	}
}

vector<int> ImageDB::addDataset(const string &path)
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
			int imageId = getOrLoadImage(filePath, directoryLabel);

			if (imageId >= 0)
				addedIds.push_back(imageId);
		}
	}

	return addedIds;
}
