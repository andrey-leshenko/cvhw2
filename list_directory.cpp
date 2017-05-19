#include "list_directory.hpp"

#ifdef __linux__

#include <dirent.h>
#include <errno.h>

#elif _WIN32

#include <Windows.h>

#endif

int listDirectory(const char* path, std::vector<std::string> &files)
{
#ifdef __linux__
	DIR *dp;
	struct dirent *dirp;

	files.resize(0);

	dp = opendir(path);

	if (!dp)
	{
		return errno;
	}

	while ((dirp = readdir(dp)))
	{
		files.push_back(dirp->d_name);
	}

	return 0;
#elif _WIN32
	// NOTE: http://stackoverflow.com/questions/2314542/listing-directory-contents-using-c-and-windows

	WIN32_FIND_DATA fdFile;
	HANDLE hFind = NULL;

	char sPath[2048];

	//Specify a file mask. *.* = We want everything!
	sprintf_s(sPath, "%s\\*.*", path);

	if ((hFind = FindFirstFile(sPath, &fdFile)) == INVALID_HANDLE_VALUE)
	{
		printf("Path not found: [%s]\n", path);
		return -1;
	}

	do
	{
		files.push_back(fdFile.cFileName);
	} while (FindNextFile(hFind, &fdFile));

	FindClose(hFind);

	return 0;
#else
#error Not Implemented
#endif
}
