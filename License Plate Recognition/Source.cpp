#include <opencv2\opencv.hpp>
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <math.h>
#include <vector>
#include <string>
#include <windows.h>
#include <filesystem>

const int    MIN_CHAR_NUM = 6;
const double CHAR_SIZE_ERROR_RATIO = 0.1;
const double CHAR_X_ERROR_RATIO = 1.5;
const double CHAR_Y_ERROR_RATIO = 0.2;
const double CHAR_ASPECT_RATIO = 0.6;
const int    MIN_PIXEL_AREA = 80;
const double MAX_ASPECT_RATIO = 1.0;
const double MIN_ASPECT_RATIO = 0.12;

using namespace cv;
using namespace std;
namespace fs = std::experimental::filesystem;

const char* supportExt[] = { ".jpg", ".jpeg" , ".bmp", ".png", ".dib", ".jpe", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff" };

bool LicensePlateRecognition(string srcPath, string dstPath, vector<vector<Vec4i>> &charContours);

int main(int argc, char** argv) {
	string in_dir, out_dir;

	cout << "License Plate Recognition v1.0 by KKK" << endl;
	cout << "Input Folder Path : ";
	cin >> in_dir;
	cout << "Output Folder Path : ";
	cin >> out_dir;
	cout << "\n";

	CreateDirectory(out_dir.c_str(), NULL);
	vector<vector<Vec4i>> charContours;
	ofstream ofile(out_dir + "\\output.txt");
	if (!ofile) return -1;

	for (auto &p : fs::directory_iterator(in_dir)) {
		int cnt = 0;
		string fName = fs::path(p).filename().string();
		string ext = fs::path(p).extension().string();

		transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
		if (find(supportExt, supportExt + sizeof(supportExt) / sizeof(*supportExt), ext) == supportExt + sizeof(supportExt) / sizeof(*supportExt))
			continue;

		if (!LicensePlateRecognition(in_dir + "\\" + fName, out_dir + "\\" + fName, charContours)) {
			cout << "Can't find " << fs::absolute(p) << endl;
			continue;
		}
		cout << "Recognizing " << fName << endl;
		ofile << fName << endl;
		for (auto &group : charContours) {
			for (auto &r : group) {
				ofile << r[0] << " " << r[1] << " " << r[2] << " " << r[3] << endl;
				cnt++;
			}
		}
		cout << "Find " << cnt << " characters." << endl;
	}

	ofile.close();

	cout << "Done.\n" << endl;
	system("pause");
	return 0;
}

bool LicensePlateRecognition(string srcPath, string dstPath, vector<vector<Vec4i>> &charContours) {
	Mat src, src_fix, dst, edge;

	if (!(src = imread(srcPath)).data)
		return false;
	charContours.clear();

	// 高斯低通
	GaussianBlur(src, src_fix, Size(5, 5), 0, 0);

	// 高通濾波
	Mat kernel = (Mat_<int>(3, 3) << 0, -1, 0, -1, 6, -1, 0, -1, 0);
	filter2D(src_fix, src_fix, src.depth(), kernel);

	// Canny邊界
	Canny(src_fix, edge, 200, 255, 3, true);
	//cvtColor(edge.clone(), dst, COLOR_GRAY2BGR);

	// Find Contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(edge, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	// Sort Contours
	vector<int> contours_x;
	for (int i = 0; i < contours.size(); i++)
		contours_x.push_back(boundingRect(contours[i]).x);
	vector<int> sort_contours;
	for (int i = contours.size() - 1, j; i >= 0; i--) {
		for (j = 0; j < sort_contours.size(); j++) {
			if (j < sort_contours.size() && contours_x[sort_contours[j]] > contours_x[i]) {
				break;
			}
		}
		sort_contours.insert(sort_contours.begin() + j, i);
	}
	contours_x.clear();

	// Group Contours
	vector<vector<int>> group;
	vector<Rect> groupRect;
	for (auto id : sort_contours) {
		Rect r1 = boundingRect(contours[id]);
		double aspect = (float)r1.width / r1.height;;
		if (r1.area() < MIN_PIXEL_AREA || aspect < MIN_ASPECT_RATIO || aspect > MAX_ASPECT_RATIO)
			continue;
		bool isClassify = false;
		for (int j = 0; j < group.size(); j++) {
			auto& g = group[j];
			Rect r2 = groupRect[j];
			float size_error_ratio = abs((double)r2.height / r1.height - 1);
			double r1_x = (double)r1.x + (double)r1.width / 2 - (double)r1.height * CHAR_ASPECT_RATIO / 2;
			double r2_x = (double)r2.x + (double)r2.width / 2 - (double)r2.height * CHAR_ASPECT_RATIO / 2;
			float x_error_ratio = (double)abs(r1_x - r2_x) / (r1.height * CHAR_ASPECT_RATIO) - 1;
			float y_error_ratio = (double)abs(r1.y - r2.y) / r1.height;

			if (size_error_ratio < CHAR_SIZE_ERROR_RATIO &&
				x_error_ratio < CHAR_X_ERROR_RATIO &&
				y_error_ratio < CHAR_Y_ERROR_RATIO) {
				if (r1.x - r2.x + r1.y - r2.y != 0) {
					// 刪除自己的父親
					auto dup = find(g.begin(), g.end(), hierarchy[id][3]);
					if (dup != g.end())
						g.erase(dup);

					// 自己不能是別人的父親
					//if (find(g.begin(), g.end(), [&hierarchy, id](const int& a) mutable {return id == hierarchy[a][3]; }) == g.end()) g.push_back(id);
					bool isParent = false;
					for (auto c : g) {
						if (id == hierarchy[c][3]) {
							isParent = true;
							break;
						}
					}
					if (!isParent) {
						g.push_back(id);
						groupRect[j] = r1;
					}
				}
				isClassify = true;
			}
		}
		if (!isClassify) {
			group.emplace_back();
			group.back().push_back(id);
			groupRect.push_back(r1);
		}
		//}
	}

	// Output Contours
	RNG rnd(1234);
	for (auto& g : group) {
		if (g.size() < MIN_CHAR_NUM)
			continue;
		charContours.emplace_back();
		Scalar c = Scalar(rnd.next() % 255, 255, rnd.next() % 255);
		for (int i : g) {
			Rect r = boundingRect(contours[i]);
			charContours.back().emplace_back(Vec4i(r.x, r.y, r.x + r.width, r.y + r.height));
			r.x = (double)r.x + (double)r.width / 2 - ((double)r.height * CHAR_ASPECT_RATIO) / 2;
			r.width = r.height * CHAR_ASPECT_RATIO;
			rectangle(src, r, c, 1);
		}
	}

	imwrite(dstPath, src);
	/*namedWindow("Source", 1);
	imshow("Source", dst);
	waitKey(0);*/
}