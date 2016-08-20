#include "ReidDescriptor.h"

#include <float.h>


Mat ReID::HSVVector(Mat img) {
    cv::Mat img_hsv, hist, hist_h, hist_s, hist_v;
    cvtColor(img, img_hsv, CV_BGR2HSV);
    
    // Normalisation ?
    vector<cv::Mat> temp;
    split(img_hsv, temp);
    
    temp[0] = temp[0].reshape(0, 1);
    temp[1] = temp[1].reshape(0, 1);
    temp[2] = temp[2].reshape(0, 1);
    
    // Histogram computation
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    float v_ranges[] = { 0, 256 };
    
    int histSize_h[] = { 180 };
    int histSize_s[] = { 256 };
    int histSize_v[] = { 256 };
    
    const float * ranges_h[] = { h_ranges };
    const float * ranges_s[] = { s_ranges };
    const float * ranges_v[] = { v_ranges };
    
    int channels[] = { 0 };
    
    calcHist(&temp[0], 1, channels, Mat(), hist_h, 1, histSize_h, ranges_h);
    normalize(hist_h, hist_h, 0, 1, NORM_MINMAX, -1, Mat());
    calcHist(&temp[1], 1, channels, Mat(), hist_s, 1, histSize_s, ranges_s);
    normalize(hist_s, hist_s, 0, 1, NORM_MINMAX, -1, Mat());
    calcHist(&temp[2], 1, channels, Mat(), hist_v, 1, histSize_v, ranges_v);
    normalize(hist_v, hist_v, 0, 1, NORM_MINMAX, -1, Mat());
    
    vconcat(hist_h, hist_s, hist);
    vconcat(hist, hist_v, hist);
    
    return hist;
    
}

cv::Mat ReID::getWhsvFeature(cv::Mat img)
{
    int offset = img.rows / 5;
    vector<cv::Mat> sub(5);
    
    // Divide the image into 5x1 cells
    for(int i = 0 ; i < 4 ; i++) {
        sub[i] = img.rowRange(offset * i, offset * (i + 1));
    }
    sub[4] = img.rowRange(offset * 4, img.rows);
    // Debug this
    
    cv::Mat conc;
    cv::Mat temp;
    for(int i = 0 ; i < 5 ; i++) {
        cv::Mat HSV = HSVVector(sub[i]);
        if(i == 0) {
            conc = HSV;
        } else {
            vconcat(conc, HSV, conc);
        }
    }
    
    return conc;
}


ReidDescriptor::ReidDescriptor()
{

}

ReidDescriptor::ReidDescriptor(cv::Mat person)
{
	Whsv = this->getWhsvFeature(person);
}

float ReidDescriptor::compare(ReidDescriptor reid)
{
	return bhattacharyya(Whsv, reid.Whsv);
}

// The bhattacharyya distance between vector and vector q;
float ReidDescriptor::bhattacharyya(cv::Mat k, cv::Mat q)
{
	cv::normalize(k, k, 1, 0, cv::NORM_L1);
	cv::normalize(q, q, 1, 0, cv::NORM_L1);

	//show the histograms
	//drawHist("hist1", k);
	//drawHist("hist2", q);

	cv::Mat temp = k.mul(q);
	sqrt(temp, temp);

	return (float)sqrt(1 - cv::sum(temp)[0]); // sqrt(1-sum(sqrt(k.*q)))
}

// Get the minimizer of the function(only consider integer value)
int ReidDescriptor::fminbnd(float(*func)(int, cv::Mat, cv::Mat, int, float), cv::Mat img, cv::Mat MSK, int loc, float alpha, int range1, int range2)
{
	float min_value = func(range1, img, MSK, loc, alpha);
	int min_pos = range1;
	for (int i = range1 + 1; i < range2; i++)
	{
		float temp = func(i, img, MSK, loc, alpha);
		if (temp < min_value){
			min_value = temp;
			min_pos = i;
		}
	}
	return min_pos;
}

// The image is seperated into two parts given the location of seperation axis x, each part's length maximum loc+1;
// Return the dissimilarity score
float ReidDescriptor::dissym_div(int x, cv::Mat img, cv::Mat MSK, int loc, float alpha)
{
	int H = img.rows;
	int W = img.cols;
	int chs = img.channels();

	cv::Mat imgUP = img.rowRange(0, x + 1); // [0,x]
	cv::Mat imgDOWN = img.rowRange(x, img.rows);
	cv::Mat MSK_U = MSK.rowRange(0, x + 1);
	cv::Mat MSK_D = MSK.rowRange(x, MSK.rows);

	int dimLoc = min(min(x + 1, MSK_D.rows), loc + 1);

	if (dimLoc != 0)
	{
		cv::Mat imgUPloc = img.rowRange(x - dimLoc + 1, x + 1); // [x-dimLoc+1,x]
		cv::Mat imgDWloc;
		cv::flip(imgDOWN.rowRange(0, dimLoc), imgDWloc, 0);
		cv::Mat temp;
		cv::pow(imgUPloc - imgDWloc, 2, temp);
		float ans = alpha * (1 - sqrt(sum(temp.reshape(1))[0]) / dimLoc) +
					(1 - alpha) * (abs(sum(MSK_U)[0] - sum(MSK_D)[0])) / max(MSK_U.rows * MSK_U.cols, MSK_D.rows * MSK_D.cols);

		return ans;
	}
	else
	{
		return 1;
	}
}

// The image is seperated into two parts, each part's length maximum loc+1;
// x is the seperation line, both two part have x;
float ReidDescriptor::sym_div(int x, cv::Mat img, cv::Mat MSK, int loc, float alpha)
{
	int H = img.rows;
	int W = img.cols;
	int chs = img.channels();

	cv::Mat imgL = img.colRange(0, x + 1); // [0,x]
	cv::Mat imgR = img.colRange(x, img.cols);
	cv::Mat MSK_L = MSK.colRange(0, x + 1);
	cv::Mat MSK_R = MSK.colRange(x, MSK.cols);

	int dimLoc = min(min(x + 1, MSK_R.cols), loc + 1);

	if (dimLoc != 0)
	{
		cv::Mat imgLloc = img.colRange(x - dimLoc + 1, x + 1);//[x-dimLoc+1,x]
		cv::Mat imgRloc;
		cv::flip(imgR.colRange(0, dimLoc), imgRloc, 1);
		cv::Mat temp;
		cv::pow(imgLloc - imgRloc, 2, temp);
		float ans = alpha * sqrt(cv::sum(temp.reshape(1))[0]) / dimLoc +
					(1 - alpha) * (abs(sum(MSK_R)[0] - sum(MSK_L)[0])) / max(MSK_R.rows * MSK_R.cols, MSK_L.rows * MSK_L.cols);

		return ans;
	}
	else
	{
		return 1;
	}
}

// The image is seperated into two parts, each part's length maximum loc+1;
// x is the seperation line, both two part have x;
float ReidDescriptor::sym_dissimilar_MSKH(int x, cv::Mat img, cv::Mat MSK, int loc, float alpha)
{
	int H = img.rows;
	int W = img.cols;
	int chs = img.channels();

	cv::Mat imgUP = img.rowRange(0, x + 1);//[0,x]
	cv::Mat imgDOWN = img.rowRange(x, img.rows);
	cv::Mat MSK_U = MSK.rowRange(0, x + 1);
	cv::Mat MSK_D = MSK.rowRange(x, MSK.rows);

	int localderU = max(x - loc, 0);
	int localderD = min(loc + 1, MSK_D.rows);

	float ans = -abs(sum(MSK_U.rowRange(localderU, x + 1))[0] - sum(MSK_D.rowRange(0, localderD))[0]);
	return ans;
}

// Get a H * W rectangle gaussian kernel, which is left-rigth symmetric with x
cv::Mat ReidDescriptor::gau_kernel(int x, float varW, int H, int W)
{
	int w = max(W - x, x + 1) * 2 - 1;
	cv::Mat temp = cv::getGaussianKernel(w, varW, CV_32F);
	cv::Mat ans(cv::Size(W, H), CV_32F);

	for (int i = 0 ; i < H ; i++) {
		ans.row(i) = temp.rowRange(w / 2 - x, w / 2 - x + W).t();
	}
	return ans;
}

// Pad matrix given number rows;
cv::Mat ReidDescriptor::padarray(cv::Mat m, int num)
{
	cv::Mat ans = m.clone();

	for (int i = 0 ; i < num ; i++) {
		ans.push_back(m.row(m.rows - 1));
	}
	return ans;
}

// Get the max value of the matrix;
float ReidDescriptor::max_value(cv::Mat m)
{
	double result;
	cv::minMaxLoc(m, NULL, &result);
	return result;
}

// Given the image of a person, get the axes, and return the map of kernel;
cv::Mat ReidDescriptor::map_kernel(cv::Mat img, int delta1, int delta2, float alpha, float varW, int H, int W, int &TLanti, int &HDanti,cv::Mat MSK)
{
	// UPLOADING AND FEATURE EXTRACTION
	if(MSK.empty()) {
		MSK = cv::Mat::ones(cv::Size(W, H), CV_8U);
	}

	// (me) Wouldn't it be quicker to just create a image of the same size rather than clone it ?
	cv::Mat img_hsv = img.clone();
	// Matlab:0:1,0:1,0:1 ; Opencv:0:255,0:255,0:180
	cv::Mat img_cielab = img.clone();
	// Matlab:0:255,0:255,0:255 ; Opencv:0:255,0:255,0:255
	cvtColor(img, img_hsv, CV_BGR2HSV);
	cvtColor(img, img_cielab, CV_BGR2Lab);

	// (me) Normalisation...?
	vector<cv::Mat> temp;
	split(img_hsv, temp);
	temp[0].convertTo(temp[0], CV_32FC1, 1 / 180.f);
	temp[1].convertTo(temp[1], CV_32FC1, 1 / 255.f);
	temp[2].convertTo(temp[2], CV_32FC1, 1 / 255.f);
	merge(temp, img_hsv);

	//double m, M;
	//cv::minMaxLoc(img_hsv,&m,&M);
	//imshow("imghsv",temp[0]);
	//cv::normalize(img_hsv,img_hsv,CV_MINMAX);

	TLanti = fminbnd(dissym_div, img_hsv, MSK, delta1, alpha, delta1, H - delta1);
	//cout << dissym_div(33, img_hsv, MSK, delta1, alpha)<<endl;
	//cout << dissym_div(34, img_hsv, MSK, delta1, alpha) << endl;
	int BUsim = fminbnd(sym_div, img_hsv.rowRange(0, img_hsv.rows), MSK.rowRange(0, img_hsv.rows), delta2, alpha, delta2, W - delta2);
	//cout << sym_div(18, img_hsv.rowRange(0, TLanti + 1), MSK.rowRange(0, TLanti + 1), delta2, alpha) << endl;
	//cout << sym_div(35, img_hsv.rowRange(0, TLanti + 1), MSK.rowRange(0, TLanti + 1), delta2, alpha) << endl;
	//int LEGsim = fminbnd(sym_div, img_hsv.rowRange(TLanti, img_hsv.rows), MSK.rowRange(TLanti, MSK.rows), delta2, alpha, delta2, W - delta2);
	HDanti = fminbnd(sym_dissimilar_MSKH, img_hsv, MSK, delta1, 0, 5, TLanti);

	//cv::Mat img_show = img.clone();
	//line(img_show, cv::Point(0, TLanti), cv::Point(W, TLanti), cv::Scalar(255, 0, 0));
	//line(img_show, cv::Point(0, HDanti), cv::Point(W, HDanti), cv::Scalar(255, 0, 0));
	//line(img_show, cv::Point(BUsim, HDanti), cv::Point(BUsim, TLanti), cv::Scalar(255, 0, 0));
	//line(img_show, cv::Point(LEGsim, TLanti), cv::Point(LEGsim, H), cv::Scalar(255, 0, 0));
	//imshow("test", img_show);
	//cv::waitKey(0);

	// Kernel - map computation
	vector<cv::Mat> img_split;
	split(img_hsv, img_split);
	img_split[2].convertTo(img_split[2], CV_8UC1, 255);
	equalizeHist(img_split[2], img_split[2]);
	img_split[2].convertTo(img_split[2], CV_32FC1, 1.0 / 255);
	merge(img_split, img_hsv);

	//cv::Mat HEADW = cv::Mat::zeros(cv::Size(W, HDanti + 1), CV_32FC1);

	//cv::Mat UP = img_hsv.rowRange(HDanti + 1, TLanti + 1);
	cv::Mat UP = img_hsv.rowRange(0, img_hsv.rows);
	cv::Mat UPW = gau_kernel(BUsim, varW, UP.rows, W);

	//cv::Mat DOWN = img_hsv.rowRange(TLanti + 1, img_hsv.rows);
	//cv::Mat DOWNW = gau_kernel(LEGsim, varW, DOWN.rows, W);

	cv::Mat MAP_KRNL = UPW / max_value(UPW);
	//cv::Mat MAP_KRNL = HEADW.clone() / max_value(HEADW);
	//MAP_KRNL.push_back(cv::Mat(UPW / max_value(UPW)));
	//MAP_KRNL.push_back(cv::Mat(DOWNW / max_value(DOWNW)));

	if (H - MAP_KRNL.rows > 0) {
		MAP_KRNL = padarray(MAP_KRNL, H - MAP_KRNL.rows);
	} else {
		MAP_KRNL = MAP_KRNL.rowRange(0, H);
	}

	//cv::imshow("map_kernel", MAP_KRNL);
	//cv::waitKey(0);
	return MAP_KRNL;
}

// Calculate the histogram.
cv::Mat ReidDescriptor::whistcY(cv::Mat raster, cv::Mat Weight, int bins)
{
	int M;
	M = raster.cols;
	cv::Mat ans(bins, 1, CV_32F, cv::Scalar(0));

	vector<float> EDGESpt(bins);
	for (int i = 0 ; i < bins ; i++) {
		EDGESpt[i] = i * 1.0 / (bins - 1);
	}

	for (int i = 0; i < M; i++){
		for (int c = 0; c < bins; c++){
			if ((c == bins - 1) || (raster.at<float>(i) < EDGESpt[c + 1] && raster.at<float>(i) >= EDGESpt[c])){
				ans.at<float>(c) += Weight.at<float>(i);
				break;
			}
		}
	}
	return ans;
}

// Given map kernel calculate the histogram of each part of the image and combine them together.
cv::Mat ReidDescriptor::Whsv_estimation(cv::Mat img, vector<int> NBINs, cv::Mat MAP_KRNL, int HDanti, int TLanti)
{
	cv::Mat img_hsv = img.clone();
	//Matlab:0:1,0:1,0:1 ; Opencv:0:255,0:255,0:180
	cvtColor(img, img_hsv, CV_BGR2HSV);
	vector<cv::Mat>img_split;
	split(img_hsv, img_split);
	img_split[2].convertTo(img_split[2], CV_8UC1, 255);
	equalizeHist(img_split[2], img_split[2]);
	img_split[0].convertTo(img_split[0], CV_32FC1, 1.0 / 180);
	img_split[1].convertTo(img_split[1], CV_32FC1, 1.0 / 255);
	img_split[2].convertTo(img_split[2], CV_32FC1, 1.0 / 255);
	merge(img_split, img_hsv);

	cv::Mat UP = img_hsv.rowRange(HDanti + 1, TLanti + 1);
	vector<cv::Mat> UP_split; split(UP, UP_split);
	cv::Mat UPW = MAP_KRNL.rowRange(HDanti + 1, TLanti + 1);

	cv::Mat DOWN = img_hsv.rowRange(TLanti + 1, img_hsv.rows);
	vector<cv::Mat> DOWN_split; split(DOWN, DOWN_split);
	cv::Mat DOWNW = MAP_KRNL.rowRange(TLanti + 1, img_hsv.rows);;

	UPW = UPW.reshape(1, 1);
	DOWNW = DOWNW.reshape(1, 1);

	cv::Mat tmph0(0, 0, CV_32FC1); cv::Mat tmph2(0, 0, CV_32FC1);
	cv::Mat tmpup0(0, 0, CV_32FC1); cv::Mat tmpup2(0, 0, CV_32FC1);
	cv::Mat tmpdown0(0, 0, CV_32FC1); cv::Mat tmpdown2(0, 0, CV_32FC1);

	for (int ch = 0 ; ch < 3 ; ch++){
		tmph2.push_back(cv::Mat(cv::Mat::zeros(NBINs[ch], 1, CV_32F)));
		cv::Mat rasterUP = UP_split[ch];
		tmpup2.push_back(whistcY(rasterUP.reshape(1, 1), UPW, NBINs[ch]));

		cv::Mat rasterDOWN = DOWN_split[ch];
		tmpdown2.push_back(whistcY(rasterDOWN.reshape(1, 1), DOWNW, NBINs[ch]));
	}

	cv::Mat ans = tmph2;
	ans.push_back(tmpup2);
	ans.push_back(tmpdown2);

	for (int row = 0; row < ans.rows; row++) {
		for (int col = 0; col < ans.cols; col++) {
#ifdef linux
			if (isnan(ans.at<float>(row, col)) == true) {
				ans.at<float>(row, col) = 0;
			}
#endif
#ifdef _WIN32
			if (_isnan(ans.at<float>(row, col)) != 0) {
				ans.at<float>(row, col) = 0;
			}
#endif
		}
	}

	return ans;
}

// Calculate the similarity between two images;
float ReidDescriptor::onebyone(cv::Mat img1, cv::Mat img2)
{
	int H = 80; int W = 64; // 128 & 128 /!\
	cv::resize(img1, img1, cv::Size(W, H), CV_BILATERAL);
	cv::resize(img2, img2, cv::Size(W, H), CV_BILATERAL);
	cv::Mat MAP_KRNL_1, MAP_KRNL_2;
	int HDanti_1, TLanti_1, HDanti_2, TLanti_2;
	MAP_KRNL_1 = map_kernel(img1, H / 4, W / 4, 0.5, W / 5, H, W, TLanti_1, HDanti_1);
	MAP_KRNL_2 = map_kernel(img2, H / 4, W / 4, 0.5, W / 5, H, W, TLanti_2, HDanti_2);

	vector<int> NBINs;
	NBINs.push_back(16);
	NBINs.push_back(16);
	NBINs.push_back(4);
	cv::Mat Whsv1 = Whsv_estimation(img1, NBINs, MAP_KRNL_1, HDanti_1, TLanti_1);
	cv::Mat Whsv2 = Whsv_estimation(img2, NBINs, MAP_KRNL_2, HDanti_2, TLanti_2);

	//ReidDescriptor::drawHist(Whsv1);
	//ReidDescriptor::drawHist(Whsv2);

	//vector<float> www1 = Whsv1;
	//vector<float> www2 = Whsv2;

	float d = bhattacharyya(Whsv1, Whsv2);
	return d;
}

/*cv::Mat ReidDescriptor::getWhsvFeature(cv::Mat img,cv::Mat MSK)
{
	int H = 80; int W = 64; // 128 & 128 /!\
	cv::resize(img, img, cv::Size(W, H), cv::INTER_LINEAR);

	if(!MSK.empty()) {
		cv::resize(MSK, MSK, cv::Size(W, H), cv::INTER_LINEAR);
	}

	cv::Mat MAP_KRNL;
	int HDanti, TLanti;
	MAP_KRNL = map_kernel(img, H / 4, W / 4, 0.5, W / 5, H, W, TLanti, HDanti, MSK);

	vector<int> NBINs;
	NBINs.push_back(16);
	NBINs.push_back(16);
	NBINs.push_back(4);
	cv::Mat Whsv1 = Whsv_estimation(img, NBINs, MAP_KRNL, HDanti, TLanti);

	return Whsv1;
}*/

Mat HSVVector(Mat img) {
    cv::Mat img_hsv, hist, hist_h, hist_s, hist_v;
    cvtColor(img, img_hsv, CV_BGR2HSV);
    
    // Normalisation ?
    vector<cv::Mat> temp;
    split(img_hsv, temp);
    
    temp[0] = temp[0].reshape(0, 1);
    temp[1] = temp[1].reshape(0, 1);
    temp[2] = temp[2].reshape(0, 1);
    
    // Histogram computation
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    float v_ranges[] = { 0, 256 };
    
    int histSize_h[] = { 180 };
    int histSize_s[] = { 256 };
    int histSize_v[] = { 256 };
    
    const float * ranges_h[] = { h_ranges };
    const float * ranges_s[] = { s_ranges };
    const float * ranges_v[] = { v_ranges };
    
    int channels[] = { 0 };
    
    calcHist(&temp[0], 1, channels, Mat(), hist_h, 1, histSize_h, ranges_h);
    normalize(hist_h, hist_h, 0, 1, NORM_MINMAX, -1, Mat());
    calcHist(&temp[1], 1, channels, Mat(), hist_s, 1, histSize_s, ranges_s);
    normalize(hist_s, hist_s, 0, 1, NORM_MINMAX, -1, Mat());
    calcHist(&temp[2], 1, channels, Mat(), hist_v, 1, histSize_v, ranges_v);
    normalize(hist_v, hist_v, 0, 1, NORM_MINMAX, -1, Mat());
    
    vconcat(hist_h, hist_s, hist);
    vconcat(hist, hist_v, hist);
    
    return hist;
}


cv::Mat ReidDescriptor::getWhsvFeature(cv::Mat img, cv::Mat MSK)
{
	int offset = img.rows / 5;
	vector<cv::Mat> sub(5);

	// Divide the image into 5x1 cells
	for(int i = 0 ; i < 4 ; i++) {
		sub[i] = img.rowRange(offset * i, offset * (i + 1));
	}
	sub[4] = img.rowRange(offset * 4, img.rows);
	// Debug this

	cv::Mat conc;
	cv::Mat temp;
	for(int i = 0 ; i < 5 ; i++) {
		cv::Mat HSV = HSVVector(sub[i]);
		if(i == 0) {
			conc = HSV;
		} else {
			vconcat(conc, HSV, conc);
		}
	}

	return conc;
    //return cv::Mat::zeros(2,2,CV_8U);
}

float ReidDescriptor::compareImg(cv::Mat img1, cv::Mat img2)
{
	return onebyone(img1, img2);
}

float ReidDescriptor::compareDiscriptor(ReidDescriptor r1,ReidDescriptor r2)
{
	return r1.compare(r2);
}

void ReidDescriptor::drawHist(string str,cv::Mat hist)
{
	double max_val;
	cv::minMaxLoc(hist, 0, &max_val, 0, 0);
	int scale = 2;
	int hist_height = 256;
	int bins = hist.rows;
	cv::Mat hist_img = cv::Mat::zeros(hist_height, bins*scale, CV_8UC3);
	for (int i = 0; i < bins; i++)
	{
		float bin_val = hist.at<float>(i);
		int intensity = cvRound(bin_val*hist_height / max_val);  //Ҫ���Ƶĸ߶�
		rectangle(hist_img, cv::Point(i*scale, hist_height - 1),
			cv::Point((i + 1)*scale - 1, hist_height - intensity),
			CV_RGB(255, 255, 255));
	}
	cv::imshow(str, hist_img);
	cv::waitKey(33);
}
