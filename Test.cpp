#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include "Image.cpp"
using namespace cv;
using namespace std;
// template <typename T>
// {
// }
class Function
{
public:
    Mat image;

private:
    Mat image1;
    void DetectImage();
};
int main()
{
    Mat img = imread("E:\\Work\\MatlabCode\\image\\12003.jpg");
    imshow("image", img);
    img.convertTo(img, CV_64FC3, 1, 0);
    Image img1(img);
    // for (int i = 0; i < img.rows; i++)
    // {
    //     for (int j = 0; j < img.cols; j++)
    //     {
    //         for (int k = 0; k < 3; k++)
    //         {
    //             img.at<Vec3d>(i, j)[k] = 0.0;
    //         }
    //     }
    // }
    double m[4][3] = {25, 20, 40, 50, 65, 75, 125, 150, 153, 200, 225, 235};
    Mat cluster = Mat(4, 3, CV_64F, m);
    Mat output = img1.FuzzyCmeans(cluster, 3.f);
    Mat outImg = img1.OutputImage(output, cluster);
    Mat output1 = img1.DeserialImage(outImg, img.rows, img.cols);
    output1.convertTo(output1, CV_8UC3);
    imshow("ouput", output1);
    waitKey();
    return 0;
};