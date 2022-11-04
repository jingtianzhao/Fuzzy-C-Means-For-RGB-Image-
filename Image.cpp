#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;
//自动对齐：shift alt F
class Image
{
public:
    void SettingImage(Mat &image);
    static void PrintMat(Mat &m1);
    Mat GettingImage();
    Image(Mat &Image);
    Mat FuzzyCmeans(Mat cluster, float m);
    Mat OutputImage(Mat &membership, Mat &cluster);
    static Mat SerialImage(Mat &img);
    void PrintImageTypeAndSize(Mat &img);
    static bool DetectHaveNan(Mat &img);
    static Mat DeserialImage(Mat &img,int row,int col);
private:
    Mat image;
};

Mat Image::FuzzyCmeans(Mat cluster, float m)
{
    Mat output(image.rows, image.cols, CV_64FC3);
    // 先把三维图像转化为一维
    resize(output, output, Size(image.rows * image.cols, 1), 0, 0);
    Mat imageA;
    imageA = SerialImage(image);
    // imageA=imageA.t();
    //  resize(image, imageA, Size(image.rows * image.cols, 1), 0, 0);
    //  定义距离函数(m*n,cluster number)
    int clusternum = cluster.rows;
    Mat dist(clusternum, image.rows * image.cols, CV_64FC1);
    Mat U(clusternum, image.rows * image.cols, CV_64FC1);
    Mat cluster_old;
    Mat D1;
    Mat U1;
    Mat UR;
    Mat C1;
    Mat mat;
    Mat U2;
    Mat channel[3];
    Mat diffierence;
    double min;
    double max;
    Point maxIndex;
    Point minIndex;
    bool testB;
    double tp;
    float K;
    float error = 0.1;
    split(imageA, channel);
    while (error > 0.0001)
    {
        cluster_old = cluster.clone();
        // 将3通道分离分别处理BGR
        
        dist = Mat::zeros(clusternum, imageA.cols, CV_64F);
        //对于每个聚类中心
        //计算距离
        //两个都是CV_64F，但是一个是单通道C1，一个是三通道C3，也不可以进行想减的运算
        //矩阵减法时类型和维度必须一致。
        //按照行列求和
        // ones(行数，列数）
        //将聚类中心的R,G,B分量拿出来分别求距离
        for (int i = 0; i < 3; i++)
        {

            absdiff(cluster.col(i) * Mat::ones(1, imageA.cols, CV_64F), (channel[i].t() * Mat::ones(1, cluster.rows, CV_64F)).t(), D1);
            pow(D1, 2, D1);
            dist = dist + D1;
        };
        for (int i = 0; i < clusternum; i++)
        {
            //出现非法值

            pow((Mat::ones(clusternum, 1, CV_64F) * dist.row(i)) / (dist + (0.0001) * Mat::ones(dist.size(), dist.type())), 2 / (m - 1), U2);
            //按列求和
            reduce(U2, UR, 0, REDUCE_SUM, CV_64F);
            UR = Mat::ones(UR.rows, UR.cols, CV_64F) / UR;

            for (int j = 0; j < UR.cols; j++)
            {
                U.at<double>(i, j) = UR.at<double>(j);
            }
        }
        pow(U, m, U1);
        reduce(U1, C1, 1, REDUCE_SUM, CV_64F);
        //全为0
        C1 = C1 * Mat::ones(1, 3, CV_64F);
        Mat U2 = U1.clone();
        // Nan非法值
        for (int i = 0; i < 3; i++)
        {
            reduce(U1.mul(Mat::ones(clusternum, 1, CV_64F) * channel[i]), U1, 1, REDUCE_SUM, CV_64F);
            PrintMat(U1);

            for (int j = 0; j < clusternum; j++)
            {
                cluster.at<double>(j, i) = U1.at<double>(j) / C1.at<double>(j, i);
            }

            U1 = U2.clone();
        }
        absdiff(cluster, cluster_old, diffierence);
        reduce(diffierence, diffierence, 0, REDUCE_SUM, CV_64F);
        reduce(diffierence, diffierence, 1, REDUCE_SUM, CV_64F);
        if (diffierence.at<double>(0, 0) <= 0.0001)
            error = 0;
    };

    return U;
};
Mat Image::OutputImage(Mat &U, Mat &cluster)
{
    double max;
    double min;
    Point minIndex;
    Point maxIndex;
    Mat output(U.cols, 1, CV_64FC3);
    Mat channel[3];
    Mat output1;
    split(output, channel);
    cout << output.type() << endl;
    for (int i = 0; i < U.cols; i++)
    {
        minMaxLoc(U.col(i), &min, &max, &minIndex, &maxIndex);
        //测试
        for (int j = 0; j < 3; j++)
        {
            channel[j].at<double>(i) = cluster.at<double>(maxIndex.y, j);
        }
    }
    merge(channel, 3, output);
    return output;
}
Mat Image::GettingImage()
{
    return image;
};
Image::Image(Mat &img)
{
    image = img;
};
void Image::PrintMat(Mat &m1)
{
    for (int i = 0; i < m1.rows; i++)
    {
        for (int j = 0; j < m1.cols; j++)
        {
            cout << m1.at<double>(i, j) << "|";
        }
        cout << endl;
    }
}
void Image::SettingImage(Mat &img)
{
    image = img;
};
Mat Image::SerialImage(Mat &img)
{
    int m = img.rows;
    int n = img.cols;
    Mat channel[3];
    Mat outputChannel[3];
    split(img, channel);
    Mat output(1, m * n, img.type());
    split(output,outputChannel);
    int index = 0;
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                outputChannel[k].at<double>(index)=channel[k].at<double>(i,j);
            }
            index++;
        }
    }
    merge(outputChannel, 3, output);
    return output;
}
Mat Image::DeserialImage(Mat &img,int row,int col)
{
    Mat output(row,col,img.type());
    Mat outputChannel[3];
    Mat channel[3];
    split(output,outputChannel);
    split(img,channel);
    Mat Channel[3];
    int index=0;
    for(int i=0;i<row;i++)
    {
        for(int j=0;j<col;j++)
        {
            for(int k=0;k<3;k++)
            {
                outputChannel[k].at<double>(i,j)=channel[k].at<double>(index);
            }
            index++;
        }
    }
    merge(outputChannel,3,output);
    return output;
}
void Image::PrintImageTypeAndSize(Mat &img)
{
    cout << img.type() << endl;
    cout << img.size() << endl;
}
//检测matrix是否有非法值
bool Image::DetectHaveNan(Mat &img)
{
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            if (img.at<double>(i, j) != img.at<double>(i, j))
            {
                return true;
            }
        }
    }

    return false;
}
