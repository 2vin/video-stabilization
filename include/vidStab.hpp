#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

class VS
{
  //Variables declaration
  public:
    Mat THISFRAME;
    Mat LASTFRAME;
    Mat CANVAS;

    cv::Ptr<cv::CLAHE> clahe;


    std::vector<Point2f> THISPOINTS;
    std::vector<Point2f> LASTPOINTS;
    std::vector<Point2f> THISGOODPOINTS;
    std::vector<Point2f> LASTGOODPOINTS;


    int MAXFEAT;
    double QUALITY;

    vector<uchar> status;
    vector<float> errors;

    Mat_<float>     rigidTransform, smoothTransform;

    double max_dist, min_dist;


  // Function declaration
  public:
    VS();
    ~VS();
    bool COLORCORRECT(Mat&, bool);
    bool CALCFEAT(Mat&, bool);
    bool MATCHFEAT(Mat&, bool);
    bool GETRELATION(Mat&, bool);
    bool SWAPFEAT();

    void LOWPASS(Mat_<float>, double, Mat_<float>&);
};
