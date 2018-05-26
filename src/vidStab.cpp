#include "vidStab.hpp"


float euclideanDist(Point2f& p, Point2f& q) {
  Point2f diff = p - q;
  return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

double SCALECORRECT(Mat_<float>& transformMat, Mat_<float>& absoluteMat)
{
  double XSCALE = (abs(transformMat[0][0])/transformMat[0][0])*sqrt(pow(transformMat[0][0],2)+pow(transformMat[0][1],2));
  double YSCALE = (abs(transformMat[1][1])/transformMat[1][1])*sqrt(pow(transformMat[1][0],2)+pow(transformMat[1][1],2));
  double ORIENT = atan2(absoluteMat[1][0],absoluteMat[1][1])*180.0/CV_PI;

  transformMat[0][0] /= XSCALE;
  transformMat[0][1] /= XSCALE;

  transformMat[1][0] /= YSCALE;
  transformMat[1][1] /= YSCALE;

  std::system("clear");
  cout<<" XSCALE : "<<XSCALE<<endl;
  cout<<" YSCALE : "<<YSCALE<<endl;
  cout<<" ORIENT : "<<ORIENT<<endl;

  return ORIENT;
}

VS::VS()
{
  clahe = cv::createCLAHE();

  MAXFEAT = 50;
  QUALITY= 0.01;

  rigidTransform = Mat::eye(3,3,CV_32FC1);
  smoothTransform = Mat::eye(3,3,CV_32FC1);

}

VS::~VS()
{
  delete this;
}

bool VS::COLORCORRECT(Mat& thisFrame, bool SHOWIMAGE = false)
{
    if(thisFrame.empty())
    {
      return false;
    }
    // READ RGB color image and convert it to Lab
    static cv::Mat lab_image;
    cv::cvtColor(thisFrame, lab_image, CV_BGR2Lab);

    // Extract the L channel
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    clahe->setClipLimit(4);
    static cv::Mat dst;
    clahe->apply(lab_planes[0], dst);
    //cout<<"test"<<endl;

    // Merge the the color planes back into an Lab image
    dst.copyTo(lab_planes[0]);
    cv::merge(lab_planes, lab_image);

    // convert back to RGB
    static cv::Mat image_clahe;
    cv::cvtColor(lab_image, thisFrame, CV_Lab2BGR);

    if(SHOWIMAGE)
    {
      imshow("COLOR CORRECTED",thisFrame);
      if(waitKey(1)>0)
        exit(0);
    }

}


bool VS::CALCFEAT(Mat &thisFrame, bool SHOWIMAGE = false)
{
  cvtColor(thisFrame,THISFRAME, CV_BGR2GRAY);
  goodFeaturesToTrack(THISFRAME,THISPOINTS,MAXFEAT,QUALITY,10);

  if(THISPOINTS.size()<4)
  {
    return false;
  }
  if(LASTPOINTS.size()==0)
  {
    LASTPOINTS = THISPOINTS;
    LASTFRAME = THISFRAME.clone();
    return false;
  }

  if(SHOWIMAGE)
  {
    for(int i=0;i<THISPOINTS.size();i++)
      circle(thisFrame,THISPOINTS[i],3,Scalar(0,255,0),2);
    for(int i=0;i<LASTPOINTS.size();i++)
      circle(thisFrame,LASTPOINTS[i],2,Scalar(0,5,250),2);

    //line(thisFrame, LASTPOINTS[i], THISPOINTS[i], Scalar(255,0,0), int thickness=1);

    imshow("KEYPOINTS", thisFrame);

    if(waitKey(1)>0)
      exit(0);
  }

  //LASTPOINTS = THISPOINTS;
  //LASTFRAME = THISFRAME.clone();
  return true;
}

bool VS::MATCHFEAT(Mat& thisFrame, bool SHOWIMAGE = false)
{
  status.clear();
  errors.clear();
  THISPOINTS.clear();
  calcOpticalFlowPyrLK(LASTFRAME,THISFRAME,LASTPOINTS,THISPOINTS,status,errors,Size(10,10));

  float avgDistance = 0;
  THISGOODPOINTS.clear();
  LASTGOODPOINTS.clear();
  if(countNonZero(status)>0)
  {
    for(int i=0;i<THISPOINTS.size();i++)
    {
      if(status[i]!=0)
      avgDistance += abs(euclideanDist(LASTPOINTS[i],THISPOINTS[i]));
    }
    avgDistance /= (1.0*countNonZero(status));

    for(int i=0;i<THISPOINTS.size();i++)
    {
      //cout<<euclideanDist(LASTPOINTS[i],THISPOINTS[i])<<"   "<<avgDistance<<endl;
      if(status[i]!=0)
      {
        if(abs(euclideanDist(LASTPOINTS[i],THISPOINTS[i]))*1.0<avgDistance)
        {
          THISGOODPOINTS.push_back(THISPOINTS[i]);
          LASTGOODPOINTS.push_back(LASTPOINTS[i]);
        }
      }
    }
  }

  if(countNonZero(status)<0.3*THISPOINTS.size())
  {
    LASTPOINTS.clear();
    THISPOINTS.clear();
  }
  if(SHOWIMAGE)
  {
    if(THISGOODPOINTS.size()>0)
    {
      Mat MATCHEIM = thisFrame.clone();

      for(int i=0;i<THISGOODPOINTS.size();i++)
      {
        //cout<<euclideanDist(LASTPOINTS[i],THISPOINTS[i])<<"   "<<avgDistance<<endl;
        line(MATCHEIM, LASTGOODPOINTS[i], THISGOODPOINTS[i], Scalar(255,0,0), 3);
        circle(MATCHEIM,THISGOODPOINTS[i],10,Scalar(0,255,0),CV_FILLED);
        circle(MATCHEIM,LASTGOODPOINTS[i],9,Scalar(0,5,250),CV_FILLED);
      }


      for(int i=0;i<THISPOINTS.size();i++)
        circle(MATCHEIM,THISPOINTS[i],3,Scalar(0,255,0),CV_FILLED);
      for(int i=0;i<LASTPOINTS.size();i++)
        circle(MATCHEIM,LASTPOINTS[i],3,Scalar(0,5,250),CV_FILLED);


      imshow("MATCHES", MATCHEIM);

      if(waitKey(1)>0)
        exit(0);
    }
  }

}

bool VS::GETRELATION(Mat& thisFrame, bool SHOWIMAGE = false)
{
  if(THISGOODPOINTS.size()>2)
  {
    Mat_<float> newRigidTransform = estimateRigidTransform(LASTGOODPOINTS,THISGOODPOINTS,false);
    Mat_<float> nrt33 = Mat_<float>::eye(3,3);

    //cout<<"41"<<endl;

    newRigidTransform.copyTo(nrt33.rowRange(0,2));
    double orient = SCALECORRECT(nrt33,rigidTransform);

    //cout<<"42"<<endl;

    Mat_<float> tempRigidTransform = rigidTransform.clone();
    //rigidTransform *= nrt33;
    //LOWPASS(rigidTransform, 0.4, smoothTransform);

    if(((abs(rigidTransform[0][2])<thisFrame.cols/10)||((abs(rigidTransform[0][2])/rigidTransform[0][2])*(abs(nrt33[0][2])/nrt33[0][2])==-1))&&((abs(rigidTransform[1][2])<thisFrame.rows/10)||((abs(rigidTransform[1][2])/rigidTransform[1][2])*(abs(nrt33[1][2])/nrt33[1][2])==-1)))
    {
      rigidTransform *= nrt33;
      LOWPASS(rigidTransform, 0.5, smoothTransform);
    }
    else
    {
      // rigidTransform = tempRigidTransform;
      // LOWPASS(Mat::eye(3,3,CV_32FC1), 0.03, smoothTransform);
    }
    // rigidTransform = nrt33;

    //cout<<"43"<<endl;

    //cout<<rigidTransform<<endl;

    // Mat invTrans = rigidTransform.inv(DECOMP_SVD);

    Mat invTrans = smoothTransform.inv(DECOMP_SVD);

    //cout<<"44"<<endl;

    Mat result;
    warpAffine(thisFrame,result,invTrans.rowRange(0,2),Size());

    //cout<<"45"<<endl;

    if(SHOWIMAGE)
    {
      Rect r;
      r.x = thisFrame.cols/10;
      r.y = thisFrame.rows/10;
      r.width = thisFrame.cols-2*r.x;
      r.height = thisFrame.rows-2*r.y;
      Mat x = result(r);
      //cout<<"46"<<endl;

      resize(x,x,Size(thisFrame.cols,thisFrame.rows));
      Mat comparison;
      //cout<<"47"<<endl;

      hconcat(thisFrame,x,comparison);
      //cout<<"48"<<endl;

      imshow( "Result", comparison );
      if(waitKey(1)>0)
        exit(0);
    }

  }
  else
  {

    LOWPASS(Mat::eye(3,3,CV_32FC1), 0.03, smoothTransform);


    Mat invTrans = smoothTransform.inv(DECOMP_SVD);

    Mat result;
    warpAffine(thisFrame,result,invTrans.rowRange(0,2),Size());


    if(SHOWIMAGE)
    {
      Rect r;
      r.x = thisFrame.cols/10;
      r.y = thisFrame.rows/10;
      r.width = thisFrame.cols-2*r.x;
      r.height = thisFrame.rows-2*r.y;
      Mat x = result(r);
      //cout<<"46"<<endl;

      resize(x,x,Size(thisFrame.cols,thisFrame.rows));
      Mat comparison;
      //cout<<"47"<<endl;

      hconcat(thisFrame,x,comparison);
      //cout<<"48"<<endl;

      imshow( "Result", comparison );
      if(waitKey(1)>0)
        exit(0);
    }
  }
}

bool VS::SWAPFEAT()
{
  LASTPOINTS = THISPOINTS;
  LASTFRAME = THISFRAME.clone();

}

//UTILITY FUNCTIONS()
void VS::LOWPASS(Mat_<float> val, double filterVal, Mat_<float>& smoothVal)
{
  addWeighted( val, filterVal, smoothVal, (1-filterVal), 0.0, smoothVal);
}
