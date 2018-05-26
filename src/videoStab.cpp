#include "vidStab.hpp"

int main(int argc, char** argv)
{
  Mat frame, last;
  VideoCapture cap;

  if(argc<2)
    cap.open(0);
  else
    cap.open(argv[1]);

  VS obj;

  while(cap.isOpened())
  {
    cap>>frame;
    if(frame.empty())
	exit(0);

    resize(frame,frame,Size(),0.5,0.5);

    //obj.COLORCORRECT(frame,false);

    obj.CALCFEAT(frame, false);

    obj.MATCHFEAT(frame, false);

    obj.GETRELATION(frame, true);

    obj.SWAPFEAT();
  }

  return 0;
}
