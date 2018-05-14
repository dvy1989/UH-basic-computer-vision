// g++ test.cc -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/plot.hpp>
using namespace cv;

#include <iostream>
using namespace std;

// see Matrix Type in /usr/include/opencv2/core/cvdef.h 
int main(int, char**) {
  cout << CV_VERSION << endl;
  
  Mat img_bgr = imread("messi5.jpg");
  cout << "rows=" << img_bgr.rows << " cols=" << img_bgr.cols 
       << " channels=" << img_bgr.channels() << " type=" << img_bgr.type()
       << " (type " << CV_8UC3 << " means CV_8UC3)" << endl;
  imshow("BGR", img_bgr);
  waitKey(0);

  Mat channel[3];
  split(img_bgr, channel);
  Mat img_g = channel[1];
  cout << "rows=" << img_g.rows << " cols=" << img_g.cols 
       << " channels=" << img_g.channels() << " type=" << img_g.type() 
       << " (type " << CV_8UC1 << " means CV_8UC1)" << endl;
  imwrite("green.jpg", img_g);
  imshow("grey", img_g);
  waitKey(0);

  Mat xData, yData, plotdisplay;
  Ptr<plot::Plot2d> plot;
  yData.create(1, img_g.rows, CV_64F);
  xData.create(1, img_g.rows, CV_64F); //1 Row, 100 columns, Double
  
  for (int i=0; i<img_g.rows; i++) {
    double s = 0.0;
    for (int j=0; j<img_g.cols; j++)
      s += img_g.at<unsigned char>(i, j)/255.0;
    
    yData.at<double>(i) = i;
    xData.at<double>(i) = s;
  }
  plot = plot::createPlot2d(xData, yData);       // this is for OpenCV 3.1.0
  // plot = plot::Plot2d::create(xData, yData);  // this is for OpenCV 3.4.1
  plot->setPlotSize(img_g.rows, 100);
  plot->setMaxX(400);
  plot->setMinX(0);
  plot->setMaxY(img_g.rows-1);
  plot->setMinY(0);
  plot->render(plotdisplay);
  imwrite("projection.png", plotdisplay);
  imshow("projection", plotdisplay);
  waitKey();
}


