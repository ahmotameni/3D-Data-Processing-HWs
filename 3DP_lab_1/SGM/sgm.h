#pragma once
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#define DEBUG false



using namespace std;
using namespace cv;

typedef vector<unsigned long> ul_array;
typedef vector<ul_array> ul_array2D;
typedef vector<ul_array2D> ul_array3D;
typedef vector<ul_array3D> ul_array4D;

namespace sgm
{
  struct path
  {
    int direction_x;
    int direction_y;
  };
  struct processing_window
  {
    int east;
    int west;
    int north;
    int south;

  };
  class SGM
  {

  public:
    SGM(unsigned int disparity_range, unsigned int p1=3, unsigned int p2=40, unsigned int window_height=3, unsigned window_width_=3);
    void set(const  cv::Mat &left_img, const  cv::Mat &right_img);
    void compute_disparity();
    void save_disparity(char* out_file_name);
    float compute_mse(const  cv::Mat &gt);


  private:
      void init_paths();
      void aggregation();
      void calculate_cost_hamming();
      void compute_path_cost(int direction_y, int direction_x, int cur_y, int cur_x, int cur_path);

      int height_;
      int width_;
      unsigned int disparity_range_;
      unsigned int p1_;
      unsigned int p2_;
      unsigned int window_height_;
      unsigned window_width_;
      cv::Mat disp_;
      cv::Mat views_[2];
      vector<path> paths_;
      processing_window pw_;
      ul_array3D cost_;
      ul_array3D aggr_cost_;
      ul_array4D path_cost_;

  };
}



