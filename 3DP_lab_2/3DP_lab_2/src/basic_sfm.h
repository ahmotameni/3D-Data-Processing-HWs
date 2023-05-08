#pragma once

#pragma once

#include <string>
#include <vector>

#include "Eigen/Dense"
#include <opencv2/opencv.hpp>

class BasicSfM
{
 public:

  ~BasicSfM();

  // Read data from file, that are observations along with the ids of the camera positions
  // where the observations have been acquired and the ids of the 3D points that generate such observations)
  // If load_initial_guess is set to true, it is assumed that the input file also includes an initial guess
  // solution of the reconstruction problem, hence load it
  // If load_colors is set to true, it is assumed that the input file also includes the RGB colors of the
  // 3D points, hence load them
  void readFromFile(const std::string& filename, bool load_initial_guess = false, bool load_colors = false );
  // Write data to file, if write_unoptimized is set to true, also the optimized parameters (camera and points
  // positions) are stored to file
  void writeToFile (const std::string& filename, bool write_unoptimized = false  ) const;

  // Save the reconstructed scene in a PLY file: you can visualize it by using for instance MeshLab
  // (execute
  // sudo apt install meshlab
  // in debian-derived linux distributions to install meshlab)
  void writeToPLYFile (const std::string& filename, bool write_unoptimized = false ) const;

  // The core of this class: it performs incremental structure from motion on the loaded data
  void solve();

  // Clear everything
  void reset();

 private:

  // Refine camera and point positions registered so far inside a global optimization problem
  void bundleAdjustmentIter( int new_cam_idx );

  // A simple strategy for eliminating outliers: just check the projection error of each point in each view,
  // if is greater than max_reproj_err_, remove the point from the solution
  int rejectOuliers();

  // Get the pointer to the 6-dimensional parameter block that defines the position of the pose_idx-th view
  inline double *cameraBlockPtr ( int pose_idx = 0 ) const
  {
    return const_cast<double *>(parameters_.data()) + (pose_idx * camera_block_size_ );
  };

  // Get the pointer to the 3-dimensional parameter block that defines the position of the point_idx-th point
  inline double *pointBlockPtr ( int point_idx = 0 ) const
  {
    return const_cast<double *>(parameters_.data()) + (num_cam_poses_ * camera_block_size_ + point_idx * point_block_size_ );
  };

  void initCamParams(int new_pose_idx, cv::Mat r_vec, cv::Mat t_vec );

  void cam2center (const double* camera, double* center) const;
  void center2cam (const double* center, double* camera) const;

  // Test the cheirality constaint for the pt_idx-th 3D point seen by the pos_idx-th camera position
  bool checkCheiralityConstraint(int pos_idx, int pt_idx );

  // Print the the 6-dimensional parameter block that defines the position of the idx-th view
  void printPose( int idx ) const;

  // Print the the 3-dimensional parameter block that defines the position of the idx-th point
  void printPointParams( int idx ) const;

  // Number of camera positions
  int num_cam_poses_ = 0;
  // Number of observed 3D points
  int num_points_ = 0;
  // Number of observation, i.e., projections of the 3D points into an image plane
  int num_observations_ = 0;
  // Total number of parameters that could be optimized (basically 6 * num_cam_poses_ + 3 * num_points_ )
  int num_parameters_ = 0;

  // For each observation (i.e., projection into an image plane) point_index_ stores the *index* of the corresponding
  // 3D point that generates such observation. point_index_ has a size equal to num_observations_
  std::vector<int> point_index_;
  // For each observation (i.e., projection into an image plane) cam_pose_index_ stores the *index* of the corresponding
  // 6-DoF position (3D axis-angle rotation and 3D translation) of the camera that made the observation.
  // cam_pose_index_ has a size equal to num_observations_
  std::vector<int> cam_pose_index_;
  // Vector of observations, i.e. 2D point projections in all images of the observed 3D points.
  // observations_ has a size equal to 2*num_observations_
  std::vector<double> observations_;
  // Vector of the RGB colors of the observed 3D points (if available). colors_ has a size equal to 3*num_points_
  std::vector<unsigned char> colors_;

  // Vector of all the parameters to be estimated: it is composed by num_cam_poses_ 6D blocks
  // (3D axis-angle rotation and 3D translation, one for each camera view) followed by num_points_
  // 3D blocks (one for each 3D point)
  std::vector<double> parameters_;

  const int camera_block_size_ = 6;
  const int point_block_size_ = 3;

  // For each camera pose, the number of optimization iterations (0 if it has not yet been estimated,
  // -1 if the pose has been rejected)
  std::vector<int> cam_pose_optim_iter_;
  // For each 3D point, the number of optimization iterations (0 if it has not yet been estimated,
  // -1 if the point has been rejected)
  std::vector<int> pts_optim_iter_;

  // Maximum allowed reprojection error used to classify an inlier
  double max_reproj_err_ = 0.01;
  // Maximum number of outliers that we can tolerate without re-optimizing all
  int max_outliers_ = 5;
};
