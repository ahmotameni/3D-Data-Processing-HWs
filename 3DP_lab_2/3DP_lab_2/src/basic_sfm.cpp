#include "basic_sfm.h"

#include <iostream>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <fstream>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;

struct ReprojectionError {
    //////////////////////////// Code to be completed (5/6) //////////////////////////////////
    // This class should include an auto-differentiable cost function (see Ceres Solver docs).
    // Remember that we are dealing with a normalized, canonical camera:
    // point projection is easy! To rotete a point given an axis-angle rotation, use
    // the Ceres function:
    // AngleAxisRotatePoint(...) (see ceres/rotation.h)
    // WARNING: When dealing with the AutoDiffCostFunction template parameters,
    // pay attention to the order of the template parameters
    //////////////////////////////////////////////////////////////////////////////////////////
    ReprojectionError(double observed_x, double observed_y)
            : observed_x(observed_x), observed_y(observed_y) {}

    template<typename T>
    bool operator()(const T *const camera, const T *const point, T *residuals) const {
        // camera[0, 1, 2] is the axis-angle rotation.
        T rotated_point[3];
        ceres::AngleAxisRotatePoint(camera, point, rotated_point);

        // camera[3, 4, 5] is the translation.
        T projected_point[2];
        projected_point[0] = rotated_point[0] / rotated_point[2] + camera[3];
        projected_point[1] = rotated_point[1] / rotated_point[2] + camera[4];

        // Compute the reprojection error.
        residuals[0] = projected_point[0] - T(observed_x);
        residuals[1] = projected_point[1] - T(observed_y);

        return true;
    }

    // Factory to hide the construction of the CostFunction object from the client code.
    static ceres::CostFunction *Create(const double observed_x, const double observed_y) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
                new ReprojectionError(observed_x, observed_y)));
    }

    double observed_x;
    double observed_y;
};
    //////////////////////////////////////////////////////////////////////////////////////////



namespace {
    typedef Eigen::Map <Eigen::VectorXd> VectorRef;
    typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

    template<typename T>
    void FscanfOrDie(FILE *fptr, const char *format, T *value) {
        int num_scanned = fscanf(fptr, format, value);
        if (num_scanned != 1) {
            cerr << "Invalid UW data file.";
            exit(-1);
        }
    }

}  // namespace


BasicSfM::~BasicSfM() {
    reset();
}

void BasicSfM::reset() {
    point_index_.clear();
    cam_pose_index_.clear();
    observations_.clear();
    colors_.clear();
    parameters_.clear();

    num_cam_poses_ = num_points_ = num_observations_ = num_parameters_ = 0;
}

void BasicSfM::readFromFile(const std::string &filename, bool load_initial_guess, bool load_colors) {
    reset();

    FILE *fptr = fopen(filename.c_str(), "r");

    if (fptr == NULL) {
        cerr << "Error: unable to open file " << filename;
        return;
    };

    // This wil die horribly on invalid files. Them's the breaks.
    FscanfOrDie(fptr, "%d", &num_cam_poses_);
    FscanfOrDie(fptr, "%d", &num_points_);
    FscanfOrDie(fptr, "%d", &num_observations_);

    cout << "Header: " << num_cam_poses_
         << " " << num_points_
         << " " << num_observations_ << std::endl;

    point_index_.resize(num_observations_);
    cam_pose_index_.resize(num_observations_);
    observations_.resize(2 * num_observations_);

    num_parameters_ = camera_block_size_ * num_cam_poses_ + point_block_size_ * num_points_;
    parameters_.resize(num_parameters_);

    for (int i = 0; i < num_observations_; ++i) {
        FscanfOrDie(fptr, "%d", cam_pose_index_.data() + i);
        FscanfOrDie(fptr, "%d", point_index_.data() + i);
        for (int j = 0; j < 2; ++j) {
            FscanfOrDie(fptr, "%lf", observations_.data() + 2 * i + j);
        }
    }

    if (load_colors) {
        colors_.resize(3 * num_points_);
        for (int i = 0; i < num_points_; ++i) {
            int r, g, b;
            FscanfOrDie(fptr, "%d", &r);
            FscanfOrDie(fptr, "%d", &g);
            FscanfOrDie(fptr, "%d", &b);
            colors_[i * 3] = r;
            colors_[i * 3 + 1] = g;
            colors_[i * 3 + 2] = b;
        }
    }

    if (load_initial_guess) {
        cam_pose_optim_iter_.resize(num_cam_poses_, 1);
        pts_optim_iter_.resize(num_points_, 1);

        for (int i = 0; i < num_parameters_; ++i) {
            FscanfOrDie(fptr, "%lf", parameters_.data() + i);
        }
    } else {
        memset(parameters_.data(), 0, num_parameters_ * sizeof(double));
        // Masks used to indicate which cameras and points have been optimized so far
        cam_pose_optim_iter_.resize(num_cam_poses_, 0);
        pts_optim_iter_.resize(num_points_, 0);
    }

    fclose(fptr);
}


void BasicSfM::writeToFile(const string &filename, bool write_unoptimized) const {
    FILE *fptr = fopen(filename.c_str(), "w");

    if (fptr == NULL) {
        cerr << "Error: unable to open file " << filename;
        return;
    };

    if (write_unoptimized) {
        fprintf(fptr, "%d %d %d\n", num_cam_poses_, num_points_, num_observations_);

        for (int i = 0; i < num_observations_; ++i) {
            fprintf(fptr, "%d %d", cam_pose_index_[i], point_index_[i]);
            for (int j = 0; j < 2; ++j) {
                fprintf(fptr, " %g", observations_[2 * i + j]);
            }
            fprintf(fptr, "\n");
        }

        if (colors_.size() == num_points_ * 3) {
            for (int i = 0; i < num_points_; ++i)
                fprintf(fptr, "%d %d %d\n", colors_[i * 3], colors_[i * 3 + 1], colors_[i * 3 + 2]);
        }

        for (int i = 0; i < num_cam_poses_; ++i) {
            const double *camera = parameters_.data() + camera_block_size_ * i;
            for (int j = 0; j < camera_block_size_; ++j) {
                fprintf(fptr, "%.16g\n", camera[j]);
            }
        }

        const double *points = pointBlockPtr();
        for (int i = 0; i < num_points_; ++i) {
            const double *point = points + i * point_block_size_;
            for (int j = 0; j < point_block_size_; ++j) {
                fprintf(fptr, "%.16g\n", point[j]);
            }
        }
    } else {
        int num_cameras = 0, num_points = 0, num_observations = 0;

        for (int i = 0; i < num_cam_poses_; ++i)
            if (cam_pose_optim_iter_[i] > 0) num_cameras++;

        for (int i = 0; i < num_points_; ++i)
            if (pts_optim_iter_[i] > 0) num_points++;

        for (int i = 0; i < num_observations_; ++i)
            if (cam_pose_optim_iter_[cam_pose_index_[i]] > 0 && pts_optim_iter_[point_index_[i]] > 0)
                num_observations++;

        fprintf(fptr, "%d %d %d\n", num_cameras, num_points, num_observations);

        for (int i = 0; i < num_observations_; ++i) {
            if (cam_pose_optim_iter_[cam_pose_index_[i]] > 0 && pts_optim_iter_[point_index_[i]] > 0) {
                fprintf(fptr, "%d %d", cam_pose_index_[i], point_index_[i]);
                for (int j = 0; j < 2; ++j) {
                    fprintf(fptr, " %g", observations_[2 * i + j]);
                }
                fprintf(fptr, "\n");
            }
        }

        if (colors_.size() == num_points_ * 3) {
            for (int i = 0; i < num_points_; ++i) {
                if (pts_optim_iter_[i] > 0)
                    fprintf(fptr, "%d %d %d\n", colors_[i * 3], colors_[i * 3 + 1], colors_[i * 3 + 2]);
            }
        }

        for (int i = 0; i < num_cam_poses_; ++i) {
            if (cam_pose_optim_iter_[i] > 0) {
                const double *camera = parameters_.data() + camera_block_size_ * i;
                for (int j = 0; j < camera_block_size_; ++j) {
                    fprintf(fptr, "%.16g\n", camera[j]);
                }
            }
        }

        const double *points = pointBlockPtr();
        for (int i = 0; i < num_points_; ++i) {
            if (pts_optim_iter_[i] > 0) {
                const double *point = points + i * point_block_size_;
                for (int j = 0; j < point_block_size_; ++j) {
                    fprintf(fptr, "%.16g\n", point[j]);
                }
            }
        }
    }

    fclose(fptr);
}

// Write the problem to a PLY file for inspection in Meshlab or CloudCompare.
void BasicSfM::writeToPLYFile(const string &filename, bool write_unoptimized) const {
    ofstream of(filename.c_str());

    int num_cameras, num_points;

    if (write_unoptimized) {
        num_cameras = num_cam_poses_;
        num_points = num_points_;
    } else {
        num_cameras = 0;
        num_points = 0;
        for (int i = 0; i < num_cam_poses_; ++i)
            if (cam_pose_optim_iter_[i] > 0) num_cameras++;

        for (int i = 0; i < num_points_; ++i)
            if (pts_optim_iter_[i] > 0) num_points++;
    }

    of << "ply"
       << '\n' << "format ascii 1.0"
       << '\n' << "element vertex " << num_cameras + num_points
       << '\n' << "property float x"
       << '\n' << "property float y"
       << '\n' << "property float z"
       << '\n' << "property uchar red"
       << '\n' << "property uchar green"
       << '\n' << "property uchar blue"
       << '\n' << "end_header" << endl;

    bool write_colors = (colors_.size() == num_points_ * 3);
    if (write_unoptimized) {
        // Export extrinsic data (i.e. camera centers) as green points.
        double center[3];
        for (int i = 0; i < num_cam_poses_; ++i) {
            const double *camera = cameraBlockPtr(i);
            cam2center(camera, center);
            of << center[0] << ' ' << center[1] << ' ' << center[2]
               << " 0 255 0" << '\n';
        }

        // Export the structure (i.e. 3D Points) as white points.
        const double *points = pointBlockPtr();
        for (int i = 0; i < num_points_; ++i) {
            const double *point = points + i * point_block_size_;
            for (int j = 0; j < point_block_size_; ++j) {
                of << point[j] << ' ';
            }
            if (write_colors)
                of << int(colors_[3 * i]) << " " << int(colors_[3 * i + 1]) << " " << int(colors_[3 * i + 2]) << "\n";
            else
                of << "255 255 255\n";
        }
    } else {
        // Export extrinsic data (i.e. camera centers) as green points.
        double center[3];
        for (int i = 0; i < num_cam_poses_; ++i) {
            if (cam_pose_optim_iter_[i] > 0) {
                const double *camera = cameraBlockPtr(i);
                cam2center(camera, center);
                of << center[0] << ' ' << center[1] << ' ' << center[2]
                   << " 0 255 0" << '\n';
            }
        }

        // Export the structure (i.e. 3D Points) as white points.
        const double *points = pointBlockPtr();;
        for (int i = 0; i < num_points_; ++i) {
            if (pts_optim_iter_[i] > 0) {
                const double *point = points + i * point_block_size_;
                for (int j = 0; j < point_block_size_; ++j) {
                    of << point[j] << ' ';
                }
                if (write_colors)
                    of << int(colors_[3 * i]) << " " << int(colors_[3 * i + 1]) << " " << int(colors_[3 * i + 2])
                       << "\n";
                else
                    of << "255 255 255\n";
            }
        }
    }
    of.close();
}

/* c_{w,cam} = R_{cam}'*[0 0 0]' - R_{cam}'*t_{cam} -> c_{w,cam} = - R_{cam}'*t_{cam} */
void BasicSfM::cam2center(const double *camera, double *center) const {
    ConstVectorRef angle_axis_ref(camera, 3);

    Eigen::VectorXd inverse_rotation = -angle_axis_ref;
    ceres::AngleAxisRotatePoint(inverse_rotation.data(), camera + 3, center);
    VectorRef(center, 3) *= -1.0;
}

/* [0 0 0]' = R_{cam}*c_{w,cam} + t_{cam} -> t_{cam} = - R_{cam}*c_{w,cam} */
void BasicSfM::center2cam(const double *center, double *camera) const {
    ceres::AngleAxisRotatePoint(camera, center, camera + 3);
    VectorRef(camera + 3, 3) *= -1.0;
}


bool BasicSfM::checkCheiralityConstraint(int pos_idx, int pt_idx) {
    double *camera = cameraBlockPtr(pos_idx),
            *point = pointBlockPtr(pt_idx);

    double p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);

    // camera[5] is the z cooordinate wrt the camera at pose pose_idx
    p[2] += camera[5];
    return p[2] > 0;
}

void BasicSfM::printPose(int idx) const {
    const double *cam = cameraBlockPtr(idx);
    std::cout << "camera[" << idx << "]" << std::endl
              << "{" << std::endl
              << "\t r_vec : (" << cam[0] << ", " << cam[1] << ", " << cam[2] << ")" << std::endl
              << "\t t_vec : (" << cam[3] << ", " << cam[4] << ", " << cam[5] << ")" << std::endl;

    std::cout << "}" << std::endl;
}


void BasicSfM::printPointParams(int idx) const {
    const double *pt = pointBlockPtr(idx);
    std::cout << "point[" << idx << "] : (" << pt[0] << ", " << pt[1] << ", " << pt[2] << ")" << std::endl;
}


void BasicSfM::solve() {
    // Canonical camera so identity K
    cv::Mat_<double> intrinsics_matrix = cv::Mat_<double>::eye(3, 3);

    // For each camera pose, prepare a map that reports the pairs [point index, observation index]
    // This map is used to quickly retrieve the observation index given a 3D point index
    // For instance, to query if the camera pose with index i_cam observed the
    // 3D point with index i_pt, check if cam_observation[i_cam].find( i_pt ) is not cam_observation[i_cam].end(), i.e.,:
    // if(cam_observation[i_cam].find( i_pt ) != cam_observation[i_cam].end())  { .... }
    // In case of success, you can retrieve the observation index obs_id simply with:
    // obs_id = cam_observation[i_cam][i_pt]
    vector<map<int, int> > cam_observation(num_cam_poses_);
    for (int i_obs = 0; i_obs < num_observations_; i_obs++) {
        int i_cam = cam_pose_index_[i_obs], i_pt = point_index_[i_obs];
        cam_observation[i_cam][i_pt] = i_obs;
    }

    // Compute a (symmetric) num_cam_poses_ X num_cam_poses_ matrix
    // that counts the number of correspondences between pairs of camera poses
    Eigen::MatrixXi corr = Eigen::MatrixXi::Zero(num_cam_poses_, num_cam_poses_);

    for (int r = 0; r < num_cam_poses_; r++) {
        for (int c = r + 1; c < num_cam_poses_; c++) {
            int nc = 0;
            for (auto const &co_iter: cam_observation[r]) {
                if (cam_observation[c].find(co_iter.first) != cam_observation[c].end())
                    nc++;
            }
            corr(r, c) = nc;
        }
    }

    // num_cam_poses_ X num_cam_poses_ matrix to mask already tested seed pairs
    // already_tested_pair(r,c) == 0 -> not tested pair
    // already_tested_pair(r,c) != 0 -> already tested pair
    Eigen::MatrixXi already_tested_pair = Eigen::MatrixXi::Zero(num_cam_poses_, num_cam_poses_);

    // From the correspondence matrix corr select a seed pair of poses that both a) maximize the number of
    // correspondences b) define an Essential Matrix as geometrical model c) define a "good" translation
    bool seed_found = false;
    // Indices of the two camera poses that define the initial seed pair (default
    int ref_cam_pose_idx, new_cam_pose_idx;
    // Init R,t between the seed pair
    cv::Mat init_r_mat, init_r_vec, init_t_vec;

    std::vector<cv::Point2d> points0, points1;
    cv::Mat inlier_mask_E, inlier_mask_H;

    while (!seed_found) {
        points0.clear();
        points1.clear();

        int max_corr = -1;
        for (int r = 0; r < num_cam_poses_; r++) {
            for (int c = r + 1; c < num_cam_poses_; c++) {
                if (!already_tested_pair(r, c) && corr(r, c) > max_corr) {
                    max_corr = corr(r, c);
                    ref_cam_pose_idx = r;
                    new_cam_pose_idx = c;
                }
            }
        }

        if (max_corr < 0) {
            std::cout << "No seed pair found, exiting" << std::endl;
            return;
        }
        already_tested_pair(ref_cam_pose_idx, new_cam_pose_idx) = 1;

        for (auto const &co_iter: cam_observation[ref_cam_pose_idx]) {
            if (cam_observation[new_cam_pose_idx].find(co_iter.first) != cam_observation[new_cam_pose_idx].end()) {
                points0.emplace_back(observations_[2 * co_iter.second], observations_[2 * co_iter.second + 1]);
                points1.emplace_back(observations_[2 * cam_observation[new_cam_pose_idx][co_iter.first]],
                                     observations_[2 * cam_observation[new_cam_pose_idx][co_iter.first] + 1]);
            }
        }

        //////////////////////////// Code to be completed (3/6) /////////////////////////////////
        // Extract both Essential matrix E and Homograph matrix H.
        // Check that the number of inliers for the model E is higher than the number of
        // inliers for the model H (-> use inlier_mask_E and inlier_mask_H defined above <-).
        // As threshold in the functions to estimate both models, you may use 0.001 or similar.
        // If true, recover from E the initial rigid body transformation between ref_cam_pose_idx
        // and new_cam_pose_idx by using the cv::recoverPose() OpenCV function (use inlier_mask_E as input/output param)
        // Check if the recovered transformation is mainly given by a sideward motion, which is better than forward one.
        // Store the transformation into init_r_mat and  init_t_vec; defined above and set the seed_found flag to true
        // Otherwise, test a different [ref_cam_pose_idx, new_cam_pose_idx] pair (while( !seed_found ) loop)
        // The dummy condition here:
        if (true) seed_found = true;
        // should be replaced with the criteria described above
        /////////////////////////////////////////////////////////////////////////////////////////
        // Extract both Essential matrix E and Homography matrix H.
        cv::Mat E = cv::findEssentialMat(points0, points1, intrinsics_matrix, cv::RANSAC, 0.99, 1.0, inlier_mask_E);
        cv::Mat H = cv::findHomography(points0, points1, cv::RANSAC, 1.0, inlier_mask_H);

        // Count the number of inliers for both E and H.
        int num_inliers_E = cv::countNonZero(inlier_mask_E);
        int num_inliers_H = cv::countNonZero(inlier_mask_H);

        // Check that the number of inliers for E is higher than the number of inliers for H.
        if (num_inliers_E > num_inliers_H) {
            // Recover the initial rigid body transformation between ref_pose_idx and new_pose_idx
            // from the Essential matrix and set the seed_found flag to true.
            cv::recoverPose(E, points0, points1, intrinsics_matrix, init_r_mat, init_t, inlier_mask_E);
            seed_found = true;
        }
        /////////////////////////////////////////////////////////////////////////////////////////
    }

    // Initialize the first optimized poses, by integrating them into the registration
    // cam_pose_optim_iter_ and pts_optim_iter_ are simple mask vectors that define which camera poses and
    // which point positions have been already registered, specifically
    // if cam_pose_optim_iter_[pos_id] or pts_optim_iter_[id] are:
    // > 0 ---> The corresponding pose or point position has been already been estimated
    // == 0 ---> The corresponding pose or point position has not yet been estimated
    // == -1 ---> The corresponding pose or point position has been rejected due to e.g. outliers, etc...
    cam_pose_optim_iter_[ref_cam_pose_idx] = cam_pose_optim_iter_[new_cam_pose_idx] = 1;

    //Initialize the first RT wrt the reference position
    cv::Mat r_vec;
    // Recover the axis-angle rotation from init_r_mat
    cv::Rodrigues(init_r_mat, r_vec);

    // And update the parameters_ vector
    // First camera pose of the seed pair: just the identity transformation for now
    cv::Mat_<double> ref_rt_vec = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    initCamParams(ref_cam_pose_idx, ref_rt_vec, ref_rt_vec);
    // Second camera pose of the seed pair: the just recovered transformation
    initCamParams(new_cam_pose_idx, r_vec, init_t_vec);

    printPose(ref_cam_pose_idx);
    printPose(new_cam_pose_idx);

    // Triangulate the 3D points observed by both cameras
    cv::Mat_<double> proj_mat0 = cv::Mat_<double>::zeros(3, 4), proj_mat1(3, 4), hpoints4D;
    // First camera pose of the seed pair: just the identity transformation for now
    proj_mat0(0, 0) = proj_mat0(1, 1) = proj_mat0(2, 2) = 1.0;
    // Second camera pose of the seed pair: the just recovered transformation
    init_r_mat.copyTo(proj_mat1(cv::Rect(0, 0, 3, 3)));
    init_t_vec.copyTo(proj_mat1(cv::Rect(3, 0, 1, 3)));

    cv::triangulatePoints(proj_mat0, proj_mat1, points0, points1, hpoints4D);

    int r = 0;
    // Initialize the first optimized points
    for (auto const &co_iter: cam_observation[ref_cam_pose_idx]) {
        auto &pt_idx = co_iter.first;
        if (cam_observation[new_cam_pose_idx].find(pt_idx) !=
            cam_observation[new_cam_pose_idx].end()) {
            if (inlier_mask_E.at<unsigned char>(r)) {
                // Initialize the new point into the optimization
                double *pt = pointBlockPtr(pt_idx);

                // H-normalize the point
                pt[0] = hpoints4D.at<double>(0, r) / hpoints4D.at<double>(3, r);
                pt[1] = hpoints4D.at<double>(1, r) / hpoints4D.at<double>(3, r);
                pt[2] = hpoints4D.at<double>(2, r) / hpoints4D.at<double>(3, r);

                // Check the cheirality constraint
                if (pt[2] > 0.0) {
                    // Try to reproject the estimated 3D point in both cameras
                    cv::Mat_<double> pt_3d = (cv::Mat_<double>(3, 1) << pt[0], pt[1], pt[2]);

                    pt_3d = init_r_mat * pt_3d + init_t_vec;

                    cv::Point2d p0(pt[0] / pt[2], pt[1] / pt[2]),
                            p1(pt_3d(0, 0) / pt_3d(2, 0), pt_3d(1, 0) / pt_3d(2, 0));

                    // If the reprojection error is small, add the point to the reconstruction
                    if (cv::norm(p0 - points0[r]) < max_reproj_err_ && cv::norm(p1 - points1[r]) < max_reproj_err_)
                        pts_optim_iter_[pt_idx] = 1;
                    else
                        pts_optim_iter_[pt_idx] = -1;
                }
            }
        }
        r++;
    }

    // First bundle adjustment iteration: here we have only two camera poses, i.e., the seed pair
    bundleAdjustmentIter(new_cam_pose_idx);

    // Start to register new poses and observations...
    for (int iter = 1; iter < num_cam_poses_ - 1; iter++) {
        // The vector n_init_pts stores the number of points already being optimized
        // that are projected in a new camera pose when is optimized for the first time
        std::vector<int> n_init_pts(num_cam_poses_, 0);
        int max_init_pts = -1;

        // Select the new camera (new_cam_pose_idx) to be included in the optimization as the one that has
        // more projected points in common with the cameras already included in the optimization
        for (int i_p = 0; i_p < num_points_; i_p++) {
            if (pts_optim_iter_[i_p] > 0) // Point already added
            {
                for (int i_c = 0; i_c < num_cam_poses_; i_c++) {
                    if (cam_pose_optim_iter_[i_c] == 0 && // New camera pose not yet registered
                        cam_observation[i_c].find(i_p) !=
                        cam_observation[i_c].end()) // Dees camera i_c see this 3D point?
                        n_init_pts[i_c]++;
                }
            }
        }

        // Select the new camera pose that see more already registered 3D points
        for (int i_c = 0; i_c < num_cam_poses_; i_c++) {
            if (cam_pose_optim_iter_[i_c] == 0 && n_init_pts[i_c] > max_init_pts) {
                max_init_pts = n_init_pts[i_c];
                new_cam_pose_idx = i_c;
            }
        }

        // Now new_cam_pose_idx is the index of the next camera pose to be registered
        // Extract the 3D points that are projected in the new_cam_pose_idx-th pose and that are already registered
        std::vector<cv::Point3d> scene_pts;
        std::vector<cv::Point2d> img_pts;
        for (int i_p = 0; i_p < num_points_; i_p++) {
            if (pts_optim_iter_[i_p] > 0 &&
                cam_observation[new_cam_pose_idx].find(i_p) != cam_observation[new_cam_pose_idx].end()) {
                double *pt = pointBlockPtr(i_p);
                scene_pts.emplace_back(pt[0], pt[1], pt[2]);
                img_pts.emplace_back(observations_[cam_observation[new_cam_pose_idx][i_p] * 2],
                                     observations_[cam_observation[new_cam_pose_idx][i_p] * 2 + 1]);
            }
        }
        if (scene_pts.size() <= 3) {
            std::cout << "No other positions can be optimized, exiting" << std::endl;
            return;
        }

        // Estimate an initial R,t by using PnP + RANSAC
        cv::solvePnPRansac(scene_pts, img_pts, intrinsics_matrix, cv::Mat(),
                           init_r_vec, init_t_vec, false, 100, max_reproj_err_);
        // ... and add to the pool of optimized camera positions
        initCamParams(new_cam_pose_idx, init_r_vec, init_t_vec);
        cam_pose_optim_iter_[new_cam_pose_idx] = 1;

        // Extract the new points that, thanks to the new camera, are going to be optimized
        int n_new_pts = 0;
        std::vector<cv::Point2d> points0(1), points1(1);
        cv::Mat_<double> proj_mat0(3, 4), proj_mat1(3, 4), hpoints4D;
        for (int cam_idx = 0; cam_idx < num_cam_poses_; cam_idx++) {
            if (cam_pose_optim_iter_[cam_idx] > 0) {
                for (auto const &co_iter: cam_observation[cam_idx]) {
                    auto &pt_idx = co_iter.first;
                    if (pts_optim_iter_[pt_idx] == 0 &&
                        cam_observation[new_cam_pose_idx].find(pt_idx) != cam_observation[new_cam_pose_idx].end()) {
                        double *cam0_data = cameraBlockPtr(new_cam_pose_idx),
                                *cam1_data = cameraBlockPtr(cam_idx);

                        //////////////////////////// Code to be completed (4/6) /////////////////////////////////
                        // Triangulate the 3D point with index pt_idx by using the observation of this point in the
                        // camera poses with indices new_cam_pose_idx and cam_idx. The pointers cam0_data and cam1_data
                        // point to the 6D pose blocks for these inside the parameters vector (e.g.,
                        // cam0_data[0], cam0_data[1], cam0_data[2] hold the axis-angle representation fo the rotation of the
                        // camera with index new_cam_pose_idx.
                        // Use the OpenCV cv::triangulatePoints() function, remembering to check the cheirality constraint
                        // for both cameras
                        // In case of success (cheirality constrant satisfied) execute the following instructions (decomment and
                        // cut&paste):

                        // n_new_pts++;
                        // pts_optim_iter_[pt_idx] = 1;
                        // double *pt = pointBlockPtr(pt_idx);
                        // pt[0] = /*X coordinate of the estimated point */;
                        // pt[1] = /*X coordinate of the estimated point */;
                        // pt[2] = /*X coordinate of the estimated point */;
                        /////////////////////////////////////////////////////////////////////////////////////////
                        // Task 4: Triangulate the 3D point with index pt_idx using the observations of this point in the camera poses.
                        cv::Mat cam0_proj_matrix(3, 4, CV_64F, cam0_data);
                        cv::Mat cam1_proj_matrix(3, 4, CV_64F, cam1_data);
                        cv::Mat points4D;

                        cv::triangulatePoints(cam0_proj_matrix, cam1_proj_matrix, points0, points1, points4D);

                        // Check the cheirality constraint for both cameras
                        cv::Mat point3D = points4D.col(0) / points4D.at<double>(3, 0);
                        double z_cam0 = cam0_proj_matrix.row(2).dot(point3D);
                        double z_cam1 = cam1_proj_matrix.row(2).dot(point3D);

                        if (z_cam0 > 0 && z_cam1 > 0) {
                            // Cheirality constraint satisfied
                            n_new_pts++;
                            pts_optim_iter_[pt_idx] = 1;
                            double *pt = pointBlockPtr(pt_idx);
                            pt[0] = point3D.at<double>(0);
                            pt[1] = point3D.at<double>(1);
                            pt[2] = point3D.at<double>(2);
                        }
                        /////////////////////////////////////////////////////////////////////////////////////////
                    }
                }
            }
        }

        cout << "ADDED " << n_new_pts << " new points" << endl;

        cout << "Using " << iter + 2 << " over " << num_cam_poses_ << " cameras" << endl;
        for (int i = 0; i < int(cam_pose_optim_iter_.size()); i++)
            cout << int(cam_pose_optim_iter_[i]) << " ";
        cout << endl;

        // Execute an iteration of bundle adjustment
        bundleAdjustmentIter(new_cam_pose_idx);

        Eigen::Vector3d vol_min = Eigen::Vector3d::Constant((std::numeric_limits<double>::max())),
                vol_max = Eigen::Vector3d::Constant((-std::numeric_limits<double>::max()));
        for (int i_c = 0; i_c < num_cam_poses_; i_c++) {
            if (cam_pose_optim_iter_[i_c]) {
                double *camera = cameraBlockPtr(i_c);
                if (camera[3] > vol_max(0)) vol_max(0) = camera[3];
                if (camera[4] > vol_max(1)) vol_max(1) = camera[4];
                if (camera[5] > vol_max(2)) vol_max(2) = camera[5];
                if (camera[3] < vol_min(0)) vol_min(0) = camera[3];
                if (camera[4] < vol_min(1)) vol_min(1) = camera[4];
                if (camera[5] < vol_min(2)) vol_min(2) = camera[5];
            }
        }

        double max_dist = 5 * (vol_max - vol_min).norm();
        if (max_dist < 10.0) max_dist = 10.0;

        double *pts = parameters_.data() + num_cam_poses_ * camera_block_size_;
        for (int i = 0; i < num_points_; i++) {
            if (pts_optim_iter_[i] > 0 &&
                (fabs(pts[i * point_block_size_]) > max_dist ||
                 fabs(pts[i * point_block_size_ + 1]) > max_dist ||
                 fabs(pts[i * point_block_size_ + 2]) > max_dist)) {
                pts_optim_iter_[i] = -1;
            }
        }
    }
}

void BasicSfM::initCamParams(int new_pose_idx, cv::Mat r_vec, cv::Mat t_vec) {
    double *camera = cameraBlockPtr(new_pose_idx);

    cv::Mat_<double> r_vec_d(r_vec), t_vec_d(t_vec);
    for (int r = 0; r < 3; r++) {
        camera[r] = r_vec_d(r, 0);
        camera[r + 3] = t_vec_d(r, 0);
    }
}

void BasicSfM::bundleAdjustmentIter(int new_cam_idx) {
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;
    options.max_num_iterations = 200;

    std::vector<double> bck_parameters;

    bool keep_optimize = true;

    // Global optimization
    while (keep_optimize) {
        bck_parameters = parameters_;
        ceres::Problem problem;
        ceres::Solver::Summary summary;

        // For each observation....
        for (int i_obs = 0; i_obs < num_observations_; i_obs++) {
            // check if this observation has bem already registered (both checking camera pose and point pose)
            if (cam_pose_optim_iter_[cam_pose_index_[i_obs]] > 0 && pts_optim_iter_[point_index_[i_obs]] > 0) {
                //////////////////////////// Code to be completed (6/6) /////////////////////////////////
                //... in case, add a residual block inside the Ceres solver problem.
                // You should define a suitable functor (i.e., see the ReprojectionError struct at the
                // beginning of this file)
                // You may try a Cauchy loss function with parameters, say, 2*max_reproj_err_
                // Remember that the parameter blocks are stored starting from the
                // parameters_.data() double* pointer.
                // The camera position blocks have size (camera_block_size_) of 6 elements,
                // while the point position blocks have size (point_block_size_) of 3 elements.
                //////////////////////////////////////////////////////////////////////////////////
                // Create a residual block using the ReprojectionError cost function
                ceres::CostFunction *cost_function = ReprojectionError::Create(observed_x[i_obs], observed_y[i_obs]);

                // Define a Cauchy loss function with the given parameters
                ceres::LossFunction *loss_function = new ceres::CauchyLoss(2 * max_reproj_err_);

                // Get pointers to the camera and point parameter blocks
                double *camera_params = &parameters_[camera_block_size_ * cam_pose_index_[i_obs]];
                double *point_params = &parameters_[camera_block_size_ * num_cameras_ + point_block_size_ * point_index_[i_obs]];

                // Add the residual block to the Ceres problem
                problem.AddResidualBlock(cost_function, loss_function, camera_params, point_params);
                /////////////////////////////////////////////////////////////////////////////////////////
            }
        }

        Solve(options, &problem, &summary);

        // WARNING Here poor optimization ... :(
        // CHeck the cheirality constraint
        int n_cheirality_violation = 0;
        for (int i_obs = 0; i_obs < num_observations_; i_obs++) {
            if (cam_pose_optim_iter_[cam_pose_index_[i_obs]] > 0 &&
                pts_optim_iter_[point_index_[i_obs]] == 1 &&
                !checkCheiralityConstraint(cam_pose_index_[i_obs], point_index_[i_obs])) {
                // Penalize the point..
                pts_optim_iter_[point_index_[i_obs]] -= 2;
                n_cheirality_violation++;
            }
        }

        int n_outliers;
        if (n_cheirality_violation > max_outliers_) {
            std::cout << "****************** OPTIM CHEIRALITY VIOLATION for " << n_cheirality_violation
                      << " points : redoing optim!!" << std::endl;
            parameters_ = bck_parameters;
        } else if ((n_outliers = rejectOuliers()) > max_outliers_) {
            std::cout << "****************** OPTIM FOUND " << n_outliers << " OUTLIERS : redoing optim!!" << std::endl;
            parameters_ = bck_parameters;
        } else
            keep_optimize = false;
    }

    printPose(new_cam_idx);
}

int BasicSfM::rejectOuliers() {
    int num_ouliers = 0;
    for (int i_obs = 0; i_obs < num_observations_; i_obs++) {
        if (cam_pose_optim_iter_[cam_pose_index_[i_obs]] > 0 && pts_optim_iter_[point_index_[i_obs]] > 0) {
            double *camera = cameraBlockPtr(cam_pose_index_[i_obs]),
                    *point = pointBlockPtr(point_index_[i_obs]),
                    *observation = observations_.data() + (i_obs * 2);

            double p[3];
            ceres::AngleAxisRotatePoint(camera, point, p);

            // camera[3,4,5] are the translation.
            p[0] += camera[3];
            p[1] += camera[4];
            p[2] += camera[5];

            double predicted_x = p[0] / p[2];
            double predicted_y = p[1] / p[2];

            if (fabs(predicted_x - observation[0]) > max_reproj_err_ ||
                fabs(predicted_y - observation[1]) > max_reproj_err_) {
                // Penalize the point
                pts_optim_iter_[point_index_[i_obs]] -= 2;
                num_ouliers++;
            }
        }
    }
    return num_ouliers;
}
