/**********************************************************************
 * Software License Agreement (BSD License)
 *
 *  NuBot workshop, NUDT China - http://nubot.trustie.com and https://github.com/nubot-nudt
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 **********************************************************************/
#ifndef COLOR_CODED_DEPTH_FEATURE_H
#define COLOR_CODED_DEPTH_FEATURE_H
//PCL includes
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/io/pcd_io.h>

//OpenCV includes
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include <string>
#include <sstream>

/// \brief The ColorCoding class
/// Encode a RGB color to a code by indexing the R, G and B values.
/// The code may be a BYTE, a WORD, or a DWORD which is defined by the \variable method_ during initialization.
class ColorCoding
{
public:
    enum METHOD
    {
        HAMMING_HSV422 = 0,
        HAMMING_HSV655,
        HAMMING_GRAY8
    };
    class CodeBase
    {
    public:
        uint32_t rgb2code_[16][16][16];
        std::vector<uint32_t> code2rgba_;
        uint32_t INVALID_CODE;
        template<typename code_t> bool initByHSV( const std::vector<code_t> &H_code, const std::vector<code_t> &S_code, const std::vector<code_t> &V_code, const code_t &EMPITY_H );
        CodeBase(){}
        virtual ~CodeBase(){}
    };
    class HSV422Code : public CodeBase
    {
    public:
        typedef uchar code_t;
        HSV422Code();
    };
    class HSV655Code : public CodeBase
    {
    public:
        typedef u_int16_t code_t;
        HSV655Code();
    };
    boost::shared_ptr<CodeBase> coder_;
public:
    METHOD method_;
    int code_type_;//CV_8U, CV_16U, etc.
    ColorCoding( const METHOD &_method=HAMMING_HSV655 );
    uint encode( void* p_code, const uchar _r, const uchar _g, const uchar _b) const;///return the sizeof the color code
    uint invalidCode( void* p_code ) const;///return the sizeof the color code
    uchar rgb2IntCode(const uchar _r, const uchar _g, const uchar _b, const uchar _bit_length=8) const;
    uint machCode(void* _code1, void* _code2, const uint _cells=1) const;
    uint32_t decode(const void * _code) const;///return rgba formate color
};

/// \brief The PerspectiveInvariantFeature class
/// The main class of PIFT feture
///
/// Usage example:
/// \code
///     cv::Mat rgb, depth;
///     std::vector<cv::KeyPoint> keypoints;
///     cv::Mat descriptors;
///
///     PerspectiveInvariantFeature pift;
///     cv::BRISK BRISK;
///     BRISK.detect( rgb, keypoints );
///     pift.prepareFrame( rgb, depth );
///     descriptors = pift.process( keypoints );
///
///     cv::Mat descriptors_last;
///     std::vector<cv::DMatch> matches;
///     cv::Ptr<cv::DescriptorMatcher> p_matcher = new PIFTMatcher( pift.color_encoder_.method_, true );
///     p_matcher->match(descriptors_last,descriptors,matches);
///
///     pift.restore_descriptor( descriptors );
///     cv::imshow( "img_matches_restore", pift.features_restore_ );
/// \endcode
class PerspectiveInvariantFeature
{
public:
    enum DESCRIPTOR_TYPE
    {
        D_TYPE_BEEHIVE = 0,
        D_TYPE_ANNULAR,
        D_TYPE_HISTOGRAM,
        D_TYPE_CUBE3,
        D_TYPE_SURF  ,
        D_TYPE_BRIEF  ,
        D_TYPE_ORB    ,
        D_TYPE_BRISK ,
        D_TYPE_EMPTY= 255
    }patch_type_;
    cv::Mat rgb_img_;
    cv::Mat depth_16U_;                             //unit:mm
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_;  //unit:mm
    pcl::PointCloud<pcl::Normal>::Ptr normals_;     //unit:mm
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree_;     //the Kd-tree for cloud_
    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB,pcl::Normal>::Ptr normal_est_;
    double camera_fx, camera_fy,camera_cx,camera_cy;//camera model parameters
    double _1_camera_fx, _1_camera_fy;              //calculate the 1/camera_fx and 1/camera_fy off-line
    int _256_camera_fx, _256_camera_fy;             //calculate the 256/camera_fx and 256/camera_fy off-line

    PerspectiveInvariantFeature(const uint _max_keypoints=200, const uint _spatial_radius=100, const DESCRIPTOR_TYPE _method=D_TYPE_BEEHIVE);
    bool prepareFrame( const cv::Mat _rgb_image, const cv::Mat _depth_16U );

    uint calcPt6d(const cv::Point& _pt, cv::Point3f &_pt3d, cv::Vec4d &_plane_coef, double &_plane_err );
    uint warpPerspectivePatch( const cv::Point3f &_pt3d, const cv::Vec4d _plane_coef, cv::Mat &_feature_patch, const uint &SPATIAL_RADIUS );
    uint sampleCubeEvenly( const cv::Point3f &_pt3d, const cv::Vec4d _plane_coef, std::vector<cv::Vec3i> &_cube, const uint &SPATIAL_RADIUS, const double &_main_angle=0 );
    std::vector<cv::Vec3i> PyramidCube(const std::vector<cv::Vec3i> &_cube_hi_res );
    uint calcFeatureDir(const cv::Mat& _feature_patch, cv::Point2d &_main_dir, const double& _dense_thresh=0.2);
    uint generateFeatureCode(const cv::Mat& _feature_patch, const cv::Point2d &_main_dir, cv::Mat& _color_code, const double& _dense_thresh=0.2);
    uint generateFeatureCode_hov(const cv::Mat& _feature_patch, cv::Mat &_color_code, const uchar& _method=0);
    cv::Mat process(std::vector<cv::KeyPoint> &m_keypoints);
    cv::Mat processFPFH(std::vector<cv::KeyPoint> &m_keypoints, const uint &SPATIAL_RADIUS=70 );
    uint MAX_KEYPOINTS;
    std::vector<cv::Point3f> keypoints_3D_;         //The spatial coordinates of the keypoints. unit:mm
    cv::Mat descriptors_;
    uint height;
    uint width;
    const uint PATCH_SIZE;          //The feature patches are normalized to the same size
    uint SPATIAL_R;                 //The main parameter. unit:mm

private:
    std::vector<cv::KeyPoint> keypoints_filtered_;

public:
    ColorCoding color_encoder_;

    /// \brief The PatchMask class
    /// The mask is a image of the same size of the feature patch.
    /// The mask is segmented to several cells.
    /// A cell of pixels are used together to generate one union color.
    /// The color of a cell is then generated to be a color code.
    /// All the codes form the descriptor of the feature patch.
    class PatchMask
    {
    public:
        uint ANGLE_RES;     // The angular resolution of the patch mask
        uint TOTAL_CELLS;   // How many cells in the patch mask, which means the length of the generated descriptor
        const uchar BAD_CELL;
        PatchMask( const uint& _angle_res=12, const uint& _total_cells=64 ) : ANGLE_RES(_angle_res), TOTAL_CELLS(_total_cells), BAD_CELL(0xff) {}
        virtual const uchar getCellID(const uint& _xp, const uint& _yp, const uint& _main_angle_deg = 0 ) const = 0;
        virtual ~PatchMask(){}
    };
    class AnnularMask : public PatchMask
    {
    public:
        AnnularMask(const uint &_patch_size, const uint &_angle_res, const uint &_dist_res);
        const uint DIST_RES;        // The radial resolution
        std::vector<cv::Point2d> DIR_PER_CELL;//The directional vector of each cell, used to calcualte the main direction of the feature patch
        const uchar getCellID(const uint& _xp, const uint& _yp, const uint& _main_angle_deg = 0 ) const;
    private:
        std::vector<double> TAN_ANGLE_THRESH;//calculate the tangent value of each direction off-line
        cv::Mat mask_;// The amount of the cells in an annular mask is ANGLE_RES*DIST_RES+1, with the IDs of 0~ANGLE_RES*DIST_RES, and the ivalide pixels is set to 0xff.
    };
    class BeehiveMask : public PatchMask
    {
    public:
        BeehiveMask(const uint &_patch_size, const uint &_layers=4, const uint &_angle_res=6*6);
        const uchar getCellID(const uint& _xp, const uint& _yp, const uint& _main_angle_deg = 0 ) const;
    private:
        std::vector<cv::Mat> masks_;// The amount of the cells in a beehive mask is 61 with 0~60, and the mask is rotated off-line and stored in a vetor.
        cv::Mat rotateCellID_;
    };
    boost::shared_ptr<PatchMask> patch_mask_;
    boost::shared_ptr<AnnularMask> annular_mask_;
    boost::shared_ptr<BeehiveMask> beehive_mask_;
    ///////////////////////////////////////////////////
    class CubeMask
    {
    public:
        const int SIDE_LENGTH;
        enum CELL_FLAG
        {
            CELL_SHADOW = -1,
            CELL_SPACE = 0,
            CELL_SOLID = 1
        };

        CubeMask();
        const uchar getCellID(const uint& _xp, const uint& _yp, const uint& _zp, const uint& _main_angle_deg = 0 ) const;
    private:
        std::vector<uint32_t> cube3d_;

    };

    //The parameters used for visualization
    bool DRAW_IMAG;             //If restore the feature patches from the descriptors
    cv::Mat rgb_show_;          //The color image with keypoints
    cv::Mat features_show_;     //The pespective invariant feature patches
    cv::Mat features_restore_;  //The restored feature patches
    bool restore_descriptor(const cv::Mat& _descriptor);
};

/// \brief The PIFTMatcher class
/// Derived from the cv::DescriptorMatcher Class
/// The distance of two descriptors is the Hamming-distance of the two binary, except for the Invalid Color Code.
class CV_EXPORTS_W PIFTMatcher : public cv::DescriptorMatcher
{
public:
    CV_WRAP PIFTMatcher( const ColorCoding::METHOD &_method, const bool &_cross_check=true ):color_encoder_(_method), cross_check_(_cross_check){}
    virtual ~PIFTMatcher(){}
    CV_WRAP void matchDescriptors( const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors, CV_OUT std::vector<cv::DMatch>& matches ) const;
    virtual bool isMaskSupported() const { return false; }
    virtual cv::Ptr<cv::DescriptorMatcher> clone( bool emptyTrainData=false ) const;

protected:
    virtual void knnMatchImpl( const cv::Mat& queryDescriptors, std::vector<std::vector<cv::DMatch> >& matches, int k,
           const std::vector<cv::Mat>& masks=std::vector<cv::Mat>(), bool compactResult=false );
    virtual void radiusMatchImpl( const cv::Mat& queryDescriptors, std::vector<std::vector<cv::DMatch> >& matches, float maxDistance,
           const std::vector<cv::Mat>& masks=std::vector<cv::Mat>(), bool compactResult=false );

protected:
    ColorCoding color_encoder_;
    bool cross_check_;
};
#endif
