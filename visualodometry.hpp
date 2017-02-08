#ifndef VISUAL_ODOMETRY_HPP
#define VISUAL_ODOMETRY_HPP

#include <boost/function.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include "transformprocess.hpp"

/////////////////////////////////////////
/// \brief The VisualOdometry class
///
class VisualOdometery
{
public:
    VisualOdometery();
    ////////////////////////////////////////////////////////////////////////
    /// \brief setMatcher   Customize the feature match method. Note that the process will not work without this setting
    /// \param _matcher     The cv::BFMatcher as an example
    /// \return             void
    ///
    void setMatcher( cv::Ptr<cv::DescriptorMatcher> _matcher ){ p_matcher = _matcher; }
    void setMatcher( const cv::DescriptorMatcher &_matcher ){ p_matcher = _matcher.clone(); }
    ////////////////////////////////////////////////////////////////////////
    /// \brief setInnerDistance
    /// \param _inner
    ///
    void setInnerDistance(const double &_inner ){ inner_dist = _inner; }
    bool process(const cv::Mat &_descriptors, const std::vector<cv::Point3f> &_keypoints3, Eigen::Matrix4f &_trans);
    void clearMap(){ map_size_=0; }

protected:
    const size_t MAP_SIZE;
    size_t       map_size_;
    cv::Mat                     descriptors_;
    std::vector<cv::Point3f>    keypoints3_;

//    typedef uchar Weight;
//    typedef std::pair<Weight,uint> WeightCnt;   // Note the sort() method is valid
//    std::vector<Weight>         weights_;       // range in [ 1, DEFAULT_W*2 ] and 0 means invalid
//    std::list<WeightCnt>        weights_cnt_;   // always sorted by weight
//    const Weight                DEFAULT_W;
    double                      inner_dist;
    uint oldest_data_id;
    cv::Ptr<cv::DescriptorMatcher> p_matcher;

};

#endif
