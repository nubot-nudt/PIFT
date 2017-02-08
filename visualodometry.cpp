#include "visualodometry.hpp"

VisualOdometery::VisualOdometery()
                : MAP_SIZE(1000), map_size_(0), inner_dist(50)//, DEFAULT_W(10)
{
//    weights_.reserve( MAP_SIZE );
    keypoints3_.reserve( MAP_SIZE );
//    weights_cnt_.push_back( WeightCnt(0,MAP_SIZE) );
}
bool VisualOdometery::process(const cv::Mat &_descriptors, const std::vector<cv::Point3f> &_keypoints3, Eigen::Matrix4f &_trans)
{
    if( !p_matcher )
        return false;
    Eigen::Matrix4f eye;
    eye.setZero();
    eye(0,0) = eye(1,1) = eye(2,2) = eye(3,3) = 1;

    /// Note that the query-data/frame-from is the map, and the train-data/frame-to is the input-frame
    /// to get the incremental tansform
    cv::Mat data_to( _descriptors, cv::Rect(0,0,_descriptors.cols,std::min(_descriptors.rows,(int)MAP_SIZE)) );
    std::vector<cv::Point3f> pt3d_to( _keypoints3.begin(), _keypoints3.begin()+std::min(_keypoints3.size(),MAP_SIZE) );
    assert( data_to.rows == (int)pt3d_to.size() );

    /// 0. Init map (first frame)
    if( map_size_==0 )
    {
        descriptors_.create( MAP_SIZE, data_to.cols, data_to.type() );
        memcpy( descriptors_.data, data_to.data, data_to.rows*data_to.step[0] );
//        weights_.resize( data_to.rows, DEFAULT_W );
//        weights_.resize( MAP_SIZE, 0 );
//        weights_cnt_.push_back( WeightCnt(DEFAULT_W,data_to.rows) );
//        weights_cnt_.begin()->second -= data_to.rows;
        keypoints3_ = pt3d_to;
        keypoints3_.resize( MAP_SIZE );
        _trans = eye;
        oldest_data_id = data_to.rows;
        map_size_ = data_to.rows;
        return true;
    }

    cv::Mat data_from( descriptors_, cv::Rect(0,0,descriptors_.cols,map_size_) );
    std::vector<cv::Point3f> pt3d_from( keypoints3_.begin(), keypoints3_.begin()+map_size_ );

    /// 2. Odometry
    std::vector<cv::DMatch> matches;
    p_matcher->match( data_from, data_to, matches );
    Eigen::Matrix4f trans = Tf::getTransformByMatchs( pt3d_from, pt3d_to, matches, inner_dist );

    if( trans!=eye )///succeed
    {
        matches = Tf::getMatchsByTransform( pt3d_from, pt3d_to, trans, true, inner_dist);
        /// Compute accurate transform
        Eigen::Matrix4f trans2 = Tf::getTransformByMatchs( pt3d_from, pt3d_to, matches, inner_dist/2 );
        if( trans2!=eye ) trans = trans2;
        _trans = trans;
        /// 3. Update map
        /// Record the current data into the map: [oldest_data_id, oldest_data_id+data_to.rows]
        const size_t & step = data_to.step[0];
        if( data_to.rows <= MAP_SIZE - oldest_data_id )
        {
            std::memcpy( descriptors_.data+oldest_data_id*step, data_to.data, data_to.rows*step );
            std::memcpy( &keypoints3_[oldest_data_id],         &pt3d_to[0],   data_to.rows*sizeof(cv::Point3f) );
        }
        else
        {
            std::memcpy( descriptors_.data+oldest_data_id*step, data_to.data, (MAP_SIZE-oldest_data_id)*step );
            std::memcpy( &keypoints3_[oldest_data_id],     &pt3d_to[0],  (MAP_SIZE-oldest_data_id)*sizeof(cv::Point3f) );
            std::memcpy( descriptors_.data, data_to.data+(MAP_SIZE-oldest_data_id)*step, (data_to.rows+oldest_data_id-MAP_SIZE)*step );
            std::memcpy( &keypoints3_[0],        &pt3d_to[MAP_SIZE-oldest_data_id],      (data_to.rows+oldest_data_id-MAP_SIZE)*sizeof(cv::Point3f) );
        }
        if( map_size_<MAP_SIZE )
        {
            map_size_ += data_to.rows;
            if( map_size_ > MAP_SIZE )
                map_size_ = MAP_SIZE;
        }
        /// transform the rest of the map to the current pos: [oldest_data_id-(map_size_-data_to.rows), oldest_data_id]
        const int start_id = oldest_data_id - (map_size_-data_to.rows);
        for(int i=start_id; i<oldest_data_id; i++)
        {
            if( i <    0     ) i += MAP_SIZE;
            if( i >=MAP_SIZE ) i -= MAP_SIZE;
            cv::Point3f &pt = pt3d_from[i];
            Eigen::Vector4f pt4( pt.x, pt.y, pt.z, 1 );
            pt4 = trans * pt4;
            pt.x=pt4[0], pt.y=pt4[1], pt.z=pt4[2];
        }
        /// the oldest data now
        oldest_data_id += data_to.rows;
        if( oldest_data_id >= MAP_SIZE )
            oldest_data_id -= MAP_SIZE;
    }
    else///failed
    {
        /// do nothing
        /// It's users authority to restart the odometry or not.
        _trans = eye;
        return false;
    }

    return true;

}
