#ifndef TRANSFORMPROCESS_HPP
#define TRANSFORMPROCESS_HPP

#include <opencv2/opencv.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/features/narf.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/shot.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/range_image/range_image.h>

class Tf
{
public:

static Eigen::Matrix4f
computeTfByPos7( std::vector<double> _pos_fr, std::vector<double> _pos_to )
{
    /// pos7 = [qw qx qy qz tx ty tz]
    /// qw = cos(angle/2)
    /// [qx qy qz] = [axis_x axis_y axis_z]*sin(angle/2)
    /// The pos here is the position of the camera in the word, so the inverse is needed.
    Eigen::Quaternionf q1( _pos_fr[0], -_pos_fr[1], -_pos_fr[2], -_pos_fr[3] );//world to camera1
    Eigen::Quaternionf q2( _pos_to[0], -_pos_to[1], -_pos_to[2], -_pos_to[3] );//world to camera2
    Eigen::Matrix3f R1 = q1.toRotationMatrix();
    Eigen::Matrix3f R2 = q2.toRotationMatrix();
    Eigen::Matrix3f R1_inv = R1.inverse();
    Eigen::Matrix3f R2_inv = R2.inverse();
    Eigen::Vector3f T1_inv( _pos_fr[4], _pos_fr[5], _pos_fr[6] );
    Eigen::Vector3f T2_inv( _pos_to[4], _pos_to[5], _pos_to[6] );
    Eigen::Vector3f T1 = -R1*T1_inv;
    Eigen::Vector3f T2 = -R2*T2_inv;

    Eigen::MatrixXf tf(R2*R1_inv);
    Eigen::Vector3f T = R2*T1_inv + T2;
    tf.conservativeResize(4,4);
                                     tf(0,3) = T[0];
                                     tf(1,3) = T[1];
                                     tf(2,3) = T[2];
    tf(3,0) = tf(3,1) = tf(3,2) = 0; tf(3,3) = 1;
    return tf;
}

template<class PointT> static bool
regeisterCloudPFH( pcl::PointCloud<PointT> cloud_src, pcl::PointCloud<PointT> cloud_tgt, Eigen::Matrix4f &tf )
{
    const uint min_res    = 15;//mm
    const uint norm_r     = 75;
    const uint feature_r  = 200;
    const uint SAC_Thresh = 15;

    /// 1. reprocess
    //remove NAN-Points
    std::vector<int> indices1,indices2;
    pcl::removeNaNFromPointCloud (cloud_src, cloud_src, indices1);
    pcl::removeNaNFromPointCloud (cloud_tgt, cloud_tgt, indices2);
    //Downsampling
    pcl::PointCloud<PointT> ds_src;
    pcl::PointCloud<PointT> ds_tgt;
    pcl::VoxelGrid<PointT> grid;
    grid.setLeafSize (min_res, min_res, min_res);//mm
    grid.setInputCloud ( cloud_src.makeShared());
    grid.filter (ds_src);
    grid.setInputCloud ( cloud_tgt.makeShared() );
    grid.filter (ds_tgt);
    // Normal-Estimation
    pcl::PointCloud<pcl::Normal>::Ptr norm_src (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr norm_tgt (new pcl::PointCloud<pcl::Normal>);
    boost::shared_ptr<pcl::search::KdTree<PointT> > tree_src ( new pcl::search::KdTree<PointT> );
    boost::shared_ptr<pcl::search::KdTree<PointT> > tree_tgt ( new pcl::search::KdTree<PointT> );
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud ( ds_src.makeShared() );
    ne.setSearchSurface ( cloud_src.makeShared() );
    ne.setSearchMethod ( tree_src );
    ne.setRadiusSearch ( norm_r );//mm
    ne.compute (*norm_src);
    ne.setInputCloud (ds_tgt.makeShared());
    ne.setSearchSurface (cloud_tgt.makeShared());
    ne.setSearchMethod (tree_tgt);
    ne.setRadiusSearch ( norm_r );
    ne.compute (*norm_tgt);

    /// 2. Keypoints NARF
    pcl::RangeImage range_src;
    pcl::RangeImage range_tgt;
    //Range Image
    float angularResolution = (float) (  0.2f * (M_PI/180.0f));  //   0.5 degree in radians
    float maxAngleWidth     = (float) (360.0f * (M_PI/180.0f));  // 360.0 degree in radians
    float maxAngleHeight    = (float) (180.0f * (M_PI/180.0f));  // 180.0 degree in radians
    Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
    pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
    float noiseLevel = 0.00;
    float minRange = 0.0f;
    int borderSize = 1;
    range_src.createFromPointCloud (cloud_src, angularResolution, maxAngleWidth, maxAngleHeight, sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);
    range_tgt.createFromPointCloud (cloud_tgt, angularResolution, maxAngleWidth, maxAngleHeight, sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);
    //Extract NARF-Keypoints
    pcl::RangeImageBorderExtractor range_image_ba;
    float support_size = min_res; //mm
    pcl::NarfKeypoint narf_keypoint_src (&range_image_ba);
    narf_keypoint_src.setRangeImage (&range_src);
    narf_keypoint_src.getParameters ().support_size = support_size;
    pcl::PointCloud<int> keypoints_ind_src;
    narf_keypoint_src.compute (keypoints_ind_src);
    pcl::NarfKeypoint narf_keypoint_tgt (&range_image_ba);
    narf_keypoint_tgt.setRangeImage (&range_tgt);
    narf_keypoint_tgt.getParameters ().support_size = support_size;
    pcl::PointCloud<int> keypoints_ind_tgt;
    narf_keypoint_tgt.compute (keypoints_ind_tgt);
    //get Keypoints as cloud
    pcl::PointCloud<PointT> keypoints_src;
    pcl::PointCloud<PointT> keypoints_tgt;
    keypoints_src.width = keypoints_ind_src.points.size();
    keypoints_src.height = 1;
    keypoints_src.is_dense = false;
    keypoints_src.points.resize (keypoints_src.width * keypoints_src.height);
    keypoints_tgt.width = keypoints_ind_tgt.points.size();
    keypoints_tgt.height = 1;
    keypoints_tgt.is_dense = false;
    keypoints_tgt.points.resize (keypoints_tgt.width * keypoints_tgt.height);
    for (size_t i = 0; i < keypoints_ind_src.points.size(); i++)
    {
        const int &ind_count = keypoints_ind_src.points[i];
        keypoints_src.points[i].x = range_src.points[ind_count].x;
        keypoints_src.points[i].y = range_src.points[ind_count].y;
        keypoints_src.points[i].z = range_src.points[ind_count].z;
    }
    for (size_t i = 0; i < keypoints_ind_tgt.points.size(); i++)
    {
        const int &ind_count = keypoints_ind_tgt.points[i];
        keypoints_tgt.points[i].x = range_tgt.points[ind_count].x;
        keypoints_tgt.points[i].y = range_tgt.points[ind_count].y;
        keypoints_tgt.points[i].z = range_tgt.points[ind_count].z;
    }
    std::cout << "NARF keypoints num=" << keypoints_src.size() << " " << keypoints_tgt.size() << std::endl;
    /// 3. Feature-Descriptor
    pcl::PFHEstimation<PointT, pcl::Normal, pcl::PFHSignature125> pfh_est_src;
    boost::shared_ptr<pcl::search::KdTree<PointT> > tree_pfh_src ( new pcl::search::KdTree<PointT> );
    pfh_est_src.setSearchMethod (tree_pfh_src);
    pfh_est_src.setRadiusSearch ( feature_r );//mm
    pfh_est_src.setSearchSurface (ds_src.makeShared());
    pfh_est_src.setInputNormals (norm_src);
    pfh_est_src.setInputCloud (keypoints_src.makeShared());
    pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_src (new pcl::PointCloud<pcl::PFHSignature125>);
    pfh_est_src.compute (*pfh_src);
    pcl::PFHEstimation<PointT, pcl::Normal, pcl::PFHSignature125> pfh_est_tgt;
    boost::shared_ptr<pcl::search::KdTree<PointT> > tree_pfh_tgt ( new pcl::search::KdTree<PointT> );
    pfh_est_tgt.setSearchMethod (tree_pfh_tgt);
    pfh_est_tgt.setRadiusSearch ( feature_r );//mm
    pfh_est_tgt.setSearchSurface (ds_tgt.makeShared());
    pfh_est_tgt.setInputNormals (norm_tgt);
    pfh_est_tgt.setInputCloud (keypoints_tgt.makeShared());
    pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_tgt (new pcl::PointCloud<pcl::PFHSignature125>);
    pfh_est_tgt.compute (*pfh_tgt);

    /// 4. Match
    Eigen::Matrix4f transformation;
    // Correspondence Estimation
    pcl::registration::CorrespondenceEstimation<pcl::PFHSignature125, pcl::PFHSignature125> corEst;
    corEst.setInputSource (pfh_src);
    corEst.setInputTarget (pfh_tgt);
    boost::shared_ptr<pcl::Correspondences> cor_all_ptr (new pcl::Correspondences);
    corEst.determineCorrespondences (*cor_all_ptr);
    //SAC
    pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> sac;
    boost::shared_ptr<pcl::Correspondences> cor_inliers_ptr (new pcl::Correspondences);
    sac.setInputSource (keypoints_src.makeShared());
    sac.setInputTarget (keypoints_tgt.makeShared());
    sac.setInlierThreshold ( SAC_Thresh );
    sac.setMaximumIterations (100);
    sac.setInputCorrespondences (cor_all_ptr);
    sac.getCorrespondences (*cor_inliers_ptr);
    transformation = sac.getBestTransformation();

    /// end
//    pcl::PointCloud<PointT> cloud_tmp;
//    pcl::transformPointCloud (*cloud_src, *cloud_tmp, transformation);
    tf = transformation;

    return true;
}

template<class Pointxyz> static std::vector<cv::DMatch>
getMatchsByTransform( const std::vector<Pointxyz> &_from, const std::vector<Pointxyz> &_to, const Eigen::Matrix4f &_trans, const bool _cross_check = false, const double dis_thresh=20  )
{
    std::vector<cv::DMatch> matches;     //new matches
    matches.reserve( _from.size() );
    for(int i=0; i<_from.size(); i++ )
    {
        Eigen::Vector4f pt_from( _from[i].x, _from[i].y, _from[i].z, 1 );
        pt_from = _trans*pt_from;
        int min_dist = INFINITY;
        int min_id_to = -1;
        for(int j=0; j<_to.size(); j++ )
        {
            Eigen::Vector4f pt_to( _to[j].x, _to[j].y, _to[j].z, 1 );
            double temp_dist = (pt_from-pt_to).norm();
            if( temp_dist < min_dist )
            {
                min_dist = temp_dist;
                min_id_to = j;
            }
         }
        if( min_id_to != -1 && min_dist < dis_thresh)//mm
            matches.push_back( cv::DMatch( i, min_id_to, min_dist) );
    }
    if( ! _cross_check )
        return matches;
    ///cross check
    Eigen::Matrix4f trans_inv = _trans.inverse();
    for( std::vector<cv::DMatch>::iterator p_match = matches.begin(); p_match != matches.end();  )
    {
        Eigen::Vector4f pt_to( _to[p_match->trainIdx].x, _to[p_match->trainIdx].y, _to[p_match->trainIdx].z, 1 );
        pt_to = trans_inv * pt_to;
        uint min_dist = p_match->distance;
        bool reject = false;
        for(int id_from=0; id_from<_from.size(); id_from++ )
        {
            Eigen::Vector4f pt_from( _from[id_from].x, _from[id_from].y, _from[id_from].z, 1 );

            if( id_from != p_match->queryIdx )
            if( min_dist >= (pt_from-pt_to).norm() )
            {
                reject = true;
                break;
            }
        }
        if( reject )
            p_match = matches.erase( p_match );
        else
            p_match ++;
    }
    return matches;
}

template<class Pointxyz> static cv::Mat
extractFPFH( std::vector<cv::KeyPoint> _keypoints, const boost::shared_ptr<pcl::PointCloud<Pointxyz> > _cloud, timeval *_time_start )
{

    const uint min_res    = 15;//mm
    const uint norm_r     = 75;
    const uint feature_r  = 200;

    /// 1. preprocess
    //remove NAN-Points
    std::vector<int> indices;
    pcl::PointCloud<Pointxyz> cloud_ds;
    pcl::removeNaNFromPointCloud ( *_cloud, cloud_ds, indices);
    //Downsampling
    pcl::VoxelGrid<Pointxyz> grid;
    grid.setLeafSize ( min_res, min_res, min_res );//mm
    grid.setInputCloud ( cloud_ds.makeShared() );
    grid.filter ( cloud_ds );
    // Normal-Estimation
    pcl::PointCloud<pcl::Normal>::Ptr norms (new pcl::PointCloud<pcl::Normal>);
    boost::shared_ptr<pcl::search::KdTree<Pointxyz> > tree ( new pcl::search::KdTree<Pointxyz> );
    pcl::NormalEstimation<Pointxyz, pcl::Normal> ne;
    ne.setInputCloud ( cloud_ds.makeShared() );
    ne.setSearchSurface ( _cloud );
    ne.setSearchMethod ( tree );
    ne.setRadiusSearch ( norm_r );//mm
    ne.compute (*norms);

    /// 2. Keypoints 3D
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr key_points3( new pcl::PointCloud<pcl::PointXYZRGB>(1,_keypoints.size()) );
    std::vector<cv::KeyPoint> keypoints_filtered_;
    keypoints_filtered_.reserve( _keypoints.size() );
    int valid_cnt = 0;
    for( int i = 0; i < (int)_keypoints.size(); i++)
    {
        const pcl::PointXYZRGB &pt = _cloud->at(_keypoints[i].pt.x,_keypoints[i].pt.y);
        if( pt.z!=0 && std::isfinite(pt.z) )
        {
            key_points3->at(valid_cnt) = pt;
            keypoints_filtered_.push_back( _keypoints[i] );
            valid_cnt ++;
        }
    }
    key_points3->resize( valid_cnt );
    _keypoints = keypoints_filtered_;
    /// 3. Feature-Descriptor
    gettimeofday(_time_start,NULL);
    pcl::FPFHEstimation<Pointxyz, pcl::Normal, pcl::FPFHSignature33> pfh_est;
    boost::shared_ptr<pcl::search::KdTree<Pointxyz> > tree_pfh ( new pcl::search::KdTree<Pointxyz> );
    pfh_est.setSearchMethod (tree_pfh);
    pfh_est.setRadiusSearch ( feature_r );//mm
    pfh_est.setSearchSurface (cloud_ds.makeShared());
    pfh_est.setInputNormals (norms);
    pfh_est.setInputCloud ( key_points3 );
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_signature (new pcl::PointCloud<pcl::FPFHSignature33>);
    pfh_est.compute (*fpfh_signature);

    cv::Mat descriptors_( fpfh_signature->size(), 33, CV_32F );
    for( int i = 0; i < fpfh_signature->size(); i++)
        memcpy( descriptors_.data+descriptors_.step[0]*i, fpfh_signature->at(i).histogram, sizeof(pcl::FPFHSignature33) );
    return descriptors_;
}

template<class Pointxyzrgb> static cv::Mat
extractCSHOT( std::vector<cv::KeyPoint> _keypoints, const boost::shared_ptr<pcl::PointCloud<Pointxyzrgb> > _cloud, timeval *_time_start )
{

    const uint min_res    = 15;//mm
    const uint norm_r     = 75;
    const uint feature_r  = 200;

    /// 1. preprocess
    //remove NAN-Points
    std::vector<int> indices;
    pcl::PointCloud<Pointxyzrgb> cloud_ds;
    pcl::removeNaNFromPointCloud ( *_cloud, cloud_ds, indices);
    //Downsampling
    pcl::VoxelGrid<Pointxyzrgb> grid;
    grid.setLeafSize ( min_res, min_res, min_res );//mm
    grid.setInputCloud ( cloud_ds.makeShared() );
    grid.filter ( cloud_ds );
    // Normal-Estimation
    pcl::PointCloud<pcl::Normal>::Ptr norms (new pcl::PointCloud<pcl::Normal>);
    boost::shared_ptr<pcl::search::KdTree<Pointxyzrgb> > tree ( new pcl::search::KdTree<Pointxyzrgb> );
    pcl::NormalEstimation<Pointxyzrgb, pcl::Normal> ne;
    ne.setInputCloud ( cloud_ds.makeShared() );
    ne.setSearchSurface ( _cloud );
    ne.setSearchMethod ( tree );
    ne.setRadiusSearch ( norm_r );//mm
    ne.compute (*norms);

    /// 2. Keypoints 3D
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr key_points3( new pcl::PointCloud<pcl::PointXYZRGB>(1,_keypoints.size()) );
    std::vector<cv::KeyPoint> keypoints_filtered_;
    keypoints_filtered_.reserve( _keypoints.size() );
    int valid_cnt = 0;
    for( int i = 0; i < (int)_keypoints.size(); i++)
    {
        const pcl::PointXYZRGB &pt = _cloud->at(_keypoints[i].pt.x,_keypoints[i].pt.y);
        if( pt.z!=0 && std::isfinite(pt.z) )
        {
            key_points3->at(valid_cnt) = pt;
            keypoints_filtered_.push_back( _keypoints[i] );
            valid_cnt ++;
        }
    }
    key_points3->resize( valid_cnt );
    _keypoints = keypoints_filtered_;
    /// 3. Feature-Descriptor
    gettimeofday(_time_start,NULL);
    pcl::SHOTColorEstimation<Pointxyzrgb,pcl::Normal,pcl::SHOT1344> cshot_est;
    boost::shared_ptr<pcl::search::KdTree<Pointxyzrgb> > tree_pfh ( new pcl::search::KdTree<Pointxyzrgb> );
    cshot_est.setSearchMethod (tree_pfh);
    cshot_est.setRadiusSearch ( feature_r );//mm
    cshot_est.setSearchSurface (cloud_ds.makeShared());
    cshot_est.setInputNormals (norms);
    cshot_est.setInputCloud ( key_points3 );
    pcl::PointCloud<pcl::SHOT1344>::Ptr fpfh_signature (new pcl::PointCloud<pcl::SHOT1344>);
    cshot_est.compute (*fpfh_signature);

    cv::Mat descriptors_( fpfh_signature->size(), 1344, CV_32F );// pcl::SHOT1344 contains 1344+9 float, where the ReferenceFrame occupies 9
    for( int i = 0; i < fpfh_signature->size(); i++)
        memcpy( descriptors_.data+descriptors_.step[0]*i, fpfh_signature->at(i).descriptor, 1344*sizeof(float) );
    return descriptors_;
}

template<class Pointxyz> static std::vector<cv::DMatch>
filterMatchsByTransform( const std::vector<Pointxyz> &_from, const std::vector<Pointxyz> &_to, const std::vector<cv::DMatch> _matches, const Eigen::Matrix4f &_trans, const double dis_thresh=50  )
{
    std::vector<cv::DMatch> matches;     //new matches
    matches.reserve( _matches.size() );
    for( std::vector<cv::DMatch>::const_iterator p_match = _matches.begin(); p_match != _matches.end(); p_match++ )
    {
        Eigen::Vector4f pt_from( _from[p_match->queryIdx].x, _from[p_match->queryIdx].y, _from[p_match->queryIdx].z, 1 );
        Eigen::Vector4f pt_to  ( _to  [p_match->trainIdx].x, _to  [p_match->trainIdx].y, _to  [p_match->trainIdx].z, 1 );
        if( (_trans*pt_from-pt_to).norm() <= dis_thresh )
            matches.push_back( *p_match );
    }
    return matches;
}
template<class Pointxyz> static Eigen::Matrix4f
getTransformByMatchs( const std::vector<Pointxyz> &_from, const std::vector<Pointxyz> &_to, const std::vector<cv::DMatch> _matches, const double dis_thresh=50  )
{
    pcl::PointCloud<pcl::PointXYZ> from( _from.size(), 1 );
    pcl::PointCloud<pcl::PointXYZ> to  (   _to.size(), 1 );
    boost::shared_ptr<pcl::Correspondences>    matches( new pcl::Correspondences );
    for( int i=0; i<_from.size(); i++ )
    {
        from.at(i).x = _from[i].x;
        from.at(i).y = _from[i].y;
        from.at(i).z = _from[i].z;
    }
    for( int i=0; i<_to.size(); i++ )
    {
        to.at(i).x = _to[i].x;
        to.at(i).y = _to[i].y;
        to.at(i).z = _to[i].z;
    }
    matches->reserve(_matches.size() );
    for( int i=0; i<_matches.size(); i++ )
    {
        pcl::PointXYZ &pt_from = from.at(_matches[i].queryIdx);
        if( pt_from.z==0 || std::isinf(pt_from.z) ) continue;
        pcl::PointXYZ &pt_to   = to  .at(_matches[i].trainIdx);
        if( pt_to.z==0 || std::isinf(pt_to.z) ) continue;
        matches->push_back( pcl::Correspondence( _matches[i].queryIdx, _matches[i].trainIdx,_matches[i].distance) );
    }
    Eigen::Matrix4f transformation;
    pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> sac;
    boost::shared_ptr<pcl::Correspondences> cor_inliers_ptr (new pcl::Correspondences);
    sac.setInputSource (from.makeShared());
    sac.setInputTarget (to.makeShared());
    sac.setInlierThreshold ( dis_thresh );
    sac.setMaximumIterations (1000);
    sac.setInputCorrespondences ( matches );
    sac.getCorrespondences (*cor_inliers_ptr);//necessary step
    transformation = sac.getBestTransformation();
    if( fabs( transformation(3,3) ) < 0.1 )///invalid tansform
    {
        transformation.setZero();
        transformation(0,0) = transformation(1,1) = transformation(2,2) = transformation(3,3) = 1;
    }
    return transformation;
}

static std::vector<cv::DMatch>
getMatchsByhomography( const std::vector<cv::KeyPoint> &_from, const std::vector<cv::KeyPoint> &_to, const cv::Mat& homography, const bool _cross_check = false )
{
    assert( homography.type() == CV_64F );
    std::vector<cv::DMatch> matches;     //new matches
    matches.reserve( _from.size() );
    std::vector<cv::Point2f> pts_from( _from.size() );
    for(int i=0; i<_from.size(); i++ )
        pts_from[i] = _from[i].pt;
    std::vector<cv::Point2f> pts_from_t;
    cv::perspectiveTransform( pts_from, pts_from_t, homography);

    for(int i=0; i<_from.size(); i++ )
    {
        cv::Point2f &pt_from = pts_from_t[i];
        std::cout << _from[i].pt << "->" << pt_from <<std::endl;
        int min_dist = INFINITY;
        int min_id_to = -1;
        for(int j=0; j<_to.size(); j++ )
        {
            const cv::Point2f & pt_to = _to[j].pt;
            double temp_dist = hypot( pt_from.x-pt_to.x, pt_from.y-pt_to.y );
            if( temp_dist < min_dist )
            {
                min_dist = temp_dist;
                min_id_to = j;
            }
         }
        if( min_id_to != -1 )//mm
            matches.push_back( cv::DMatch( i, min_id_to, min_dist) );
    }
    if( ! _cross_check )
        return matches;
    ///cross check
    cv::Mat homography_R = cv::Mat(homography, cv::Rect(0,0,2,2));
    cv::Mat homo_inv_R = homography_R.inv();
    double  homo_inv_T[2] = { -homography.at<double>(0,2), -homography.at<double>(1,2) };
    for( std::vector<cv::DMatch>::iterator p_match = matches.begin(); p_match != matches.end();  )
    {
        const cv::Point2f & pt = _to[p_match->trainIdx].pt;
        cv::Point2f pt_to;
        pt_to.x = pt.x * homo_inv_R.at<double>(0,0) + pt.y * homo_inv_R.at<double>(0,1) + homo_inv_T[0];
        pt_to.y = pt.x * homo_inv_R.at<double>(1,0) + pt.y * homo_inv_R.at<double>(1,1) + homo_inv_T[1];
        uint min_dist = p_match->distance;
        bool reject = false;
        for(int id_from=0; id_from<_from.size(); id_from++ )
        {
            const cv::Point2f & pt_from = _from[id_from].pt;
            if( id_from != p_match->queryIdx )
            if( min_dist >= hypot( pt_from.x-pt_to.x, pt_from.y-pt_to.y ) )
            {
                reject = true;
                break;
            }
        }
        if( reject )
            p_match = matches.erase( p_match );
        else
            p_match ++;
    }
    return matches;
}

static std::vector<cv::DMatch>
refineMatchesWithHomography( const std::vector<cv::KeyPoint>& _src, const std::vector<cv::KeyPoint>& _dst,
        float _project_thresh, const std::vector<cv::DMatch> _matches, cv::Mat& _homography )
{
    const int minNumberMatchesAllowed = 8;
    std::vector<cv::DMatch> inliers;
    if (_matches.size() < minNumberMatchesAllowed)
        return inliers;
    // Prepare data for cv::findHomography
    std::vector<cv::Point2f> srcPoints(_matches.size());
    std::vector<cv::Point2f> dstPoints(_matches.size());
    for (size_t i = 0; i < _matches.size(); i++)
    {
        srcPoints[i] = _src[_matches[i].trainIdx].pt;
        dstPoints[i] = _dst[_matches[i].queryIdx].pt;
    }
    // Find homography matrix and get inliers mask
    std::vector<unsigned char> inliersMask(srcPoints.size());
    _homography = cv::findHomography(srcPoints,
                                    dstPoints,
                                    CV_FM_RANSAC,
                                    _project_thresh,
                                    inliersMask);
    for (size_t i=0; i<inliersMask.size(); i++)
    {
        if (inliersMask[i])
            inliers.push_back(_matches[i]);
    }
    return inliers;
}
};
#endif // TRANSFORMPROCESS_HPP
