#include <iostream>
#include <bitset>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "pift.h"
#include "datasetplayer.hpp"
#include "visualodometry.hpp"
#include "transformprocess.hpp"


/// uint of millimeter in this experiment
int main(int argc, char **argv)
{
#if 0 //washington dataset
    // scene4 scene4_high
    std::string dataset_path( "rgbd-data/scene4_high/" );
    std::string save_path( "rgbd-data/scene4_high/results/" );         // Where to save the experiment results?
    const double DEPTH_SCALE    = 0.1;      //   -> mm
    const double GT_SCALE       = 1000;     // m -> mm
    DataSetPlayer data_reader;
    data_reader.setColorPath( dataset_path + "rgb" );
    data_reader.setDepthPath( dataset_path + "depth" );
    data_reader.setGroundTruthPath( dataset_path + "pos.txt" );
    data_reader.setFormat( DataSetPlayer::QwQxQyQzTxTyTz, DEPTH_SCALE, GT_SCALE );
    const int  SENSOR_ERR = 25;
    const uint   spatial_radius     = 117;  //unit: mm 117 70
    const uint   START_FRAME    = 0;
    const uint   TOTAL_FRAMES   = 200;
#else // tum datasets
    std::string dataset_path( "/media/nubot22/QQ320G/[data-sets]/vision.in.tum.de_data_datasets/Handheld SLAM/rgbd_dataset_freiburg2_desk/" );
    std::string save_path( "rgbd-data/desk/results/" );         // Where to save the experiment results?
    const double DEPTH_SCALE    = 0.2;      //   -> mm
    const double GT_SCALE       = 1000;     // m -> mm
    const double SYNC_TIME_ERR  = 0.017;    // valid only when the dataset contains time-stamp (INFINITY as default)
    DataSetPlayer data_reader;
    data_reader.setColorPath( dataset_path + "rgb" );
    data_reader.setDepthPath( dataset_path + "depth" );
    data_reader.setGroundTruthPath( dataset_path + "pos.txt" );
    data_reader.setFormat( DataSetPlayer::TimeTxTyTzQxQyQzQw, DEPTH_SCALE, GT_SCALE, SYNC_TIME_ERR );
    const int  SENSOR_ERR = 40;     // The ground truth is not accurate at all in this dataset
    const uint   spatial_radius     = 117;  //unit: mm 117 70
    const uint   START_FRAME    = 1400;
    const uint   TOTAL_FRAMES   = 400;
//    data_reader.setDownsample( 1.0 );
#endif


    const bool SAME_KEY_POINTS = true;
    const bool INCREMENTAL = false;
    const bool GT_INSTEAD_OF_RANSAC = false;// Use the ground truth to check the crrect match instead of using RANSAC
    const uint max_key_ponts = 1000;

    if( SAME_KEY_POINTS )
        std::cerr << "Warning! All Features Use The SAME_KEY_POINTS." << std::endl;
    else
        std::cerr << "Warning! All Features NOT Use The SAME_KEY_POINTS." << std::endl;
    if( INCREMENTAL )
        std::cerr << "Warning! Use incremental transform." << std::endl;
    else
        std::cerr << "Warning! Use absolute transform." << std::endl;

    cv::initModule_nonfree();
    pcl::console::setVerbosityLevel( pcl::console::L_ALWAYS );
    enum DescriptorType
    {
        DT_SURF,
        DT_BRIEF,
        DT_BRISK,
        DT_FREAK,
        DT_ORB,
        DT_FPFH,
        DT_PIFT,
        DT_DEFAULT
    }descriptor_type = DT_PIFT;
    PerspectiveInvariantFeature::DESCRIPTOR_TYPE PIFT_TYPE = PerspectiveInvariantFeature::D_TYPE_BEEHIVE;

    for(int i=1;i<argc;++i)
    {
        if(strcmp(argv[i], "pift") == 0 )
            descriptor_type = DT_PIFT;
        if(strcmp(argv[i], "pift-annular") == 0 )
            descriptor_type = DT_PIFT,
            PIFT_TYPE = PerspectiveInvariantFeature::D_TYPE_ANNULAR;
        if(strcmp(argv[i], "pift-brisk") == 0 )
            descriptor_type = DT_PIFT,
            PIFT_TYPE = PerspectiveInvariantFeature::D_TYPE_BRISK;
        if(strcmp(argv[i], "pift-surf") == 0 )
            descriptor_type = DT_PIFT,
            PIFT_TYPE = PerspectiveInvariantFeature::D_TYPE_SURF;
        if(strcmp(argv[i], "pift-orb") == 0 )
            descriptor_type = DT_PIFT,
            PIFT_TYPE = PerspectiveInvariantFeature::D_TYPE_ORB;
        if(strcmp(argv[i], "pift-hist") == 0 )
            descriptor_type = DT_PIFT,
            PIFT_TYPE = PerspectiveInvariantFeature::D_TYPE_HISTOGRAM;
        else if(strcmp(argv[i], "orb") == 0 || strcmp(argv[i], "ORB") ==0 )
            descriptor_type = DT_ORB;
        else if(strcmp(argv[i], "brief") == 0 || strcmp(argv[i], "BRIEF") ==0 )
            descriptor_type = DT_BRIEF;
        else if(strcmp(argv[i], "surf") == 0 || strcmp(argv[i], "SURF") ==0 )
            descriptor_type = DT_SURF;
        else if(strcmp(argv[i], "brisk") == 0 || strcmp(argv[i], "BRIEF") ==0 )
            descriptor_type = DT_BRISK;
        else if(strcmp(argv[i], "freak") == 0 || strcmp(argv[i], "FREAK") ==0 )
            descriptor_type = DT_FREAK;
        else if(strcmp(argv[i], "fpfh") == 0 )
            descriptor_type = DT_FPFH;
    }

//    cv::StarDetector StarDetector;
//    cv::FastFeatureDetector FastFeatureDetector;
//    cv::GFTTDetector GFTTDetector( 300, 0.03, 2 );

    cv::SURF Surf;
    cv::BriefDescriptorExtractor BriefExtractor;
    cv::BRISK BRISK;
    cv::FREAK FreakExtractor;
    cv::ORB ORB;
    PerspectiveInvariantFeature pift( max_key_ponts, spatial_radius, PIFT_TYPE );

    cv::Mat rgb_last;
    pcl::PointCloud<pcl::PointXYZRGB> cloud_last;
    std::vector<cv::KeyPoint> keypoints_last;
    std::vector<cv::Point3f> keypoints3_last;
    cv::Mat descriptors_last;
    cv::Mat features_show_last;
    cv::Mat features_restore_last;
    bool RESTORE_PATCH = descriptor_type == DT_PIFT
                    && ( pift.patch_type_==PerspectiveInvariantFeature::D_TYPE_BEEHIVE
                      || pift.patch_type_==PerspectiveInvariantFeature::D_TYPE_ANNULAR
                      || pift.patch_type_==PerspectiveInvariantFeature::D_TYPE_CUBE3 );
    pcl::visualization::PCLVisualizer viewer_pift("pift key points");
    viewer_pift.setCameraPosition(0,-1500,-1000, 0,0,2000, 0,-1,0);
    pcl::visualization::PCLVisualizer viewer_noob("noob key points");
    viewer_noob.setCameraPosition(0,-1500,-1000, 0,0,2000, 0,-1,0);
//    pcl::visualization::PCLVisualizer viewer_norm("normals");
//    viewer_norm.setCameraPosition(0,-1500,-1000, 0,0,2000, 0,-1,0);
    pcl::visualization::PCLVisualizer viewer_reg("reg_tf");
    viewer_reg .setCameraPosition(0,-1500,-1000, 0,0,2000, 0,-1,0);

    cv::VideoWriter match_video;
    cv::VideoWriter reg_video;
    cv::VideoCapture comp_video;
    std::ofstream result_file( (save_path+"result.txt").c_str() );
    std::stringstream result_str;
    result_str << std::left <<std::setw(5) << "cnt"
               << std::left <<std::setw(15) << "time"
               << std::right<<std::setw(10) << "Tx"
               << std::right<<std::setw(10) << "Ty"
               << std::right<<std::setw(10) << "Tz"
               << std::right<<std::setw(10) << "Qx"
               << std::right<<std::setw(10) << "Qy"
               << std::right<<std::setw(10) << "Qz"
               << std::right<<std::setw(10) << "Qw"
               << std::right<<std::setw(10) << "Precision"
               << std::right<<std::setw(10) << "Recall"
               << std::right<<std::setw(10) << "TransErr"
               << std::right<<std::setw(10) << "CameTrans";
    result_file << result_str.str() << std::endl;
    std::cout   << result_str.str() << std::endl;
    result_str.str("");
    result_str.setf( std::ios::fixed );
    result_str.precision(4);


    cv::Ptr<cv::DescriptorMatcher> p_matcher;
    switch (descriptor_type)
    {
    case DT_PIFT:
        if( pift.patch_type_ == PerspectiveInvariantFeature::D_TYPE_SURF
          ||pift.patch_type_ == PerspectiveInvariantFeature::D_TYPE_HISTOGRAM
          ||pift.descriptors_.type() == CV_32F
          ||pift.descriptors_.type() == CV_64F)
            p_matcher = new cv::BFMatcher(cv::NORM_L2, true);
        else
            p_matcher = new PIFTMatcher( pift.color_encoder_.method_, true );
        break;
    case DT_SURF:
    case DT_FPFH:
        p_matcher = new cv::BFMatcher(cv::NORM_L2, true);
        break;
    case DT_BRIEF:
    case DT_BRISK:
    case DT_FREAK:
    case DT_ORB:
        p_matcher = new cv::BFMatcher(cv::NORM_HAMMING, true);
        break;
    default:
        return -1;
        break;
    }

    VisualOdometery odometery;
    odometery.setMatcher( p_matcher );
    odometery.setInnerDistance( SENSOR_ERR );

    double time_stamp;
    cv::Mat rgb, depth;
    std::vector<double>        GT_Pos7(7);//stored as QwQxQyQzTxTyTz
    static std::vector<double> GT_Pos7_last(7);
    Eigen::Matrix4f            GT_transform;
    Eigen::Quaternionf         GT_PosR;
    Eigen::Vector3f            GT_PosT;
    if( START_FRAME > 1)
    for( int data_cnt=1; data_reader.getFrame( rgb, depth, GT_PosR, GT_PosT, &time_stamp ); data_cnt++ )
        if( data_cnt >= START_FRAME ) break;
    for( int data_cnt=1; data_reader.getFrame( rgb, depth, GT_PosR, GT_PosT, &time_stamp ) && data_cnt<=TOTAL_FRAMES; data_cnt++ )
    {
        GT_Pos7[0] = GT_PosR.w();//angle
        GT_Pos7[1] = GT_PosR.x();
        GT_Pos7[2] = GT_PosR.y();
        GT_Pos7[3] = GT_PosR.z();
        GT_Pos7[4] = GT_PosT[0];
        GT_Pos7[5] = GT_PosT[1];
        GT_Pos7[6] = GT_PosT[2];
        if( data_cnt>1 )
            GT_transform = Tf::computeTfByPos7( GT_Pos7_last, GT_Pos7 );

        //////////// 1. Extrct key points & descriptor///////////////
        std::vector<cv::KeyPoint> keypoints0;
        cv::Mat descriptors;
        cv::Mat mask = cv::Mat::zeros(rgb.rows,rgb.cols,CV_8UC1);
        const int &BORDER = 50;
        cv::Mat(mask,cv::Rect(BORDER, BORDER, rgb.cols-2*BORDER, rgb.rows-2*BORDER)).setTo(1);
        BRISK.detect( rgb, keypoints0, mask );

        pift.prepareFrame( rgb, depth );
        std::vector<cv::KeyPoint> keypoints;
        if( !SAME_KEY_POINTS && descriptor_type == DT_PIFT )
            keypoints = keypoints0;
        else
        {
            ///filter keypoints by depth
            keypoints.reserve( keypoints0.size() );
            for( int i = 0;  i < keypoints0.size(); ++i )
            {
                const pcl::PointXYZRGB &pt = pift.cloud_->at(keypoints0[i].pt.x,keypoints0[i].pt.y);
                if( pt.z!=0 && std::isfinite(pt.z) )
                    keypoints.push_back( keypoints0[i] );
            }
            if(  SAME_KEY_POINTS )///filter key points, resrve valid for all
            {
                Surf.compute(  rgb, keypoints, descriptors );
                ORB.compute( rgb, keypoints, descriptors );
                BriefExtractor.compute( rgb, keypoints, descriptors);
                BRISK.compute( rgb, keypoints, descriptors );
                FreakExtractor.compute( rgb, keypoints, descriptors );
                pift.process( keypoints );
            }
        }
        if( keypoints.size()<4 )
        {
            std::cerr << "Too few key points!!! = " << keypoints.size() << std::endl;
            continue;
        }

        switch (descriptor_type)
        {
        case DT_SURF:
            Surf.compute( rgb, keypoints, descriptors);
            break;
        case DT_ORB:
            ORB.compute( rgb, keypoints, descriptors);
            break;
        case DT_BRIEF:
            BriefExtractor.compute(rgb, keypoints, descriptors);
            break;
        case DT_BRISK:
            BRISK.compute(rgb, keypoints, descriptors);
            break;
        case DT_FREAK:
            FreakExtractor.compute(rgb, keypoints, descriptors);
            break;
        case DT_FPFH:
        {
            pift.prepareFrame( rgb, depth );
            descriptors = Tf::extractFPFH( keypoints, pift.cloud_ );
        }
            break;
        case DT_PIFT:
            pift.prepareFrame( rgb, depth );
            descriptors = pift.process( keypoints );
            pift.restore_descriptor( descriptors );
            break;
        default:
            pift.prepareFrame( rgb, depth );
            descriptors = pift.processFPFH( keypoints );
            break;
        }
        if( keypoints.size()<3 )
        {
            std::cerr << "Too few key points!!! = " << keypoints.size() << std::endl;
            continue;
        }

        std::vector<cv::Point3f> keypoints3;
        if( descriptor_type == DT_PIFT || SAME_KEY_POINTS )
            keypoints3 = pift.keypoints_3D_;
        else
        {
            keypoints3.reserve( keypoints.size() );
            for( int i = 0;  i < keypoints.size(); ++i )
            {
                const pcl::PointXYZRGB &pt = pift.cloud_->at(keypoints[i].pt.x,keypoints[i].pt.y);
                keypoints3.push_back( cv::Point3f(pt.x,pt.y,pt.z) );
            }
        }

        /// test Odometery
//        Eigen::Matrix4f trans_odom;
//        odometery.process( descriptors, keypoints3, trans_odom );

        /////////////////// 2. Keypoint match ///////////////////////////
        std::vector<cv::DMatch> matches;
        std::vector<cv::DMatch> matches_correct;//correct matches
        std::vector<cv::DMatch> matches_GT;     //ground truth matches
        Eigen::Matrix4f transform;
        if( data_cnt>1 )
        {
            /// 2.1 descriptor match
            p_matcher->match(descriptors_last,descriptors,matches);

            /// 2.2 calculate ground truth
            matches_GT = Tf::getMatchsByTransform( keypoints3_last, keypoints3, GT_transform, true, SENSOR_ERR);

            /// 2.3 calculate performance
            matches_correct = Tf::filterMatchsByTransform( keypoints3_last, keypoints3, matches, GT_transform, SENSOR_ERR );
            if( GT_INSTEAD_OF_RANSAC )
                transform = Tf::getTransformByMatchs( keypoints3_last, keypoints3, matches_correct, SENSOR_ERR );
            else
                transform = Tf::getTransformByMatchs( keypoints3_last, keypoints3, matches, SENSOR_ERR );
//            transform = trans_odom;// test Odometery

            Eigen::Vector3f pos_GT( GT_transform(0,3), GT_transform(1,3), GT_transform(2,3) );
            Eigen::Vector3f pos   ( transform   (0,3), transform   (1,3), transform   (2,3) );
            Eigen::MatrixXf R_cur( transform );
            R_cur.conservativeResize( 3, 3 );
            Eigen::Matrix3f R_cur1 = R_cur;
            Eigen::Quaternionf q_cur( R_cur1 );
            result_str << std::left <<std::setw(4)  << data_cnt << " "
                       << std::left <<std::setw(15) << time_stamp << " "
                       << std::right<<std::setw(10) << transform(0,3) << " "
                       << std::right<<std::setw(10) << transform(1,3) << " "
                       << std::right<<std::setw(10) << transform(2,3) << " "
                       << std::right<<std::setw(8) << q_cur.x() << " "
                       << std::right<<std::setw(8) << q_cur.y() << " "
                       << std::right<<std::setw(8) << q_cur.z() << " "
                       << std::right<<std::setw(8) << q_cur.w() << " "
                       << std::right<<std::setw(8) << (double)matches_correct.size()/matches.size() << " "
                       << std::right<<std::setw(8) << std::min(1.0,(double)matches_correct.size()/matches_GT.size()) << " "
                       << std::right<<std::setw(10) << (pos-pos_GT).norm() << " "
                       << std::right<<std::setw(10) << pos_GT.norm();
            result_file << result_str.str() << std::endl;
            if( pos.norm() == 0 )
                std::cout << "\033[33m" << result_str.str() << "\033[m" << std::endl;
            else
                std::cout   << result_str.str() << std::endl;
            result_str.str("");
        }

        ////////////////// 3. Draw Key Points /////////////////////////
        cv::Mat keypoints_show = rgb.clone();
        if( descriptor_type != DT_PIFT )
            cv::drawKeypoints( rgb, keypoints0, keypoints_show, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
        else
        {
            ///draw the same kepoints with the same color in different images/clouds
            pcl::PointCloud<pcl::PointXYZRGB> pift_cloud(keypoints.size(),1);
            pcl::PointCloud<pcl::PointXYZRGB> noob_cloud(keypoints0.size(),1);
            cv::RNG rng;
            pcl::RGB color_draw;
            int cnt_pift=0;
            int cnt_noob=0;
            for( int i = 0;  i < noob_cloud.size(); ++i )//draw original key points
            {
                color_draw.rgba = (uint32_t)rng.uniform(0x00000000,0x00ffffff);
                std::vector<cv::KeyPoint> cur_keypoint(1,keypoints0[i]);
                cv::drawKeypoints( rgb, cur_keypoint, keypoints_show, CV_RGB(color_draw.r,color_draw.g,color_draw.b), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG );
                const pcl::PointXYZRGB &noob_pt = pift.cloud_->at(keypoints0[i].pt.x,keypoints0[i].pt.y);
                if( noob_pt.z!=0 && std::isfinite(noob_pt.z) )
                {
                    noob_cloud.at(cnt_noob) = noob_pt;
                    noob_cloud.at(cnt_noob).rgba = color_draw.rgba;
                    cnt_noob++;
                }
                for( int j=0; j<keypoints.size(); j++)                    //draw corresponding PIFT points
                    if( keypoints[j].pt == keypoints0[i].pt && cnt_pift < pift_cloud.size() )
                    {
                        pift_cloud.at(cnt_pift).x = keypoints3[j].x;
                        pift_cloud.at(cnt_pift).y = keypoints3[j].y;
                        pift_cloud.at(cnt_pift).z = keypoints3[j].z;
                        pift_cloud.at(cnt_pift).rgba = color_draw.rgba;
                        cnt_pift++; break;
                    }
            }
            pift_cloud.resize(cnt_pift);
            noob_cloud.resize(cnt_noob);
            ///show all clouds
            if( data_cnt==1 )
            {
                viewer_pift.addPointCloud( pift.cloud_, "cloud0" );
                viewer_pift.addPointCloud( pift_cloud.makeShared(),"key_points" );
                viewer_pift.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud0");
                viewer_pift.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "key_points");
                viewer_noob.addPointCloud( pift.cloud_, "cloud0" );
                viewer_noob.addPointCloud( noob_cloud.makeShared(),"key_points" );
                viewer_noob.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud0");
                viewer_noob.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "key_points");
            }
            else
            {
                viewer_pift.updatePointCloud( pift.cloud_, "cloud0" );
                viewer_pift.updatePointCloud( pift_cloud.makeShared(),"key_points" );
                viewer_noob.updatePointCloud( pift.cloud_, "cloud0" );
                viewer_noob.updatePointCloud( noob_cloud.makeShared(),"key_points" );
            }
//            viewer_norm.removeAllPointClouds();
//            viewer_norm.addPointCloud( pift.cloud_, "cloud0" );
//            viewer_norm.addPointCloudNormals<pcl::PointXYZRGB,pcl::Normal>( pift.cloud_, pift.normals_, 20, 10, "normals" );
            cv::imshow("keypoints_pift",pift.rgb_show_);
        }
        cv::imshow("keypoints_noob",keypoints_show);
        ///show cloud_reg
        if( data_cnt > 1 )
        {
            Eigen::Matrix4f trans_inv = transform.inverse();
            pcl::PointCloud<pcl::PointXYZRGB> cloud_reg;
            pcl::transformPointCloud ( *pift.cloud_, cloud_reg, trans_inv );
            if( data_cnt ==2 )
            {
                viewer_reg.addPointCloud( cloud_last.makeShared(),"points" );
                viewer_reg.addPointCloud( cloud_reg.makeShared(),"reg_points" );
            }
            else
            {
                viewer_reg.updatePointCloud( cloud_last.makeShared(),"points" );
                viewer_reg.updatePointCloud( cloud_reg.makeShared(),"reg_points" );
            }
        }

        //////////////// 4. Draw maches //////////////
        cv::Mat img_matches;
        cv::Mat img_matches_correct;
        cv::Mat img_matches_patch;
        cv::Mat img_matches_patch_restore;
        cv::Mat img_matches_GT;
        if( data_cnt==1 )
        {
            cv::imshow("img_matches_patch"  ,pift.features_show_);
            cv::imshow("img_matches_restore",pift.features_restore_);
        }
        else
        {
            cv::drawMatches( rgb_last, keypoints_last, rgb, keypoints, matches,         img_matches);//, CV_RGB(255,  0,0) );
//            cv::drawMatches( rgb_last, keypoints_last, rgb, keypoints, matches_correct, img_matches, CV_RGB(  0,255,0), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG );
            cv::drawMatches( rgb_last, keypoints_last, rgb, keypoints, matches_correct, img_matches_correct );
            cv::drawMatches( rgb_last, keypoints_last, rgb, keypoints, matches_GT, img_matches_GT );
            std::vector<cv::KeyPoint> keypoints_last_temp = keypoints_last;
            std::vector<cv::KeyPoint> keypoints_temp = keypoints;
            for(int i=0; i<keypoints_last_temp.size(); i++)
            {
                keypoints_last_temp[i].pt.x = (i%10)*(features_show_last.cols/10)+features_show_last.cols/10/2+1;
                keypoints_last_temp[i].pt.y = (i/10)*(features_show_last.rows/10)+features_show_last.rows/10/2+1;
            }
            for(int i=0; i<keypoints_temp.size(); i++)
            {
                keypoints_temp[i].pt.x = (i%10)*(pift.features_show_.cols/10)+pift.features_show_.cols/10/2+1;
                keypoints_temp[i].pt.y = (i/10)*(pift.features_show_.rows/10)+pift.features_show_.rows/10/2+1;
            }
            if( descriptor_type == DT_PIFT )
            {
                cv::Mat temp_img, temp_img_last;
                cv::cvtColor( pift.features_show_, temp_img, CV_RGBA2RGB);
                cv::cvtColor( features_show_last, temp_img_last, CV_RGBA2RGB);
                cv::drawMatches( temp_img_last, keypoints_last_temp, temp_img, keypoints_temp, matches, img_matches_patch );
                cv::line( img_matches_patch, cv::Point(img_matches_patch.cols/2,0), cv::Point(img_matches_patch.cols/2,img_matches_patch.rows), CV_RGB(255,255,255), 3);
                for( int i=0; i<matches.size(); i++ )
                {
                    std::stringstream dist_str;
                    cv::Point pos_text;
                    dist_str << matches[i].distance;
                    pos_text = keypoints_last_temp[ matches[i].queryIdx ].pt;
                    cv::putText( img_matches_patch, dist_str.str(), pos_text, cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(255,0,0) );
                    pos_text = keypoints_temp[ matches[i].trainIdx ].pt;
                    pos_text.x += img_matches_patch.cols/2;
                    cv::putText( img_matches_patch, dist_str.str(), pos_text, cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(255,0,0) );
                }
                if( RESTORE_PATCH )
                {
                    cv::cvtColor( pift.features_restore_, temp_img, CV_RGBA2RGB);
                    cv::cvtColor( features_restore_last, temp_img_last, CV_RGBA2RGB);
                    cv::drawMatches( temp_img_last, keypoints_last_temp, temp_img, keypoints_temp, matches, img_matches_patch_restore );
                    cv::line( img_matches_patch_restore, cv::Point(img_matches_patch_restore.cols/2,0), cv::Point(img_matches_patch_restore.cols/2,img_matches_patch_restore.rows), CV_RGB(255,255,255), 3);
                    for( int i=0; i<matches.size(); i++ )
                    {
                        std::stringstream dist_str;
                        cv::Point pt_temp;
                        dist_str << matches[i].distance;
                        pt_temp = keypoints_last_temp[ matches[i].queryIdx ].pt;
                        cv::putText( img_matches_patch_restore, dist_str.str(), pt_temp, cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(255,0,0) );
                        pt_temp = keypoints_temp[ matches[i].trainIdx ].pt;
                        pt_temp.x += img_matches_patch_restore.cols/2;
                        cv::putText( img_matches_patch_restore, dist_str.str(), pt_temp, cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(255,0,0) );
                    }
                    cv::imshow("img_matches_restore",img_matches_patch_restore);
                }
                cv::imshow("img_matches_patch",img_matches_patch);
            }
            std::stringstream info_str;
//            info_str << data_cnt << std::setiosflags(std::_S_fixed) << std::setprecision(2)
//                     << ": CorrectRate= " << (double)matches_correct.size()/matches.size()
//                     << " RecallRate= " << (double)matches_correct.size()/matches_GT.size();
            cv::Point info_pt(0,20);
            cv::putText( img_matches, info_str.str(), info_pt, cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(255,255,255),2 );
            cv::imshow("img_matches",img_matches);
            cv::imshow("img_matches_correct",img_matches_correct);
            cv::imshow("img_matches_GT",img_matches_GT);
        }

        ////////////////////// 5. Save Image ///////////////////////////
        static bool AUTO_PROGRESS = false;
        char key;
        do
        {
            key = cv::waitKey(5);
            viewer_pift.spinOnce(5);
            viewer_noob.spinOnce(5);
//            viewer_norm.spinOnce(5);
            viewer_reg.spinOnce(5);
        }while( -1==key && !AUTO_PROGRESS );
        if( key=='\r' || key=='\n' )//Enter
            AUTO_PROGRESS = true;
        else if( key!=-1 )
            AUTO_PROGRESS = false;
        if( key == 's' )
        {
            std::stringstream cnt_str;
            cnt_str << data_cnt;
            if( descriptor_type == DT_PIFT )
            {
                cv::imwrite(save_path+"keypoints_noob"+cnt_str.str()+".jpg",keypoints_show);
                cv::imwrite(save_path+"keypoints_pift"+cnt_str.str()+".jpg",pift.rgb_show_);
                cv::imwrite(save_path+"features_show"+cnt_str.str()+".jpg",pift.features_show_);
            }
            if( data_cnt>1 )
            {
                cv::imwrite(save_path+"img_matches"  +cnt_str.str()+".jpg",img_matches);
                cv::imwrite(save_path+"img_matches_correct"+cnt_str.str()+".jpg",img_matches_correct);
            }
            if( RESTORE_PATCH )
                cv::imwrite(save_path+"features_restore"+cnt_str.str()+".jpg",pift.features_restore_);
            if( descriptor_type == DT_PIFT )
                pcl::io::savePCDFile(save_path+"pointcloud"+cnt_str.str()+".pcd",*pift.cloud_);
        }
        else if( '\e' == key )//Esc
            break;

        if(data_cnt>1)
        {
            if( descriptor_type == DT_PIFT )
            {
                if( !match_video.isOpened() )
                    match_video.open( save_path+"pift_match.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, img_matches.size() );
                match_video.write( img_matches );
            }
            else
            {
                cv::Mat video_frame( img_matches.rows*2, img_matches.cols, CV_8UC3 );
                if( !match_video.isOpened() )
                {
                    comp_video.open( save_path+"pift_match.avi" );
                    if( comp_video.isOpened() )
                        match_video.open( save_path+"match.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, video_frame.size() );
                    else
                        match_video.open( save_path+"match.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, img_matches.size() );
                }
                if( comp_video.isOpened() )
                {
                    img_matches.copyTo( cv::Mat( video_frame, cv::Rect(0,0,video_frame.cols,video_frame.rows/2) ) );
                    cv::Mat cmp_img;
                    comp_video >> cmp_img;
                    if( cmp_img.rows>0 )
                        cmp_img.copyTo( cv::Mat(video_frame,cv::Rect(0, video_frame.rows/2, video_frame.cols, video_frame.rows/2)) );
                    match_video.write( video_frame );
                }
                else
                    match_video.write( img_matches );
            }

            std::stringstream reg_name;
            reg_name << save_path << "reg.png";
            viewer_reg.saveScreenshot( reg_name.str() );
            sleep(0.1);
            cv::Mat reg_img = cv::imread( reg_name.str() );
            if( !reg_video.isOpened() )
                reg_video.open( save_path+"reg.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, reg_img.size() );
            reg_video.write( reg_img );
        }
        if( INCREMENTAL || data_cnt==1 )
        {
            rgb_last = rgb.clone();
            pcl::copyPointCloud( *pift.cloud_, cloud_last );
            keypoints_last = keypoints;
            keypoints3_last = keypoints3;
            descriptors_last = descriptors.clone();
            features_show_last = pift.features_show_.clone();
            features_restore_last = pift.features_restore_.clone();
            GT_Pos7_last = GT_Pos7;
        }
    }

    std::cout << "\033[31mEnd of procsse normally! " << "\033[m" << std::endl;
    return 0;
}
