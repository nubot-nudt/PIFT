/* License
*/
#ifndef DATA_SET_PLAYER_HPP
#define DATA_SET_PLAYER_HPP

#include <fstream>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>


class DataSetPlayer
{
public:
    enum GT_file_format
    {
        QwQxQyQzTxTyTz,
        TimeTxTyTzQxQyQzQw,
        UNKNOWN
    };
protected:
    std::vector<boost::filesystem::path> color_files;
    std::vector<boost::filesystem::path> depth_files;
    std::vector<boost::filesystem::path> getDirFiles( const std::string &_dir) const;
    double depth_scale;//depth->mm
    double GT_scale;//depth->mm
    std::ifstream GT_file;
    GT_file_format GT_FORMAT;
    double sync_time_err;
    double down_sample_interval;//time or frames
    double time_last;// used to down sample

public:
    DataSetPlayer();
    bool setColorPath( const std::string & _color );
    bool setDepthPath( const std::string & _depth );
    bool setGroundTruthPath(const std::string &_GT );
    bool setFormat(const GT_file_format &_GT_format, const double &_depth_scale=1, const double &_GT_scale=1, const double &_sync_time_err=INFINITY );///The fps is valid only when the GT_file contains the time-stamp
    void setDownsample(const double &_interval){ down_sample_interval=_interval; }
    bool getFrame(cv::Mat &rgb, cv::Mat &_depth, Eigen::Quaternionf &_rotat, Eigen::Vector3f &_trans , double *_time_stamp=NULL );
private:
    uint id_c;
    uint id_d;
    double          GT_time[2];
    Eigen::Vector4f GT_rotat[2];
    Eigen::Vector3f GT_trans[2];


};

DataSetPlayer::DataSetPlayer()
        : GT_FORMAT(UNKNOWN), id_c(0), id_d(0), down_sample_interval(0), time_last(0), sync_time_err(0)
{
    GT_time[0] = 0;
}
bool DataSetPlayer::setColorPath(const std::string &_color )
{
    color_files = getDirFiles( _color );
    if( color_files.size() == 0 )
        return false;
   return true;
}
bool DataSetPlayer::setDepthPath( const std::string & _depth )
{
    depth_files = getDirFiles( _depth );
    if( depth_files.size() == 0 )
        return false;
    return true;
}
bool DataSetPlayer::setGroundTruthPath( const std::string & _GT )
{
    GT_file.open( _GT.c_str() );
    if( GT_file.is_open() )
        return true;
    else
        return false;
}
bool DataSetPlayer::setFormat(const GT_file_format &_GT_format, const double &_depth_scale, const double &_GT_scale, const double &_sync_time_err )
{
    GT_FORMAT = _GT_format;
    depth_scale = _depth_scale;
    GT_scale = _GT_scale;
    sync_time_err = _sync_time_err;
    if( GT_FORMAT==UNKNOWN )
        return false;
    if( GT_FORMAT==TimeTxTyTzQxQyQzQw && sync_time_err==0 )
        return false;
    return true;
}

bool DataSetPlayer::getFrame(cv::Mat &_rgb, cv::Mat &_depth, Eigen::Quaternionf &_rotat, Eigen::Vector3f &_trans, double *_time_stamp )
{
    if( ( GT_FORMAT==UNKNOWN )
     || ( GT_FORMAT==TimeTxTyTzQxQyQzQw && sync_time_err==0 )
     || color_files.size() == 0
     || depth_files.size() == 0
     || !GT_file.is_open()
            )
    {
        std::cout << "\033[31mERROR: No data set! " << "\033[m" << std::endl;
        return false;
    }
    if( id_c>=color_files.size() || id_d>=depth_files.size() )
    {
        return false;
    }

    double time_cur=0;
    if( GT_FORMAT==TimeTxTyTzQxQyQzQw ) /// synchronise color and depth
    {
        double time_depth, time_color;
        std::stringstream( color_files[id_c].stem().string() ) >> time_color;
        std::stringstream( depth_files[id_d].stem().string() ) >> time_depth;
        while( fabs(time_color-time_depth) >= sync_time_err || time_color-time_last<down_sample_interval || time_depth-time_last<down_sample_interval )
        {
            if( time_color>time_depth )
            {
                if( ++id_d < depth_files.size() )
                    std::stringstream( depth_files[id_d].stem().string() ) >> time_depth;
                else
                    return false;
            }
            else
            {
                if( ++id_c < color_files.size() )
                    std::stringstream( color_files[id_c].stem().string() ) >> time_color;
                else
                    return false;
            }
        }
        time_cur = (time_depth+time_color)/2;
//        std::cout << std::fixed << "\t(" << time_color << " " << time_depth << ")=" << time_cur;
    }

    Eigen::Vector4f GT_Q;//[ qw qx qy qz ]
    Eigen::Vector3f GT_T;//[ tx ty tz ]
    switch ( GT_FORMAT )
    {
    case QwQxQyQzTxTyTz:
        for( int i=0; i<std::max(1.0, down_sample_interval); i++ )
            GT_file >> GT_Q[0] >> GT_Q[1] >> GT_Q[2] >> GT_Q[3] >> GT_T[0] >> GT_T[1] >> GT_T[2];
        break;
    case TimeTxTyTzQxQyQzQw:
    {
        while( time_cur>GT_time[0] && GT_file.good() )
        {
            GT_time[1]  = GT_time[0];
            GT_rotat[1] = GT_rotat[0];
            GT_trans[1] = GT_trans[0];
            GT_file >> GT_time[0] >> GT_trans[0][0] >> GT_trans[0][1] >> GT_trans[0][2]
                    >> GT_rotat[0][1] >> GT_rotat[0][2] >> GT_rotat[0][3] >> GT_rotat[0][0];
        }
        if( !GT_file.good() )
            return false;
        if( GT_time[1]==0 )///first frame
        {
            GT_Q = GT_rotat[0];
            GT_T = GT_trans[0];
        }
        else/// linear interpolation
        {
            if( GT_time[0]-GT_time[1] >= 2*sync_time_err )
                std::cout << "\033[33mWarnning: Ground Truth with big interval! Linear interpolation applied: " << std::fixed << GT_time[1] << " ~ " << GT_time[0] << "\033[m" << std::endl;
            if( fabs(GT_rotat[0][0]-GT_rotat[1][0]) > 0.9 )//angle across PI/2
                GT_rotat[1] = -GT_rotat[1];
            double w1 = (GT_time[0]-time_cur  )/(GT_time[0]-GT_time[1]);
            double w0   = 1 - w1;
            GT_Q = GT_rotat[1] * w1 + GT_rotat[0] * w0;
            GT_T = GT_trans[1] * w1 + GT_trans[0] * w0;
            GT_Q /= GT_Q.norm();
        }
//        std::cout << std::fixed << "=(" << GT_time[1] << " " << GT_time[0] << ")" << std::endl;
    }
        break;
    default:
        return false;
        break;
    }

    if( _time_stamp!=NULL )
        *_time_stamp = time_cur;
    _rgb   = cv::imread( color_files[id_c].string() );
    _depth = cv::imread( depth_files[id_d].string(), cv::IMREAD_ANYDEPTH ) * depth_scale;
    assert(   _rgb.type() ==  CV_8UC3 );
    assert( _depth.type() == CV_16UC1 );
    _rotat = Eigen::Quaternionf( GT_Q[0], GT_Q[1], GT_Q[2], GT_Q[3] );
    _trans = GT_T*GT_scale;

    time_last = time_cur;
    if( GT_FORMAT==TimeTxTyTzQxQyQzQw )
    {
        id_c++;
        id_d++;
    }
    else
    {
        id_c += std::max(1.0, down_sample_interval);
        id_d += std::max(1.0, down_sample_interval);
    }
    return true;

}

std::vector<boost::filesystem::path> DataSetPlayer::getDirFiles( const std::string &_dir) const
{
    std::vector<std::string> files_str;
    boost::filesystem::recursive_directory_iterator end_iter;
    for(boost::filesystem::recursive_directory_iterator iter(_dir);iter!=end_iter;iter++)
        if ( !boost::filesystem::is_directory( *iter ) )
            files_str.push_back( iter->path().string() );
    std::sort( files_str.begin(), files_str.end() );
    std::vector<boost::filesystem::path> files;
    files.reserve( files_str.size() );
    for( std::vector<std::string>::iterator it = files_str.begin(); it != files_str.end(); it++)
        files.push_back( boost::filesystem::path(*it) );
    return files;
}
#endif
