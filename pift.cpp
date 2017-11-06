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
#include "pift.h"

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_circle.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/fpfh.h>

//#define USE_OPENMP
#ifdef USE_OPENMP
#include "omp.h"
#endif
ColorCoding::ColorCoding(const METHOD &_method)
    : method_(_method)
{
    switch ( method_ )
    {
    case HAMMING_HSV422:
        coder_ = boost::shared_ptr<CodeBase> (new HSV422Code);
        code_type_ = CV_8UC1;
        break;
    case HAMMING_HSV655:
        coder_ = boost::shared_ptr<CodeBase> (new HSV655Code);
        code_type_ = CV_16UC1;
        break;
    case HAMMING_GRAY8:
        code_type_ = CV_8UC1;
        break;
    default:
        code_type_ = CV_8UC1;
        break;
    }
}
ColorCoding::HSV422Code::HSV422Code()
{
    INVALID_CODE = 0b10101010;// 0xaa == 170
    ///Init color coding table in HSV color space
    const code_t H_code[8] = {0x00,0x10,0x30,0x70,0xF0,0xE0,0xC0,0x80};//high 4 bits
    const code_t S_code[3] = {0x00,0x04,0x0C};//the 5th and 6th bits
    const code_t V_code[3] = {0x00,0x01,0x03};//the last 2 bits
    const code_t EMPITY_H = 0b01010000;

    std::vector<code_t> H_vec(8), S_vec(3), V_vec(3);
    std::memcpy( H_vec.data(), H_code, 8*sizeof(code_t) );
    std::memcpy( S_vec.data(), S_code, 3*sizeof(code_t) );
    std::memcpy( V_vec.data(), V_code, 3*sizeof(code_t) );
    initByHSV<code_t>( H_vec, S_vec, V_vec, EMPITY_H );
}

ColorCoding::HSV655Code::HSV655Code()
{
    const code_t H_code[12] ={0B0000000000000000,
                              0B0000010000000000,
                              0B0000110000000000,
                              0B0001110000000000,
                              0B0011110000000000,
                              0B0111110000000000,
                              0B1111110000000000,
                              0B1111100000000000,
                              0B1111000000000000,
                              0B1110000000000000,
                              0B1100000000000000,
                              0B1000000000000000};//high 6 bits
    const code_t S_code[6] = {0B0000000000000000,
                              0B0000000000100000,
                              0B0000000001100000,
                              0B0000000011100000,
                              0B0000000111100000,
                              0B0000001111100000};//middle 5 bits
    const code_t V_code[6] = {0B0000000000000000,
                              0B0000000000000001,
                              0B0000000000000011,
                              0B0000000000000111,
                              0B0000000000001111,
                              0B0000000000011111};//low 5 bits
    const code_t EMPITY_H  =  0B0101010000000000;
    INVALID_CODE           =  0B1010101010101010;

    std::vector<code_t> H_vec(12), S_vec(6), V_vec(6);
    std::memcpy( H_vec.data(), H_code, 12*sizeof(code_t) );
    std::memcpy( S_vec.data(), S_code, 6*sizeof(code_t) );
    std::memcpy( V_vec.data(), V_code, 6*sizeof(code_t) );
    initByHSV<code_t>( H_vec, S_vec, V_vec, EMPITY_H );
}
template<typename code_t>
bool ColorCoding::CodeBase::initByHSV( const std::vector<code_t> &H_code, const std::vector<code_t> &S_code, const std::vector<code_t> &V_code, const code_t &EMPITY_H )
{
    size_t H_size = H_code.size();
    size_t S_size = S_code.size();
    size_t V_size = V_code.size();

    const uchar GRAY_THRESH_S = 50;//take it as gray if the S value is lower than this thresh
    const uchar GRAY_THRESH_V = 20;

    cv::Mat rgb2hsv_img(16,16*16,CV_8UC3);
    for(int r=0; r<16; r++)
    {
        uchar * p_img = rgb2hsv_img.data + r*rgb2hsv_img.step[0];
        for(int g=0; g<16; g++)
        for(int b=0; b<16; b++)
        {
            *p_img = b*16;
            *(p_img+1) = g*16;
            *(p_img+2) = r*16;
            p_img += rgb2hsv_img.step[1];
        }
    }
    cv::cvtColor(rgb2hsv_img, rgb2hsv_img, CV_BGR2HSV_FULL);
    for(int r=0; r<16; r++)
    {
        uchar * p_img = rgb2hsv_img.data + r*rgb2hsv_img.step[0];
        for(int g=0; g<16; g++)
        for(int b=0; b<16; b++)
        {
            uchar H = *p_img;
            uchar S = *(p_img+1);
            uchar V = *(p_img+2);
            if( S<=GRAY_THRESH_S || V<=GRAY_THRESH_V )
            {
                V = V*V_size/256;
                rgb2code_[r][g][b] = EMPITY_H | V_code[V];
            }
            else
            {
                H = H*H_size/256;
                V = V*V_size/256;
                S = (int)(S-GRAY_THRESH_S)*S_size/(256-GRAY_THRESH_S);
                rgb2code_[r][g][b] = H_code[H] | S_code[S] | V_code[V];
            }
            p_img += rgb2hsv_img.step[1];
        }
    }

    /// Init code2rgba_ for visualization
    code2rgba_.resize( 1<<(sizeof(code_t)*8), 0 );
    cv::Mat hsv_img(1,H_size*(S_size+1)*V_size,CV_8UC3);
    uchar H_show[H_size];
    uchar S_show[S_size+1];//empty H means S=0, so an extra S_show is added
    uchar V_show[V_size];
    for(int h=0; h<H_size; h++)
        H_show[h] = (h+0.5)*256/H_size;
    for(int s=0; s<S_size; s++)
        S_show[s] = (s+0.5)*(256-GRAY_THRESH_S)/S_size+GRAY_THRESH_S;
    S_show[S_size] = 0;
    for(int v=0; v<V_size; v++)
        V_show[v] = (v+0.5)*256/V_size;
    uchar *p_hsv_img = hsv_img.data;
    for(size_t H=0; H<H_size;   H++)
    for(size_t S=0; S<S_size+1; S++)
    for(size_t V=0; V<V_size;   V++)
    {
        *p_hsv_img     = H_show[H];
        *(p_hsv_img+1) = S_show[S];
        *(p_hsv_img+2) = V_show[V];
        p_hsv_img += hsv_img.step[1];
    }

    cv::Mat hsi2rgba( hsv_img.rows, hsv_img.cols, CV_8UC4 );
    cv::cvtColor(hsv_img, hsi2rgba, CV_HSV2BGR_FULL);
    uchar *p_hsi2rgba = hsi2rgba.data;
    for(size_t H=0; H<H_size;   H++)
    for(size_t S=0; S<S_size+1; S++)
    for(size_t V=0; V<V_size;   V++)
    {
        if(S!=S_size)
            code2rgba_[ H_code[H] | S_code[S] | V_code[V] ] = *(uint32_t*)p_hsi2rgba;
        else
            code2rgba_[ EMPITY_H | V_code[V] ] = *(uint32_t*)p_hsi2rgba;
        p_hsi2rgba += hsi2rgba.step[1];
    }
    return true;
}

uint ColorCoding::encode(void *p_code, const uchar _r, const uchar _g, const uchar _b) const
{
    switch ( method_ )
    {
    case HAMMING_HSV422:
    {
        uchar &code = *(uchar*)p_code;
        code = coder_->rgb2code_[_r/16][_g/16][_b/16];
//        if( _V_MEAN != 128 )
//        {
//            int v = ((int)_r+(int)_g+(int)_b)/3 ;
//            if(      v < _V_MEAN-30 ) code = (code & 0xFC) | 0x00;
//            else if( v > _V_MEAN+30 ) code = (code & 0xFC) | 0x03;
//            else                      code = (code & 0xFC) | 0x01;
//        }
    }
        return sizeof(uchar);
    case HAMMING_HSV655:
    {
        u_int16_t &code = *(u_int16_t*)p_code;
        code = coder_->rgb2code_[_r/16][_g/16][_b/16];
    }
        return sizeof(u_int16_t);
    case HAMMING_GRAY8:
    {
        uchar &code = *(uchar*)p_code;
        int gray = ((int)_r+_g+_b)/3 * 9/256;//0~8
        code = (uchar)0xff >> (8-gray);
    }
        return 1;
    default:
        return 0;
    }
}

uint ColorCoding::invalidCode( void* p_code ) const
{
    switch ( method_ )
    {
    case HAMMING_HSV422:
        *(uchar*)p_code = coder_->INVALID_CODE;
        return 1;
    case HAMMING_HSV655:
        *(u_int16_t*)p_code = coder_->INVALID_CODE;
        return 2;
    case HAMMING_GRAY8:
        *(uchar*)p_code = 0b10101010;
        return 1;
    default:
        return 0;
    }

}

uchar ColorCoding::rgb2IntCode(const uchar _r, const uchar _g, const uchar _b, const uchar _bit_length) const
{
    int chanel_range;
    if( _bit_length%3 == 0 )
        chanel_range = 1 << (_bit_length/3) ;
    else
        chanel_range = std::pow(2, _bit_length/3.0);
    //normalize r g b to [0,chanel_range)
    uchar r = (int)_r * chanel_range / 256;
    uchar g = (int)_g * chanel_range / 256;
    uchar b = (int)_b * chanel_range / 256;
    return r*chanel_range*chanel_range + g*chanel_range + b;
}

uint ColorCoding::machCode(void* _code1, void* _code2, const uint _cells) const
{
    uint dist = 0;
    uint invalid_cnt = 0;
    uint valid_cnt = 0;
    for(uint i=0; i<_cells; i++)
    {
        switch ( method_ )
        {
        case HAMMING_HSV422:
        {
            typedef uchar code_type;
            const code_type *p1 = (code_type*)_code1 + i;
            const code_type *p2 = (code_type*)_code2 + i;
            if( *p1 == *p2 )
                dist += 0;
            else if( *p1 == coder_->INVALID_CODE || *p2 == coder_->INVALID_CODE )
                invalid_cnt ++;
            else
                dist += cv::normHamming( p1, p2, sizeof(code_type) );
        }
            break;
        case HAMMING_HSV655:
        {
            typedef u_int16_t code_type;
            const code_type *p1 = (code_type*)_code1 + i;
            const code_type *p2 = (code_type*)_code2 + i;
            if     ( *p1 == coder_->INVALID_CODE && *p2 == coder_->INVALID_CODE )
                dist += 0;
            else if( *p1 == coder_->INVALID_CODE || *p2 == coder_->INVALID_CODE )
                invalid_cnt ++;
            else
            {
                valid_cnt ++;
                dist += cv::normHamming( (uchar*)p1, (uchar*)p2, sizeof(code_type) );
            }
        }
            break;
        case HAMMING_GRAY8:

        default:
            return INFINITY;
        }
    }
    if( valid_cnt < invalid_cnt )
        return INFINITY;
    else
        return dist * _cells / (_cells-invalid_cnt);
}
uint32_t ColorCoding::decode(const void *_code) const
{
    switch ( method_ )
    {
    case HAMMING_HSV422:
        return coder_->code2rgba_[ *(uchar*)_code ];
    case HAMMING_HSV655:
        return coder_->code2rgba_[ *(u_int16_t*)_code ];
    case HAMMING_GRAY8:
    {
        uchar temp = cv::normHamming( (uchar*)_code, &temp, 1 ) *256/9;
        return (uint32_t)temp<<16 | (uint32_t)temp<<8 | (uint32_t)temp;
    }
    default:
        return 0;
    }
}

PerspectiveInvariantFeature::AnnularMask::AnnularMask(const uint &_patch_size, const uint &_angle_res, const uint &_dist_res)
    : PatchMask(_angle_res,_dist_res*_angle_res+1), DIST_RES(_dist_res)
{
    const uint RADIUS = _patch_size/2;
    TAN_ANGLE_THRESH.reserve(ANGLE_RES/4);
    for(size_t i=0; i<ANGLE_RES/4; i++)
        TAN_ANGLE_THRESH.push_back( tan( i*2*M_PI/ANGLE_RES ) );

    DIR_PER_CELL.resize(TOTAL_CELLS);
    DIR_PER_CELL[0].x =DIR_PER_CELL[0].y = 0;
    for(size_t d=0; d<DIST_RES; d++)
    for(size_t i=0; i<ANGLE_RES; i++)
    {
        DIR_PER_CELL[d*ANGLE_RES+i+1].x = cos( M_PI*2*(i+0.5)/ANGLE_RES );
        DIR_PER_CELL[d*ANGLE_RES+i+1].y = sin( M_PI*2*(i+0.5)/ANGLE_RES );
    }

    /////// Init patch_mask_annular_
    mask_.create( _patch_size, _patch_size, CV_8UC1);
    std::vector<uint> dis_res_thresh2(DIST_RES+1);
    // The same area size
//    for(uint k=0; k<DIST_RES+1; k++)
//    {
//        double curr_r = sqrt(double(ANGLE_RES*k+1)/(ANGLE_RES*DIST_RES+1))*RADIUS;//共ANGLE_RES*DIST_RES+1个单元个(含中心一个)
//        dis_res_thresh2[k] = (curr_r+0.5)*(curr_r+0.5);
//    }
    // The same incremental radius
    for(uint k=0; k<DIST_RES+1; k++)
        dis_res_thresh2[k] = ( (k+0.5d)/(DIST_RES+0.5d)*RADIUS+0.5 ) * ( (k+0.5d)/(DIST_RES+0.5d)*RADIUS+0.5 );

    for(int i=0; i<mask_.rows; i++)
    for(int j=0; j<mask_.cols; j++)
    {
        int x = j - (int)RADIUS;
        int y = (int)RADIUS - i;
        if( (uint)x*x+y*y <= dis_res_thresh2[DIST_RES])
        {
            uint pos;
            if( (uint)x*x+y*y <= dis_res_thresh2[0] )
            {
                mask_.at<uchar>(i,j) = 0;// the center cell is 0
                continue;
            }
            for(pos=1; pos<TAN_ANGLE_THRESH.size() && abs(y)>abs(x)*TAN_ANGLE_THRESH[pos]; pos++)
                ;//normalize the ID to 1~ANGLE_RES/4 in 1st quadrant.
            if     ( x>=0 && y>=0 ) pos--;                    // 1st quadrant: 0             ~ ANGLE_RES/4-1 ;
            else if( x<=0 && y>=0 ) pos = ANGLE_RES/2 - pos;  // 2nd quadrant: ANGLE_RES/4-1 ~ ANGLE_RES/2-1 ;
            else if( x<=0 && y<=0 ) pos = ANGLE_RES/2 + pos-1;// 3rd quadrant: ANGLE_RES/2-1 ~ ANGLE_RES/4*3-1;
            else if( x>=0 && y<=0 ) pos = ANGLE_RES - pos;    // 4th quadrant: ANGLE_RES/4*3-1 ~ ANGLE_RES-1;
            pos += 1;// skip the center cell
            for(uint k=1; k<DIST_RES+1; k++)
            {
                if( (uint)x*x+y*y > dis_res_thresh2[k] )
                    pos += ANGLE_RES;// 1 ~ ANGLE_RES*DIST_RES;
                else break;
            }

            mask_.at<uchar>(i,j) = pos;
        }
        else
            mask_.at<uchar>(i,j) = BAD_CELL;
    }
//    patch_mask_annular_.at<uchar>(RADIUS,RADIUS) = BAD_CELL;

//    /// show and save
//    cv::Mat mask_show, mask_edge;
//    mask_show = mask_.clone();
//    cv::Canny( mask_show, mask_edge, 1, 1 );
//    cv::threshold( mask_edge, mask_edge, 128, 255, CV_THRESH_BINARY_INV );
//    cv::Point center(mask_edge.cols/2,mask_edge.rows/2);
//    cv::Point arow( center.x*2,center.y );
//    cv::line( mask_edge,center, arow, 128, 1 );
//    cv::imshow("mask_show",mask_show);
//    cv::imshow("mask_edge",mask_edge);
//    char key = cv::waitKey();
//    if( key == 's' )
//    {
//        cv::imwrite("mask_show.bmp",mask_show);
//        cv::imwrite("mask_edge.bmp",mask_edge);
//    }/// show and save
}

const uchar
PerspectiveInvariantFeature::AnnularMask::getCellID(const uint& _xp, const uint& _yp, const uint& _main_angle_deg ) const
{
    uint dir_id = _main_angle_deg*ANGLE_RES/360;
    while( dir_id>=ANGLE_RES ) dir_id -= ANGLE_RES;
    const uchar& cell_id = mask_.at<uchar>(_yp,_xp);
    if( cell_id==BAD_CELL || cell_id==0 )
        return cell_id;
    else
    {
        if( (cell_id-1)%ANGLE_RES < dir_id )
            return cell_id + ANGLE_RES - dir_id;
        else
            return cell_id - dir_id;
    }
}

PerspectiveInvariantFeature::BeehiveMask::BeehiveMask(const uint &_patch_size, const uint &_layers, const uint &_angle_res)
    : PatchMask( _angle_res, 3*_layers*(_layers+1)+1 )
{
    assert( ANGLE_RES%6==0 );// The behive mask is invariant to the ratition of 0deg, 60deg, etc.
    assert( _patch_size%2 ==1 );
    const uint BIG_SIZE = _patch_size*3;//Enlarge the mask to make a acurrate rotation, and then down-sample it.
    const uint BIG_RADIUS = BIG_SIZE/2;
    ///// Init patch_mask_ ///////////////////////////////////////////////////////////
    /// The generation of a behive mask with 4 layers is shown as an example.
    /// The cells in the 1st quadrant are formed as follows. The whole mask is generated by mirror operation
    /// There are 5 steps in the j axes, and the step is CELL_BORDER*1.5
    /// There are 9 steps in the i axes, and the step is CELL_BORDER*sqrt(3)/2
    /// The nember in each cell indicate the cell ID
    ///    0     1      2      3      4      5
    ///   __________________________________ j
    /// 0 |0  /     \  10   /     \  43   /
    ///   |__/       \_____/       \_____/  ff
    /// 1 |  \   2   /     \  23   /     \
    ///   |   \_____/       \_____/       \ ff
    /// 2 |1  /     \   9   /     \  41   /
    ///   |__/       \_____/       \_____/  ff
    /// 3 |  \   8   /     \  22   /     \
    ///   |   \_____/       \_____/       \ ff
    /// 4 |7  /     \  21   /     \  41   /
    ///   |__/       \_____/       \_____/  ff
    /// 5 |  \  20   /     \  40   /
    ///   |   \_____/       \_____/  ff     ff
    /// 6 |19 /     \  39   /
    ///   |__/       \_____/  ff     ff     ff
    /// 7 |  \  38   /
    ///   |   \_____/  ff     ff     ff     ff
    /// 8 |37 /
    ///   |__/  ff     ff     ff     ff     ff
    /// 9 i
    cv::Mat mask0  = cv::Mat( BIG_SIZE, BIG_SIZE, CV_8UC1, BAD_CELL);
    const double CELL_BORDER =     BIG_SIZE/((_layers*2+1)*sqrt(3));
    const double STEP_J  =      CELL_BORDER*1.5;
    const double STEP_I =       CELL_BORDER*sqrt(3)/2 ;
    const double TEMP_1      =      (STEP_J*STEP_J - STEP_I*STEP_I)/2.0;
    const double TEMP_2      =      (STEP_J*STEP_J + STEP_I*STEP_I)/2.0;
    std::vector<uint> CELLS_PER_RING( _layers+1 );
    std::vector<uint> START_PER_RING( _layers+1 );
    CELLS_PER_RING[0] = 1;
    START_PER_RING[0] = 0;
    for(uint i=1; i<=_layers; i++)
    {
        CELLS_PER_RING[i] = i*6;
        START_PER_RING[i] = START_PER_RING[i-1] + CELLS_PER_RING[i-1];
    }

    for(int i=0; i<=(int)BIG_RADIUS; i++)
    for(int j=0; j<=(int)BIG_RADIUS; j++)
    {
        int cell_j = (int)(j/STEP_J);
        int cell_i = (int)(i/STEP_I);
        if((cell_j+cell_i)&1)
        {
            if((j-cell_j*STEP_J)*STEP_J-(i-cell_i*STEP_I)*STEP_I > TEMP_1) cell_j++;
        }
        else
            if((j-cell_j*STEP_J)*STEP_J+(i-cell_i*STEP_I)*STEP_I > TEMP_2) cell_j++;
        if((cell_j+cell_i)&1)
            cell_i++;

        uchar ring_id;
        for( ring_id=0; cell_j>ring_id || cell_j+cell_i>ring_id*2; ring_id++)
            ;
        if( ring_id<=_layers )
        {
            uchar pos_temp = BAD_CELL;//The position in the current ring: 0~CELLS_PER_RING/4
            if( cell_j==ring_id ) pos_temp = CELLS_PER_RING[ring_id]/4 - cell_i/2;
            else pos_temp = cell_j;
            mask0.at<uchar>(BIG_RADIUS-j,BIG_RADIUS+i) = START_PER_RING[ring_id] + pos_temp;                            //1st quadrant
            mask0.at<uchar>(BIG_RADIUS-j,BIG_RADIUS-i) = START_PER_RING[ring_id] - pos_temp + CELLS_PER_RING[ring_id]/2;//2nd quadrant
            mask0.at<uchar>(BIG_RADIUS+j,BIG_RADIUS-i) = START_PER_RING[ring_id] + pos_temp + CELLS_PER_RING[ring_id]/2;//3rd quadrant
            mask0.at<uchar>(BIG_RADIUS+j,BIG_RADIUS+i) = START_PER_RING[ring_id] - pos_temp + CELLS_PER_RING[ring_id];  //4th quadrant
            if( pos_temp==0 )// The last cell in the 4th quadrant equals to the first cell in the 1st quadrant
                mask0.at<uchar>(BIG_RADIUS+j,BIG_RADIUS+i) = START_PER_RING[ring_id] - pos_temp;
        }
    }

    masks_.resize( ANGLE_RES/6 );//Rotate the mask. The 0deg, 60deg, ... is not needed.
    double angle_deg = 0;
    cv::Mat mat_temp(BIG_SIZE, BIG_SIZE, CV_8UC1);
    for( std::vector<cv::Mat>::iterator it = masks_.begin(); it != masks_.end(); it++, angle_deg += 360/ANGLE_RES )
    {
        const cv::Mat &rotate_mat = cv::getRotationMatrix2D( cv::Point2f(BIG_RADIUS,BIG_RADIUS), angle_deg, 1 );
        cv::warpAffine( mask0, mat_temp, rotate_mat, cv::Size(BIG_SIZE,BIG_SIZE), cv::INTER_NEAREST, cv::BORDER_CONSTANT, CV_RGB(BAD_CELL,BAD_CELL,BAD_CELL) );

//        /// show and save
//        cv::Mat mask_show, mask_edge;
//        mask_show = mat_temp.clone();
//        mask_show = mask_show*2;
//        cv::Canny( mask_show, mask_edge, 1, 1 );
//        cv::threshold( mask_edge, mask_edge, 128, 255, CV_THRESH_BINARY_INV );
//        cv::Point center(mask_edge.cols/2,mask_edge.rows/2);
//        cv::Point arow( center.x*cos(angle_deg*M_PI/180),-center.x*sin(angle_deg*M_PI/180) );
//        cv::line( mask_edge,center, center+arow, 128, 1 );
//        cv::imshow("mask_show",mask_show);
//        cv::imshow("mask_edge",mask_edge);
//        char key = cv::waitKey();
//        if( key == 's' )
//        {
//            cv::imwrite("mask_show.bmp",mask_show);
//            cv::imwrite("mask_edge.bmp",mask_edge);
//        }/// show and save

        cv::resize(mat_temp,*it,cv::Size(_patch_size,_patch_size),0,0,cv::INTER_NEAREST);
    }

    rotateCellID_ = cv::Mat::zeros( 6, TOTAL_CELLS, CV_8UC1 );//6 angles of an beehive
    for(uint a=0; a<6; a++)
    for(uint pos=0; pos<TOTAL_CELLS; pos++)
    {
        uint ring_id = 0;
        for( ring_id=0; ring_id<_layers && pos>=START_PER_RING[ring_id+1]; ring_id++ )
            ;
        uint pos_inc = CELLS_PER_RING[ring_id]/(ANGLE_RES/2) * a;
        if( pos < START_PER_RING[ring_id] + pos_inc )
            rotateCellID_.at<uchar>(a,pos) = pos + CELLS_PER_RING[ring_id] - pos_inc;
        else
            rotateCellID_.at<uchar>(a,pos) = pos - pos_inc;
    }
}
const uchar
PerspectiveInvariantFeature::BeehiveMask::getCellID(const uint& _xp, const uint& _yp, const uint& _main_angle_deg ) const
{
    uint dir_id = _main_angle_deg*ANGLE_RES/360.0 + 0.5;
    while( dir_id>=ANGLE_RES ) dir_id -= ANGLE_RES;
    uchar cell_id = masks_[dir_id%(ANGLE_RES/6)].at<uchar>(_yp,_xp);
    if( cell_id != BAD_CELL )
        return rotateCellID_.at<uchar>(dir_id/(ANGLE_RES/6),cell_id);
    else
        return BAD_CELL;
}

PerspectiveInvariantFeature::PerspectiveInvariantFeature(const uint _max_keypoints, const uint _spatial_radius, const DESCRIPTOR_TYPE _method)
    : PATCH_SIZE(40*2+1), MAX_KEYPOINTS(_max_keypoints), SPATIAL_R(_spatial_radius)
{
    DRAW_IMAG = false;
    annular_mask_ = boost::shared_ptr<AnnularMask> ( new AnnularMask( PATCH_SIZE, 12, 4 ) );
    beehive_mask_ = boost::shared_ptr<BeehiveMask> ( new BeehiveMask( PATCH_SIZE, 6 ) );
    patch_type_ = _method;// D_TYPE_BEEHIVE D_TYPE_BEEHIVE_NOOB D_TYPE_BRIEF D_TYPE_ORB D_TYPE_SURF D_TYPE_HISTOGRAM
    if( patch_type_ == D_TYPE_ANNULAR )
        patch_mask_ = annular_mask_;
    else//PATCH_TYPE_BEEHIVE
        patch_mask_ = beehive_mask_;

    ///kinect2
    //QHD: fx=517.704829, cx=476.8145455, fy=518.132948, cy=275.4860225,
    //530.7519923402057, 0.0, 478.15372152637906, 0.0, 529.1110630882142, 263.21561548634605
    camera_fx = 531;
    camera_fy = 529;
    camera_cx = 478;
    camera_cy = 263;
    ///kinect1 generic param
    camera_fx = 525;
    camera_fy = 525;
    camera_cx = (640-1)/2.0;
    camera_cy = (480-1)/2.0;

    _1_camera_fx   =   1.0/camera_fx;
    _1_camera_fy   =   1.0/camera_fy;
    _256_camera_fx = 256.0/camera_fx;
    _256_camera_fy = 256.0/camera_fy;

    W2P_RATIO_256 = double(PATCH_SIZE/2+0.5)/(double)SPATIAL_R * 256;// World to Pixel transformation: mm->pixel (with the sacle of 256)
    blur_r_.resize(128);         //The depth is down sampled by the step of 100mm, and with the maximum range of 128*100mm
    for(uint i=0; i<blur_r_.size(); i++)
        blur_r_[i] = (i*100+99)*hypot(_1_camera_fx,_1_camera_fy) * W2P_RATIO_256/256  * 2 + 1;

    keypoints_filtered_.reserve( MAX_KEYPOINTS );
    keypoints_3D_.reserve(MAX_KEYPOINTS);
    features_show_   .create( 10*PATCH_SIZE, 10*PATCH_SIZE, CV_8UC4);//show 10*10 patches in total
    features_restore_.create( 10*PATCH_SIZE, 10*PATCH_SIZE, CV_8UC4);
}

uint PerspectiveInvariantFeature::calcPt6d(const cv::Point& _pt, cv::Point3f &_pt3d, cv::Vec4d &_plane_coef, double &_plane_curvature ) const
{//_plan_coef: Ax+By+Cz=D; return num of pixels

    const bool SHOW_TIME_INFO = false;
    timeval time0, timel, timen;
    if(SHOW_TIME_INFO)
    {
        gettimeofday(&timel,NULL);
        time0 = timel;
    }

    const uint r_nearest = 6;   // used to generate the depth histogram
    cv::Mat nearest_roi(depth_16U_, cv::Rect(_pt.x-r_nearest, _pt.y-r_nearest, r_nearest*2+1, r_nearest*2+1));//without copy data,only new header

    const uint THRESH_NOISE_POINTS = 1;
    /// 1. generat the depth histogram with the resoution of 10mm, range of 12800mm.
    const uint HIST_RESOLUTION = 10;//unit:mm
    const uint DEPTH_MAX = 12800;//unit:mm
    std::vector<ushort> depth_8U_hist(DEPTH_MAX/HIST_RESOLUTION,0);
    uint min_depth_8U = DEPTH_MAX/HIST_RESOLUTION-1;
    uint max_depth_8U = 0;
    uchar *pdata;
    uint point_num = 0;
    uint depth_uchar;
    const uint d_nearest = r_nearest*2+1;
    for(size_t i =0; i<d_nearest; i++)
    {
        pdata = nearest_roi.data + i*nearest_roi.step[0];
        for(size_t j=0; j<d_nearest; j++)
        {
            const u_int16_t &depth_current = *(u_int16_t*)pdata;
            depth_uchar = depth_current>=DEPTH_MAX ? 0 : depth_current/HIST_RESOLUTION;
            if( depth_uchar != 0 )
            {
                point_num ++;
                depth_8U_hist[depth_uchar]++;
                if( depth_uchar<min_depth_8U && depth_uchar!=0 ) min_depth_8U = depth_uchar;
                if( depth_uchar>max_depth_8U ) max_depth_8U = depth_uchar;
            }
            pdata += nearest_roi.step[1];
        }
    }
    if( point_num<THRESH_NOISE_POINTS*2 )
        return 0;

    /// 2. calculate the range of the depth histgram
    for( ; depth_8U_hist[min_depth_8U]<THRESH_NOISE_POINTS && min_depth_8U!=max_depth_8U; min_depth_8U++) ;
    for( ; depth_8U_hist[max_depth_8U]<THRESH_NOISE_POINTS && min_depth_8U!=max_depth_8U; max_depth_8U--) ;
    uint front_depth_8U_thresh = min_depth_8U;
    uint gap_width = 0;
    for( size_t i=min_depth_8U+1; i<=max_depth_8U; i++ )
    {
        if( depth_8U_hist[i]<THRESH_NOISE_POINTS )
            gap_width ++;
        else if(gap_width*HIST_RESOLUTION<SPATIAL_R)
            front_depth_8U_thresh = i;
    }
    float front_thresh_max = (float)front_depth_8U_thresh*HIST_RESOLUTION+HIST_RESOLUTION-1;
    float front_thresh_min = (float)min_depth_8U*HIST_RESOLUTION;
    front_thresh_max += SPATIAL_R/2;
    front_thresh_min -= SPATIAL_R/2;

    /// 3. surch for the nearest point which is on the feture plane
    pcl::PointXYZRGB surface_pt;
    bool pt_found = false;
    const pcl::PointXYZRGB &pt = cloud_->at(_pt.x, _pt.y);
    if( pt.z!=0 && pt.z<front_thresh_max )
    {
        pt_found = true;
        surface_pt.x = pt.x;
        surface_pt.y = pt.y;
        surface_pt.z = pt.z;
    }
    for(uint r=1; r<=r_nearest && !pt_found; r++)//distance
    for(uint side=0; side<4    && !pt_found; side++)//four sides: clockwise
    for(int i=-r; i<(int)r     && !pt_found; i++)
    {
        int x_off, y_off;
        if     (side==0) x_off = i, y_off=-r;//up       //  0 0 1
        else if(side==1) x_off = r, y_off= i;//right    //  3 * 1
        else if(side==2) x_off =-i, y_off= r;//down     //  3 2 2
        else if(side==3) x_off =-r, y_off=-i;//left
        const pcl::PointXYZRGB &pt = cloud_->at(_pt.x+x_off, _pt.y+y_off);
        if( pt.z!=0 && pt.z<front_thresh_max )
        {
            pt_found = true;
            surface_pt.x = pt.x;
            surface_pt.y = pt.y;
            surface_pt.z = pt.z;
        }
    }
    if( !pt_found )
        return 0;

    /// 4. get the normal of surface_pt, and calculate the surface coefficients
    std::vector<int> pointIndicesOut;
    std::vector<float> pointRadiusSquaredDistance;
    kdtree_.nearestKSearch( surface_pt, 4*r_nearest*r_nearest, pointIndicesOut, pointRadiusSquaredDistance );
    uint id_nn = 0;
    for(id_nn=0; id_nn<pointIndicesOut.size() && !pcl::isFinite( normals_->at(pointIndicesOut[id_nn]) ); id_nn++)
        ;
    if( id_nn == pointIndicesOut.size() )
        return 0;
    const pcl::Normal &plane_norm = normals_->at(pointIndicesOut[id_nn]);
    _plane_coef[0] = plane_norm.normal_x;
    _plane_coef[1] = plane_norm.normal_y;
    _plane_coef[2] = plane_norm.normal_z;
    _plane_coef[3] = surface_pt.x*_plane_coef[0]+surface_pt.y*_plane_coef[1]+surface_pt.z*_plane_coef[2];
    _plane_curvature = 1.0/plane_norm.curvature;

    /// 5. calculate the accurate 3D coordinates of the keypoint
    float pt_xn = (_pt.x-camera_cx + 0.5) * _1_camera_fx;
    float pt_yn = (_pt.y-camera_cy + 0.5) * _1_camera_fy;
    _pt3d.z = _plane_coef[3] / ( pt_xn*_plane_coef[0]+pt_yn*_plane_coef[1]+_plane_coef[2] );
    _pt3d.x = pt_xn * _pt3d.z;
    _pt3d.y = pt_yn * _pt3d.z;


    if(SHOW_TIME_INFO)
    {
        gettimeofday(&timen,NULL);
        std::cout << " norm time "  << (timen.tv_sec-timel.tv_sec)*1000+(timen.tv_usec-timel.tv_usec)/1000.0;
        timel = timen;
    }
    return point_num;
}

uint
PerspectiveInvariantFeature::warpPerspectivePatch( const cv::Point& _pt2d, const cv::Point3f &_pt3d, const cv::Vec4d _plane_coef , cv::Mat &_feature_patch)
const
{
    assert( _feature_patch.cols == PATCH_SIZE && _feature_patch.rows == PATCH_SIZE );
    assert( _feature_patch.channels()==4 );
    _feature_patch.setTo(0);

    const bool SHOW_TIME_INFO = false;
    timeval time0, timel, timen;
    if(SHOW_TIME_INFO)
    {
        gettimeofday(&timel,NULL);
        time0 = timel;
    }
    const int PATCH_R = PATCH_SIZE/2;
    const int SPATIAL_R2 = SPATIAL_R*SPATIAL_R;

    /// 1. Calculate the perspective projection matrix
    cv::Mat_<double> rotat_mat(3,3);
    cv::Vec3d axes_morm( _plane_coef[0], _plane_coef[1], _plane_coef[2] );
    cv::Vec3d axes_view( -_pt3d.x, -_pt3d.y, -_pt3d.z );
    axes_view /= cv::norm(axes_view);
    cv::Vec3d rotat_axes = axes_morm.cross( axes_view );
    double   rotat_angle = asin( cv::norm(rotat_axes) );
    rotat_axes *= rotat_angle / cv::norm(rotat_axes);
    cv::Rodrigues( rotat_axes, rotat_mat );
    cv::Mat_<int32_t> rotat_mat32S256(3,3);
    rotat_mat.convertTo( rotat_mat32S256, CV_32S, 256);

    ///test
    /// perspective transform from 2D to 2D
    /*
    cv::Mat img_after = cv::Mat::zeros(PATCH_SIZE, PATCH_SIZE, CV_8UC3);
    double W2P_RATIO = double(PATCH_SIZE/2)/(double)SPATIAL_RADIUS;// World to Pixel transformation: mm->pixel
    double perspect_scale = _pt3d.z*m_1_camera_fx * W2P_RATIO;
    cv::Mat_<double> perspec_mat = cv::Mat_<double>::eye(3,3);
    perspec_mat(0,0) = ( rotat_mat(0,0) - rotat_mat(0,2)*axes_morm(0)/axes_morm(2) ) * perspect_scale;
    perspec_mat(0,1) = ( rotat_mat(0,1) - rotat_mat(0,2)*axes_morm(1)/axes_morm(2) ) * perspect_scale;
    perspec_mat(1,0) = ( rotat_mat(1,0) - rotat_mat(1,2)*axes_morm(0)/axes_morm(2) ) * perspect_scale;
    perspec_mat(1,1) = ( rotat_mat(1,1) - rotat_mat(1,2)*axes_morm(1)/axes_morm(2) ) * perspect_scale;
    perspec_mat(0,2) =  PATCH_SIZE/2 - ( _pt.x*perspec_mat(0,0) + _pt.y*perspec_mat(0,1) );
    perspec_mat(1,2) =  PATCH_SIZE/2 - ( _pt.x*perspec_mat(1,0) + _pt.y*perspec_mat(1,1) );
    cv::warpPerspective( rgb_img_, img_after, perspec_mat, cv::Size(PATCH_SIZE,PATCH_SIZE) );
    cv::cvtColor( img_after, _feature_patch, cv::COLOR_BGR2BGRA );
//    cv::imshow( "perspectiveTransform", _feature_patch );
//    cv::waitKey();
    int neighbor_num = PATCH_SIZE*PATCH_SIZE;
    int total_num = PATCH_SIZE*PATCH_SIZE;
/*/

    /// 2. Perspective projection from 3D to 2D
    int total_num = 0;
    cv::Mat_< cv::Vec<int,5> > patch_blur( PATCH_SIZE, PATCH_SIZE, cv::Vec<int, 5>(0,0,0,0,-SPATIAL_R*2) );//r,g,b,weight,proj_z
    int hints_r = SPATIAL_R * camera_fx / _pt3d.z;
    int start_x = _pt2d.x-hints_r, end_x = _pt2d.x+hints_r;
    int start_y = _pt2d.y-hints_r, end_y = _pt2d.y+hints_r;
    if( start_x<0 ) start_x = 0;
    if( start_y<0 ) start_y = 0;
    if( end_x>=width ) end_x = width-1;
    if( end_y>=height ) end_y = height-1;
    int ind_row = start_y*width + start_x;
    for(int y=start_y; y<=end_y; y++)
    {
    int ind = ind_row;
    ind_row += width;
    for(int x=start_x; x<=end_x; x++)
    {
        const pcl::PointXYZRGB cur_point = cloud_->at(ind);
        ind ++;
        cv::Vec3i ptn( cur_point.x-_pt3d.x, cur_point.y-_pt3d.y, cur_point.z-_pt3d.z );
        if( ptn.dot(ptn)>SPATIAL_R2 )
            continue;

        const int xn = cur_point.x - _pt3d.x + 0.5;
        const int yn = cur_point.y - _pt3d.y + 0.5;
        const int zn = cur_point.z - _pt3d.z + 0.5;
        const int proj_x = ( xn*rotat_mat32S256(0,0) + yn*rotat_mat32S256(0,1) + zn*rotat_mat32S256(0,2) ) / 256;
        const int proj_y = ( xn*rotat_mat32S256(1,0) + yn*rotat_mat32S256(1,1) + zn*rotat_mat32S256(1,2) ) / 256;
        const int proj_z = ( xn*rotat_mat32S256(2,0) + yn*rotat_mat32S256(2,1) + zn*rotat_mat32S256(2,2) ) / 256;

        ///test
//        if( proj_x!=0 && proj_y!=0 )
//        {
//            double proj_ratio = sqrt( xn*xn + yn*yn + zn*zn ) / sqrt( proj_x*proj_x + proj_y*proj_y );
//            proj_x *= proj_ratio;
//            proj_y *= proj_ratio;
//        }

        ///test
//        uint32_t cur_color = proj_z *256 / (int)SPATIAL_RADIUS;
//        if     ( cur_color >  127 ) cur_color =  127;
//        else if( cur_color < -127 ) cur_color = -127;
//         cur_color = cur_color + 127;
//        cur_color = cur_color<<16 | cur_color<<8 | cur_color<<0;

        const int x_pixel = proj_x * W2P_RATIO_256 /256 + PATCH_R;
        const int y_pixel = proj_y * W2P_RATIO_256 /256 + PATCH_R;

       /// feature-patch blur
        const uint id_r = std::min( (uint)cur_point.z/100, (uint)blur_r_.size()-1 );
        const int blur_r = blur_r_[id_r];
        int start_x = x_pixel-blur_r, end_x = x_pixel+blur_r;
        int start_y = y_pixel-blur_r, end_y = y_pixel+blur_r;
        if( start_x<0 ) start_x = 0;
        if( start_y<0 ) start_y = 0;
        if( end_x>=PATCH_SIZE ) end_x = PATCH_SIZE-1;
        if( end_y>=PATCH_SIZE ) end_y = PATCH_SIZE-1;
        uchar* p_row =  patch_blur.data + start_y*patch_blur.step[0] + start_x*patch_blur.step[1];
        for(int y=start_y; y<=end_y; y++)
        {
            int *p_pixel = (int*)p_row;
            for(int x=start_x; x<=end_x; x++)
            {
                int &r = p_pixel[2], &g = p_pixel[1], &b = p_pixel[0];
                int &w = p_pixel[3], &z = p_pixel[4];
                if( proj_z > z + SPATIAL_R/16 )//The nearer point replaces the farer point
                {
                    r = cur_point.r;
                    g = cur_point.g;
                    b = cur_point.b;
                    z = proj_z;
                    w = 1;
                }
                else//blur
                {
                    r += cur_point.r;
                    g += cur_point.g;
                    b += cur_point.b;
                    if( z < proj_z )
                        z = proj_z;
                    w ++;
                }
                p_pixel += 5; // five channels of integer
            }
            p_row += patch_blur.step[0];
        }
        total_num ++;
    }
    }
    uchar* p_row_blur =  patch_blur.data;
    uchar* p_row_patch =  _feature_patch.data;
    for(int y=0; y<PATCH_SIZE; y++)
    {
        int *p_blur = (int*)p_row_blur;
        uchar *p_patch = p_row_patch;
        for(int x=0; x<PATCH_SIZE; x++)
        {
            int &r = p_blur[2], &g = p_blur[1], &b = p_blur[0];
            int &w = p_blur[3], &z = p_blur[4];
            if( w==1 )
                p_patch[0] = b, p_patch[1] = g, p_patch[2] = r;
            else if( w!=0 )
                p_patch[0] = b/w, p_patch[1] = g/w, p_patch[2] = r/w;
            p_blur  += 5; // five channels of integer
            p_patch += 4; // four channels of byte
        }
        p_row_blur += patch_blur.step[0];
        p_row_patch += _feature_patch.step[0];
    }

    if(SHOW_TIME_INFO)
    {
        gettimeofday(&timen,NULL);
        std::cout << " Project "  << total_num << "pts in " << (timen.tv_sec-timel.tv_sec)*1000+(timen.tv_usec-timel.tv_usec)/1000.0 << "ms.";
        std::cout << " Totla:" << (timen.tv_sec-time0.tv_sec)*1000+(timen.tv_usec-time0.tv_usec)/1000.0<<"ms"<<std::endl;
    }
    return total_num;
}

uint
PerspectiveInvariantFeature::sampleCubeEvenly( const cv::Point3f &_pt3d, const cv::Vec4d _plane_coef , std::vector<cv::Vec3i> & _cube, const double &_main_angle )
const
{
    const uint LAYERS = 3;
    ///Calculate perspective projection matrix
    cv::Mat_<double> rotat_mat(3,3);
    cv::Vec3d axes_morm( _plane_coef[0], _plane_coef[1], _plane_coef[2] );
    cv::Vec3d axes_view( -_pt3d.x, -_pt3d.y, -_pt3d.z );
    axes_view /= cv::norm(axes_view);
    cv::Vec3d rotat_axes = axes_morm.cross( axes_view );
    double   rotat_angle = asin( cv::norm(rotat_axes) );
    rotat_axes *= rotat_angle / cv::norm(rotat_axes);
    cv::Rodrigues( rotat_axes, rotat_mat );
    if( _main_angle!=0 )
    {
        cv::Mat_<double> rotat_mat2(3,3);
        rotat_axes = cv::Vec3d(0,0,_main_angle);
        cv::Rodrigues( rotat_axes, rotat_mat2 );
        rotat_mat = rotat_mat2 * rotat_mat;
    }
    cv::Mat_<int32_t> rotat_mat32S256(3,3);
    rotat_mat.convertTo( rotat_mat32S256, CV_32S, 256);

    /// Project the 3D points and sample them into the cube
    int BORDER_LENGTH = std::pow(3,LAYERS);
    _cube.resize( BORDER_LENGTH*BORDER_LENGTH*BORDER_LENGTH, cv::Vec3i(0,0,0) );
    std::vector<int> cube_hi_cnt( _cube.size(), 0 );

    pcl::PointXYZRGB center3d;
    center3d.x = _pt3d.x, center3d.y = _pt3d.y, center3d.z = _pt3d.z;
    std::vector<int> pointIndicesOut;
    std::vector<float> pointRadiusSquaredDistance;
    kdtree_.radiusSearch( center3d, SPATIAL_R, pointIndicesOut, pointRadiusSquaredDistance );///std::max(cos(rotat_angle),0.5)
    int neighbor_num = pointIndicesOut.size();

    uint valid_cubes = 0;
    for(int i=0; i< neighbor_num; i++)
    {
        pcl::PointXYZRGB &cur_point = cloud_->at(pointIndicesOut[i]);
        int xn = cur_point.x - center3d.x + 0.5;
        int yn = cur_point.y - center3d.y + 0.5;
        int zn = cur_point.z - center3d.z + 0.5;
        long proj_x = ( xn*rotat_mat32S256(0,0) + yn*rotat_mat32S256(0,1) + zn*rotat_mat32S256(0,2) ) / 256;
        long proj_y = ( xn*rotat_mat32S256(1,0) + yn*rotat_mat32S256(1,1) + zn*rotat_mat32S256(1,2) ) / 256;
        long proj_z = ( xn*rotat_mat32S256(2,0) + yn*rotat_mat32S256(2,1) + zn*rotat_mat32S256(2,2) ) / 256;
        int cube_idx = (proj_x+SPATIAL_R) * BORDER_LENGTH / (SPATIAL_R*2+1);
        int cube_idy = (proj_y+SPATIAL_R) * BORDER_LENGTH / (SPATIAL_R*2+1);
        int cube_idz = (proj_z+SPATIAL_R) * BORDER_LENGTH / (SPATIAL_R*2+1);
        assert( cube_idx<BORDER_LENGTH && cube_idy<BORDER_LENGTH && cube_idz<BORDER_LENGTH );
        int cube_id = cube_idz*BORDER_LENGTH*BORDER_LENGTH + cube_idy*BORDER_LENGTH + cube_idx;
        _cube[ cube_id ] += cv::Vec3i( cur_point.r, cur_point.g, cur_point.b );
        cube_hi_cnt[ cube_id ] ++;
    }
    for(size_t i=0; i<_cube.size(); i++ )
        if( cube_hi_cnt.at(i) > 0 )
        {
            _cube[i] /= cube_hi_cnt[i];
            valid_cubes ++;
        }

    return valid_cubes;
}

std::vector<cv::Vec3i> PerspectiveInvariantFeature::PyramidCube( const std::vector<cv::Vec3i> & _cube_hi_res )
{

    const int BORDER_LENGTH = std::pow( _cube_hi_res.size(), 1.0/3 ) + 0.5;
    const uint LAYERS = std::log(BORDER_LENGTH)/std::log(3) + 0.5;
    std::vector<cv::Vec3i> _cube;
    _cube.resize( LAYERS*26 +1 );//每层是一个26临域
    uint cube_id = 0;
    _cube[ cube_id++ ] = _cube_hi_res[ _cube_hi_res.size()/2 ];
    for( size_t l=1; l<=LAYERS; l++ )
    {
        const uint cell_length = std::pow(3,l-1);
        for(int neighbor_x=-1; neighbor_x<=1; neighbor_x++ )
        for(int neighbor_y=-1; neighbor_y<=1; neighbor_y++ )
        for(int neighbor_z=-1; neighbor_z<=1; neighbor_z++ )
        {///26 neighbors
            if( neighbor_x==0 && neighbor_y==0 && neighbor_z==0 )
                continue;
            const uint cell_center_x = neighbor_x * cell_length + BORDER_LENGTH/2;
            const uint cell_center_y = neighbor_y * cell_length + BORDER_LENGTH/2;
            const uint cell_center_z = neighbor_z * cell_length + BORDER_LENGTH/2;
            cv::Vec3i cell_rgb(0,0,0);
            uint cell_rgb_cnt = 0;
            for( int x=(int)cell_center_x-(int)cell_length/2; x<=(int)cell_center_x+(int)cell_length/2; x++ )
            for( int y=(int)cell_center_y-(int)cell_length/2; y<=(int)cell_center_y+(int)cell_length/2; y++ )
            for( int z=(int)cell_center_z-(int)cell_length/2; z<=(int)cell_center_z+(int)cell_length/2; z++ )
            {
                if( x<0 || x>=BORDER_LENGTH || y<0 || y>=BORDER_LENGTH || z<0 || z>=BORDER_LENGTH )
                    continue;
                const cv::Vec3i &cur_rgb = _cube_hi_res[ z*BORDER_LENGTH*BORDER_LENGTH + y*BORDER_LENGTH + x ];
                if( cur_rgb != cv::Vec3i::zeros() )
                {
                    cell_rgb_cnt ++;
                    cell_rgb += cur_rgb;
                }
            }
            cell_rgb_cnt = cell_rgb_cnt == 0 ?  1 : cell_rgb_cnt;
            _cube[ cube_id++ ] = cell_rgb / (int)cell_rgb_cnt;
        }
    }
    return _cube;
}


uint PerspectiveInvariantFeature::calcFeatureDir(const cv::Mat& _feature_patch, cv::Point2d &_main_dir, const double &_dense_thresh)
{
    /// 1. Merge the color of the pixels per mask cell
    std::vector<int64> color_r(annular_mask_->TOTAL_CELLS,0);
    std::vector<int64> color_g(annular_mask_->TOTAL_CELLS,0);
    std::vector<int64> color_b(annular_mask_->TOTAL_CELLS,0);
    std::vector<int> color_weight(annular_mask_->TOTAL_CELLS,0);
    for(int h=0; h<_feature_patch.rows; h++)
    {
        uchar * p_cur_patch = _feature_patch.data + h*_feature_patch.step[0];
        for(int w=0; w<_feature_patch.cols; w++)
        {
            if( *(uint32_t*)p_cur_patch != 0 )
            {
                uchar pos = annular_mask_->getCellID(w,h);
                if(pos!=annular_mask_->BAD_CELL)
                {
                    color_r[pos] += p_cur_patch[2];
                    color_g[pos] += p_cur_patch[1];
                    color_b[pos] += p_cur_patch[0];
                    color_weight[pos] ++;
                }
            }
            p_cur_patch += _feature_patch.step[1];
        }
    }


    /// 2. Calculate the main direction of the feature patch
    std::vector<uint> dir_bright(annular_mask_->ANGLE_RES,0);// Intensity of all the directions calculated by the intensity of the cells
    cv::Point2d main_dir_vec(0,0);//main intensity direction
    cv::Point2d valid_dir_vec(0,0);//
    uint valid_cells_cnt = 0;
    const int dense_thresh = PATCH_SIZE*PATCH_SIZE/annular_mask_->TOTAL_CELLS * _dense_thresh;
    for(uint i=0; i<annular_mask_->TOTAL_CELLS; i++)
    {
        if( color_weight[i] > dense_thresh )
        {
            color_r[i] /= color_weight[i];
            color_g[i] /= color_weight[i];
            color_b[i] /= color_weight[i];

            int64 &r = color_r[i];
            int64 &g = color_g[i];
            int64 &b = color_b[i];
            int I = (r+g+b)/3;
            if( i!=0 )    //i==0 means the center cell
            {
//                if( i > annular_mask_->ANGLE_RES )//step over the first layer
                if( i <= annular_mask_->ANGLE_RES ) //only use  the first layer
                    dir_bright[(i-1)%annular_mask_->ANGLE_RES] +=  I;//[0~255]
                main_dir_vec += annular_mask_->DIR_PER_CELL[i] * I;
                valid_dir_vec+= annular_mask_->DIR_PER_CELL[i];
            }
            valid_cells_cnt ++;
        }
     }
    if( valid_cells_cnt <= 6 )
        return 0;

    /// 3. Check fake keypoits. Check if the border of the patch is straight, which means there are 50% valid directions in the patch
    uint main_cnt = 0;
    uint else_cnt = 0;
    for(size_t i=0; i<dir_bright.size(); i++)
    {
        if(dir_bright[i])
        {
            cv::Point2d cur_dir = annular_mask_->DIR_PER_CELL[i+1];
            double cos_angle = cur_dir.dot( valid_dir_vec ) / hypot(cur_dir.x,cur_dir.y) / hypot(valid_dir_vec.x,valid_dir_vec.y);
            if( cos_angle > cos( (90-20)*M_PI/180.0 ) )
                main_cnt ++;
            else if( cos_angle < cos( (90+20)*M_PI/180.0 ) )
                else_cnt ++;
        }
    }
    if( main_cnt > annular_mask_->ANGLE_RES*0.35 && else_cnt <= 2 )//fake keypoint check
    {
//        double cos_theta = valid_dir_vec.dot( main_dir_vec ) / hypot(valid_dir_vec.x,valid_dir_vec.y) / hypot(main_dir_vec.x,main_dir_vec.y);
//        if( fabs(cos_theta)>cos(M_PI/9) )// The main direction and the valid direction are the same, means the color are even along the border.
            return 0;
    }

    _main_dir = main_dir_vec;
    return valid_cells_cnt;
}

uint
PerspectiveInvariantFeature::generateFeatureCode(const cv::Mat& _feature_patch, const cv::Point2d &_main_dir, cv::Mat& _color_code, const double &_dense_thresh)
{
    double main_dir_deg = _main_dir.x==0 ? (_main_dir.y>0?M_PI_2:3*M_PI_2) : atan( _main_dir.y / _main_dir.x );
    if( _main_dir.x<0 ) main_dir_deg += M_PI;
    if( main_dir_deg<0 ) main_dir_deg += 2*M_PI;
    main_dir_deg = main_dir_deg*180.0/M_PI;
    /// 1. Merge the color of the pixels per mask cell
    std::vector<uint64>    color_r(patch_mask_->TOTAL_CELLS,0);
    std::vector<uint64>    color_g(patch_mask_->TOTAL_CELLS,0);
    std::vector<uint64>    color_b(patch_mask_->TOTAL_CELLS,0);
    std::vector<uint> color_weight(patch_mask_->TOTAL_CELLS,0);
    for(int h=0; h<_feature_patch.rows; h++)
    {
        uchar * p_cur_patch = _feature_patch.data + h*_feature_patch.step[0];
        for(int w=0; w<_feature_patch.cols; w++)
        {
            if( *(uint32_t*)p_cur_patch != 0 )
            {
                uchar cell_id = patch_mask_->getCellID( w, h, main_dir_deg );
                if( cell_id != patch_mask_->BAD_CELL )
                {
                    color_r[cell_id] += p_cur_patch[2];
                    color_g[cell_id] += p_cur_patch[1];
                    color_b[cell_id] += p_cur_patch[0];
                    color_weight[cell_id] ++;
                }
            }
            p_cur_patch += _feature_patch.step[1];
        }
    }
    /// 2. Color coding of each cell
    if( _color_code.cols!=(int)patch_mask_->TOTAL_CELLS || _color_code.type()!=color_encoder_.code_type_ )
        _color_code.create( 1, patch_mask_->TOTAL_CELLS, color_encoder_.code_type_ );
    uint mean_V = 0, pixel_cnt = 0;
    for(uint i=0; i<patch_mask_->TOTAL_CELLS; i++)
    {
        mean_V += ( color_r[i] +  color_g[i] +  color_b[i] ) / 3;
        pixel_cnt += color_weight[i];
    }
    if( pixel_cnt==0 ) return 0;
    mean_V /= pixel_cnt;
    uint valid_cells_cnt = 0;
    const uint dense_thresh = PATCH_SIZE*PATCH_SIZE/patch_mask_->TOTAL_CELLS * _dense_thresh;
    uchar *p_code = _color_code.data;
    for(uint i=0; i<patch_mask_->TOTAL_CELLS; i++)
    {
        if( color_weight[i] > dense_thresh )
        {
            color_r[i] /= color_weight[i];
            color_g[i] /= color_weight[i];
            color_b[i] /= color_weight[i];
            uint64 &r = color_r[i];
            uint64 &g = color_g[i];
            uint64 &b = color_b[i];
            color_encoder_.encode( p_code, r, g, b );
            valid_cells_cnt ++;
        }
        else
            color_encoder_.invalidCode( p_code );
        p_code += _color_code.step[1];
     }
    return valid_cells_cnt;
}

uint PerspectiveInvariantFeature::generateFeatureCode_hov(const cv::Mat& _feature_patch, cv::Mat& _color_code, const uchar& _method)
{
    if( _color_code.cols != (int)patch_mask_->TOTAL_CELLS || _color_code.type() != CV_8UC1 )
        _color_code.create( 1, patch_mask_->TOTAL_CELLS, CV_8UC1 );

    std::vector<uint> color_hist(64,0);
    for(int h=0; h<_feature_patch.rows; h++)
    {
        uchar * p_cur_patch = _feature_patch.data + h*_feature_patch.step[0];
        for(int w=0; w<_feature_patch.cols; w++)
        {
            if( h*h+w*w < _feature_patch.rows*_feature_patch.cols )
            if( *(uint32_t*)p_cur_patch != 0 )
            {
                switch (_method)
                {
                case 0:
                default:
                    uchar id = color_encoder_.rgb2IntCode( p_cur_patch[0], p_cur_patch[1], p_cur_patch[2], 6 );
//                    std::cout<<"id="<<(int)id<<std::endl;
//                    id = ( (int)p_cur_patch[0] + p_cur_patch[1] + p_cur_patch[2] ) / 3 * 64 / 256;
                    color_hist[id] ++;
                    break;
                }
            }
            p_cur_patch += _feature_patch.step[1];
        }
    }
    for(int i=0; i<64; i++)//nomorlize the histogram
        _color_code.at<uchar>(0,i) = color_hist[i] * 255 / (_feature_patch.rows*_feature_patch.cols);
    return 64;
}
bool
PerspectiveInvariantFeature::prepareFrame( const cv::Mat _rgb_image, const cv::Mat _depth_16U)
{
    const bool SHOW_TIME_INFO = false;
    timeval time_start, time_temp;
    gettimeofday(&time_start,NULL);

    assert( _rgb_image.type()==CV_8UC1 || _rgb_image.type()==CV_8UC3 );
    assert( _depth_16U.channels()==1 );
    if( _depth_16U.type()!=CV_16UC1 ) _depth_16U.convertTo(depth_16U_,CV_16U);
    else                              _depth_16U.copyTo(   depth_16U_ );
    height = _depth_16U.rows;
    width  = _depth_16U.cols;
    assert( width>0 && height>0 );
    assert( width==_rgb_image.cols && height==_rgb_image.rows );
    if( _rgb_image.channels()==1 )
        cv::cvtColor(_rgb_image,rgb_img_,CV_GRAY2RGB);
    else if( _rgb_image.channels()==3 )
        rgb_img_ = _rgb_image.clone();
    rgb_show_ = rgb_img_.clone();

    /// generat the 3D point cloud
    if(!cloud_)
    {
      cloud_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>(width, height));
      cloud_->is_dense = false;
    }
    else if( cloud_->width != width || cloud_->height != height )
    {
        cloud_->resize( width*height );
        cloud_->width = width;
        cloud_->height= height;
    }
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for( int y = 0; y < (int)height; ++y )
    {
        uchar *prgb   = _rgb_image.data + y*_rgb_image.step[0];
        uchar *pdepth = depth_16U_.data + y*depth_16U_.step[0];
        pcl::PointXYZRGB *p_cloud = &cloud_->points[0] + y*width;
        for(int x = 0; x < (int)width; x++ )
        {
            const u_int16_t &depth_current = *(u_int16_t*)pdepth;// / 1000.0f;
            if( std::isnan(depth_current) || depth_current<10 )
            {
                p_cloud->x = p_cloud->y = p_cloud->z = std::numeric_limits<float>::quiet_NaN();
            }
            else
            {
                p_cloud->z = depth_current;
                p_cloud->x = (x-camera_cx+0.5) * _1_camera_fx * depth_current;
                p_cloud->y = (y-camera_cy+0.5) * _1_camera_fy * depth_current;
                p_cloud->b = prgb[0];
                p_cloud->g = prgb[1];
                p_cloud->r = prgb[2];
            }
            prgb   += _rgb_image.step[1];
            pdepth += depth_16U_.step[1];
            p_cloud++;
        }
    }
    if(!normal_est_)
    {
        normals_ = (pcl::PointCloud<pcl::Normal>::Ptr) new pcl::PointCloud<pcl::Normal>;
        normal_est_ = (pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB,pcl::Normal>::Ptr) new pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB,pcl::Normal>;
        normal_est_->setNormalEstimationMethod( normal_est_->COVARIANCE_MATRIX);//AVERAGE_3D_GRADIENT
        normal_est_->setMaxDepthChangeFactor( SPATIAL_R );//threshold for computing object borders
        normal_est_->setNormalSmoothingSize( 50 );//unit: pixel
    }
    normal_est_->setInputCloud( cloud_ );
    normal_est_->compute( *normals_ );
    kdtree_.setInputCloud (cloud_);

    gettimeofday(&time_temp,NULL);
    int init_time = (time_temp.tv_sec-time_start.tv_sec)*1000+(time_temp.tv_usec-time_start.tv_usec)/1000;
    if( SHOW_TIME_INFO )
        std::cout << "prepare time(ms):" << init_time << std::endl;

    return true;
}
cv::Mat
PerspectiveInvariantFeature::process( std::vector<cv::KeyPoint> &m_keypoints )
{
    const bool SHOW_TIME_INFO = false;
    timeval time_start;
    gettimeofday(&time_start,NULL);

    keypoints_filtered_.clear();
    keypoints_3D_.clear();
    features_show_.setTo(0);
    features_restore_.setTo(0);

    uint keypoint_valid_cnt =0;
    uint fake_point_cnt[3] = {0};

    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for( int key_id = 0; key_id < (int)m_keypoints.size() && keypoint_valid_cnt < MAX_KEYPOINTS; key_id++)
    {
        cv::Point2f keypoint2d = m_keypoints[key_id].pt;

        ///1. extract feature patch: cur_patch
        cv::Point3f keypoint3d;
        cv::Mat cur_patch;
        cv::Vec4d plan_coef(0,0,0,0);//Ax+By+Cz=D ,(A B C) is the normal vector
        double plane_curvature;
        cur_patch = cv::Mat::zeros( PATCH_SIZE, PATCH_SIZE, CV_8UC4 );

        uint front_pt_num;
        front_pt_num = calcPt6d( keypoint2d, keypoint3d, plan_coef, plane_curvature );//0.03ms
        if( !front_pt_num  )
        {
            fake_point_cnt[0] ++;
//            std::cout<< "\t\t\t\t\tPoint loss data."<< std::endl;
            if( DRAW_IMAG ) cv::circle(rgb_show_,  keypoint2d,2,CV_RGB(255,0,0),CV_FILLED );
            continue;
        }

        if(TEST_NOT_USE_PROJECTION)
        {
            cv::Rect roi( keypoint2d.x-PATCH_SIZE/2, keypoint2d.y-PATCH_SIZE/2, PATCH_SIZE, PATCH_SIZE );
            cv::cvtColor( cv::Mat(rgb_img_, roi), cur_patch, CV_BGR2BGRA );
        }
        else
            warpPerspectivePatch( keypoint2d, keypoint3d, plan_coef, cur_patch );
        bool is_fake = fabs(plan_coef[2]) < sin(M_PI/20) && plane_curvature<SPATIAL_R;
        if( is_fake && !TEST_NOT_USE_FAKE_FILTER )
        {
            fake_point_cnt[1] ++;
//            std::cout<< "\t\t\t\t\tPoint at curve."<< std::endl;
            if( DRAW_IMAG ) cv::circle(rgb_show_,  keypoint2d,2,CV_RGB(255,0,255),CV_FILLED );
            continue;
        }

        ///2. calculate the main direction of the feature patch
        cv::Point2d main_dir_vec(1,0);
        is_fake = !calcFeatureDir( cur_patch, main_dir_vec );
        if( is_fake && !TEST_NOT_USE_FAKE_FILTER )
        {
//            if( DRAW_IMAG ) cv::circle(rgb_show_,  keypoint2d,2,CV_RGB(100,100,255),CV_FILLED );
//            cv::imshow( "img", rgb_show_ );
//            cv::imshow( "fake point", cur_patch );
//            cv::waitKey();
            if( DRAW_IMAG ) cv::circle(rgb_show_,  keypoint2d,2,CV_RGB(255,0,255),CV_FILLED );
            fake_point_cnt[2] ++;
            continue;
        }
        double main_dir_rad = main_dir_vec.x==0 ? (main_dir_vec.y>0?M_PI_2:3*M_PI_2) : atan( main_dir_vec.y / main_dir_vec.x );
        if( main_dir_vec.x<0 ) main_dir_rad += M_PI;
        if( main_dir_rad<0 ) main_dir_rad += 2*M_PI;
        double main_dir_deg = main_dir_rad*(180.0/M_PI);

        std::vector<cv::Vec3i> cube3;
        if( patch_type_ == D_TYPE_CUBE3 )
        {
            sampleCubeEvenly( keypoint3d, plan_coef, cube3, 0 );
            std::cout << "cube size=" << cube3.size();
            cube3 = PyramidCube( cube3 );
            std::cout << " ->" << cube3.size() << std::endl;
        }
        ///3. Generate the descriptor from the feature patch
        cv::Mat cur_descriptor;
        switch (patch_type_)
        {
        case D_TYPE_ANNULAR:
        case D_TYPE_BEEHIVE:
        {
            /// color coding
            const uint valid_cells_cnt = generateFeatureCode( cur_patch, main_dir_vec, cur_descriptor );
            if( valid_cells_cnt < patch_mask_->TOTAL_CELLS/16.0 )
            {
//                std::cout<< "\t\t\t\t\tSmall Patch."<< std::endl;
                if( DRAW_IMAG ) cv::circle( rgb_show_, keypoint2d, 2, CV_RGB(0,0,255), CV_FILLED );
                continue;
            }
        }
            break;
        case D_TYPE_BRIEF:
        {
            cv::BriefDescriptorExtractor brief_extractor(32);
            const uint PATCH_SIZEE = 63;
            cv::Mat image_temp;
            cv::resize(cur_patch,image_temp,cv::Size(PATCH_SIZEE,PATCH_SIZEE),0,0,cv::INTER_NEAREST);
            cv::vector<cv::KeyPoint> key_point_temp( 1, m_keypoints[key_id] );
            key_point_temp[0].pt.x = image_temp.cols/2;
            key_point_temp[0].pt.y = image_temp.rows/2;
            key_point_temp[0].size = PATCH_SIZEE;
            key_point_temp[0].angle= main_dir_deg;
            brief_extractor.compute(image_temp, key_point_temp, cur_descriptor);
            if( !cur_descriptor.rows )
                continue;
        }
            break;
        case D_TYPE_ORB:
        {
            cv::OrbDescriptorExtractor orb_extractor;
            const uint PATCH_SIZEE = 63;
            cv::Mat image_temp;
            cv::resize(cur_patch,image_temp,cv::Size(PATCH_SIZEE,PATCH_SIZEE),0,0,cv::INTER_NEAREST);
            cv::vector<cv::KeyPoint> key_point_temp( 1, m_keypoints[key_id] );
            key_point_temp[0].pt.x = image_temp.cols/2;
            key_point_temp[0].pt.y = image_temp.rows/2;
            key_point_temp[0].size = PATCH_SIZEE;
            key_point_temp[0].angle= main_dir_deg;
            orb_extractor( image_temp, cv::Mat::ones(image_temp.rows,image_temp.cols,CV_8UC1), key_point_temp, cur_descriptor, true);
            if( !cur_descriptor.rows )
                continue;
        }
            break;
        case D_TYPE_BRISK:
        {
            cv::BRISK brisk_extractor;
            const uint PATCH_SIZEE = PATCH_SIZE;
            cv::Mat image_temp;
            cv::resize(cur_patch,image_temp,cv::Size(PATCH_SIZEE,PATCH_SIZEE),0,0,cv::INTER_NEAREST);
            cv::vector<cv::KeyPoint> key_point_temp( 1, m_keypoints[key_id] );
            key_point_temp[0].pt.x = image_temp.cols/2;
            key_point_temp[0].pt.y = image_temp.rows/2;
            key_point_temp[0].size = PATCH_SIZEE;
            key_point_temp[0].angle= main_dir_deg;
            brisk_extractor( image_temp, cv::Mat::ones(image_temp.rows,image_temp.cols,CV_8UC1), key_point_temp, cur_descriptor, true);
            if( !cur_descriptor.rows )
                continue;
        }
            break;
        case D_TYPE_SURF:
        {
            cv::SurfDescriptorExtractor surf_extractor;
            const uint PATCH_SIZEE = 63;
            cv::Mat image_temp;
            cv::resize(cur_patch,image_temp,cv::Size(PATCH_SIZEE,PATCH_SIZEE),0,0,cv::INTER_NEAREST);
            cv::vector<cv::KeyPoint> key_point_temp( 1, m_keypoints[key_id] );
            key_point_temp[0].pt.x = image_temp.cols/2;
            key_point_temp[0].pt.y = image_temp.rows/2;
            key_point_temp[0].size = PATCH_SIZEE;
            key_point_temp[0].angle= main_dir_deg;
            surf_extractor( image_temp, cv::Mat::ones(image_temp.rows,image_temp.cols,CV_8UC1), key_point_temp, cur_descriptor, true);
            if( !cur_descriptor.rows )
                continue;
        }
            break;
        case D_TYPE_HISTOGRAM:
            generateFeatureCode_hov( cur_patch, cur_descriptor );
            break;
        case D_TYPE_CUBE3:
        {
            assert( cube3.size()>0 );
            cur_descriptor.create( 1, cube3.size(), color_encoder_.code_type_ );
            uchar *p_code = cur_descriptor.data;
            for(uint i=0; i<cube3.size(); i++)
            {
                int &r = cube3.at(i)(0);
                int &g = cube3.at(i)(1);
                int &b = cube3.at(i)(2);
                if( r>0 && g>0 && b>0 )
                {
                    color_encoder_.encode( p_code, r, g, b );
                }
                else
                    color_encoder_.invalidCode( p_code );
                p_code += cur_descriptor.step[1];
             }
        }
            break;
        default:
        {
        }
            break;
        }
        /// 3. Save the descriptor into the matrix
        #ifdef USE_OPENMP
        #pragma omp critical
        #endif
        {
        if( keypoint_valid_cnt < MAX_KEYPOINTS )
        {
            if( descriptors_.rows  != (int)MAX_KEYPOINTS
             || descriptors_.cols  != cur_descriptor.cols
             || descriptors_.type()!= cur_descriptor.type() )
                descriptors_.create( MAX_KEYPOINTS, cur_descriptor.cols, cur_descriptor.type() );

            memcpy( descriptors_.data+descriptors_.step[0]*keypoint_valid_cnt, cur_descriptor.data, cur_descriptor.step[0] );
            keypoints_filtered_.push_back( m_keypoints[key_id] );
            keypoints_3D_.push_back( keypoint3d );

            if( DRAW_IMAG )
            {
                const uint x_max = features_show_.cols / PATCH_SIZE;
                const uint y_max = features_show_.rows / PATCH_SIZE;
                if( keypoint_valid_cnt < x_max*y_max )
                {
                    cv::line( cur_patch, cv::Point(PATCH_SIZE/2,PATCH_SIZE/2), cv::Point(PATCH_SIZE/2+main_dir_vec.x,PATCH_SIZE/2-main_dir_vec.y),CV_RGB(255,0,0) );
                    cv::Rect draw_mask = cv::Rect( (keypoint_valid_cnt%x_max)*PATCH_SIZE,
                                                   (keypoint_valid_cnt/y_max)*PATCH_SIZE,
                                                    PATCH_SIZE, PATCH_SIZE );
                    cur_patch.copyTo( cv::Mat(features_show_,draw_mask) );
                }
            }
            keypoint_valid_cnt++;
        }
        }

        if( DRAW_IMAG )
            cv::circle(rgb_show_,  keypoint2d,2,CV_RGB(0,255,0),CV_FILLED );

    }//end of for(it_keypoints=m_keypoints.begin(); it_keypoints!=m_keypoints.end(); it_keypoints++)



    if(SHOW_TIME_INFO)
    {
        std::cout <<"KeyPointNum:"<<m_keypoints.size()<<"-"<<fake_point_cnt[0]<<"-"<<fake_point_cnt[1]<<"-"<<fake_point_cnt[2]<<"="<<keypoints_filtered_.size();
        timeval time_end;
        gettimeofday(&time_end,NULL);
        int total_time = (time_end.tv_sec-time_start.tv_sec)*1000+(time_end.tv_usec-time_start.tv_usec)/1000;
        std::cout << "Total(ms):" << total_time << "="<< m_keypoints.size() << "*" << (double)(total_time)/m_keypoints.size() << std::endl<< std::endl;
    }
    m_keypoints = keypoints_filtered_;
    return cv::Mat( descriptors_, cv::Rect(0, 0, descriptors_.cols, m_keypoints.size()) ).clone();
}

cv::Mat
PerspectiveInvariantFeature::processFPFH(std::vector<cv::KeyPoint> &m_keypoints, const uint &SPATIAL_RADIUS )
{
    timeval time_start, time_temp;
    gettimeofday(&time_start,NULL);

    keypoints_filtered_.clear();
    keypoints_3D_.clear();
    features_show_.setTo(0);
    features_restore_.setTo(0);

    gettimeofday(&time_temp,NULL);
//    int init_time = (time_temp.tv_sec-time_start.tv_sec)*1000+(time_temp.tv_usec-time_start.tv_usec)/1000;


    ///fpfh
    pcl::FPFHEstimation<pcl::PointXYZRGB,pcl::Normal,pcl::FPFHSignature33> fpfh_est;
    fpfh_est.setSearchSurface( cloud_->makeShared() );
    fpfh_est.setInputNormals( normals_->makeShared() );
    fpfh_est.setRadiusSearch( SPATIAL_RADIUS );

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr key_points( new pcl::PointCloud<pcl::PointXYZRGB>(1,m_keypoints.size()) );
    int valid_cnt = 0;
    for( int key_id = 0; key_id < (int)m_keypoints.size(); key_id++)
    {
        cv::Point2f keypoint = m_keypoints[key_id].pt;
        const uint &border = 50;
        if( keypoint.x<border || keypoint.y<border || keypoint.x>width-1-border || keypoint.y>height-1-border )
            continue;
        cv::Point3f keypoint_3d;
        cv::Vec4d plan_coef(0,0,0,0);//Ax+By+Cz=D ,(A B C) is the normal vector
        double plane_err;
        uint front_pt_num = calcPt6d( keypoint, keypoint_3d, plan_coef, plane_err );
        if( front_pt_num>0 )
        {
            key_points->at(valid_cnt).x = keypoint_3d.x;
            key_points->at(valid_cnt).y = keypoint_3d.y;
            key_points->at(valid_cnt).z = keypoint_3d.z;
            keypoints_filtered_.push_back( m_keypoints[key_id] );
            valid_cnt ++;
        }
    }
    key_points->resize( valid_cnt );
    m_keypoints = keypoints_filtered_;

    fpfh_est.setInputCloud( key_points->makeShared() );
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_signature(new pcl::PointCloud<pcl::FPFHSignature33>() );
    fpfh_est.compute( *fpfh_signature );

    if( descriptors_.rows  != (int)MAX_KEYPOINTS
     || descriptors_.cols  != 33
     || descriptors_.type()!= CV_32F )
        descriptors_.create( MAX_KEYPOINTS, 33, CV_32F );
    for( int i = 0; i < valid_cnt; i++)
        memcpy( descriptors_.data+descriptors_.step[0]*i, fpfh_signature->at(i).histogram, sizeof(pcl::FPFHSignature33) );

    timeval time_end;
    gettimeofday(&time_end,NULL);
//    int total_time = (time_end.tv_sec-time_start.tv_sec)*1000+(time_end.tv_usec-time_start.tv_usec)/1000;
//    std::cout << "FPFH time(ms):" << total_time << "="<< init_time << "+" << m_keypoints.size() << "*" << (double)(total_time-init_time)/m_keypoints.size() << std::endl<< std::endl;

    return cv::Mat( descriptors_, cv::Rect(0, 0, descriptors_.cols, valid_cnt) );
}

bool
PerspectiveInvariantFeature::restore_descriptor(const cv::Mat& _descriptor)
{
    const uint& size = PATCH_SIZE;
    std::vector<uint32_t> cur_color_show;
    features_restore_.setTo(0);
    for(uint cnt=0; cnt<(uint)_descriptor.rows; cnt++)
    {
        cur_color_show.resize(patch_mask_->TOTAL_CELLS,0);
        const uint x_max = features_restore_.cols / size;
        const uint y_max = features_restore_.rows / size;
        cv::Rect cur_patch_mask = cv::Rect( (cnt%x_max)*size, (cnt/y_max)*size, size, size );
        cv::Mat cur_patch;
        if( cnt < x_max*y_max )  cur_patch = cv::Mat(features_restore_, cur_patch_mask );
        else            return true;//cur_patch = cv::Mat::zeros(height, width, CV_8UC4);

        if( patch_type_ == D_TYPE_BEEHIVE || patch_type_ == D_TYPE_ANNULAR )
        {
            /// restore color
            uchar *p_color_code = _descriptor.data + cnt*_descriptor.step[0];
            for(int i=0; i<_descriptor.cols; i++)
            {
                cur_color_show[i] =  color_encoder_.decode( p_color_code );
                p_color_code += _descriptor.step[1];
            }
            /// draw color to the image
            for(uint y=0; y<size; y++)
            {
                uchar* pdata_patch =   cur_patch.data + y*  cur_patch.step[0];
                for(uint x=0; x<size; x++)
                {
                    uchar cell_id = patch_mask_->getCellID(x,y);
                    if( cell_id != patch_mask_->BAD_CELL )
                        *(uint32_t*)pdata_patch = cur_color_show[cell_id];
                    else
                        *(uint32_t*)pdata_patch = 0;
                    pdata_patch +=   cur_patch.step[1];
                }
            }
        }
        else if( patch_type_ == D_TYPE_CUBE3 )
        {
            const int CUBE_RES = std::pow( _descriptor.cols, 1.0/3) + 0.5;
            if( _descriptor.cols != CUBE_RES*CUBE_RES*CUBE_RES )
                return false;
            cv::Mat cube_show = cv::Mat::zeros( CUBE_RES, CUBE_RES, CV_8UC4 );
            uchar *p_color_code = _descriptor.data + cnt*_descriptor.step[0];
            for(int z=0; z<CUBE_RES; z++)
            for(int y=0; y<CUBE_RES; y++)
            {
                uchar *p_img = cube_show.data + y*cube_show.step[0];
                for(int x=0; x<CUBE_RES; x++)
                {
                    uint32_t rgb = color_encoder_.decode( p_color_code );
                    if(   rgb != 0 && *(uint32_t*)p_img==0 )
                        *(uint32_t*)p_img = rgb;
                    p_img += cube_show.step[1];
                    p_color_code += _descriptor.step[1];
                }
            }
            cur_patch = cv::Mat( cur_patch, cv::Rect(1,1,cur_patch.cols-2,cur_patch.rows-2) );//make a blank border
            cv::resize( cube_show, cur_patch, cv::Size(cur_patch.cols,cur_patch.rows), 0, 0, cv::INTER_NEAREST );
        }
        else
            return false;
    }
    return true;
}

CV_WRAP void
PIFTMatcher::matchDescriptors( const cv::Mat& _queryDescriptors, const cv::Mat& _trainDescriptors, CV_OUT std::vector<std::vector<cv::DMatch> >& _matches ) const
{
    std::vector<cv::DMatch> matches_temp;
    matches_temp.reserve( _queryDescriptors.rows );
    uchar *p_from = _queryDescriptors.data;
    for(int id_from=0; id_from<_queryDescriptors.rows; id_from++ )
    {
        uint min_dist = INFINITY;
        int min_id = -1;
        uchar *p_to =  _trainDescriptors.data;
        for(int id_to=0; id_to<_trainDescriptors.rows; id_to++ )
        {
            uint temp_dist = color_encoder_.machCode( p_from, p_to, _trainDescriptors.cols );
            if( temp_dist < min_dist )
            {
                min_dist = temp_dist;
                min_id = id_to;
            }
            p_to += _trainDescriptors.step[0];
        }
        if( min_id != -1 )
            matches_temp.push_back( cv::DMatch( id_from, min_id, min_dist) );
        p_from += _queryDescriptors.step[0];
    }

    _matches.reserve( matches_temp.size() );
    for( std::vector<cv::DMatch>::iterator p_match = matches_temp.begin(); p_match != matches_temp.end();  )
    {
        uchar *p_to =  _trainDescriptors.data + p_match->trainIdx * _trainDescriptors.step[0];
        uint min_dist = p_match->distance;
        bool reject = false;
        uchar *p_from =  _queryDescriptors.data;
        if( cross_check_ )///cross check
        for(int id_from=0; id_from<_queryDescriptors.rows; id_from++ )
        {
            if( id_from != p_match->queryIdx )
            if( min_dist >= color_encoder_.machCode( p_to, p_from, _queryDescriptors.cols ) )
            {
                reject = true;
                break;
            }
            p_from += _queryDescriptors.step[0];
        }
        if( !reject )
            _matches.push_back( std::vector<cv::DMatch>( 1,*p_match ) );
        p_match ++;
    }
}

cv::Ptr<cv::DescriptorMatcher> PIFTMatcher::clone( bool emptyTrainData ) const
{
    PIFTMatcher* matcher = new PIFTMatcher(this->color_encoder_.method_,this->cross_check_);
//    if( !emptyTrainData )
//    {
//        matcher->trainDescCollection.resize(trainDescCollection.size());
//        std::transform( trainDescCollection.begin(), trainDescCollection.end(), matcher->trainDescCollection.begin(), clone_op );
//    }
    return matcher;
}


void PIFTMatcher::knnMatchImpl( const cv::Mat& queryDescriptors, std::vector<std::vector<cv::DMatch> >& matches, int knn,
                              const std::vector<cv::Mat>& masks, bool compactResult )
{
    if( queryDescriptors.empty() || trainDescCollection.empty() )
    {
        matches.clear();
        return;
    }
    CV_Assert( queryDescriptors.type() == trainDescCollection[0].type() );
    matchDescriptors( queryDescriptors, trainDescCollection[0], matches );
}


void PIFTMatcher::radiusMatchImpl( const cv::Mat& queryDescriptors, std::vector<std::vector<cv::DMatch> >& matches,
                                 float maxDistance, const std::vector<cv::Mat>& masks, bool compactResult )
{
    assert( false );
}
