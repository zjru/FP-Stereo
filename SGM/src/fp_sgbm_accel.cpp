#include "fp_sgbm_accel.h"

#if (NUM_DIR==4)
/* For 4 paths aggregation */
void semiglobalbm_accel(xf::Mat<IN_T, HEIGHT, WIDTH, XF_NPPC1> &_srcL, xf::Mat<IN_T, HEIGHT, WIDTH, XF_NPPC1> &_srcR, xf::Mat<OUT_T, HEIGHT, WIDTH, XF_NPPC1> &_dst)
{
    fp::SemiGlobalBM<WINDOW_SIZE,SHD_WINDOW,NUM_DISPARITY,PARALLEL_DISPARITIES,FilterWin,IN_T,OUT_T,HEIGHT,WIDTH,XF_NPPC1,SMALL_PENALTY,LARGE_PENALTY>(_srcL,_srcR,_dst);
}
#endif		
