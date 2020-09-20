/* Similar to the xxx_config.h in the xfOpenCV library */

#ifndef _FP_SGBM_ACCEL_H_
#define _FP_SGBM_ACCEL_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.h"
#include "common/xf_utility.h"

#include "lib_accel/fp_sgbm.hpp"
#include "fp_config_params.h"
#include "fp_config_arch.h"

#define IN_T XF_8UC1
#define OUT_T XF_8UC1

#if (NUM_DIR==4)
/* For 4 paths aggregation */
void semiglobalbm_accel(xf::Mat<IN_T, HEIGHT, WIDTH, XF_NPPC1> &_srcL, xf::Mat<IN_T, HEIGHT, WIDTH, XF_NPPC1> &_srcR, xf::Mat<OUT_T, HEIGHT, WIDTH, XF_NPPC1> &_dst);

#endif

#endif  // end of _FP_SGBM_ACCEL_H_
