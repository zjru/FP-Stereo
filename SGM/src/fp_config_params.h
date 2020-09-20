#ifndef _FP_CONFIG_PARAMS_H_
#define _FP_CONFIG_PARAMS_H_

#ifndef __cplusplus
#error C++ is needed to use this file!
#endif

/* Algorithmic parameters */
/*-------------------------------------------------------------------------------------------------*/
/* set the height and width */
#define HEIGHT 375
#define WIDTH  1242

/* NO_OF_DISPARITIES must be greater than '0' and less than the image width */
#define NUM_DISPARITY 128

/* set penalties for SGM */
#define SMALL_PENALTY   7
#define LARGE_PENALTY 	86

/* Window size for cost computation: census, rank, SAD, ZSAD...*/
#define WINDOW_SIZE   7 

/* Window size for sum of hamming distance */
#define SHD_WINDOW   3 

/* Median filter window size */
#define FilterWin   5 


/* Gap Interplation threshold */
#define GAP_THRESHOLD NUM_DISPARITY
/*-------------------------------------------------------------------------------------------------*/


/* Hardware parameters */
/*-------------------------------------------------------------------------------------------------*/
/* NO_OF_DISPARITIES must not be lesser than PARALLEL_UNITS and NO_OF_DISPARITIES/PARALLEL_UNITS must be a non-fractional number */
#define PARALLEL_DISPARITIES 16 
/*-------------------------------------------------------------------------------------------------*/


#endif //_FP_CONFIG_PARAMS_H_