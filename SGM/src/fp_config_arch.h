#ifndef _FP_CONFIG_ARCH_H_
#define _FP_CONFIG_ARCH_H_

#ifndef __cplusplus
#error C++ is needed to use this file!
#endif

#include "lib_accel/fp_common.h"

/* Architecture parameters */
/*-------------------------------------To decide the components-------------------------------------*/
/* Number of aggregation directions */
#define NUM_DIR 4

/* Select the cost function */
#define COST_FUNCTION 0

/* Uniqueness check or not */
#define UNIQ 0

/* Left-right check or not */
#define LR_CHECK 0

/*-------------------------------------To decide the interface--------------------------------------*/
/* The bandwidth of HP AXI port for the target platform */
#define MAX_PORT_BW 128

/* Degree of parallelism (PARALLEL_DISPARITIES) */
#define PARALLELISM 16 

/* Penalty (P2) */
#define PENALTY2 96
/* Window size of cost function (WINDOW_SIZE) */
#define COST_WIN 5
/* Window size of shd cost function (SHD_WINDOW) */
#define SHD_WIN 3

#define PORT_BW MAX_PORT_BW
#define EXTRA_BW AGGR_WIDTH(COST_MAP(COST_FUNCTION,8,COST_WIN,SHD_WIN),PENALTY2,PARALLELISM,PORT_BW)


#define LOG_1(n) (((n) >= 2) ? 1 : 0)
#define LOG_2(n) (((n) >= 1<<2) ? (2 + LOG_1((n)>>2)) : LOG_1(n))
#define LOG_4(n) (((n) >= 1<<4) ? (4 + LOG_2((n)>>4)) : LOG_2(n))
#define LOG_8(n) (((n) >= 1<<8) ? (8 + LOG_4((n)>>8)) : LOG_4(n))
#define LOG(n)   (((n) >= 1<<16) ? (16 + LOG_8((n)>>16)) : LOG_8(n))

#if COST_FUNCTION==0
#define DATA_WIDTH LOG(COST_WIN*COST_WIN-1+PENALTY2) + 3	
#elif COST_FUNCTION==1
#define DATA_WIDTH LOG(COST_WIN*COST_WIN-1+PENALTY2) + 3		
#elif COST_FUNCTION==2
#define DATA_WIDTH LOG(255*COST_WIN*COST_WIN+PENALTY2) + 3 		
#elif COST_FUNCTION==3
#define DATA_WIDTH LOG(255*COST_WIN*COST_WIN*2+PENALTY2) + 3
#elif COST_FUNCTION==4
#define DATA_WIDTH LOG((COST_WIN*COST_WIN-1)*SHD_WIN*SHD_WIN+PENALTY2) + 3
#endif

#if (DATA_WIDTH <= (MAX_PORT_BW/PARALLELISM))
#define ARGUMENTS_NUM 1
#elif (DATA_WIDTH > MAX_PORT_BW/PARALLELISM) && (DATA_WIDTH <= (MAX_PORT_BW/PARALLELISM)*2)
#define ARGUMENTS_NUM 2
#elif (DATA_WIDTH > (MAX_PORT_BW/PARALLELISM)*2) && (DATA_WIDTH <= (MAX_PORT_BW/PARALLELISM)*3)
#define ARGUMENTS_NUM 3
#elif (DATA_WIDTH > (MAX_PORT_BW/PARALLELISM)*3) && (DATA_WIDTH <= (MAX_PORT_BW/PARALLELISM)*4)
#define ARGUMENTS_NUM 4
#elif (DATA_WIDTH > (MAX_PORT_BW/PARALLELISM)*4)
#define ARGUMENTS_NUM 5
#endif

#endif //_FP_CONFIG_ARCH_H_