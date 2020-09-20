/*
 *  FP-Stereo
 *  Copyright (C) 2020  RCSL, HKUST
 *  
 *  GPL-3.0 License
 *
 */
 
#ifndef _FP_COMMON_H_
#define _FP_COMMON_H_

#ifndef __cplusplus
#error C++ is needed to use this file!
#endif

#include "ap_int.h"
#include "ap_fixed.h"
#include <stdint.h>


/*The maximum value with all bits equal to 1*/
#define MAX_VALUE_BOUND 1048575


/* Compute bitwidth */
template<int N, int Count>
class BITWIDTH_COUNT {
public:
    static const int value = BITWIDTH_COUNT<(N>>1),(Count+1)>::value;
};
template<int Count>
class BITWIDTH_COUNT<1,Count> {
public:
    static const int value = Count;
};
template<int Count>
class BITWIDTH_COUNT<0,Count> {
public:
    static const int value = Count;
};

#define BIT_WIDTH(flags) (BITWIDTH_COUNT<(flags),1>::value)
#define DATA_TYPE(flags) ap_uint<(BITWIDTH_COUNT<(flags),1>::value)>


#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif


/* The bitwidth of the aggregated costs in 1 path */
template<int N1, int N2>
class AggrDataWidth {
public:
    static const int width = BITWIDTH_COUNT<(N1+N2),1>::value;
};

#define AGGR_DISPARITY_WIDTH(DisparityFlags, PenaltyFlags)\
    AggrDataWidth<DisparityFlags, PenaltyFlags>::width
#define AGGR_DISPARITY_TYPE(DisparityFlags, PenaltyFlags)\
    ap_uint<AggrDataWidth<DisparityFlags, PenaltyFlags>::width>



/* The bitwidth of the aggregated costs for 4 paths */
template<int N1, int N2> 
class AggrDataWidth4 {
public:
    static const int width = AGGR_DISPARITY_WIDTH(N1, N2) + 2;
};

#define AGGR4_DISPARITY_WIDTH(DisparityFlags, PenaltyFlags)\
    AggrDataWidth4<DisparityFlags, PenaltyFlags>::width
#define AGGR4_DISPARITY_TYPE(DisparityFlags, PenaltyFlags)\
    ap_uint<AggrDataWidth4<DisparityFlags, PenaltyFlags>::width>



/* The bitwidth of the aggregated costs for 5 paths */
template<int N1, int N2> 
class AggrDataWidth5 {
public:
    static const int width = AGGR_DISPARITY_WIDTH(N1, N2) + 3;
};

#define AGGR5_DISPARITY_WIDTH(DisparityFlags, PenaltyFlags)\
    AggrDataWidth5<DisparityFlags, PenaltyFlags>::width
#define AGGR5_DISPARITY_TYPE(DisparityFlags, PenaltyFlags)\
    ap_uint<AggrDataWidth5<DisparityFlags, PenaltyFlags>::width>



/* The bitwidth of the aggregated costs for 8 paths */
template<int N1, int N2> 
class AggrDataWidth8 {
public:
    static const int width = AGGR_DISPARITY_WIDTH(N1, N2) + 3;
};

#define AGGR8_DISPARITY_WIDTH(DisparityFlags, PenaltyFlags)\
    AggrDataWidth8<DisparityFlags, PenaltyFlags>::width
#define AGGR8_DISPARITY_TYPE(DisparityFlags, PenaltyFlags)\
    ap_uint<AggrDataWidth8<DisparityFlags, PenaltyFlags>::width>


/* Auto-computed bitwidth based on input parameters */
template<int NUM_DIR_Flags, int COST_VALUE_Flags, int PENALTY_Flags> struct AggrMap {};

template<int COST_VALUE_Flags, int PENALTY_Flags>
struct AggrMap<4, COST_VALUE_Flags, PENALTY_Flags> {
	static const int width = AGGR4_DISPARITY_WIDTH(COST_VALUE_Flags, PENALTY_Flags);
};

template<int COST_VALUE_Flags, int PENALTY_Flags>
struct AggrMap<5, COST_VALUE_Flags, PENALTY_Flags> {
	static const int width = AGGR5_DISPARITY_WIDTH(COST_VALUE_Flags, PENALTY_Flags);
};

template<int COST_VALUE_Flags, int PENALTY_Flags>
struct AggrMap<8, COST_VALUE_Flags, PENALTY_Flags> {
	static const int width = AGGR8_DISPARITY_WIDTH(COST_VALUE_Flags, PENALTY_Flags);
};

#define AGGR_MAP(NUM_DIR_Flags,COST_VALUE_Flags,PENALTY_Flags) AggrMap<NUM_DIR_Flags,COST_VALUE_Flags,PENALTY_Flags>::width


/* The maximum value that can be represented by a data with N bitwidth. */
template<int T> struct MaxValue{};
template<> struct MaxValue<1> {static const int value=1;};
template<> struct MaxValue<2> {static const int value=3;};
template<> struct MaxValue<3> {static const int value=7;};
template<> struct MaxValue<4> {static const int value=15;};
template<> struct MaxValue<5> {static const int value=31;};
template<> struct MaxValue<6> {static const int value=63;};
template<> struct MaxValue<7> {static const int value=127;};
template<> struct MaxValue<8> {static const int value=255;};
template<> struct MaxValue<9> {static const int value=511;};
template<> struct MaxValue<10> {static const int value=1023;};
template<> struct MaxValue<11> {static const int value=2047;};
template<> struct MaxValue<12> {static const int value=4095;};
template<> struct MaxValue<13> {static const int value=8191;};
template<> struct MaxValue<14> {static const int value=16383;};
template<> struct MaxValue<15> {static const int value=32767;};
template<> struct MaxValue<16> {static const int value=65535;};
template<> struct MaxValue<17> {static const int value=131071;};
template<> struct MaxValue<18> {static const int value=262143;};
template<> struct MaxValue<19> {static const int value=524287;};
template<> struct MaxValue<20> {static const int value=1048575;};

#define BW_VALUE(N) MaxValue<N>::value

/* Max value and necessary bitwith for census transform given window size */
template<int N> 
class CENSUSCost {
public:
    static const int value = N*N-1;
    static const int bitwidth = BIT_WIDTH(N*N-1);
};
#define CENSUS_COST(WINDOW_SIZE) CENSUSCost<WINDOW_SIZE>::value
#define CENSUS_COST_BW(WINDOW_SIZE) CENSUSCost<WINDOW_SIZE>::bitwidth

/* Max value and necessary bitwith for rank transform given window size */
template<int N> 
class RANKCost {
public:
    static const int value = N*N-1;
    static const int bitwidth = BIT_WIDTH(N*N-1);
};
#define RANK_COST(WINDOW_SIZE) RANKCost<WINDOW_SIZE>::value
#define RANK_COST_BW(WINDOW_SIZE) RANKCost<WINDOW_SIZE>::bitwidth

/* Max value and necessary bitwith for SAD given window size and input bitwidth */
template<int N1, int N2> 
class SADCost {
public:
    static const int value = BW_VALUE(N1)*(N2*N2);
    static const int bitwidth = N1+BIT_WIDTH(N2*N2-1);
};
#define SAD_COST(BW_INPUT, WINDOW_SIZE) SADCost<BW_INPUT, WINDOW_SIZE>::value
#define SAD_COST_BW(BW_INPUT, WINDOW_SIZE) SADCost<BW_INPUT, WINDOW_SIZE>::bitwidth

/* Max value and necessary bitwith for ZSAD given window size and input bitwidth */
template<int N1, int N2> 
class ZSADCost {
public:
    static const int value = BW_VALUE(N1)*(N2*N2)*2;
    static const int bitwidth = BIT_WIDTH(BW_VALUE(N1)*(N2*N2)*2);
};
#define ZSAD_COST(BW_INPUT, WINDOW_SIZE) ZSADCost<BW_INPUT, WINDOW_SIZE>::value
#define ZSAD_COST_BW(BW_INPUT, WINDOW_SIZE) ZSADCost<BW_INPUT, WINDOW_SIZE>::bitwidth

/* Max value and necessary bitwith for SHD cost function given window size and the max value of census transform */
template<int N1, int N2> 
class SHDCost {
public:
    static const int value = N1*N2*N2;
    static const int bitwidth = BIT_WIDTH(N1*N2*N2);
};
#define SHD_COST(CENSUS_VALUE, SHD_WINDOW) SHDCost<CENSUS_VALUE, SHD_WINDOW>::value
#define SHD_COST_BW(CENSUS_VALUE, SHD_WINDOW) SHDCost<CENSUS_VALUE, SHD_WINDOW>::bitwidth


/* Maintain the consistent interface for different cost functions */
template<int COST_FUNCTION_Flags, int BW_INPUT_Flags, int WINDOW_SIZE_Flags, int SHD_WINDOW_Flags> struct CostMap {};

template<int BW_INPUT_Flags, int WINDOW_SIZE_Flags, int SHD_WINDOW_Flags> 
struct CostMap<0, BW_INPUT_Flags, WINDOW_SIZE_Flags, SHD_WINDOW_Flags> {
    static const int cost_value = CENSUS_COST(WINDOW_SIZE_Flags);
};
template<int BW_INPUT_Flags, int WINDOW_SIZE_Flags, int SHD_WINDOW_Flags> 
struct CostMap<1, BW_INPUT_Flags, WINDOW_SIZE_Flags, SHD_WINDOW_Flags> {
    static const int cost_value = RANK_COST(WINDOW_SIZE_Flags);
};
template<int BW_INPUT_Flags, int WINDOW_SIZE_Flags, int SHD_WINDOW_Flags> 
struct CostMap<2, BW_INPUT_Flags, WINDOW_SIZE_Flags, SHD_WINDOW_Flags> {
    static const int cost_value = SAD_COST(BW_INPUT_Flags,WINDOW_SIZE_Flags);
};
template<int BW_INPUT_Flags, int WINDOW_SIZE_Flags, int SHD_WINDOW_Flags> 
struct CostMap<3, BW_INPUT_Flags, WINDOW_SIZE_Flags, SHD_WINDOW_Flags> {
    static const int cost_value = ZSAD_COST(BW_INPUT_Flags,WINDOW_SIZE_Flags);
};
template<int BW_INPUT_Flags, int WINDOW_SIZE_Flags, int SHD_WINDOW_Flags> 
struct CostMap<4, BW_INPUT_Flags, WINDOW_SIZE_Flags, SHD_WINDOW_Flags> {
    static const int cost_value = SHD_COST(CENSUS_COST(WINDOW_SIZE_Flags),SHD_WINDOW_Flags);
};

#define COST_MAP(COST_FUNCTION_Flags,BW_INPUT_Flags,WINDOW_SIZE_Flags,SHD_WINDOW_Flags) CostMap<COST_FUNCTION_Flags,BW_INPUT_Flags,WINDOW_SIZE_Flags,SHD_WINDOW_Flags>::cost_value


template<int N, int Count>
class PowerTwo {
public:
    static const int value = PowerTwo<N|(N>>1), Count-1>::value;
};

template<int N>
class PowerTwo<N, 0> {
public:
    static const int value = N;
};

template<int COST_VALUE_Flags, int PENALTY_Flags, int PARALLEL_DISPARITIES_Flags, int PORT_BW_Flags> 
struct AGGR_BW {
    static const int actual_data_width = AGGR4_DISPARITY_WIDTH(COST_VALUE_Flags, PENALTY_Flags);
    static const int max_bw = PORT_BW_Flags/PARALLEL_DISPARITIES_Flags;
    static const int modulo = actual_data_width % max_bw;
    static const int N = MAX(modulo-1, 0);
    static const int bw = BIT_WIDTH(N);
    static const int data_width = ( (modulo==0) ? max_bw : (PowerTwo<N,bw>::value + 1) ) * PARALLEL_DISPARITIES_Flags;
};

#define AGGR_WIDTH(COST_VALUE_Flags,PENALTY_Flags,PARALLEL_DISPARITIES_Flags,PORT_BW_Flags) AGGR_BW<COST_VALUE_Flags,PENALTY_Flags,PARALLEL_DISPARITIES_Flags,PORT_BW_Flags>::data_width


#endif//_FP_COMMON_H_