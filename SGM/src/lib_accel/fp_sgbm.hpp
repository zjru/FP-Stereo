/*
 *  FP-Stereo
 *  Copyright (C) 2020  RCSL, HKUST
 *  
 *  GPL-3.0 License
 *
 */

#ifndef _FP_SGBM_HPP_
#define _FP_SGBM_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

#include "hls_video.h"
#include "common/xf_common.h"
#include "common/xf_utility.h"
#include "lib_accel/fp_common.h"
#include "lib_accel/fp_ComputeCost.hpp"
#include "lib_accel/fp_AggregateCost.hpp"
#include "lib_accel/fp_ComputeDisparity.hpp"
#include "lib_accel/fp_PostProcessing.hpp"
#include "fp_config_arch.h"

namespace fp{

template<int COST_VALUE, int BW_INPUT, int ROWS, int COLS, int WINDOW_SIZE, int SHD_WINDOW, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpComputeCost(hls::stream< ap_uint<BW_INPUT> > &src_l, hls::stream< ap_uint<BW_INPUT> > &src_r, 
		hls::stream< DATA_TYPE(COST_VALUE) > cost[PARALLEL_DISPARITIES], ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
#pragma HLS INLINE
#if COST_FUNCTION==0
	fpComputeCensusCost<BW_INPUT,ROWS,COLS,WINDOW_SIZE,NUM_DISPARITY,PARALLEL_DISPARITIES>(src_l, src_r, cost, img_height, img_width);
#elif COST_FUNCTION==1
	fpComputeRankCost<BW_INPUT,ROWS,COLS,WINDOW_SIZE,NUM_DISPARITY,PARALLEL_DISPARITIES>(src_l, src_r, cost, img_height, img_width);
#elif COST_FUNCTION==2
	fpComputeSADCost<BW_INPUT,ROWS,COLS,WINDOW_SIZE,NUM_DISPARITY,PARALLEL_DISPARITIES>(src_l, src_r, cost, img_height, img_width);
#elif COST_FUNCTION==3
	fpComputeZSADCost<BW_INPUT,ROWS,COLS,WINDOW_SIZE,NUM_DISPARITY,PARALLEL_DISPARITIES>(src_l, src_r, cost, img_height, img_width);
#elif COST_FUNCTION==4
	fpComputeSHDCost<BW_INPUT,ROWS,COLS,WINDOW_SIZE,SHD_WINDOW,NUM_DISPARITY,PARALLEL_DISPARITIES>(src_l, src_r, cost, img_height, img_width);
#endif
}

template<int COST_VALUE, int BW_INPUT, int ROWS, int COLS, int WINDOW_SIZE, int SHD_WINDOW, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpLRComputeCost(hls::stream< ap_uint<BW_INPUT> > &src_l, hls::stream< ap_uint<BW_INPUT> > &src_r, 
		hls::stream< DATA_TYPE(COST_VALUE) > left_cost[PARALLEL_DISPARITIES], hls::stream< DATA_TYPE(COST_VALUE) > right_cost[PARALLEL_DISPARITIES],
        ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
#pragma HLS INLINE
#if COST_FUNCTION==0
	fpLRComputeCensusCost<BW_INPUT,ROWS,COLS,WINDOW_SIZE,NUM_DISPARITY,PARALLEL_DISPARITIES>(src_l, src_r, left_cost, right_cost, img_height, img_width);
#elif COST_FUNCTION==1
	fpLRComputeRankCost<BW_INPUT,ROWS,COLS,WINDOW_SIZE,NUM_DISPARITY,PARALLEL_DISPARITIES>(src_l, src_r, left_cost, right_cost, img_height, img_width);
#elif COST_FUNCTION==2
	fpLRComputeSADCost<BW_INPUT,ROWS,COLS,WINDOW_SIZE,NUM_DISPARITY,PARALLEL_DISPARITIES>(src_l, src_r, left_cost, right_cost, img_height, img_width);
#elif COST_FUNCTION==3
	fpLRComputeZSADCost<BW_INPUT,ROWS,COLS,WINDOW_SIZE,NUM_DISPARITY,PARALLEL_DISPARITIES>(src_l, src_r, left_cost, right_cost, img_height, img_width);
#elif COST_FUNCTION==4
	fpLRComputeSHDCost<BW_INPUT,ROWS,COLS,WINDOW_SIZE,SHD_WINDOW,NUM_DISPARITY,PARALLEL_DISPARITIES>(src_l, src_r, left_cost, right_cost, img_height, img_width);
#endif	
}

template<typename T, int ROWS, int COLS, int COST_VALUE, int NUM_DISPARITY, int PARALLEL_DISPARITIES, int P1, int P2>
void fpAggregateCostRasterPath(hls::stream< DATA_TYPE(COST_VALUE) > cost[PARALLEL_DISPARITIES], hls::stream< T > aggregated_cost[PARALLEL_DISPARITIES], 
		ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
#pragma HLS INLINE
	fpAggregateCost4Path<ROWS,COLS,COST_VALUE,NUM_DISPARITY,PARALLEL_DISPARITIES,P1,P2>(cost, aggregated_cost, img_height, img_width);
}

template<typename T, int ROWS, int COLS, int COST_VALUE, int NUM_DISPARITY, int PARALLEL_DISPARITIES, int P1, int P2>
void fpAggregateCostRasterPath_copy(hls::stream< DATA_TYPE(COST_VALUE) > cost[PARALLEL_DISPARITIES], hls::stream< T > aggregated_cost[PARALLEL_DISPARITIES], 
		ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
#pragma HLS INLINE
	fpAggregateCost4Path_copy<ROWS,COLS,COST_VALUE,NUM_DISPARITY,PARALLEL_DISPARITIES,P1,P2>(cost, aggregated_cost, img_height, img_width);
}

template<typename T, int ROWS, int COLS, int DST_TYPE, int NPC, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpComputeDisparityMap(hls::stream< T > aggregated_cost[PARALLEL_DISPARITIES], hls::stream< XF_TNAME(DST_TYPE,NPC) > &dst_fifo, 
		ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
#pragma HLS INLINE	
#if UNIQ==0
	fpComputeDisparity<T,ROWS,COLS,DST_TYPE,NPC,NUM_DISPARITY,PARALLEL_DISPARITIES>(aggregated_cost, dst_fifo, img_height, img_width);
#elif UNIQ==1
	fpComputeDisparityUniqueness<T,ROWS,COLS,DST_TYPE,NPC,NUM_DISPARITY,PARALLEL_DISPARITIES>(aggregated_cost, dst_fifo, img_height, img_width);
#endif
}

template<typename T, int ROWS, int COLS, int DST_TYPE, int NPC, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpLRComputeDisparityMap(hls::stream< T > aggregated_cost[PARALLEL_DISPARITIES], hls::stream< XF_TNAME(DST_TYPE,NPC) > &left_dst_fifo, 
		hls::stream< XF_TNAME(DST_TYPE,NPC) > &right_dst_fifo, ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
#pragma HLS INLINE	
#if UNIQ==0
	fpLRComputeDisparity<T,ROWS,COLS,DST_TYPE,NPC,NUM_DISPARITY,PARALLEL_DISPARITIES>(aggregated_cost, left_dst_fifo, right_dst_fifo, img_height, img_width);
#elif UNIQ==1
	fpLRComputeDisparityUniqueness<T,ROWS,COLS,DST_TYPE,NPC,NUM_DISPARITY,PARALLEL_DISPARITIES>(aggregated_cost, left_dst_fifo, right_dst_fifo, img_height, img_width);
#endif
}

// SGM without L-R check
template<int WINDOW_SIZE, int SHD_WINDOW, int NUM_DISPARITY, int PARALLEL_DISPARITIES, int FilterWin, int SRC_TYPE, int DST_TYPE, int ROWS, int COLS, int NPC, int P1, int P2>
void SemiGlobalBMNLR(xf::Mat<SRC_TYPE, ROWS, COLS, NPC> &src_mat_l, xf::Mat<SRC_TYPE, ROWS, COLS, NPC> &src_mat_r, xf::Mat<DST_TYPE, ROWS, COLS, NPC> &dst_mat)
{
	#pragma HLS INLINE

	hls::stream< XF_TNAME(SRC_TYPE,NPC) > src_l_fifo;
	hls::stream< XF_TNAME(SRC_TYPE,NPC) > src_r_fifo;
	hls::stream< XF_TNAME(DST_TYPE,NPC) > out_dst_fifo;
	hls::stream< XF_TNAME(DST_TYPE,NPC) > dst_fifo;

	const int COST_VALUE = COST_MAP(COST_FUNCTION,XF_DTPIXELDEPTH(SRC_TYPE,NPC),WINDOW_SIZE,SHD_WINDOW);
	const int AGGR_WIDTH = AGGR_MAP(NUM_DIR,COST_VALUE,P2);

	hls::stream< DATA_TYPE(COST_VALUE) > cost[PARALLEL_DISPARITIES];
	#pragma HLS ARRAY_PARTITION variable=cost complete dim=1

	hls::stream< ap_uint<AGGR_WIDTH> > aggregated_cost[PARALLEL_DISPARITIES];
	#pragma HLS ARRAY_PARTITION variable=aggregated_cost complete dim=1	

	ap_uint<BIT_WIDTH(ROWS)> height = src_mat_l.rows;
	ap_uint<BIT_WIDTH(COLS)> width = src_mat_l.cols;	

	loop_access_src:
	for(ap_uint<BIT_WIDTH(ROWS)> i = 0; i < height; i++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS //This pragma is to get the HLS estimation.
		for(ap_uint<BIT_WIDTH(COLS)> j = 0; j < width; j++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			#pragma HLS PIPELINE 
			src_l_fifo.write(*(src_mat_l.data+i*width+j));
			src_r_fifo.write(*(src_mat_r.data+i*width+j));
		}
	}

	fpComputeCost<COST_VALUE,XF_DTPIXELDEPTH(SRC_TYPE,NPC),ROWS,COLS,WINDOW_SIZE,SHD_WINDOW,NUM_DISPARITY,PARALLEL_DISPARITIES>(src_l_fifo,src_r_fifo,cost,height,width);

	fpAggregateCostRasterPath<ap_uint<AGGR_WIDTH>,ROWS,COLS,COST_VALUE,NUM_DISPARITY,PARALLEL_DISPARITIES,P1,P2>(cost, aggregated_cost, height, width);

	fpComputeDisparityMap<ap_uint<AGGR_WIDTH>,ROWS,COLS,DST_TYPE,NPC,NUM_DISPARITY,PARALLEL_DISPARITIES>(aggregated_cost,out_dst_fifo,height,width);

	fpMedianFilter<ROWS,COLS,DST_TYPE,NPC,FilterWin>(out_dst_fifo, dst_fifo, height, width);

	// write back from stream to Mat
	for(int i=0; i<dst_mat.rows;i++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for(int j=0; j<dst_mat.cols; j++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			#pragma HLS PIPELINE
			*(dst_mat.data + i*dst_mat.cols +j) = (dst_fifo.read());
		}
	}
}

// SGM with L-R consistency check (LR1 method)
template<int WINDOW_SIZE, int SHD_WINDOW, int NUM_DISPARITY, int PARALLEL_DISPARITIES, int FilterWin, int SRC_TYPE, int DST_TYPE, int ROWS, int COLS, int NPC, int P1, int P2>
void SemiGlobalBMLR1(xf::Mat<SRC_TYPE, ROWS, COLS, NPC> &src_mat_l, xf::Mat<SRC_TYPE, ROWS, COLS, NPC> &src_mat_r, xf::Mat<DST_TYPE, ROWS, COLS, NPC> &dst_mat)
{
	#pragma HLS INLINE 

	hls::stream< XF_TNAME(SRC_TYPE,NPC) > src_l_fifo;
	hls::stream< XF_TNAME(SRC_TYPE,NPC) > src_r_fifo;
	hls::stream< XF_TNAME(DST_TYPE,NPC) > left_dst_fifo;
	hls::stream< XF_TNAME(DST_TYPE,NPC) > right_dst_fifo;
	static hls::stream< XF_TNAME(DST_TYPE,NPC) > l_dst_fifo;
	#pragma HLS STREAM variable=l_dst_fifo depth=NUM_DISPARITY
	hls::stream< XF_TNAME(DST_TYPE,NPC) > r_dst_fifo;
	hls::stream< XF_TNAME(DST_TYPE,NPC) > dst_fifo;

	const int COST_VALUE = COST_MAP(COST_FUNCTION,XF_DTPIXELDEPTH(SRC_TYPE,NPC),WINDOW_SIZE,SHD_WINDOW);
	const int AGGR_WIDTH = AGGR_MAP(NUM_DIR,COST_VALUE,P2);

	hls::stream< DATA_TYPE(COST_VALUE) > cost[PARALLEL_DISPARITIES];
	#pragma HLS ARRAY_PARTITION variable=cost complete dim=1

	hls::stream< ap_uint<AGGR_WIDTH> > aggregated_cost[PARALLEL_DISPARITIES];
	#pragma HLS ARRAY_PARTITION variable=aggregated_cost complete dim=1	

	ap_uint<BIT_WIDTH(ROWS)> height = src_mat_l.rows;
	ap_uint<BIT_WIDTH(COLS)> width = src_mat_l.cols;	

	loop_access_src:
	for(ap_uint<BIT_WIDTH(ROWS)> i = 0; i < height; i++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS //This pragma is to get the HLS estimation.
		for(ap_uint<BIT_WIDTH(COLS)> j = 0; j < width; j++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			#pragma HLS PIPELINE 
			src_l_fifo.write(*(src_mat_l.data+i*width+j));
			src_r_fifo.write(*(src_mat_r.data+i*width+j));
		}
	}

	fpComputeCost<COST_VALUE,XF_DTPIXELDEPTH(SRC_TYPE,NPC),ROWS,COLS,WINDOW_SIZE,SHD_WINDOW,NUM_DISPARITY,PARALLEL_DISPARITIES>(src_l_fifo,src_r_fifo,cost,height,width);

	fpAggregateCostRasterPath<ap_uint<AGGR_WIDTH>,ROWS,COLS,COST_VALUE,NUM_DISPARITY,PARALLEL_DISPARITIES,P1,P2>(cost, aggregated_cost, height, width);

	fpLRComputeDisparityMap<ap_uint<AGGR_WIDTH>,ROWS,COLS,DST_TYPE,NPC,NUM_DISPARITY,PARALLEL_DISPARITIES>(aggregated_cost, left_dst_fifo, right_dst_fifo, height, width);

	fpMedianFilter<ROWS,COLS,DST_TYPE,NPC,FilterWin>(left_dst_fifo, l_dst_fifo, height, width);
	fpMedianFilter<ROWS,COLS,DST_TYPE,NPC,FilterWin>(right_dst_fifo, r_dst_fifo, height, width);

	fpLRCheckConsistency<ROWS,COLS,DST_TYPE,NPC,NUM_DISPARITY>(l_dst_fifo, r_dst_fifo, dst_fifo, height, width);

	// write back from stream to Mat
	for(int i=0; i<dst_mat.rows;i++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for(int j=0; j<dst_mat.cols; j++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			#pragma HLS PIPELINE
			*(dst_mat.data + i*dst_mat.cols +j) = (dst_fifo.read());
		}
	}
}

// SGM with L-R consistency check (LR2 method)
template<int WINDOW_SIZE, int SHD_WINDOW, int NUM_DISPARITY, int PARALLEL_DISPARITIES, int FilterWin, int SRC_TYPE, int DST_TYPE, int ROWS, int COLS, int NPC, int P1, int P2>
void SemiGlobalBMLR2(xf::Mat<SRC_TYPE, ROWS, COLS, NPC> &src_mat_l, xf::Mat<SRC_TYPE, ROWS, COLS, NPC> &src_mat_r, xf::Mat<DST_TYPE, ROWS, COLS, NPC> &dst_mat)
{
	#pragma HLS INLINE

	hls::stream< XF_TNAME(SRC_TYPE,NPC) > src_l_fifo;
	hls::stream< XF_TNAME(SRC_TYPE,NPC) > src_r_fifo;
	hls::stream< XF_TNAME(DST_TYPE,NPC) > left_dst_fifo;
	hls::stream< XF_TNAME(DST_TYPE,NPC) > right_dst_fifo;
	hls::stream< XF_TNAME(DST_TYPE,NPC) > r_dst_fifo;
	hls::stream< XF_TNAME(DST_TYPE,NPC) > dst_fifo;

	const int COST_VALUE = COST_MAP(COST_FUNCTION,XF_DTPIXELDEPTH(SRC_TYPE,NPC),WINDOW_SIZE,SHD_WINDOW);
	const int AGGR_WIDTH = AGGR_MAP(NUM_DIR,COST_VALUE,P2);

	hls::stream< DATA_TYPE(COST_VALUE) > left_cost[PARALLEL_DISPARITIES];
	#pragma HLS ARRAY_PARTITION variable=left_cost complete dim=1
	hls::stream< DATA_TYPE(COST_VALUE) > right_cost[PARALLEL_DISPARITIES];
	#pragma HLS ARRAY_PARTITION variable=right_cost complete dim=1

	hls::stream< ap_uint<AGGR_WIDTH> > left_aggregated_cost[PARALLEL_DISPARITIES];
	#pragma HLS ARRAY_PARTITION variable=left_aggregated_cost complete dim=1	
	hls::stream< ap_uint<AGGR_WIDTH> > right_aggregated_cost[PARALLEL_DISPARITIES];
	#pragma HLS ARRAY_PARTITION variable=right_aggregated_cost complete dim=1	

	ap_uint<BIT_WIDTH(ROWS)> height = src_mat_l.rows;
	ap_uint<BIT_WIDTH(COLS)> width = src_mat_l.cols;	

	loop_access_src:
	for(ap_uint<BIT_WIDTH(ROWS)> i = 0; i < height; i++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS //This pragma is to get the HLS estimation.
		for(ap_uint<BIT_WIDTH(COLS)> j = 0; j < width; j++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			#pragma HLS PIPELINE 
			src_l_fifo.write(*(src_mat_l.data+i*width+j));
			src_r_fifo.write(*(src_mat_r.data+i*width+j));
		}
	}

	fpLRComputeCost<COST_VALUE,XF_DTPIXELDEPTH(SRC_TYPE,NPC),ROWS,COLS,WINDOW_SIZE,SHD_WINDOW,NUM_DISPARITY,PARALLEL_DISPARITIES>(src_l_fifo,src_r_fifo,left_cost,right_cost,height,width);

	fpAggregateCostRasterPath<ap_uint<AGGR_WIDTH>,ROWS,COLS,COST_VALUE,NUM_DISPARITY,PARALLEL_DISPARITIES,P1,P2>(left_cost, left_aggregated_cost, height, width);
	fpAggregateCostRasterPath_copy<ap_uint<AGGR_WIDTH>,ROWS,COLS,COST_VALUE,NUM_DISPARITY,PARALLEL_DISPARITIES,P1,P2>(right_cost, right_aggregated_cost, height, width);

	fpComputeDisparityMap<ap_uint<AGGR_WIDTH>,ROWS,COLS,DST_TYPE,NPC,NUM_DISPARITY,PARALLEL_DISPARITIES>(left_aggregated_cost,left_dst_fifo,height,width);
	fpComputeDisparityMap<ap_uint<AGGR_WIDTH>,ROWS,COLS,DST_TYPE,NPC,NUM_DISPARITY,PARALLEL_DISPARITIES>(right_aggregated_cost,right_dst_fifo,height,width);

	fpMedianFilter<ROWS,COLS,DST_TYPE,NPC,FilterWin>(right_dst_fifo, r_dst_fifo, height, width);

	static hls::stream< XF_TNAME(DST_TYPE,NPC) > l_dst_fifo;
	#pragma HLS STREAM variable=l_dst_fifo depth=NUM_DISPARITY
	fpMedianFilter<ROWS,COLS,DST_TYPE,NPC,FilterWin>(left_dst_fifo, l_dst_fifo, height, width);
	fpLRCheckConsistency<ROWS,COLS,DST_TYPE,NPC,NUM_DISPARITY>(l_dst_fifo, r_dst_fifo, dst_fifo, height, width);

	// write back from stream to Mat
	for(int i=0; i<dst_mat.rows;i++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for(int j=0; j<dst_mat.cols; j++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			#pragma HLS PIPELINE
			*(dst_mat.data + i*dst_mat.cols +j) = (dst_fifo.read());
		}
	}
}


// Top function for SGM accelerator

#pragma SDS data mem_attribute("src_mat_l.data":NON_CACHEABLE|PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute("src_mat_r.data":NON_CACHEABLE|PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute("dst_mat.data":NON_CACHEABLE|PHYSICAL_CONTIGUOUS)
#pragma SDS data access_pattern("src_mat_l.data":SEQUENTIAL, "src_mat_r.data":SEQUENTIAL, "dst_mat.data":SEQUENTIAL)
#pragma SDS data copy("src_mat_l.data"[0:"src_mat_l.size"], "src_mat_r.data"[0:"src_mat_r.size"], "dst_mat.data"[0:"dst_mat.size"])

template<int WINDOW_SIZE, int SHD_WINDOW, int NUM_DISPARITY, int PARALLEL_DISPARITIES, int FilterWin, int SRC_TYPE, int DST_TYPE, int ROWS, int COLS, int NPC, int P1, int P2>
void SemiGlobalBM(xf::Mat<SRC_TYPE, ROWS, COLS, NPC> &src_mat_l, xf::Mat<SRC_TYPE, ROWS, COLS, NPC> &src_mat_r, xf::Mat<DST_TYPE, ROWS, COLS, NPC> &dst_mat)
{
	assert((SRC_TYPE == XF_8UC1) && " WORDWIDTH_SRC must be XF_8UC1 ");
	assert((DST_TYPE == XF_8UC1) && " WORDWIDTH_DST must be XF_8UC1 ");
	assert((NPC == XF_NPPC1) && " NPC must be XF_NPPC1 ");	
	assert(((NUM_DISPARITY > 1) && (NUM_DISPARITY <= 256)) && " The number of disparities must be greater than '1' and less than or equal to '256' ");
	assert((NUM_DISPARITY >= PARALLEL_DISPARITIES) && " The number of disparities must not be lesser than (parallel units)");
	assert((((NUM_DISPARITY/PARALLEL_DISPARITIES)*PARALLEL_DISPARITIES) == NUM_DISPARITY) && " NUM_DISPARITY/PARALLEL_DISPARITIES must be a non-fractional number ");
	assert(((ROWS/2)*2 == ROWS) && ((COLS/2)*2 == COLS) && "ROWS and COLS must be a even number ");
	assert((P1 < P2) && "P1 must be always less than P2");
	assert((WINDOW_SIZE==3)||(WINDOW_SIZE==5)||(WINDOW_SIZE==7)||(WINDOW_SIZE==9)||(WINDOW_SIZE==11)||(WINDOW_SIZE==13)||(WINDOW_SIZE==15) && " WSIZE must be set to '3,5,7,9,11,13,15' ");

	#pragma HLS INLINE OFF
	#pragma HLS DATAFLOW
#if LR_CHECK==0
	SemiGlobalBMNLR<WINDOW_SIZE,SHD_WINDOW,NUM_DISPARITY,PARALLEL_DISPARITIES,FilterWin,SRC_TYPE,DST_TYPE,ROWS,COLS,NPC,P1,P2>(src_mat_l,src_mat_r,dst_mat);	
#elif LR_CHECK==1
	SemiGlobalBMLR1<WINDOW_SIZE,SHD_WINDOW,NUM_DISPARITY,PARALLEL_DISPARITIES,FilterWin,SRC_TYPE,DST_TYPE,ROWS,COLS,NPC,P1,P2>(src_mat_l,src_mat_r,dst_mat);
#elif LR_CHECK==2
	SemiGlobalBMLR2<WINDOW_SIZE,SHD_WINDOW,NUM_DISPARITY,PARALLEL_DISPARITIES,FilterWin,SRC_TYPE,DST_TYPE,ROWS,COLS,NPC,P1,P2>(src_mat_l,src_mat_r,dst_mat);
#endif
}


}
#endif

