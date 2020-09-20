/*
 *  FP-Stereo
 *  Copyright (C) 2020  RCSL, HKUST
 *  
 *  GPL-3.0 License
 *
 */

#ifndef _FP_COMPUTEDISPARITY_HPP_
#define _FP_COMPUTEDISPARITY_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

//#include "hls_stream.h"
#include "hls_video.h"
#include "common/xf_common.h"
#include "common/xf_utility.h"
#include "lib_accel/fp_common.h"

namespace fp{

template<int SIZE>
class fpMinArrIndexVal
{
public:
	template <typename T, typename T_idx>
	static void find(T a[SIZE], T_idx &loc, T &val)
	{
#pragma HLS INLINE
#pragma HLS array_partition variable=a complete dim=1

		T a1[SIZE/2];
		T a2[SIZE-SIZE/2];
#pragma HLS array_partition variable=a1 complete dim=1
#pragma HLS array_partition variable=a2 complete dim=1
		for(int i = 0; i < SIZE/2; i++)
		{
#pragma HLS UNROLL
			a1[i] = a[i];
		}
		for(int i = 0; i < SIZE-SIZE/2; i++)
		{
#pragma HLS UNROLL
			a2[i] = a[i+SIZE/2];
		}

		T_idx l1,l2;
		T v1,v2;
		fpMinArrIndexVal<SIZE/2>::find(a1,l1,v1);
		fpMinArrIndexVal<SIZE-SIZE/2>::find(a2,l2,v2);

		if(v2 < v1)
		{
			val = v2;
			loc = l2+SIZE/2;
		}
		else
		{
			val = v1;
			loc = l1;
		}
	}
};

template<>
class fpMinArrIndexVal<1>
{
public:
	template <typename T, typename T_idx>
	static void find(T a[1], T_idx &loc, T &val)
	{
#pragma HLS INLINE
		loc = 0;
		val = a[0];
	}
};

template<>
class fpMinArrIndexVal<2>
{
public:
	template <typename T, typename T_idx>
	static void find(T a[2], T_idx &loc, T &val)
	{
#pragma HLS INLINE
#pragma HLS array_partition variable=a complete dim=0

		T_idx l1=0, l2=1;
		T v1=a[0], v2=a[1];
		if(v2 < v1)
		{
			val = v2;
			loc = l2;
		}
		else
		{
			val = v1;
			loc = l1;
		}
	}
};

template<typename T, int ROWS, int COLS, int DST_TYPE, int NPC, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpComputeDisparity(hls::stream< T > aggregated_cost[PARALLEL_DISPARITIES], hls::stream< XF_TNAME(DST_TYPE,NPC) > &dst_fifo, 
		ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS array_partition variable=aggregated_cost complete dim=1
	const int ITERATION = NUM_DISPARITY/PARALLEL_DISPARITIES;
	const T max_value_bound = (T)MAX_VALUE_BOUND;
	
	ap_uint<BIT_WIDTH(ROWS)> row;
	ap_uint<BIT_WIDTH(COLS)> col;
	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for (col = 0; col < img_width; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			if (NUM_DISPARITY == PARALLEL_DISPARITIES)
			{
				#pragma HLS PIPELINE II=1 //If equal, pipeline the outer loop. 
			}

			T min_aggregated_cost = max_value_bound;
			XF_TNAME(DST_TYPE,NPC) min_disp;
			for(ap_uint<BIT_WIDTH(ITERATION)> iter = 0; iter < ITERATION; iter++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_flatten
				T tmp[PARALLEL_DISPARITIES];
				#pragma HLS ARRAY_PARTITION variable=tmp complete dim=1
				for(int num = 0; num < PARALLEL_DISPARITIES; num++)
				{
					#pragma HLS UNROLL
					tmp[num]=aggregated_cost[num].read();
				}
				T min_aggregated_cost_tmp;
				ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> min_disp_tmp;
				fpMinArrIndexVal<PARALLEL_DISPARITIES>::find(tmp,min_disp_tmp,min_aggregated_cost_tmp);
				if(min_aggregated_cost_tmp < min_aggregated_cost){
					min_disp = iter*PARALLEL_DISPARITIES+min_disp_tmp;
					min_aggregated_cost = min_aggregated_cost_tmp;
				}								
			}
			dst_fifo.write(min_disp);
		}
	}
}


template<typename T, int ROWS, int COLS, int DST_TYPE, int NPC, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpLRComputeDisparity(hls::stream< T > aggregated_cost[PARALLEL_DISPARITIES], hls::stream< XF_TNAME(DST_TYPE,NPC) > &left_dst_fifo, 
		hls::stream< XF_TNAME(DST_TYPE,NPC) > &right_dst_fifo, ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS array_partition variable=aggregated_cost complete dim=1
	const int ITERATION = NUM_DISPARITY/PARALLEL_DISPARITIES;
	const T max_value_bound = (T)MAX_VALUE_BOUND;
	
	T right_min[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=right_min complete dim=1
	
	XF_TNAME(DST_TYPE,NPC) right_min_disp[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=right_min_disp complete dim=1

	for(int i = 0; i < NUM_DISPARITY; i++)
	{
		#pragma HLS UNROLL
		right_min[i] = max_value_bound;
		right_min_disp[i] = 0;
	}

	ap_uint<BIT_WIDTH(ROWS)> row;
	ap_uint<BIT_WIDTH(COLS+NUM_DISPARITY-1)> col;
	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for (col = 0; col < img_width + NUM_DISPARITY-1; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS+NUM_DISPARITY-1 max=COLS+NUM_DISPARITY-1
			if (NUM_DISPARITY == PARALLEL_DISPARITIES)
			{
				#pragma HLS PIPELINE II=1 //If equal, pipeline the outer loop. 
			}
			T min_aggregated_cost = max_value_bound;
			XF_TNAME(DST_TYPE,NPC) min_disp;
			for(ap_uint<BIT_WIDTH(ITERATION)> iter = 0; iter < ITERATION; iter++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_flatten
				T tmp[PARALLEL_DISPARITIES];
				#pragma HLS ARRAY_PARTITION variable=tmp complete dim=1
				for(int num = 0; num < PARALLEL_DISPARITIES; num++)
				{
					#pragma HLS UNROLL
					if(col<img_width){
						tmp[num]=aggregated_cost[num].read();
					}
					else{
						tmp[num]=max_value_bound;
					}					
				}
				T min_aggregated_cost_tmp;
				ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> min_disp_tmp;
				fpMinArrIndexVal<PARALLEL_DISPARITIES>::find(tmp,min_disp_tmp,min_aggregated_cost_tmp);
				if(min_aggregated_cost_tmp < min_aggregated_cost){
					min_disp = iter*PARALLEL_DISPARITIES+min_disp_tmp;
					min_aggregated_cost = min_aggregated_cost_tmp;
				}
				for(int num = 0; num < PARALLEL_DISPARITIES; num++)
				{
					#pragma HLS UNROLL
					ap_uint<BIT_WIDTH(NUM_DISPARITY)> disparity_idx = iter*PARALLEL_DISPARITIES+num;
					if(tmp[num]<right_min[disparity_idx]){
						right_min[disparity_idx] = tmp[num];
						right_min_disp[disparity_idx] = disparity_idx;
					}
				}				
				if(iter>=(ITERATION-1)){
					if(col<img_width){
						left_dst_fifo.write(min_disp);
					}
					if(col>=NUM_DISPARITY-1){
						right_dst_fifo.write(right_min_disp[NUM_DISPARITY-1]);
					}
					for(int i = NUM_DISPARITY-1; i > 0; i--)
					{
						#pragma HLS UNROLL
						right_min[i] = right_min[i-1];
						right_min_disp[i] = right_min_disp[i-1];
					}
					right_min[0] = max_value_bound;	
					right_min_disp[0] = 0;
				}
			}					
		}
	}
}


template<int SIZE, typename T, typename T_width>
void fpSortArray(T sort_array[SIZE], T min_value[3], T_width &index, T_width &second_index)
{
	#pragma HLS INLINE
	#pragma HLS array_partition variable=sort_array complete dim=1
	#pragma HLS array_partition variable=min_value complete dim=1
	
	const T max_value_bound = (T)MAX_VALUE_BOUND;
	min_value[2] = max_value_bound;
	min_value[1] = max_value_bound;
	min_value[0] = max_value_bound;

	for(T_width i=0; i<SIZE; i++)
	{
		#pragma HLS UNROLL
		if(sort_array[i]<min_value[0]){
			min_value[2] = min_value[1];
			min_value[1] = min_value[0];
			min_value[0] = sort_array[i];
			second_index = index;
			index = i;
		}
		else if(sort_array[i]<min_value[1]){
			min_value[2] = min_value[1];
			min_value[1] = sort_array[i];
			second_index = i;
		}
		else if(sort_array[i]<min_value[2]){
			min_value[2] = sort_array[i];
		}
	}
}

template<typename T, int ROWS, int COLS, int DST_TYPE, int NPC, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpComputeDisparityUniqueness(hls::stream< T > aggregated_cost[PARALLEL_DISPARITIES], hls::stream< XF_TNAME(DST_TYPE,NPC) > &dst_fifo, 
		ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS array_partition variable=aggregated_cost complete dim=1
	const int ITERATION = NUM_DISPARITY/PARALLEL_DISPARITIES;
	const T max_value_bound = (T)MAX_VALUE_BOUND;
	
	ap_uint<BIT_WIDTH(ROWS)> row;
	ap_uint<BIT_WIDTH(COLS)> col;
	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for (col = 0; col < img_width; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			if (NUM_DISPARITY == PARALLEL_DISPARITIES)
			{
				#pragma HLS PIPELINE II=1 //If equal, pipeline the outer loop. 
			}

			T min_aggregated_cost[3];
			#pragma HLS ARRAY_PARTITION variable=min_aggregated_cost complete dim=1
			min_aggregated_cost[0] = max_value_bound;
			min_aggregated_cost[1] = max_value_bound;
			min_aggregated_cost[2] = max_value_bound;
			XF_TNAME(DST_TYPE,NPC) min_disp = 0;
			XF_TNAME(DST_TYPE,NPC) second_min_disp = 0;
			
			for(ap_uint<BIT_WIDTH(ITERATION)> iter = 0; iter < ITERATION; iter++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_flatten
				T tmp[PARALLEL_DISPARITIES];
				#pragma HLS ARRAY_PARTITION variable=tmp complete dim=1
				for(int num = 0; num < PARALLEL_DISPARITIES; num++)
				{
					#pragma HLS UNROLL
					tmp[num]=aggregated_cost[num].read();
				}
				T min_aggregated_cost_tmp[3];
				#pragma HLS ARRAY_PARTITION variable=min_aggregated_cost_tmp complete dim=1
				ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> min_disp_tmp = 0;
				ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> second_min_disp_tmp = 0;
				fpSortArray<PARALLEL_DISPARITIES,T,ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> >(tmp,min_aggregated_cost_tmp,min_disp_tmp,second_min_disp_tmp);

				if(min_aggregated_cost_tmp[0]<min_aggregated_cost[0]){
					min_aggregated_cost[2] = min_aggregated_cost[1];
					min_aggregated_cost[1] = min_aggregated_cost[0];
					min_aggregated_cost[0] = min_aggregated_cost_tmp[0];
					second_min_disp = min_disp;
					min_disp = iter*PARALLEL_DISPARITIES+min_disp_tmp;
				}
				else if(min_aggregated_cost_tmp[0]<min_aggregated_cost[1]){
					min_aggregated_cost[2] = min_aggregated_cost[1];
					min_aggregated_cost[1] = min_aggregated_cost_tmp[0];
					second_min_disp = iter*PARALLEL_DISPARITIES+min_disp_tmp;
				}
				else if(min_aggregated_cost_tmp[0]<min_aggregated_cost[2]){
					min_aggregated_cost[2] = min_aggregated_cost_tmp[0];
				}
				if(min_aggregated_cost_tmp[1]<min_aggregated_cost[1]){
					min_aggregated_cost[2] = min_aggregated_cost[1];
					min_aggregated_cost[1] = min_aggregated_cost_tmp[1];
					second_min_disp = iter*PARALLEL_DISPARITIES+second_min_disp_tmp;
				}
				else if(min_aggregated_cost_tmp[1]<min_aggregated_cost[2]){
					min_aggregated_cost[2] = min_aggregated_cost_tmp[1];
				}
				if(min_aggregated_cost_tmp[2]<min_aggregated_cost[2]){
					min_aggregated_cost[2] = min_aggregated_cost_tmp[2];
				}
				if(iter>=ITERATION-1){
					XF_TNAME(DST_TYPE,NPC) abs_diff = fpABSdiff<XF_TNAME(DST_TYPE,NPC) >(min_disp,second_min_disp);
					int min0 = (int(min_aggregated_cost[0])<<2)+(int(min_aggregated_cost[0])<<4);
					int min1 = int(min_aggregated_cost[1])+(int(min_aggregated_cost[1])<<1)+(int(min_aggregated_cost[1])<<4);
					int min2 = int(min_aggregated_cost[2])+(int(min_aggregated_cost[2])<<1)+(int(min_aggregated_cost[2])<<4);
					if( (abs_diff>1) && (min0>min1) ){
						min_disp = 0;
					}
					else if(min0>min2){
						min_disp = 0;
					}
				}							
			}
			dst_fifo.write(min_disp);
		}
	}
}

template<typename T, int ROWS, int COLS, int DST_TYPE, int NPC, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpLRComputeDisparityUniqueness(hls::stream< T > aggregated_cost[PARALLEL_DISPARITIES], hls::stream< XF_TNAME(DST_TYPE,NPC) > &left_dst_fifo, 
		hls::stream< XF_TNAME(DST_TYPE,NPC) > &right_dst_fifo, ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS array_partition variable=aggregated_cost complete dim=1
	const int ITERATION = NUM_DISPARITY/PARALLEL_DISPARITIES;
	const T max_value_bound = (T)MAX_VALUE_BOUND;
	
	T right_min[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=right_min complete dim=1

	XF_TNAME(DST_TYPE,NPC) right_min_disp[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=right_min_disp complete dim=1

	for(int i = 0; i < NUM_DISPARITY; i++)
	{
		#pragma HLS UNROLL
		right_min[i] = max_value_bound;
		right_min_disp[i] = 0;
	}

	ap_uint<BIT_WIDTH(ROWS)> row;
	ap_uint<BIT_WIDTH(COLS+NUM_DISPARITY-1)> col;
	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for (col = 0; col < img_width + NUM_DISPARITY-1; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS+NUM_DISPARITY-1 max=COLS+NUM_DISPARITY-1
			if (NUM_DISPARITY == PARALLEL_DISPARITIES)
			{
				#pragma HLS PIPELINE II=1 //If equal, pipeline the outer loop. 
			}

			T min_aggregated_cost[3];
			#pragma HLS ARRAY_PARTITION variable=min_aggregated_cost complete dim=1
			min_aggregated_cost[0] = max_value_bound;
			min_aggregated_cost[1] = max_value_bound;
			min_aggregated_cost[2] = max_value_bound;
			XF_TNAME(DST_TYPE,NPC) min_disp = 0;
			XF_TNAME(DST_TYPE,NPC) second_min_disp = 0;

			for(ap_uint<BIT_WIDTH(ITERATION)> iter = 0; iter < ITERATION; iter++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_flatten
				T tmp[PARALLEL_DISPARITIES];
				#pragma HLS ARRAY_PARTITION variable=tmp complete dim=1
				for(int num = 0; num < PARALLEL_DISPARITIES; num++)
				{
					#pragma HLS UNROLL
					if(col<img_width){
						tmp[num]=aggregated_cost[num].read();
					}
					else{
						tmp[num]=max_value_bound;
					}					
				}
				T min_aggregated_cost_tmp[3];
				#pragma HLS ARRAY_PARTITION variable=min_aggregated_cost_tmp complete dim=1
				ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> min_disp_tmp = 0;
				ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> second_min_disp_tmp = 0;
				fpSortArray<PARALLEL_DISPARITIES,T,ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> >(tmp,min_aggregated_cost_tmp,min_disp_tmp,second_min_disp_tmp);
				if(min_aggregated_cost_tmp[0]<min_aggregated_cost[0]){
					min_aggregated_cost[2] = min_aggregated_cost[1];
					min_aggregated_cost[1] = min_aggregated_cost[0];
					min_aggregated_cost[0] = min_aggregated_cost_tmp[0];
					second_min_disp = min_disp;
					min_disp = iter*PARALLEL_DISPARITIES+min_disp_tmp;
				}
				else if(min_aggregated_cost_tmp[0]<min_aggregated_cost[1]){
					min_aggregated_cost[2] = min_aggregated_cost[1];
					min_aggregated_cost[1] = min_aggregated_cost_tmp[0];
					second_min_disp = iter*PARALLEL_DISPARITIES+min_disp_tmp;
				}
				else if(min_aggregated_cost_tmp[0]<min_aggregated_cost[2]){
					min_aggregated_cost[2] = min_aggregated_cost_tmp[0];
				}
				if(min_aggregated_cost_tmp[1]<min_aggregated_cost[1]){
					min_aggregated_cost[2] = min_aggregated_cost[1];
					min_aggregated_cost[1] = min_aggregated_cost_tmp[1];
					second_min_disp = iter*PARALLEL_DISPARITIES+second_min_disp_tmp;
				}
				else if(min_aggregated_cost_tmp[1]<min_aggregated_cost[2]){
					min_aggregated_cost[2] = min_aggregated_cost_tmp[1];
				}
				if(min_aggregated_cost_tmp[2]<min_aggregated_cost[2]){
					min_aggregated_cost[2] = min_aggregated_cost_tmp[2];
				}
				for(int num = 0; num < PARALLEL_DISPARITIES; num++)
				{
					#pragma HLS UNROLL
					ap_uint<BIT_WIDTH(NUM_DISPARITY)> disparity_idx = iter*PARALLEL_DISPARITIES+num;
					if(tmp[num]<right_min[disparity_idx]){
						right_min[disparity_idx] = tmp[num];
						right_min_disp[disparity_idx] = disparity_idx;
					}
				}				
				if(iter>=(ITERATION-1)){
					XF_TNAME(DST_TYPE,NPC) abs_diff = fpABSdiff<XF_TNAME(DST_TYPE,NPC) >(min_disp,second_min_disp);
					int min0 = (int(min_aggregated_cost[0])<<2)+(int(min_aggregated_cost[0])<<4);
					int min1 = int(min_aggregated_cost[1])+(int(min_aggregated_cost[1])<<1)+(int(min_aggregated_cost[1])<<4);
					int min2 = int(min_aggregated_cost[2])+(int(min_aggregated_cost[2])<<1)+(int(min_aggregated_cost[2])<<4);
					if( (abs_diff>1) && (min0>min1) ){
						min_disp = 0;
					}
					else if(min0>min2){
						min_disp = 0;
					}					
					if(col<img_width){
						left_dst_fifo.write(min_disp);
					}
					if(col>=NUM_DISPARITY-1){
						right_dst_fifo.write(right_min_disp[NUM_DISPARITY-1]);
					}
					for(int i = NUM_DISPARITY-1; i > 0; i--)
					{
						#pragma HLS UNROLL
						right_min[i] = right_min[i-1];
						right_min_disp[i] = right_min_disp[i-1];
					}
					right_min[0] = max_value_bound;	
					right_min_disp[0] = 0;
				}
			}					
		}
	}
}

}
#endif

