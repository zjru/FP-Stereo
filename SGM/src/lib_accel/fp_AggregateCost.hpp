/*
 *  FP-Stereo
 *  Copyright (C) 2020  RCSL, HKUST
 *  
 *  GPL-3.0 License
 *	
 *	This contains functions for cost aggregation.
 */

#ifndef _FP_AGGREGATECOST_HPP_
#define _FP_AGGREGATECOST_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

//#include "hls_stream.h"
#include "hls_video.h"
#include "common/xf_common.h"
#include "common/xf_utility.h"
#include "lib_accel/fp_common.h"
#include "lib_accel/fp_PostProcessing.hpp"

namespace fp{


template<int SIZE>
class fpMinArrVal
{
public:
	template <typename T>
	static void find(T a[SIZE], T &val)
	{
#pragma HLS INLINE
#pragma HLS array_partition variable=a complete dim=0

		T a1[SIZE/2];
		T a2[SIZE-SIZE/2];
#pragma HLS array_partition variable=a1 complete dim=0
#pragma HLS array_partition variable=a2 complete dim=0

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

		T v1,v2;
		fpMinArrVal<SIZE/2>::find(a1,v1);
		fpMinArrVal<SIZE-SIZE/2>::find(a2,v2);

		if(v2 < v1)
		{
			val = v2;
		}
		else
		{
			val = v1;
		}
	}
};

template<>
class fpMinArrVal<1>
{
public:
	template <typename T>
	static void find(T a[1], T &val)
	{
#pragma HLS INLINE
		val = a[0];
	}
};

template<>
class fpMinArrVal<2>
{
public:
	template <typename T>
	static void find(T a[2], T &val)
	{
#pragma HLS INLINE
#pragma HLS array_partition variable=a complete dim=0
		T v1=a[0], v2=a[1];
		if(v2 < v1)
		{
			val = v2;
		}
		else
		{
			val = v1;
		}
	}
};

/*
Input data types given different cost functions:
DATA_TYPE(SAD_COST(BW_INPUT,WINDOW_SIZE))
DATA_TYPE(ZSAD_COST(BW_INPUT,WINDOW_SIZE))
DATA_TYPE(RANK_VALUE)
DATA_TYPE(CENSUS_VALUE)
DATA_TYPE(SHD_COST(CENSUS_VALUE,SHD_WINDOW))
*/

// Four-path cost aggregation
template<int ROWS, int COLS, int COST_VALUE, int NUM_DISPARITY, int PARALLEL_DISPARITIES, int P1, int P2>
void fpAggregateCost4Path(hls::stream< DATA_TYPE(COST_VALUE) > cost[PARALLEL_DISPARITIES], hls::stream< AGGR4_DISPARITY_TYPE(COST_VALUE,P2) > aggregated_cost[PARALLEL_DISPARITIES], 
		ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	assert((AGGR_DISPARITY_WIDTH(COST_VALUE,P2) <= 20 ) && "The bit width of the aggregated cost should not exceed 20");

	#pragma HLS DATAFLOW 
	#pragma HLS INLINE OFF
	#pragma HLS ARRAY_PARTITION variable=cost complete dim=1
	#pragma HLS ARRAY_PARTITION variable=aggregated_cost complete dim=1
	
    ap_uint<BIT_WIDTH(ROWS+1)> row;
	ap_uint<BIT_WIDTH(COLS+1)> col;
	
	const int ITERATION = NUM_DISPARITY/PARALLEL_DISPARITIES;

	/* Two FIFOs to reschedule the pixels at intervals */	
	static hls::stream< ap_uint<BIT_WIDTH(COST_VALUE)*PARALLEL_DISPARITIES> > cost_tmp_0;
	#pragma HLS STREAM variable=cost_tmp_0 depth=COLS*ITERATION/4 //The minimum length to disable stall

	static hls::stream< ap_uint<BIT_WIDTH(COST_VALUE)*PARALLEL_DISPARITIES> > cost_tmp_1;
	#pragma HLS STREAM variable=cost_tmp_1 depth=COLS*ITERATION/2 //The minimum length to disable stall

	/* Two FIFOs to reorder the pixels and output them in the original order */
	static hls::stream< ap_uint<AGGR4_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> > aggregated_cost_tmp_0;
	#pragma HLS STREAM variable=aggregated_cost_tmp_0 depth=COLS*ITERATION/2

	static hls::stream< ap_uint<AGGR4_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> > aggregated_cost_tmp_1;
	#pragma HLS STREAM variable=aggregated_cost_tmp_1 depth=COLS*ITERATION/4

	/* Array to store the temporary costs from previous pixels in r1, r2, r3 directions */
	ap_uint<AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> Lr[3][(COLS+2)*ITERATION];
	#pragma HLS ARRAY_PARTITION variable=Lr complete dim=1
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_Disparity[3][(COLS+2)*ITERATION];
	#pragma HLS ARRAY_PARTITION variable=Lr_Disparity complete dim=1

	/* Array to store the temporary costs from the left upper pixel (r1 direction) */
	ap_uint<AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> Lr_r1_0[ITERATION];
	#pragma HLS ARRAY_PARTITION variable=Lr_r1_0 complete dim=1
	ap_uint<AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> Lr_r1_1[ITERATION];
	#pragma HLS ARRAY_PARTITION variable=Lr_r1_1 complete dim=1
	ap_uint<AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> Lr_r1_border[ITERATION];
	#pragma HLS ARRAY_PARTITION variable=Lr_r1_border complete dim=1

	/* Array to store the temporary costs from the left pixel (r0 direction) */
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_r0_0[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=Lr_r0_0 complete dim=1
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_r0_1[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=Lr_r0_1 complete dim=1	
	/* Registers to store the temporary costs at the right border for r0 direction computation */
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_r0_border[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=Lr_r0_border complete dim=1

	/* Temporary array to store the data for cost aggregation */
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_tmp[4][PARALLEL_DISPARITIES+2];
	#pragma HLS ARRAY_PARTITION variable=Lr_tmp complete dim=0

	/* Array to store the minimum cost from previous pixels in r1, r2, r3 directions */
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_min[3][COLS+2];
	#pragma HLS ARRAY_PARTITION variable=Lr_min complete dim=1

	/* Registers to store the minimum cost from the left upper pixel (r1 direction) */
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_r1_min_0;
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_r1_min_1;

	/* Registers to store the minimum cost from the left pixel (r0 direction) */
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_r0_min_0;
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_r0_min_1;

	/* Registers to store the temporary minimum value for cost aggregation */
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_min_tmp[4];
	#pragma HLS ARRAY_PARTITION variable=Lr_min_tmp complete dim=1

	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_min_post_tmp[4];
	#pragma HLS ARRAY_PARTITION variable=Lr_min_post_tmp complete dim=1

	/* Registers to store lr for minimum computation */
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) store_lr_for_min[4][PARALLEL_DISPARITIES];
	#pragma HLS ARRAY_PARTITION variable=store_lr_for_min complete dim=0	

	const AGGR_DISPARITY_TYPE(COST_VALUE,P2) max_value_bound = (AGGR_DISPARITY_TYPE(COST_VALUE,P2))MAX_VALUE_BOUND;

	/* Interleave the matching costs */
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
			for(ap_uint<BIT_WIDTH(ITERATION)> iter = 0; iter < ITERATION; iter++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_flatten
				ap_uint<BIT_WIDTH(COST_VALUE)*PARALLEL_DISPARITIES> input_data = 0;
				for(int num = 0; num < PARALLEL_DISPARITIES; num++)
				{
					#pragma HLS UNROLL
					input_data.range((num+1)*BIT_WIDTH(COST_VALUE)-1,num*BIT_WIDTH(COST_VALUE)) = cost[num].read();
				}
				if (col<(img_width>>1))
				{
					cost_tmp_0.write(input_data);
				}
				else{
					cost_tmp_1.write(input_data);
				}								
			}
		}
	}

	/* Process matching costs every clock cycle after interleaving */
	for (row = 0; row < img_height + 1; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS+1 max=ROWS+1
		for (col = 1; col < img_width + 1; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			if (NUM_DISPARITY == PARALLEL_DISPARITIES)
			{
				#pragma HLS PIPELINE II=2 //If equal, pipeline the outer loop. 
			}			
			for(ap_uint<BIT_WIDTH(ITERATION)> iter = 0; iter < ITERATION; iter++)
			{
				if(PARALLEL_DISPARITIES < NUM_DISPARITY){
					#pragma HLS PIPELINE II=1
				}
				#pragma HLS loop_flatten
				#pragma HLS DEPENDENCE variable=Lr array inter false
				#pragma HLS DEPENDENCE variable=Lr_Disparity array inter false
				if(NUM_DISPARITY == PARALLEL_DISPARITIES){
					#pragma HLS DEPENDENCE variable=Lr_r0_0 inter distance=2 true
					#pragma HLS DEPENDENCE variable=Lr_r0_1 inter distance=2 true
				}
				else{
					#pragma HLS DEPENDENCE variable=Lr_r0_0 inter distance=ITERATION*2-1 true
					#pragma HLS DEPENDENCE variable=Lr_r0_1 inter distance=ITERATION*2-1 true					
				}
				#pragma HLS DEPENDENCE variable=Lr_r0_border inter distance=ITERATION true
				#pragma HLS DEPENDENCE variable=Lr_r1_0 inter distance=ITERATION*2 true
				#pragma HLS DEPENDENCE variable=Lr_r1_1 inter distance=ITERATION*2 true
				#pragma HLS DEPENDENCE variable=Lr_r1_border inter distance=ITERATION true				
				
				#pragma HLS DEPENDENCE variable=Lr_min array inter false
				#pragma HLS DEPENDENCE variable=Lr_r0_min_0 inter distance=ITERATION+1 true
				#pragma HLS DEPENDENCE variable=Lr_r0_min_1 inter distance=ITERATION true
				#pragma HLS DEPENDENCE variable=Lr_r1_min_0 inter distance=ITERATION+1 true
				#pragma HLS DEPENDENCE variable=Lr_r1_min_1 inter distance=ITERATION true

				/* Pack data to save memory usage */
				ap_uint< AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES > Lr0 = Lr[0][(COLS+2)*iter+(col-2)];
				ap_uint< AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES > Lr1 = Lr[1][(COLS+2)*iter+(col)];
				ap_uint< AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES > Lr2 = Lr[2][(COLS+2)*iter+(col+2)];

				if (iter == 0)
				{
					/* Get the costs of previous pixels */
					for(int r = 0; r < 4; r++)
					{
						#pragma HLS UNROLL
						Lr_tmp[r][0] = 0;
						Lr_min_post_tmp[r] = max_value_bound;
					}
					for(int num = 0; num < PARALLEL_DISPARITIES; num++)
					{
						#pragma HLS UNROLL
						if(col.range(0,0)==0){
							Lr_tmp[0][num+1] = Lr_r0_0[num];
						}
						else if(col.range(0,0)==1){
							Lr_tmp[0][num+1] = Lr_r0_1[num];
						}
						Lr_tmp[1][num+1] = Lr0.range((num+1)*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR_DISPARITY_WIDTH(COST_VALUE,P2));
						Lr_tmp[2][num+1] = Lr1.range((num+1)*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR_DISPARITY_WIDTH(COST_VALUE,P2));
						Lr_tmp[3][num+1] = Lr2.range((num+1)*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR_DISPARITY_WIDTH(COST_VALUE,P2));						
					}
					if (PARALLEL_DISPARITIES < NUM_DISPARITY)
					{
						if(col.range(0,0)==0){
							Lr_tmp[0][PARALLEL_DISPARITIES+1] = Lr_r0_0[PARALLEL_DISPARITIES];
						}
						else if(col.range(0,0)==1){
							Lr_tmp[0][PARALLEL_DISPARITIES+1] = Lr_r0_1[PARALLEL_DISPARITIES];
						}
						Lr_tmp[1][PARALLEL_DISPARITIES+1] = Lr_Disparity[0][(COLS+2)+(col-2)];
						Lr_tmp[2][PARALLEL_DISPARITIES+1] = Lr_Disparity[1][(COLS+2)+(col)];
						Lr_tmp[3][PARALLEL_DISPARITIES+1] = Lr_Disparity[2][(COLS+2)+(col+2)];					
					}
					else
					{
						Lr_tmp[0][PARALLEL_DISPARITIES+1] = 0;
						Lr_tmp[1][PARALLEL_DISPARITIES+1] = 0;
						Lr_tmp[2][PARALLEL_DISPARITIES+1] = 0;
						Lr_tmp[3][PARALLEL_DISPARITIES+1] = 0;					
					}
					/* Get the minimum aggregated cost of previous pixels */
					if(col.range(0,0)==0){
						Lr_min_tmp[0] = Lr_r0_min_0;
					}
					else if(col.range(0,0)==1){
						Lr_min_tmp[0] = Lr_r0_min_1;
					}
					Lr_min_tmp[1] = Lr_min[0][col-2];
					Lr_min_tmp[2] = Lr_min[1][col];
					Lr_min_tmp[3] = Lr_min[2][col+2];				
				}
				else
				{
					for(int r = 0; r < 4; r++)
					{
						#pragma HLS UNROLL
						Lr_tmp[r][0] = Lr_tmp[r][PARALLEL_DISPARITIES];
						Lr_tmp[r][1] = Lr_tmp[r][PARALLEL_DISPARITIES+1];
					}
					for(int num = 1; num < PARALLEL_DISPARITIES; num++)
					{
						#pragma HLS UNROLL
						ap_uint<BIT_WIDTH(NUM_DISPARITY)> disparity_idx = iter*PARALLEL_DISPARITIES + num;

						if(col.range(0,0)==0){
							Lr_tmp[0][num+1] = Lr_r0_0[disparity_idx];
						}
						else if(col.range(0,0)==1){
							Lr_tmp[0][num+1] = Lr_r0_1[disparity_idx];
						}
						Lr_tmp[1][num+1] = Lr0.range((num+1)*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR_DISPARITY_WIDTH(COST_VALUE,P2));
						Lr_tmp[2][num+1] = Lr1.range((num+1)*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR_DISPARITY_WIDTH(COST_VALUE,P2));
						Lr_tmp[3][num+1] = Lr2.range((num+1)*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR_DISPARITY_WIDTH(COST_VALUE,P2));
					}
					ap_uint<BIT_WIDTH(NUM_DISPARITY)> disparity_idx = iter*PARALLEL_DISPARITIES + PARALLEL_DISPARITIES;									
					if (disparity_idx < NUM_DISPARITY)
					{
						if(col.range(0,0)==0){
							Lr_tmp[0][PARALLEL_DISPARITIES+1] = Lr_r0_0[disparity_idx];
						}
						else if(col.range(0,0)==1){
							Lr_tmp[0][PARALLEL_DISPARITIES+1] = Lr_r0_1[disparity_idx];
						}
						Lr_tmp[1][PARALLEL_DISPARITIES+1] = Lr_Disparity[0][(COLS+2)*(iter+1)+(col-2)];
						Lr_tmp[2][PARALLEL_DISPARITIES+1] = Lr_Disparity[1][(COLS+2)*(iter+1)+(col)];
						Lr_tmp[3][PARALLEL_DISPARITIES+1] = Lr_Disparity[2][(COLS+2)*(iter+1)+(col+2)];				
					}
					else
					{
						Lr_tmp[0][PARALLEL_DISPARITIES+1] = 0;
						Lr_tmp[1][PARALLEL_DISPARITIES+1] = 0;
						Lr_tmp[2][PARALLEL_DISPARITIES+1] = 0;
						Lr_tmp[3][PARALLEL_DISPARITIES+1] = 0;					
					}
				}	

				/* Read input matching costs */
				ap_uint<BIT_WIDTH(COST_VALUE)*PARALLEL_DISPARITIES> input_data = 0;
				if (col.range(0,0)==1)
				{
					if(row!=img_height){
						input_data = cost_tmp_0.read();
					}
				}
				else{
					if(row!=0){
						input_data = cost_tmp_1.read();
					}
				}
				// Do the computation for aggregation
				ap_uint<AGGR4_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> aggregated_data = 0;
				ap_uint<AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> combined_lr0 = 0;
				ap_uint<AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> combined_lr1 = 0;
				ap_uint<AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> combined_lr2 = 0;
				for(int num = 0; num < PARALLEL_DISPARITIES; num++)
				{
					#pragma HLS UNROLL
					AGGR_DISPARITY_TYPE(COST_VALUE,P2) initial_cost = (AGGR_DISPARITY_TYPE(COST_VALUE,P2)) input_data.range((num+1)*BIT_WIDTH(COST_VALUE)-1,num*BIT_WIDTH(COST_VALUE));
					AGGR4_DISPARITY_TYPE(COST_VALUE,P2) aggregated_val = 0;
					for(int r=0; r<4; r++)
					{
						#pragma HLS UNROLL
						AGGR_DISPARITY_TYPE(COST_VALUE,P2) lr, lr_d, lr_dp, lr_dn, lr_minimum = max_value_bound;
						lr_d = Lr_tmp[r][num+1];
						lr_dp = Lr_tmp[r][num];
						lr_dn = Lr_tmp[r][num+2];
						lr_minimum = Lr_min_tmp[r];
						
						ap_uint<BIT_WIDTH(NUM_DISPARITY)> disparity_idx = iter*PARALLEL_DISPARITIES + num;

						if (disparity_idx==0){
							lr_dp = max_value_bound - P1;
						}
						else if (disparity_idx >= (NUM_DISPARITY-1)){
							lr_dn = max_value_bound - P1;
						}

						AGGR_DISPARITY_TYPE(COST_VALUE,P2*2) min_val;
						AGGR_DISPARITY_TYPE(COST_VALUE,P2*2) min_array[4];
						#pragma HLS ARRAY_PARTITION variable=min_array complete dim=1
						min_array[0] = lr_d;
						min_array[1] = lr_dp + P1;
						min_array[2] = lr_dn + P1;
						min_array[3] = lr_minimum + P2;
	
						fpMinArrVal<4>::find(min_array,min_val);

						AGGR_DISPARITY_TYPE(COST_VALUE,P2) lr_tmp;
						#pragma HLS RESOURCE variable=lr_tmp core=AddSub_DSP
						lr_tmp = initial_cost - lr_minimum; //unsigned substraction follows modulo computation: will not overflow.
						lr = AGGR_DISPARITY_TYPE(COST_VALUE,P2)(min_val) + lr_tmp;

						if ( (((r==0)||(r==1))&&(col==1)) || (((r==1)||(r==2)||(r==3))&&(row==0)) || (((r==1)||(r==2)||(r==3))&&(row==1)&&(col.range(0,0)==0)) || ((r==3)&&(col==img_width)))
						{
							lr = initial_cost;
						}
						// Store aggregated results along each direction
						// r0 direction
						if (r==0){
							if(col.range(0,0)==0){
								Lr_r0_0[disparity_idx] = lr;
								if(col==img_width){
									Lr_r0_0[disparity_idx] = Lr_r0_border[disparity_idx];
								}							
							}
							else if(col.range(0,0)==1){
								Lr_r0_1[disparity_idx] = lr;
								if(col==(img_width-1)){
									Lr_r0_border[disparity_idx] = lr;
								}								
							}
						}
						// r1 direction
						else if(r==1){
							combined_lr0.range((num+1)*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)) = lr;
						}
						// r2 direction
						else if(r==2){
							combined_lr1.range((num+1)*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)) = lr;
						}
						// r3 direction
						else if(r==3){
							combined_lr2.range((num+1)*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)) = lr;
						}						
						store_lr_for_min[r][num] = lr;
						// accumulate results from all the directions
						aggregated_val += lr;
					}
					aggregated_data.range((num+1)*AGGR4_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR4_DISPARITY_WIDTH(COST_VALUE,P2))=aggregated_val;
				}
				/* Reorder aggregated costs */
				if (col.range(0,0)==1) 
				{
					if (row!=img_height){
						aggregated_cost_tmp_0.write(aggregated_data);
					}
				}
				else{ 
					if (row!=0){
						aggregated_cost_tmp_1.write(aggregated_data);
					}
				}

				/* Update aggregated costs in r2 and r3 directions */
				if(col==(img_width-1)){
					Lr_r1_border[iter] = combined_lr0;
				}
				Lr[1][(COLS+2)*iter+col] = combined_lr1;
				Lr_Disparity[1][(COLS+2)*iter+col] = combined_lr1.range(AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,0);
				if (col==2){
					Lr[2][(COLS+2)*iter+(COLS+1)] = combined_lr2;
					Lr_Disparity[2][(COLS+2)*iter+(COLS+1)] = combined_lr2.range(AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,0);
				}
				else if((col>2)&&(col<(COLS+1))){
					Lr[2][(COLS+2)*iter+col] = combined_lr2;
					Lr_Disparity[2][(COLS+2)*iter+col] = combined_lr2.range(AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,0);
				}
				// updating the previous col-2 cost in r1 direction
				if(col.range(0,0)==0){
					Lr[0][(COLS+2)*iter+col-2] = Lr_r1_0[iter];
					Lr_Disparity[0][(COLS+2)*iter+(col-2)] = Lr_r1_0[iter].range(AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,0);
					Lr_r1_0[iter] = combined_lr0;
					if(col==img_width){
						Lr_r1_0[iter] = Lr_r1_border[iter];
					}
				}
				else{
					Lr[0][(COLS+2)*iter+col-2] = Lr_r1_1[iter];
					Lr_Disparity[0][(COLS+2)*iter+(col-2)] = Lr_r1_1[iter].range(AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,0);
					Lr_r1_1[iter] = combined_lr0;
				}

				// compute min value for all sets of disparities
				for (int r=0; r<4; r++)
				{
					#pragma HLS UNROLL
					AGGR_DISPARITY_TYPE(COST_VALUE,P2) min_cost;
					fpMinArrVal<PARALLEL_DISPARITIES>::find(store_lr_for_min[r], min_cost);
					if (min_cost < Lr_min_post_tmp[r])
						Lr_min_post_tmp[r] = min_cost;
				}

				if (iter >= (ITERATION-1))// when its the last set of disparities update the min arrays
				{
					if(col.range(0,0)==0){
						Lr_r0_min_0 = Lr_min_post_tmp[0];
						if(col==img_width){
							Lr_r0_min_0 = Lr_r0_min_1;
						}
					}
					else{
						Lr_r0_min_1 = Lr_min_post_tmp[0];
					}
					// for the pixel in the col-2 update the min values
					if(col.range(0,0)==0){
						Lr_min[0][col-2] = Lr_r1_min_0;
						Lr_r1_min_0 = Lr_min_post_tmp[1];
						if(col==img_width){
							Lr_r1_min_0 = Lr_r1_min_1;
						}
					}
					else{
						Lr_min[0][col-2] = Lr_r1_min_1;
						Lr_r1_min_1 = Lr_min_post_tmp[1];
					}

					Lr_min[1][col] = Lr_min_post_tmp[2];
					
					if (col==2){
						Lr_min[2][COLS+1] = Lr_min_post_tmp[3];
					}
					else if((col>2)&&(col<(COLS+1))){
						Lr_min[2][col] = Lr_min_post_tmp[3];
					}
				}
			}			
		}		
	}

	/* Output aggregated costs to the next stage after reordering */
	for (row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for (col = 0; col < img_width; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			if (NUM_DISPARITY == PARALLEL_DISPARITIES)
			{
				#pragma HLS PIPELINE II=1 //If equal, pipeline the outer loop. 
			}			
			for(ap_uint<BIT_WIDTH(ITERATION)> iter = 0; iter < ITERATION; iter++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_flatten
				ap_uint<AGGR4_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> aggregated_data = 0;
				if (col<(img_width>>1))
				{
					aggregated_data = aggregated_cost_tmp_0.read();
				}
				else{
					aggregated_data = aggregated_cost_tmp_1.read();
				}
				for(int num = 0; num < PARALLEL_DISPARITIES; num++)
				{
					#pragma HLS UNROLL
					aggregated_cost[num].write(aggregated_data.range((num+1)*AGGR4_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR4_DISPARITY_WIDTH(COST_VALUE,P2)));
				}
			}
		}
	}

}

/* The same copy of fpAggregateCost4Path, which is used to ensure parallel path aggregation with HLS tools when considering L-R consistency check. */
template<int ROWS, int COLS, int COST_VALUE, int NUM_DISPARITY, int PARALLEL_DISPARITIES, int P1, int P2>
void fpAggregateCost4Path_copy(hls::stream< DATA_TYPE(COST_VALUE) > cost[PARALLEL_DISPARITIES], hls::stream< AGGR4_DISPARITY_TYPE(COST_VALUE,P2) > aggregated_cost[PARALLEL_DISPARITIES], 
		ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	assert((AGGR_DISPARITY_WIDTH(COST_VALUE,P2) <= 20 ) && "The bit width of the aggregated cost should not exceed 20");

	#pragma HLS DATAFLOW 
	#pragma HLS INLINE OFF
	#pragma HLS ARRAY_PARTITION variable=cost complete dim=1
	#pragma HLS ARRAY_PARTITION variable=aggregated_cost complete dim=1
	
    ap_uint<BIT_WIDTH(ROWS+1)> row;
	ap_uint<BIT_WIDTH(COLS+1)> col;
	
	const int ITERATION = NUM_DISPARITY/PARALLEL_DISPARITIES;

	/* Two FIFOs to reschedule the pixels at intervals */	
	static hls::stream< ap_uint<BIT_WIDTH(COST_VALUE)*PARALLEL_DISPARITIES> > cost_tmp_0;
	#pragma HLS STREAM variable=cost_tmp_0 depth=COLS*ITERATION/4 //The minimum length to disable stall

	static hls::stream< ap_uint<BIT_WIDTH(COST_VALUE)*PARALLEL_DISPARITIES> > cost_tmp_1;
	#pragma HLS STREAM variable=cost_tmp_1 depth=COLS*ITERATION/2 //The minimum length to disable stall

	/* Two FIFOs to reorder the pixels and output them in the original order */
	static hls::stream< ap_uint<AGGR4_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> > aggregated_cost_tmp_0;
	#pragma HLS STREAM variable=aggregated_cost_tmp_0 depth=COLS*ITERATION/2

	static hls::stream< ap_uint<AGGR4_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> > aggregated_cost_tmp_1;
	#pragma HLS STREAM variable=aggregated_cost_tmp_1 depth=COLS*ITERATION/4

	/* Array to store the temporary costs from previous pixels in r1, r2, r3 directions */
	ap_uint<AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> Lr[3][(COLS+2)*ITERATION];
	#pragma HLS ARRAY_PARTITION variable=Lr complete dim=1
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_Disparity[3][(COLS+2)*ITERATION];
	#pragma HLS ARRAY_PARTITION variable=Lr_Disparity complete dim=1

	/* Array to store the temporary costs from the left upper pixel (r1 direction) */
	ap_uint<AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> Lr_r1_0[ITERATION];
	#pragma HLS ARRAY_PARTITION variable=Lr_r1_0 complete dim=1
	ap_uint<AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> Lr_r1_1[ITERATION];
	#pragma HLS ARRAY_PARTITION variable=Lr_r1_1 complete dim=1
	ap_uint<AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> Lr_r1_border[ITERATION];
	#pragma HLS ARRAY_PARTITION variable=Lr_r1_border complete dim=1

	/* Array to store the temporary costs from the left pixel (r0 direction) */
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_r0_0[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=Lr_r0_0 complete dim=1
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_r0_1[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=Lr_r0_1 complete dim=1	
	/* Registers to store the temporary costs at the right border for r0 direction computation */
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_r0_border[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=Lr_r0_border complete dim=1

	/* Temporary array to store the data for cost aggregation */
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_tmp[4][PARALLEL_DISPARITIES+2];
	#pragma HLS ARRAY_PARTITION variable=Lr_tmp complete dim=0

	/* Array to store the minimum cost from previous pixels in r1, r2, r3 directions */
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_min[3][COLS+2];
	#pragma HLS ARRAY_PARTITION variable=Lr_min complete dim=1

	/* Registers to store the minimum cost from the left upper pixel (r1 direction) */
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_r1_min_0;
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_r1_min_1;

	/* Registers to store the minimum cost from the left pixel (r0 direction) */
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_r0_min_0;
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_r0_min_1;

	/* Registers to store the temporary minimum value for cost aggregation */
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_min_tmp[4];
	#pragma HLS ARRAY_PARTITION variable=Lr_min_tmp complete dim=1

	AGGR_DISPARITY_TYPE(COST_VALUE,P2) Lr_min_post_tmp[4];
	#pragma HLS ARRAY_PARTITION variable=Lr_min_post_tmp complete dim=1

	/* Registers to store lr for minimum computation */
	AGGR_DISPARITY_TYPE(COST_VALUE,P2) store_lr_for_min[4][PARALLEL_DISPARITIES];
	#pragma HLS ARRAY_PARTITION variable=store_lr_for_min complete dim=0	

	const AGGR_DISPARITY_TYPE(COST_VALUE,P2) max_value_bound = (AGGR_DISPARITY_TYPE(COST_VALUE,P2))MAX_VALUE_BOUND;

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
			for(ap_uint<BIT_WIDTH(ITERATION)> iter = 0; iter < ITERATION; iter++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_flatten
				ap_uint<BIT_WIDTH(COST_VALUE)*PARALLEL_DISPARITIES> input_data = 0;
				for(int num = 0; num < PARALLEL_DISPARITIES; num++)
				{
					#pragma HLS UNROLL
					input_data.range((num+1)*BIT_WIDTH(COST_VALUE)-1,num*BIT_WIDTH(COST_VALUE)) = cost[num].read();
				}
				if (col<(img_width>>1))
				{
					cost_tmp_0.write(input_data);
				}
				else{
					cost_tmp_1.write(input_data);
				}								
			}
		}
	}

	for (row = 0; row < img_height + 1; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS+1 max=ROWS+1
		for (col = 1; col < img_width + 1; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			if (NUM_DISPARITY == PARALLEL_DISPARITIES)
			{
				#pragma HLS PIPELINE II=2 //If equal, pipeline the outer loop. 
			}			
			for(ap_uint<BIT_WIDTH(ITERATION)> iter = 0; iter < ITERATION; iter++)
			{
				if(PARALLEL_DISPARITIES < NUM_DISPARITY){
					#pragma HLS PIPELINE II=1
				}
				#pragma HLS loop_flatten
				#pragma HLS DEPENDENCE variable=Lr array inter false
				#pragma HLS DEPENDENCE variable=Lr_Disparity array inter false
				if(NUM_DISPARITY == PARALLEL_DISPARITIES){
					#pragma HLS DEPENDENCE variable=Lr_r0_0 inter distance=2 true
					#pragma HLS DEPENDENCE variable=Lr_r0_1 inter distance=2 true
				}
				else{
					#pragma HLS DEPENDENCE variable=Lr_r0_0 inter distance=ITERATION*2-1 true
					#pragma HLS DEPENDENCE variable=Lr_r0_1 inter distance=ITERATION*2-1 true					
				}
				#pragma HLS DEPENDENCE variable=Lr_r0_border inter distance=ITERATION true
				#pragma HLS DEPENDENCE variable=Lr_r1_0 inter distance=ITERATION*2 true
				#pragma HLS DEPENDENCE variable=Lr_r1_1 inter distance=ITERATION*2 true
				#pragma HLS DEPENDENCE variable=Lr_r1_border inter distance=ITERATION true				
				
				#pragma HLS DEPENDENCE variable=Lr_min array inter false
				#pragma HLS DEPENDENCE variable=Lr_r0_min_0 inter distance=ITERATION+1 true
				#pragma HLS DEPENDENCE variable=Lr_r0_min_1 inter distance=ITERATION true
				#pragma HLS DEPENDENCE variable=Lr_r1_min_0 inter distance=ITERATION+1 true
				#pragma HLS DEPENDENCE variable=Lr_r1_min_1 inter distance=ITERATION true

				ap_uint< AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES > Lr0 = Lr[0][(COLS+2)*iter+(col-2)];
				ap_uint< AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES > Lr1 = Lr[1][(COLS+2)*iter+(col)];
				ap_uint< AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES > Lr2 = Lr[2][(COLS+2)*iter+(col+2)];

				if (iter == 0)
				{
					/*Get the costs of previous pixels.*/
					for(int r = 0; r < 4; r++)
					{
						#pragma HLS UNROLL
						Lr_tmp[r][0] = 0;
						Lr_min_post_tmp[r] = max_value_bound;
					}
					for(int num = 0; num < PARALLEL_DISPARITIES; num++)
					{
						#pragma HLS UNROLL
						if(col.range(0,0)==0){
							Lr_tmp[0][num+1] = Lr_r0_0[num];
						}
						else if(col.range(0,0)==1){
							Lr_tmp[0][num+1] = Lr_r0_1[num];
						}
						Lr_tmp[1][num+1] = Lr0.range((num+1)*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR_DISPARITY_WIDTH(COST_VALUE,P2));
						Lr_tmp[2][num+1] = Lr1.range((num+1)*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR_DISPARITY_WIDTH(COST_VALUE,P2));
						Lr_tmp[3][num+1] = Lr2.range((num+1)*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR_DISPARITY_WIDTH(COST_VALUE,P2));						
					}
					if (PARALLEL_DISPARITIES < NUM_DISPARITY)
					{
						if(col.range(0,0)==0){
							Lr_tmp[0][PARALLEL_DISPARITIES+1] = Lr_r0_0[PARALLEL_DISPARITIES];
						}
						else if(col.range(0,0)==1){
							Lr_tmp[0][PARALLEL_DISPARITIES+1] = Lr_r0_1[PARALLEL_DISPARITIES];
						}
						Lr_tmp[1][PARALLEL_DISPARITIES+1] = Lr_Disparity[0][(COLS+2)+(col-2)];
						Lr_tmp[2][PARALLEL_DISPARITIES+1] = Lr_Disparity[1][(COLS+2)+(col)];
						Lr_tmp[3][PARALLEL_DISPARITIES+1] = Lr_Disparity[2][(COLS+2)+(col+2)];					
					}
					else
					{
						Lr_tmp[0][PARALLEL_DISPARITIES+1] = 0;
						Lr_tmp[1][PARALLEL_DISPARITIES+1] = 0;
						Lr_tmp[2][PARALLEL_DISPARITIES+1] = 0;
						Lr_tmp[3][PARALLEL_DISPARITIES+1] = 0;					
					}
					/*Get the minimum disparity cost of previous pixels.*/
					if(col.range(0,0)==0){
						Lr_min_tmp[0] = Lr_r0_min_0;
					}
					else if(col.range(0,0)==1){
						Lr_min_tmp[0] = Lr_r0_min_1;
					}
					Lr_min_tmp[1] = Lr_min[0][col-2];
					Lr_min_tmp[2] = Lr_min[1][col];
					Lr_min_tmp[3] = Lr_min[2][col+2];				
				}
				else
				{
					for(int r = 0; r < 4; r++)
					{
						#pragma HLS UNROLL
						Lr_tmp[r][0] = Lr_tmp[r][PARALLEL_DISPARITIES];
						Lr_tmp[r][1] = Lr_tmp[r][PARALLEL_DISPARITIES+1];
					}
					for(int num = 1; num < PARALLEL_DISPARITIES; num++)
					{
						#pragma HLS UNROLL
						ap_uint<BIT_WIDTH(NUM_DISPARITY)> disparity_idx = iter*PARALLEL_DISPARITIES + num;

						if(col.range(0,0)==0){
							Lr_tmp[0][num+1] = Lr_r0_0[disparity_idx];
						}
						else if(col.range(0,0)==1){
							Lr_tmp[0][num+1] = Lr_r0_1[disparity_idx];
						}
						Lr_tmp[1][num+1] = Lr0.range((num+1)*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR_DISPARITY_WIDTH(COST_VALUE,P2));
						Lr_tmp[2][num+1] = Lr1.range((num+1)*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR_DISPARITY_WIDTH(COST_VALUE,P2));
						Lr_tmp[3][num+1] = Lr2.range((num+1)*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR_DISPARITY_WIDTH(COST_VALUE,P2));
					}
					ap_uint<BIT_WIDTH(NUM_DISPARITY)> disparity_idx = iter*PARALLEL_DISPARITIES + PARALLEL_DISPARITIES;									
					if (disparity_idx < NUM_DISPARITY)
					{
						if(col.range(0,0)==0){
							Lr_tmp[0][PARALLEL_DISPARITIES+1] = Lr_r0_0[disparity_idx];
						}
						else if(col.range(0,0)==1){
							Lr_tmp[0][PARALLEL_DISPARITIES+1] = Lr_r0_1[disparity_idx];
						}
						Lr_tmp[1][PARALLEL_DISPARITIES+1] = Lr_Disparity[0][(COLS+2)*(iter+1)+(col-2)];
						Lr_tmp[2][PARALLEL_DISPARITIES+1] = Lr_Disparity[1][(COLS+2)*(iter+1)+(col)];
						Lr_tmp[3][PARALLEL_DISPARITIES+1] = Lr_Disparity[2][(COLS+2)*(iter+1)+(col+2)];				
					}
					else
					{
						Lr_tmp[0][PARALLEL_DISPARITIES+1] = 0;
						Lr_tmp[1][PARALLEL_DISPARITIES+1] = 0;
						Lr_tmp[2][PARALLEL_DISPARITIES+1] = 0;
						Lr_tmp[3][PARALLEL_DISPARITIES+1] = 0;					
					}
				}	

				ap_uint<BIT_WIDTH(COST_VALUE)*PARALLEL_DISPARITIES> input_data = 0;
				if (col.range(0,0)==1)
				{
					if(row!=img_height){
						input_data = cost_tmp_0.read();
					}
				}
				else{
					if(row!=0){
						input_data = cost_tmp_1.read();
					}
				}
				ap_uint<AGGR4_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> aggregated_data = 0;
				ap_uint<AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> combined_lr0 = 0;
				ap_uint<AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> combined_lr1 = 0;
				ap_uint<AGGR_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> combined_lr2 = 0;
				for(int num = 0; num < PARALLEL_DISPARITIES; num++)
				{
					#pragma HLS UNROLL
					AGGR_DISPARITY_TYPE(COST_VALUE,P2) initial_cost = (AGGR_DISPARITY_TYPE(COST_VALUE,P2)) input_data.range((num+1)*BIT_WIDTH(COST_VALUE)-1,num*BIT_WIDTH(COST_VALUE));
					AGGR4_DISPARITY_TYPE(COST_VALUE,P2) aggregated_val = 0;
					for(int r=0; r<4; r++)
					{
						#pragma HLS UNROLL
						AGGR_DISPARITY_TYPE(COST_VALUE,P2) lr, lr_d, lr_dp, lr_dn, lr_minimum = max_value_bound;
						lr_d = Lr_tmp[r][num+1];
						lr_dp = Lr_tmp[r][num];
						lr_dn = Lr_tmp[r][num+2];
						lr_minimum = Lr_min_tmp[r];
						
						ap_uint<BIT_WIDTH(NUM_DISPARITY)> disparity_idx = iter*PARALLEL_DISPARITIES + num;

						if (disparity_idx==0){
							lr_dp = max_value_bound - P1;
						}
						else if (disparity_idx >= (NUM_DISPARITY-1)){
							lr_dn = max_value_bound - P1;
						}

						AGGR_DISPARITY_TYPE(COST_VALUE,P2*2) min_val;
						AGGR_DISPARITY_TYPE(COST_VALUE,P2*2) min_array[4];
						#pragma HLS ARRAY_PARTITION variable=min_array complete dim=1
						min_array[0] = lr_d;
						min_array[1] = lr_dp + P1;
						min_array[2] = lr_dn + P1;
						min_array[3] = lr_minimum + P2;
	
						fpMinArrVal<4>::find(min_array,min_val);

						AGGR_DISPARITY_TYPE(COST_VALUE,P2) lr_tmp;
						#pragma HLS RESOURCE variable=lr_tmp core=AddSub_DSP
						lr_tmp = initial_cost - lr_minimum; //unsigned substraction follows modulo computation: will not overflow.
						lr = AGGR_DISPARITY_TYPE(COST_VALUE,P2)(min_val) + lr_tmp;

						if ( (((r==0)||(r==1))&&(col==1)) || (((r==1)||(r==2)||(r==3))&&(row==0)) || (((r==1)||(r==2)||(r==3))&&(row==1)&&(col.range(0,0)==0)) || ((r==3)&&(col==img_width)))
						{
							lr = initial_cost;
						}

						if (r==0){
							if(col.range(0,0)==0){
								Lr_r0_0[disparity_idx] = lr;
								if(col==img_width){
									Lr_r0_0[disparity_idx] = Lr_r0_border[disparity_idx];
								}							
							}
							else if(col.range(0,0)==1){
								Lr_r0_1[disparity_idx] = lr;
								if(col==(img_width-1)){
									Lr_r0_border[disparity_idx] = lr;
								}								
							}
						}
						else if(r==1){
							combined_lr0.range((num+1)*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)) = lr;
						}
						else if(r==2){
							combined_lr1.range((num+1)*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)) = lr;
						}
						else if(r==3){
							combined_lr2.range((num+1)*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR_DISPARITY_WIDTH(COST_VALUE,P2)) = lr;
						}						
						store_lr_for_min[r][num] = lr;
						aggregated_val += lr;
					}
					aggregated_data.range((num+1)*AGGR4_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR4_DISPARITY_WIDTH(COST_VALUE,P2))=aggregated_val;
				}
				/*Reorder the disparities*/
				if (col.range(0,0)==1) 
				{
					if (row!=img_height){
						aggregated_cost_tmp_0.write(aggregated_data);
					}
				}
				else{ 
					if (row!=0){
						aggregated_cost_tmp_1.write(aggregated_data);
					}
				}

				if(col==(img_width-1)){
					Lr_r1_border[iter] = combined_lr0;
				}
				Lr[1][(COLS+2)*iter+col] = combined_lr1;
				Lr_Disparity[1][(COLS+2)*iter+col] = combined_lr1.range(AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,0);
				if (col==2){
					Lr[2][(COLS+2)*iter+(COLS+1)] = combined_lr2;
					Lr_Disparity[2][(COLS+2)*iter+(COLS+1)] = combined_lr2.range(AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,0);
				}
				else if((col>2)&&(col<(COLS+1))){
					Lr[2][(COLS+2)*iter+col] = combined_lr2;
					Lr_Disparity[2][(COLS+2)*iter+col] = combined_lr2.range(AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,0);
				}
				// updating the previous col-2 cost in r1 direction
				if(col.range(0,0)==0){
					Lr[0][(COLS+2)*iter+col-2] = Lr_r1_0[iter];
					Lr_Disparity[0][(COLS+2)*iter+(col-2)] = Lr_r1_0[iter].range(AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,0);
					Lr_r1_0[iter] = combined_lr0;
					if(col==img_width){
						Lr_r1_0[iter] = Lr_r1_border[iter];
					}
				}
				else{
					Lr[0][(COLS+2)*iter+col-2] = Lr_r1_1[iter];
					Lr_Disparity[0][(COLS+2)*iter+(col-2)] = Lr_r1_1[iter].range(AGGR_DISPARITY_WIDTH(COST_VALUE,P2)-1,0);
					Lr_r1_1[iter] = combined_lr0;
				}

				// compute min value for all sets of disparities
				for (int r=0; r<4; r++)
				{
					#pragma HLS UNROLL
					AGGR_DISPARITY_TYPE(COST_VALUE,P2) min_cost;
					fpMinArrVal<PARALLEL_DISPARITIES>::find(store_lr_for_min[r], min_cost);
					if (min_cost < Lr_min_post_tmp[r])
						Lr_min_post_tmp[r] = min_cost;
				}

				if (iter >= (ITERATION-1))// when its the last set of disparities update the min arrays
				{
					if(col.range(0,0)==0){
						Lr_r0_min_0 = Lr_min_post_tmp[0];
						if(col==img_width){
							Lr_r0_min_0 = Lr_r0_min_1;
						}
					}
					else{
						Lr_r0_min_1 = Lr_min_post_tmp[0];
					}
					// for the pixel in the col-2 update the min values
					if(col.range(0,0)==0){
						Lr_min[0][col-2] = Lr_r1_min_0;
						Lr_r1_min_0 = Lr_min_post_tmp[1];
						if(col==img_width){
							Lr_r1_min_0 = Lr_r1_min_1;
						}
					}
					else{
						Lr_min[0][col-2] = Lr_r1_min_1;
						Lr_r1_min_1 = Lr_min_post_tmp[1];
					}

					Lr_min[1][col] = Lr_min_post_tmp[2];
					
					if (col==2){
						Lr_min[2][COLS+1] = Lr_min_post_tmp[3];
					}
					else if((col>2)&&(col<(COLS+1))){
						Lr_min[2][col] = Lr_min_post_tmp[3];
					}
				}
			}			
		}		
	}

	for (row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for (col = 0; col < img_width; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			if (NUM_DISPARITY == PARALLEL_DISPARITIES)
			{
				#pragma HLS PIPELINE II=1 //If equal, pipeline the outer loop. 
			}			
			for(ap_uint<BIT_WIDTH(ITERATION)> iter = 0; iter < ITERATION; iter++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_flatten
				ap_uint<AGGR4_DISPARITY_WIDTH(COST_VALUE,P2)*PARALLEL_DISPARITIES> aggregated_data = 0;
				if (col<(img_width>>1))
				{
					aggregated_data = aggregated_cost_tmp_0.read();
				}
				else{
					aggregated_data = aggregated_cost_tmp_1.read();
				}
				for(int num = 0; num < PARALLEL_DISPARITIES; num++)
				{
					#pragma HLS UNROLL
					aggregated_cost[num].write(aggregated_data.range((num+1)*AGGR4_DISPARITY_WIDTH(COST_VALUE,P2)-1,num*AGGR4_DISPARITY_WIDTH(COST_VALUE,P2)));
				}
			}
		}
	}

}


}
#endif

