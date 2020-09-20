/*
 *  FP-Stereo
 *  Copyright (C) 2020  RCSL, HKUST
 *  
 *  GPL-3.0 License
 *
 */
 
#ifndef _FP_COMPUTECOST_HPP_
#define _FP_COMPUTECOST_HPP_

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

/*------------------------------------------------SAD: Sum of Absolute Differences-------------------------------------------------*/
// Matching cost computation: SAD
template<int BW_INPUT, int ROWS, int COLS, int WINDOW_SIZE, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpComputeSADCost(hls::stream< ap_uint<BW_INPUT> > &src_l, hls::stream< ap_uint<BW_INPUT> > &src_r, 
		hls::stream< DATA_TYPE(SAD_COST(BW_INPUT,WINDOW_SIZE)) > cost[PARALLEL_DISPARITIES], 
        ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS DATAFLOW
	#pragma HLS ARRAY_PARTITION variable=cost complete dim=1

	const int ITERATION = NUM_DISPARITY/PARALLEL_DISPARITIES;

	hls::Window<WINDOW_SIZE, WINDOW_SIZE, ap_uint<BW_INPUT> > left_window_buf;
	hls::Window<WINDOW_SIZE, NUM_DISPARITY+WINDOW_SIZE-1, ap_uint<BW_INPUT> > right_window_buf;  

    hls::Window<WINDOW_SIZE, PARALLEL_DISPARITIES+WINDOW_SIZE-1, ap_uint<BW_INPUT> > tmp_right_window_buf;

	hls::LineBuffer<WINDOW_SIZE-1, COLS, ap_uint<BW_INPUT> > left_line_buf;
    #pragma HLS RESOURCE variable=left_line_buf.val core=RAM_S2P_BRAM 
    hls::LineBuffer<WINDOW_SIZE-1, COLS, ap_uint<BW_INPUT> > right_line_buf;
	#pragma HLS RESOURCE variable=right_line_buf.val core=RAM_S2P_BRAM

	ap_uint<BIT_WIDTH(WINDOW_SIZE)> half_win = WINDOW_SIZE >> 1;
	ap_uint<BIT_WIDTH(WINDOW_SIZE)> initial_row;
	ap_uint<BIT_WIDTH(COLS+(WINDOW_SIZE>>1))> col;
	ap_uint<BIT_WIDTH(ROWS)> row;

	//Initialize the line buffer
	for( col = 0; col < img_width; col++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
		#pragma HLS pipeline
		for(initial_row = 0; initial_row < half_win; initial_row++)
		{
			#pragma HLS UNROLL
			left_line_buf.val[initial_row][col] = 0;
            right_line_buf.val[initial_row][col] = 0;
		}
		left_line_buf.val[half_win][col]=src_l.read();
        right_line_buf.val[half_win][col]=src_r.read();
	}
    ap_uint<BIT_WIDTH(WINDOW_SIZE)> next_row = half_win + 1;
	for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> i = 0; i < half_win-1; i++)
	{
		#pragma HLS UNROLL
		for( col = 0; col < img_width; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			#pragma HLS pipeline
            left_line_buf.val[next_row][col]=src_l.read();
            right_line_buf.val[next_row][col]=src_r.read();
		}
		next_row++;
	}
	//Initialize the window buffer
    for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
	{
		#pragma HLS UNROLL
		for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_col = 0; win_col < WINDOW_SIZE; win_col++)
		{
			#pragma HLS UNROLL
			left_window_buf.val[win_row][win_col] = 0;
		}
	}
	//Process the image based on window and line buffers
	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
		{
			#pragma HLS UNROLL
			for(ap_uint<BIT_WIDTH(NUM_DISPARITY+WINDOW_SIZE-1)> win_col = 0; win_col < NUM_DISPARITY+WINDOW_SIZE-1; win_col++)
			{
				#pragma HLS UNROLL
				right_window_buf.val[win_row][win_col] = 0;
			}
		}	
        for(col = 0; col < img_width+half_win; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS+(WINDOW_SIZE>>1) max=COLS+(WINDOW_SIZE>>1)
			if (NUM_DISPARITY == PARALLEL_DISPARITIES)
			{
				#pragma HLS PIPELINE II=1 //If equal, pipeline the outer loop. 
			}			
			for(ap_uint<BIT_WIDTH(ITERATION)> iter = 0; iter < ITERATION; iter++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_flatten
				#pragma HLS DEPENDENCE variable=left_line_buf.val array inter false
				#pragma HLS DEPENDENCE variable=right_line_buf.val array inter false				
				if (iter==0)
				{
                    left_window_buf.shift_pixels_left();
					right_window_buf.shift_pixels_right();                                         
                    if(col<img_width){
                        for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> line_row = 0; line_row < WINDOW_SIZE-1; line_row++)
                        {
                            #pragma HLS UNROLL
                            left_window_buf.val[line_row][WINDOW_SIZE-1] = left_line_buf.val[line_row][col];
							right_window_buf.val[line_row][0] = right_line_buf.val[line_row][col];
                        }
                        ap_uint<BW_INPUT> tmp_l = 0;
                        ap_uint<BW_INPUT> tmp_r = 0;
                        if(row < img_height-half_win){
                            tmp_l = src_l.read();
                            tmp_r = src_r.read();
                        }
                        left_window_buf.val[WINDOW_SIZE-1][WINDOW_SIZE-1] = tmp_l;
						right_window_buf.val[WINDOW_SIZE-1][0] = tmp_r;
                        left_line_buf.shift_pixels_up(col);
                        right_line_buf.shift_pixels_up(col);
                        left_line_buf.val[WINDOW_SIZE-2][col] = tmp_l;
                        right_line_buf.val[WINDOW_SIZE-2][col] = tmp_r;
                    }
                    else{
                        for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
                        {
                            #pragma HLS UNROLL
                            left_window_buf.val[win_row][WINDOW_SIZE-1] = 0;
							right_window_buf.val[win_row][0] = 0;
                        }
                    }
				}
				for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
				{
					#pragma HLS UNROLL
					for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> num = 0; num < PARALLEL_DISPARITIES; num++)
					{
						#pragma HLS UNROLL
						tmp_right_window_buf.val[win_row][num] = right_window_buf.val[win_row][iter*PARALLEL_DISPARITIES+num];
					}
					for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> num = 0; num < WINDOW_SIZE-1; num++)
					{
						#pragma HLS UNROLL
						tmp_right_window_buf.val[win_row][PARALLEL_DISPARITIES+num] = right_window_buf.val[win_row][(iter+1)*PARALLEL_DISPARITIES+num];
					}
				}
				DATA_TYPE(SAD_COST(BW_INPUT,WINDOW_SIZE)) SAD_value_buf[PARALLEL_DISPARITIES+WINDOW_SIZE-1];
				#pragma HLS ARRAY_PARTITION variable=SAD_value_buf complete dim=1
				for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES+WINDOW_SIZE-1)> num = 0; num < PARALLEL_DISPARITIES+WINDOW_SIZE-1; num++)
				{
					#pragma HLS UNROLL
					SAD_value_buf[num] = 0;
				}

				for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
				{
					#pragma HLS UNROLL
					for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_col = 0; win_col < WINDOW_SIZE; win_col++)
					{
						#pragma HLS UNROLL
						ap_uint<BW_INPUT> left_pixel = left_window_buf.getval(win_row,win_col);
						for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES+WINDOW_SIZE-1)> num = 0; num < PARALLEL_DISPARITIES+WINDOW_SIZE-1; num++)
						{
							#pragma HLS UNROLL
							ap_uint<BW_INPUT> right_pixel = tmp_right_window_buf.val[win_row][num];
							if(num>=WINDOW_SIZE-1-win_col){
								ap_uint<BW_INPUT> delta = fpABSdiff< ap_uint<BW_INPUT> >(left_pixel,right_pixel);												
								SAD_value_buf[num-(WINDOW_SIZE-1-win_col)]+=delta;
							}
						}
					}
				}
				if (col >= half_win)
				{
					for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> num = 0; num < PARALLEL_DISPARITIES; num++)
					{
						#pragma HLS UNROLL
						cost[num].write(SAD_value_buf[num]);
					}
				}
			}
		}
	}	
}

// Matching cost computation: SAD for L-R consistency check (LR2 method)
template<int BW_INPUT, int ROWS, int COLS, int WINDOW_SIZE, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpLRComputeSADCost(hls::stream< ap_uint<BW_INPUT> > &src_l, hls::stream< ap_uint<BW_INPUT> > &src_r, 
		hls::stream< DATA_TYPE(SAD_COST(BW_INPUT,WINDOW_SIZE)) > left_cost[PARALLEL_DISPARITIES], 
        hls::stream< DATA_TYPE(SAD_COST(BW_INPUT,WINDOW_SIZE)) > right_cost[PARALLEL_DISPARITIES],
        ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS DATAFLOW	
	#pragma HLS ARRAY_PARTITION variable=left_cost complete dim=1
    #pragma HLS ARRAY_PARTITION variable=right_cost complete dim=1

	const int ITERATION = NUM_DISPARITY/PARALLEL_DISPARITIES;

	hls::Window<WINDOW_SIZE, NUM_DISPARITY+WINDOW_SIZE-1, ap_uint<BW_INPUT> > left_window_buf;
	hls::Window<WINDOW_SIZE, NUM_DISPARITY+WINDOW_SIZE-1, ap_uint<BW_INPUT> > right_window_buf; 

    hls::Window<WINDOW_SIZE, PARALLEL_DISPARITIES+WINDOW_SIZE-1, ap_uint<BW_INPUT> > tmp_left_window_buf;
    hls::Window<WINDOW_SIZE, PARALLEL_DISPARITIES+WINDOW_SIZE-1, ap_uint<BW_INPUT> > tmp_right_window_buf;	
     
	hls::LineBuffer<WINDOW_SIZE-1, COLS, ap_uint<BW_INPUT> > left_line_buf;
    #pragma HLS RESOURCE variable=left_line_buf.val core=RAM_S2P_BRAM 
    hls::LineBuffer<WINDOW_SIZE-1, COLS, ap_uint<BW_INPUT> > right_line_buf;
	#pragma HLS RESOURCE variable=right_line_buf.val core=RAM_S2P_BRAM

	ap_uint<BIT_WIDTH(WINDOW_SIZE)> half_win = WINDOW_SIZE >> 1;
	ap_uint<BIT_WIDTH(WINDOW_SIZE)> initial_row;
	ap_uint<BIT_WIDTH(COLS+(WINDOW_SIZE>>1)+NUM_DISPARITY-1)> col;
	ap_uint<BIT_WIDTH(ROWS)> row;

	//Initialize the line buffer
	for( col = 0; col < img_width; col++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
		#pragma HLS pipeline
		for(initial_row = 0; initial_row < half_win; initial_row++)
		{
			#pragma HLS UNROLL
			left_line_buf.val[initial_row][col] = 0;
            right_line_buf.val[initial_row][col] = 0;
		}
		left_line_buf.val[half_win][col]=src_l.read();
        right_line_buf.val[half_win][col]=src_r.read();
	}
    ap_uint<BIT_WIDTH(WINDOW_SIZE)> next_row = half_win + 1;
	for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> i = 0; i < half_win-1; i++)
	{
		#pragma HLS UNROLL
		for( col = 0; col < img_width; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			#pragma HLS pipeline
            left_line_buf.val[next_row][col]=src_l.read();
            right_line_buf.val[next_row][col]=src_r.read();
		}
		next_row++;
	}

	//Process the image
	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
		{
			#pragma HLS UNROLL
			for(ap_uint<BIT_WIDTH(NUM_DISPARITY+WINDOW_SIZE-1)> win_col = 0; win_col < NUM_DISPARITY+WINDOW_SIZE-1; win_col++)
			{
				#pragma HLS UNROLL
				left_window_buf.val[win_row][win_col] = 0;
				right_window_buf.val[win_row][win_col] = 0;
			}
		}		
        for(col = 0; col < img_width+half_win+NUM_DISPARITY-1; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS+(WINDOW_SIZE>>1)+NUM_DISPARITY-1 max=COLS+(WINDOW_SIZE>>1)+NUM_DISPARITY-1
			if (NUM_DISPARITY == PARALLEL_DISPARITIES)
			{
				#pragma HLS PIPELINE II=1 //If equal, pipeline the outer loop. 
			}			
			for(ap_uint<BIT_WIDTH(ITERATION)> iter = 0; iter < ITERATION; iter++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_flatten
				#pragma HLS DEPENDENCE variable=left_line_buf.val array inter false
				#pragma HLS DEPENDENCE variable=right_line_buf.val array inter false				
				if (iter==0)
				{
					left_window_buf.shift_pixels_left();
					right_window_buf.shift_pixels_right();
                    if(col<img_width){
                        for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> line_row = 0; line_row < WINDOW_SIZE-1; line_row++)
                        {
                            #pragma HLS UNROLL
                            left_window_buf.val[line_row][NUM_DISPARITY+WINDOW_SIZE-2] = left_line_buf.val[line_row][col];
							right_window_buf.val[line_row][0] = right_line_buf.val[line_row][col];
                        }						
                        ap_uint<BW_INPUT> tmp_l = 0; 
                        ap_uint<BW_INPUT> tmp_r = 0;
                        if(row < img_height-half_win){
                            tmp_l = src_l.read();
                            tmp_r = src_r.read();
                        }
                        left_window_buf.val[WINDOW_SIZE-1][NUM_DISPARITY+WINDOW_SIZE-2] = tmp_l;
						right_window_buf.val[WINDOW_SIZE-1][0] = tmp_r;
                        left_line_buf.shift_pixels_up(col);
                        right_line_buf.shift_pixels_up(col);
                        left_line_buf.val[WINDOW_SIZE-2][col] = tmp_l;
                        right_line_buf.val[WINDOW_SIZE-2][col] = tmp_r;
                    }
                    else{
                        for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
                        {
                            #pragma HLS UNROLL
                            left_window_buf.val[win_row][NUM_DISPARITY+WINDOW_SIZE-2] = 0;
							right_window_buf.val[win_row][0] = 0;
                        }
                    } 
				}
				for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
				{
					#pragma HLS UNROLL
					for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> num = 0; num < PARALLEL_DISPARITIES; num++)
					{
						#pragma HLS UNROLL
						tmp_left_window_buf.val[win_row][num] = left_window_buf.val[win_row][iter*PARALLEL_DISPARITIES+num];
						tmp_right_window_buf.val[win_row][num] = right_window_buf.val[win_row][iter*PARALLEL_DISPARITIES+num];
					}
					for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> num = 0; num < WINDOW_SIZE-1; num++)
					{
						#pragma HLS UNROLL
						tmp_left_window_buf.val[win_row][PARALLEL_DISPARITIES+num] = left_window_buf.val[win_row][(iter+1)*PARALLEL_DISPARITIES+num];
						tmp_right_window_buf.val[win_row][PARALLEL_DISPARITIES+num] = right_window_buf.val[win_row][(iter+1)*PARALLEL_DISPARITIES+num];
					}
				}

				DATA_TYPE(SAD_COST(BW_INPUT,WINDOW_SIZE)) left_SAD_value_buf[PARALLEL_DISPARITIES+WINDOW_SIZE-1];
				#pragma HLS ARRAY_PARTITION variable=left_SAD_value_buf complete dim=1
				DATA_TYPE(SAD_COST(BW_INPUT,WINDOW_SIZE)) right_SAD_value_buf[PARALLEL_DISPARITIES+WINDOW_SIZE-1];
				#pragma HLS ARRAY_PARTITION variable=right_SAD_value_buf complete dim=1
				for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES+WINDOW_SIZE-1)> num = 0; num < PARALLEL_DISPARITIES+WINDOW_SIZE-1; num++)
				{
					#pragma HLS UNROLL
					left_SAD_value_buf[num] = 0;
					right_SAD_value_buf[num] = 0;
				}

				for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
				{
					#pragma HLS UNROLL
					for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_col = 0; win_col < WINDOW_SIZE; win_col++)
					{
						#pragma HLS UNROLL
						ap_uint<BW_INPUT> left_pixel = left_window_buf.getval(win_row,win_col+NUM_DISPARITY-1);
						for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES+WINDOW_SIZE-1)> num = 0; num < PARALLEL_DISPARITIES+WINDOW_SIZE-1; num++)
						{
							#pragma HLS UNROLL
							ap_uint<BW_INPUT> right_pixel = tmp_right_window_buf.val[win_row][num];
							if(num>=WINDOW_SIZE-1-win_col){	
								ap_uint<BW_INPUT> delta = fpABSdiff< ap_uint<BW_INPUT> >(left_pixel,right_pixel);														
								left_SAD_value_buf[num-(WINDOW_SIZE-1-win_col)]+=delta;
							}
						}
					}
				}
				hls::Window<WINDOW_SIZE, WINDOW_SIZE, ap_uint<BW_INPUT> > right_window;
				for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
				{
					#pragma HLS UNROLL
					for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_col = 0; win_col < WINDOW_SIZE; win_col++)
					{
						#pragma HLS UNROLL
						right_window.val[win_row][win_col] = right_window_buf.getval(win_row,NUM_DISPARITY+WINDOW_SIZE-2-win_col);
					}
				}				
				for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
				{
					#pragma HLS UNROLL
					for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_col = 0; win_col < WINDOW_SIZE; win_col++)
					{
						#pragma HLS UNROLL
						ap_uint<BW_INPUT> right_pixel = right_window.getval(win_row,win_col);
						for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES+WINDOW_SIZE-1)> num = 0; num < PARALLEL_DISPARITIES+WINDOW_SIZE-1; num++)
						{
							#pragma HLS UNROLL
							ap_uint<BW_INPUT> left_pixel = tmp_left_window_buf.val[win_row][num];
							if(num>=win_col){
								ap_uint<BW_INPUT> delta = fpABSdiff< ap_uint<BW_INPUT> >(left_pixel,right_pixel);															
								right_SAD_value_buf[num-win_col]+=delta;
							}
						}
					}
				}
				if ((col<img_width+half_win)&&(col>=half_win))
				{
					for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> num = 0; num < PARALLEL_DISPARITIES; num++)
					{
						#pragma HLS UNROLL
						left_cost[num].write(left_SAD_value_buf[num]);
					}
				}
				if (col>=(half_win+NUM_DISPARITY-1)) 
				{
					for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> num = 0; num < PARALLEL_DISPARITIES; num++)
					{
						#pragma HLS UNROLL
						right_cost[num].write(right_SAD_value_buf[num]);
					}					
				}
			}
		}
	}	
}


/*------------------------------------------------ZSAD: Zero-Mean Sum of Absolute Differences-----------------------------------------------*/
template<int BW_INPUT, int WINDOW_SIZE>
ap_ufixed<BW_INPUT+3,BW_INPUT> fpComputeMean(hls::Window<WINDOW_SIZE, WINDOW_SIZE, ap_uint<BW_INPUT> > window)
{
    #pragma HLS ARRAY_PARTITION variable=window.val complete dim=0    
	ap_ufixed<BW_INPUT+3,BW_INPUT> mean_value = 0;
    ap_uint<BW_INPUT+BIT_WIDTH(WINDOW_SIZE*WINDOW_SIZE-1)> sum = 0;
	for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
	{
		#pragma HLS UNROLL
		for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_col = 0; win_col < WINDOW_SIZE; win_col++)
		{
			#pragma HLS UNROLL
            sum += window.val[win_row][win_col];
		}
	}
	mean_value = ap_ufixed<BW_INPUT+BIT_WIDTH(WINDOW_SIZE*WINDOW_SIZE-1)+3,BW_INPUT+BIT_WIDTH(WINDOW_SIZE*WINDOW_SIZE-1)>(sum)/(WINDOW_SIZE*WINDOW_SIZE);

	return mean_value;
}

// Matching cost computation: ZSAD 
template<int BW_INPUT, int ROWS, int COLS, int WINDOW_SIZE, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpComputeZSADCost(hls::stream< ap_uint<BW_INPUT> > &src_l, hls::stream< ap_uint<BW_INPUT> > &src_r, 
		hls::stream< DATA_TYPE(ZSAD_COST(BW_INPUT,WINDOW_SIZE)) > cost[PARALLEL_DISPARITIES], 
        ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS DATAFLOW	
	#pragma HLS ARRAY_PARTITION variable=cost complete dim=1

	const int ITERATION = NUM_DISPARITY/PARALLEL_DISPARITIES;
	
	hls::Window<WINDOW_SIZE, WINDOW_SIZE, ap_uint<BW_INPUT> > left_window_buf;
	hls::Window<WINDOW_SIZE, NUM_DISPARITY+WINDOW_SIZE-1, ap_uint<BW_INPUT> > right_window_buf; 

    hls::Window<WINDOW_SIZE, PARALLEL_DISPARITIES+WINDOW_SIZE-1, ap_uint<BW_INPUT> > tmp_right_window_buf;

	ap_ufixed<BW_INPUT+3,BW_INPUT> right_mean_buf[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=right_mean_buf complete dim=1

	ap_ufixed<BW_INPUT+3,BW_INPUT> left_mean = 0;

	hls::LineBuffer<WINDOW_SIZE-1, COLS, ap_uint<BW_INPUT> > left_line_buf;
    #pragma HLS RESOURCE variable=left_line_buf.val core=RAM_S2P_BRAM 
    hls::LineBuffer<WINDOW_SIZE-1, COLS, ap_uint<BW_INPUT> > right_line_buf;
	#pragma HLS RESOURCE variable=right_line_buf.val core=RAM_S2P_BRAM

	ap_uint<BIT_WIDTH(WINDOW_SIZE)> half_win = WINDOW_SIZE >> 1;
	ap_uint<BIT_WIDTH(WINDOW_SIZE)> initial_row;
	ap_uint<BIT_WIDTH(COLS+(WINDOW_SIZE>>1))> col;
	ap_uint<BIT_WIDTH(ROWS)> row;

	//Initialize the line buffer
	for( col = 0; col < img_width; col++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
		#pragma HLS pipeline
		for(initial_row = 0; initial_row < half_win; initial_row++)
		{
			#pragma HLS UNROLL
			left_line_buf.val[initial_row][col] = 0;
            right_line_buf.val[initial_row][col] = 0;
		}
		left_line_buf.val[half_win][col]=src_l.read();
        right_line_buf.val[half_win][col]=src_r.read();
	}
    ap_uint<BIT_WIDTH(WINDOW_SIZE)> next_row = half_win + 1;
	for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> i = 0; i < half_win-1; i++)
	{
		#pragma HLS UNROLL		
		for( col = 0; col < img_width; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			#pragma HLS pipeline
            left_line_buf.val[next_row][col]=src_l.read();
            right_line_buf.val[next_row][col]=src_r.read();
		}
		next_row++;
	}
	//Initialize the window buffer
    for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
	{
		#pragma HLS UNROLL
		for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_col = 0; win_col < WINDOW_SIZE; win_col++)
		{
			#pragma HLS UNROLL
			left_window_buf.val[win_row][win_col] = 0;
		}
	}
	//Process the image
	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
		{
			#pragma HLS UNROLL
			for(ap_uint<BIT_WIDTH(NUM_DISPARITY+WINDOW_SIZE-1)> win_col = 0; win_col < NUM_DISPARITY+WINDOW_SIZE-1; win_col++)
			{
				#pragma HLS UNROLL
				right_window_buf.val[win_row][win_col] = 0;
			}
		}
		for(int i=0; i<NUM_DISPARITY;i++)
		{
			#pragma HLS UNROLL
			right_mean_buf[i] = 0;
		}				
        for(col = 0; col < img_width+half_win; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS+(WINDOW_SIZE>>1) max=COLS+(WINDOW_SIZE>>1)
			if (NUM_DISPARITY == PARALLEL_DISPARITIES)
			{
				#pragma HLS PIPELINE II=1 //If equal, pipeline the outer loop. 
			}			
			for(ap_uint<BIT_WIDTH(ITERATION)> iter = 0; iter < ITERATION; iter++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_flatten
				#pragma HLS DEPENDENCE variable=left_line_buf.val array inter false
				#pragma HLS DEPENDENCE variable=right_line_buf.val array inter false				
				if (iter==0)
				{
                    left_window_buf.shift_pixels_left();
					right_window_buf.shift_pixels_right(); 
            		for(int i=NUM_DISPARITY-1; i>0; i--)
            		{
            			#pragma HLS UNROLL
            			right_mean_buf[i] = right_mean_buf[i-1];
            		}					                                        
                    if(col<img_width){
                        for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> line_row = 0; line_row < WINDOW_SIZE-1; line_row++)
                        {
                            #pragma HLS UNROLL
                            left_window_buf.val[line_row][WINDOW_SIZE-1] = left_line_buf.val[line_row][col];
							right_window_buf.val[line_row][0] = right_line_buf.val[line_row][col];
                        }
                        ap_uint<BW_INPUT> tmp_l = 0;
                        ap_uint<BW_INPUT> tmp_r = 0;
                        if(row < img_height-half_win){
                            tmp_l = src_l.read();
                            tmp_r = src_r.read();
                        }
                        left_window_buf.val[WINDOW_SIZE-1][WINDOW_SIZE-1] = tmp_l;
						right_window_buf.val[WINDOW_SIZE-1][0] = tmp_r;
                        left_line_buf.shift_pixels_up(col);
                        right_line_buf.shift_pixels_up(col);
                        left_line_buf.val[WINDOW_SIZE-2][col] = tmp_l;
                        right_line_buf.val[WINDOW_SIZE-2][col] = tmp_r;
                    }
                    else{
                        for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
                        {
                            #pragma HLS UNROLL
                            left_window_buf.val[win_row][WINDOW_SIZE-1] = 0;
							right_window_buf.val[win_row][0] = 0;
                        }
                    }
    				hls::Window<WINDOW_SIZE, WINDOW_SIZE, ap_uint<BW_INPUT> > right_window_mean;
    				for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
    				{
    					#pragma HLS UNROLL
    					for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_col = 0; win_col < WINDOW_SIZE; win_col++)
    					{
    						#pragma HLS UNROLL
    						right_window_mean.val[win_row][win_col] = right_window_buf.val[win_row][win_col];
    					}
    				}
    				right_mean_buf[0] = fpComputeMean<BW_INPUT,WINDOW_SIZE>(right_window_mean);
    				left_mean = fpComputeMean<BW_INPUT,WINDOW_SIZE>(left_window_buf);					
				}

				for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
				{
					#pragma HLS UNROLL
					for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> num = 0; num < PARALLEL_DISPARITIES; num++)
					{
						#pragma HLS UNROLL
						tmp_right_window_buf.val[win_row][num] = right_window_buf.val[win_row][iter*PARALLEL_DISPARITIES+num];
					}
					for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> num = 0; num < WINDOW_SIZE-1; num++)
					{
						#pragma HLS UNROLL
						tmp_right_window_buf.val[win_row][PARALLEL_DISPARITIES+num] = right_window_buf.val[win_row][(iter+1)*PARALLEL_DISPARITIES+num];
					}
				}
				DATA_TYPE(ZSAD_COST(BW_INPUT,WINDOW_SIZE)) ZSAD_value_buf[PARALLEL_DISPARITIES+WINDOW_SIZE-1];
				#pragma HLS ARRAY_PARTITION variable=ZSAD_value_buf complete dim=1
				for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES+WINDOW_SIZE-1)> num = 0; num < PARALLEL_DISPARITIES+WINDOW_SIZE-1; num++)
				{
					#pragma HLS UNROLL
					ZSAD_value_buf[num] = 0;
				}

				for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
				{
					#pragma HLS UNROLL
					for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_col = 0; win_col < WINDOW_SIZE; win_col++)
					{
						#pragma HLS UNROLL
						ap_uint<BW_INPUT> left_pixel = left_window_buf.getval(win_row,win_col);
						for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES+WINDOW_SIZE-1)> num = 0; num < PARALLEL_DISPARITIES+WINDOW_SIZE-1; num++)
						{
							#pragma HLS UNROLL
							ap_uint<BW_INPUT> right_pixel = tmp_right_window_buf.val[win_row][num];
							ap_ufixed<BW_INPUT+4,BW_INPUT+1> tmp2 = left_mean + right_pixel;
							if(num>=WINDOW_SIZE-1-win_col){
								int index = iter*PARALLEL_DISPARITIES + num-(WINDOW_SIZE-1-win_col);
								if(index>=NUM_DISPARITY){
									index = NUM_DISPARITY-1; 
								}
								ap_ufixed<BW_INPUT+4,BW_INPUT+1> tmp1 = right_mean_buf[index] + left_pixel;
								ap_ufixed<BW_INPUT+4,BW_INPUT+1> difference = fpABSdiff< ap_ufixed<BW_INPUT+4,BW_INPUT+1> >(tmp1,tmp2);
								ap_uint<3> fraction = difference.range(2,0);
								ap_uint<BW_INPUT+1> integer = difference.range(BW_INPUT+3,3);
								if(fraction>=4){
									integer += 1;
								}																
								ZSAD_value_buf[num-(WINDOW_SIZE-1-win_col)]+=integer;
							}
						}
					}
				}								
				if (col >= half_win)
				{
					for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> num = 0; num < PARALLEL_DISPARITIES; num++)
					{
						#pragma HLS UNROLL
						cost[num].write(ZSAD_value_buf[num]);
					}
				}
			}
		}
	}	
}

// Matching cost computation: ZSAD for L-R consistency check (LR2 method)
template<int BW_INPUT, int ROWS, int COLS, int WINDOW_SIZE, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpLRComputeZSADCost(hls::stream< ap_uint<BW_INPUT> > &src_l, hls::stream< ap_uint<BW_INPUT> > &src_r, 
		hls::stream< DATA_TYPE(ZSAD_COST(BW_INPUT,WINDOW_SIZE)) > left_cost[PARALLEL_DISPARITIES], 
        hls::stream< DATA_TYPE(ZSAD_COST(BW_INPUT,WINDOW_SIZE)) > right_cost[PARALLEL_DISPARITIES],
        ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS DATAFLOW	
	#pragma HLS ARRAY_PARTITION variable=left_cost complete dim=1
    #pragma HLS ARRAY_PARTITION variable=right_cost complete dim=1

	const int ITERATION = NUM_DISPARITY/PARALLEL_DISPARITIES;

	hls::Window<WINDOW_SIZE, NUM_DISPARITY+WINDOW_SIZE-1, ap_uint<BW_INPUT> > left_window_buf;
	hls::Window<WINDOW_SIZE, NUM_DISPARITY+WINDOW_SIZE-1, ap_uint<BW_INPUT> > right_window_buf; 

    hls::Window<WINDOW_SIZE, PARALLEL_DISPARITIES+WINDOW_SIZE-1, ap_uint<BW_INPUT> > tmp_left_window_buf;
    hls::Window<WINDOW_SIZE, PARALLEL_DISPARITIES+WINDOW_SIZE-1, ap_uint<BW_INPUT> > tmp_right_window_buf;

	ap_ufixed<BW_INPUT+3,BW_INPUT> left_mean_buf[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=left_mean_buf complete dim=1
	ap_ufixed<BW_INPUT+3,BW_INPUT> right_mean_buf[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=right_mean_buf complete dim=1	
     
	hls::LineBuffer<WINDOW_SIZE-1, COLS, ap_uint<BW_INPUT> > left_line_buf;
    #pragma HLS RESOURCE variable=left_line_buf.val core=RAM_S2P_BRAM 
    hls::LineBuffer<WINDOW_SIZE-1, COLS, ap_uint<BW_INPUT> > right_line_buf;
	#pragma HLS RESOURCE variable=right_line_buf.val core=RAM_S2P_BRAM

	ap_uint<BIT_WIDTH(WINDOW_SIZE)> half_win = WINDOW_SIZE >> 1;
	ap_uint<BIT_WIDTH(WINDOW_SIZE)> initial_row;
	ap_uint<BIT_WIDTH(COLS+(WINDOW_SIZE>>1)+NUM_DISPARITY-1)> col;
	ap_uint<BIT_WIDTH(ROWS)> row;

	//Initialize the line buffer
	for( col = 0; col < img_width; col++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
		#pragma HLS pipeline
		for(initial_row = 0; initial_row < half_win; initial_row++)
		{
			#pragma HLS UNROLL
			left_line_buf.val[initial_row][col] = 0;
            right_line_buf.val[initial_row][col] = 0;
		}
		left_line_buf.val[half_win][col]=src_l.read();
        right_line_buf.val[half_win][col]=src_r.read();
	}
    ap_uint<BIT_WIDTH(WINDOW_SIZE)> next_row = half_win + 1;
	for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> i = 0; i < half_win-1; i++)
	{
		#pragma HLS UNROLL
		for( col = 0; col < img_width; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			#pragma HLS pipeline
            left_line_buf.val[next_row][col]=src_l.read();
            right_line_buf.val[next_row][col]=src_r.read();
		}
		next_row++;
	}

	//Process the image
	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
		{
			#pragma HLS UNROLL
			for(ap_uint<BIT_WIDTH(NUM_DISPARITY+WINDOW_SIZE-1)> win_col = 0; win_col < NUM_DISPARITY+WINDOW_SIZE-1; win_col++)
			{
				#pragma HLS UNROLL
				left_window_buf.val[win_row][win_col] = 0;
				right_window_buf.val[win_row][win_col] = 0;
			}
		}
		for(int i=0; i<NUM_DISPARITY;i++)
		{
			#pragma HLS UNROLL
			left_mean_buf[i] = 0;
			right_mean_buf[i] = 0;
		}				
        for(col = 0; col < img_width+half_win+NUM_DISPARITY-1; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS+(WINDOW_SIZE>>1)+NUM_DISPARITY-1 max=COLS+(WINDOW_SIZE>>1)+NUM_DISPARITY-1
			if (NUM_DISPARITY == PARALLEL_DISPARITIES)
			{
				#pragma HLS PIPELINE II=1 //If equal, pipeline the outer loop. 
			}			
			for(ap_uint<BIT_WIDTH(ITERATION)> iter = 0; iter < ITERATION; iter++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_flatten
				#pragma HLS DEPENDENCE variable=left_line_buf.val array inter false
				#pragma HLS DEPENDENCE variable=right_line_buf.val array inter false
				if (iter==0)
				{
					left_window_buf.shift_pixels_left();
					right_window_buf.shift_pixels_right(); 
            		for(int i=NUM_DISPARITY-1; i>0; i--)
            		{
            			#pragma HLS UNROLL
						right_mean_buf[i] = right_mean_buf[i-1];
            		}
            		for(int i=0; i<NUM_DISPARITY-1; i++)
            		{
            			#pragma HLS UNROLL
						left_mean_buf[i] = left_mean_buf[i+1];
            		}	
                    if(col<img_width){
                        for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> line_row = 0; line_row < WINDOW_SIZE-1; line_row++)
                        {
                            #pragma HLS UNROLL
                            left_window_buf.val[line_row][NUM_DISPARITY+WINDOW_SIZE-2] = left_line_buf.val[line_row][col];
							right_window_buf.val[line_row][0] = right_line_buf.val[line_row][col];
                        }
                        ap_uint<BW_INPUT> tmp_l = 0; 
                        ap_uint<BW_INPUT> tmp_r = 0;
                        if(row < img_height-half_win){
                            tmp_l = src_l.read();
                            tmp_r = src_r.read();
                        }
                        left_window_buf.val[WINDOW_SIZE-1][NUM_DISPARITY+WINDOW_SIZE-2] = tmp_l;
						right_window_buf.val[WINDOW_SIZE-1][0] = tmp_r;
                        left_line_buf.shift_pixels_up(col);
                        right_line_buf.shift_pixels_up(col);
                        left_line_buf.val[WINDOW_SIZE-2][col] = tmp_l;
                        right_line_buf.val[WINDOW_SIZE-2][col] = tmp_r;
                    }
                    else{
                        for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
                        {
                            #pragma HLS UNROLL
                            left_window_buf.val[win_row][NUM_DISPARITY+WINDOW_SIZE-2] = 0;
							right_window_buf.val[win_row][0] = 0;
                        }
                    }
    				hls::Window<WINDOW_SIZE, WINDOW_SIZE, ap_uint<BW_INPUT> > right_window_mean;
					hls::Window<WINDOW_SIZE, WINDOW_SIZE, ap_uint<BW_INPUT> > left_window_mean;
    				for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
    				{
    					#pragma HLS UNROLL
    					for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_col = 0; win_col < WINDOW_SIZE; win_col++)
    					{
    						#pragma HLS UNROLL
							left_window_mean.val[win_row][win_col] = left_window_buf.val[win_row][win_col+NUM_DISPARITY-1];
    						right_window_mean.val[win_row][win_col] = right_window_buf.val[win_row][win_col];
    					}
    				}
    				right_mean_buf[0] = fpComputeMean<BW_INPUT,WINDOW_SIZE>(right_window_mean);
					left_mean_buf[NUM_DISPARITY-1] = fpComputeMean<BW_INPUT,WINDOW_SIZE>(left_window_mean);			 
				}
				for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
				{
					#pragma HLS UNROLL
					for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> num = 0; num < PARALLEL_DISPARITIES; num++)
					{
						#pragma HLS UNROLL
						tmp_left_window_buf.val[win_row][num] = left_window_buf.val[win_row][iter*PARALLEL_DISPARITIES+num];
						tmp_right_window_buf.val[win_row][num] = right_window_buf.val[win_row][iter*PARALLEL_DISPARITIES+num];
					}
					for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> num = 0; num < WINDOW_SIZE-1; num++)
					{
						#pragma HLS UNROLL
						tmp_left_window_buf.val[win_row][PARALLEL_DISPARITIES+num] = left_window_buf.val[win_row][(iter+1)*PARALLEL_DISPARITIES+num];
						tmp_right_window_buf.val[win_row][PARALLEL_DISPARITIES+num] = right_window_buf.val[win_row][(iter+1)*PARALLEL_DISPARITIES+num];
					}
				}

				DATA_TYPE(ZSAD_COST(BW_INPUT,WINDOW_SIZE)) left_ZSAD_value_buf[PARALLEL_DISPARITIES+WINDOW_SIZE-1];
				#pragma HLS ARRAY_PARTITION variable=left_ZSAD_value_buf complete dim=1
				DATA_TYPE(ZSAD_COST(BW_INPUT,WINDOW_SIZE)) right_ZSAD_value_buf[PARALLEL_DISPARITIES+WINDOW_SIZE-1];
				#pragma HLS ARRAY_PARTITION variable=right_ZSAD_value_buf complete dim=1
				for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES+WINDOW_SIZE-1)> num = 0; num < PARALLEL_DISPARITIES+WINDOW_SIZE-1; num++)
				{
					#pragma HLS UNROLL
					left_ZSAD_value_buf[num] = 0;
					right_ZSAD_value_buf[num] = 0;
				}

				for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
				{
					#pragma HLS UNROLL
					for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_col = 0; win_col < WINDOW_SIZE; win_col++)
					{
						#pragma HLS UNROLL
						ap_uint<BW_INPUT> left_pixel = left_window_buf.getval(win_row,win_col+NUM_DISPARITY-1);
						for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES+WINDOW_SIZE-1)> num = 0; num < PARALLEL_DISPARITIES+WINDOW_SIZE-1; num++)
						{
							#pragma HLS UNROLL
							ap_uint<BW_INPUT> right_pixel = tmp_right_window_buf.val[win_row][num];
							ap_ufixed<BW_INPUT+4,BW_INPUT+1> tmp2 = left_mean_buf[NUM_DISPARITY-1] + right_pixel;
							if(num>=WINDOW_SIZE-1-win_col){
								int index = iter*PARALLEL_DISPARITIES + num-(WINDOW_SIZE-1-win_col);
								if(index>=NUM_DISPARITY){
									index = NUM_DISPARITY-1; 
								}
								ap_ufixed<BW_INPUT+4,BW_INPUT+1> tmp1 = right_mean_buf[index] + left_pixel;
								ap_ufixed<BW_INPUT+4,BW_INPUT+1> difference = fpABSdiff< ap_ufixed<BW_INPUT+4,BW_INPUT+1> >(tmp1,tmp2);
								ap_uint<3> fraction = difference.range(2,0);
								ap_uint<BW_INPUT+1> integer = difference.range(BW_INPUT+3,3);
								if(fraction>=4){
									integer += 1;
								}																
								left_ZSAD_value_buf[num-(WINDOW_SIZE-1-win_col)]+=integer;
							}
						}
					}
				}
				hls::Window<WINDOW_SIZE, WINDOW_SIZE, ap_uint<BW_INPUT> > right_window;
				for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
				{
					#pragma HLS UNROLL
					for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_col = 0; win_col < WINDOW_SIZE; win_col++)
					{
						#pragma HLS UNROLL
						right_window.val[win_row][win_col] = right_window_buf.getval(win_row,NUM_DISPARITY+WINDOW_SIZE-2-win_col);
					}
				}
				for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
				{
					#pragma HLS UNROLL
					for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_col = 0; win_col < WINDOW_SIZE; win_col++)
					{
						#pragma HLS UNROLL
						ap_uint<BW_INPUT> right_pixel = right_window.getval(win_row,win_col);
						for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES+WINDOW_SIZE-1)> num = 0; num < PARALLEL_DISPARITIES+WINDOW_SIZE-1; num++)
						{
							#pragma HLS UNROLL
							ap_uint<BW_INPUT> left_pixel = tmp_left_window_buf.val[win_row][num];
							ap_ufixed<BW_INPUT+4,BW_INPUT+1> tmp2 = right_mean_buf[NUM_DISPARITY-1] + left_pixel;
							if(num>=win_col){
								int index = iter*PARALLEL_DISPARITIES + num - win_col;
								if(index>=NUM_DISPARITY){
									index = NUM_DISPARITY-1; 
								}
								ap_ufixed<BW_INPUT+4,BW_INPUT+1> tmp1 = left_mean_buf[index] + right_pixel;
								ap_ufixed<BW_INPUT+4,BW_INPUT+1> difference = fpABSdiff< ap_ufixed<BW_INPUT+4,BW_INPUT+1> >(tmp1,tmp2);
								ap_uint<3> fraction = difference.range(2,0);
								ap_uint<BW_INPUT+1> integer = difference.range(BW_INPUT+3,3);
								if(fraction>=4){
									integer += 1;
								}																
								right_ZSAD_value_buf[num-win_col]+=integer;
							}
						}
					}
				}

				if ((col<img_width+half_win)&&(col>=half_win))
				{
					for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> num = 0; num < PARALLEL_DISPARITIES; num++)
					{
						#pragma HLS UNROLL
						left_cost[num].write(left_ZSAD_value_buf[num]);
					}
				}
				if (col>=(half_win+NUM_DISPARITY-1)) 
				{
					for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> num = 0; num < PARALLEL_DISPARITIES; num++)
					{
						#pragma HLS UNROLL
						right_cost[num].write(right_ZSAD_value_buf[num]);
					}					
				}
			}
		}
	}	
}


/*------------------------------------------------------Rank Transform-----------------------------------------------------------*/
// Compute rank value for each window
template<int BW_INPUT, int WINDOW_SIZE, int RANK_VALUE>
DATA_TYPE(RANK_VALUE) fpComputeRank(hls::Window<WINDOW_SIZE, WINDOW_SIZE, ap_uint<BW_INPUT> > window)
{
    #pragma HLS ARRAY_PARTITION variable=window.val complete dim=0
	DATA_TYPE(RANK_VALUE) rank_value = 0;
	ap_uint<BIT_WIDTH(WINDOW_SIZE)> half_win = WINDOW_SIZE >> 1;
	ap_uint<BW_INPUT> pixel = window.getval(half_win,half_win);
	for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
	{
		#pragma HLS UNROLL
		for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_col = 0; win_col < WINDOW_SIZE; win_col++)
		{
			#pragma HLS UNROLL
			ap_uint<BW_INPUT> ref = window.getval(win_row,win_col);
			if ((win_row!=half_win)||(win_col!=half_win)) {
                if(ref<pixel){
                    rank_value++;
                }
			}
		}
	}
	return rank_value;
}
// Matching cost computation: Rank transform 
template<int BW_INPUT, int ROWS, int COLS, int WINDOW_SIZE, int RANK_VALUE>
void fpRankTransformKernel(hls::stream< ap_uint<BW_INPUT> > &src, hls::stream< DATA_TYPE(RANK_VALUE) > &dst, 
        ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE off
	assert(((img_height <= ROWS ) && (img_width <= COLS)) && "ROWS and COLS should be greater than input image");
	
	hls::Window<WINDOW_SIZE, WINDOW_SIZE, ap_uint<BW_INPUT> > window_buf; 
	hls::LineBuffer<WINDOW_SIZE-1, COLS, ap_uint<BW_INPUT> > line_buf; 
	#pragma HLS RESOURCE variable=line_buf.val core=RAM_S2P_BRAM

	ap_uint<BIT_WIDTH(WINDOW_SIZE)> half_win = WINDOW_SIZE >> 1;
	ap_uint<BIT_WIDTH(WINDOW_SIZE)> initial_row;
	ap_uint<BIT_WIDTH(COLS)> col;
	ap_uint<BIT_WIDTH(ROWS)> row;
	
	//Initialize the line buffer
	for( col = 0; col < img_width; col++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
		#pragma HLS pipeline
		for(initial_row = 0; initial_row < half_win; initial_row++)
		{
			#pragma HLS UNROLL
			line_buf.val[initial_row][col] = 0;
		}
		line_buf.val[half_win][col]=src.read();
	}
    ap_uint<BIT_WIDTH(WINDOW_SIZE)> next_row = half_win + 1;
	for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> i = 0; i < half_win-1; i++)
	{
		for( col = 0; col < img_width; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			#pragma HLS pipeline
			line_buf.val[next_row][col]=src.read();
		}
		next_row++;
	}

	//Initialize the window buffer
	for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
	{
		#pragma HLS UNROLL
		for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_col = 0; win_col < WINDOW_SIZE-1; win_col++)
		{
			#pragma HLS UNROLL
			window_buf.val[win_row][win_col] = 0;		
		}
	}
	//Process the image
	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for(col = 0; col < img_width; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			#pragma HLS pipeline
			for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> line_row = 0; line_row < WINDOW_SIZE-1; line_row++)
			{
				#pragma HLS UNROLL
				window_buf.val[line_row][WINDOW_SIZE-1] = line_buf.val[line_row][col];
			}
			ap_uint<BW_INPUT> tmp = 0;
			if(row < img_height-half_win){
				tmp = src.read();
			}
			window_buf.val[WINDOW_SIZE-1][WINDOW_SIZE-1] = tmp;
			line_buf.shift_pixels_up(col);
			line_buf.val[WINDOW_SIZE-2][col] = tmp;

            DATA_TYPE(RANK_VALUE) rank_value = fpComputeRank<BW_INPUT,WINDOW_SIZE,RANK_VALUE>(window_buf);
			window_buf.shift_pixels_left();
			if (col >= half_win) 
			{
				dst.write(rank_value);
			}
		}
		
		for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> col_rborder = 0; col_rborder < half_win; col_rborder++)
		{
			#pragma HLS pipeline
			for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
			{
				#pragma HLS UNROLL
				window_buf.val[win_row][WINDOW_SIZE-1] = 0;
			}
            DATA_TYPE(RANK_VALUE) rank_value = fpComputeRank<BW_INPUT,WINDOW_SIZE,RANK_VALUE>(window_buf);
			window_buf.shift_pixels_left();
			dst.write(rank_value);
		}
	}	
}

//compute absolute difference
template<int ROWS, int COLS, int RANK_VALUE, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpRankComputationKernel(hls::stream< DATA_TYPE(RANK_VALUE) > &src_l, hls::stream< DATA_TYPE(RANK_VALUE) > &src_r, 
		hls::stream< DATA_TYPE(RANK_VALUE) > cost[PARALLEL_DISPARITIES], ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS ARRAY_PARTITION variable=cost complete dim=1
	
	DATA_TYPE(RANK_VALUE) left_rank;
	DATA_TYPE(RANK_VALUE) right_rank;
	DATA_TYPE(RANK_VALUE) rank_buffer[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=rank_buffer complete dim=1
	
	ap_uint<BIT_WIDTH(COLS)> col;
	ap_uint<BIT_WIDTH(ROWS)> row;

	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for(int i = 0; i < NUM_DISPARITY; i++)
		{
			#pragma HLS UNROLL
			rank_buffer[i] = 0;
		}
		for(col = 0; col < img_width; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			if (NUM_DISPARITY == PARALLEL_DISPARITIES)
			{
				#pragma HLS PIPELINE II=1 //If equal, pipeline the outer loop. 
			}			
			for(ap_uint<BIT_WIDTH(NUM_DISPARITY/PARALLEL_DISPARITIES)> iter = 0; iter < NUM_DISPARITY/PARALLEL_DISPARITIES; iter++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_flatten
				if (iter==0)
				{
					left_rank = src_l.read();
					right_rank = src_r.read();
					for(int i = NUM_DISPARITY-1; i > 0; i--)
					{
						#pragma HLS UNROLL
						rank_buffer[i] = rank_buffer[i-1];
					}
					rank_buffer[0] = right_rank;
				}
				for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> num = 0; num < PARALLEL_DISPARITIES; num++)
				{
					#pragma HLS UNROLL
					DATA_TYPE(RANK_VALUE) sub_result = fpABSdiff< DATA_TYPE(RANK_VALUE) >(left_rank,rank_buffer[iter*PARALLEL_DISPARITIES+num]);
					cost[num].write(sub_result);
				}
			}
		}
	}
}
// compute absolute difference for LR consistency check
template<int ROWS, int COLS, int RANK_VALUE, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpLRRankComputationKernel(hls::stream< DATA_TYPE(RANK_VALUE) > &src_l, hls::stream< DATA_TYPE(RANK_VALUE) > &src_r, hls::stream< DATA_TYPE(RANK_VALUE) > left_cost[PARALLEL_DISPARITIES],
        hls::stream< DATA_TYPE(RANK_VALUE) > right_cost[PARALLEL_DISPARITIES], ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS ARRAY_PARTITION variable=left_cost complete dim=1
    #pragma HLS ARRAY_PARTITION variable=right_cost complete dim=1
	
	DATA_TYPE(RANK_VALUE) left_rank;
	DATA_TYPE(RANK_VALUE) right_rank;
	DATA_TYPE(RANK_VALUE) left_rank_buffer[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=left_rank_buffer complete dim=1
	DATA_TYPE(RANK_VALUE) right_rank_buffer[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=right_rank_buffer complete dim=1    

	ap_uint<BIT_WIDTH(COLS+NUM_DISPARITY-1)> col;
	ap_uint<BIT_WIDTH(ROWS)> row;

	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for(int i = 0; i < NUM_DISPARITY; i++)
		{
			#pragma HLS UNROLL
			left_rank_buffer[i] = 0;
            right_rank_buffer[i] = 0;
		}
		for(col = 0; col < img_width + NUM_DISPARITY-1; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS+NUM_DISPARITY-1 max=COLS+NUM_DISPARITY-1
			if (NUM_DISPARITY == PARALLEL_DISPARITIES)
			{
				#pragma HLS PIPELINE II=1 //If equal, pipeline the outer loop. 
			}			
			for(ap_uint<BIT_WIDTH(NUM_DISPARITY/PARALLEL_DISPARITIES)> iter = 0; iter < NUM_DISPARITY/PARALLEL_DISPARITIES; iter++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_flatten
				if (iter==0)
				{
					if(col<img_width){
                        left_rank = src_l.read();
					    right_rank = src_r.read();
					}
					else{
						left_rank = 0;
						right_rank = 0;
					}
					for(int i = NUM_DISPARITY-1; i > 0; i--)
					{
						#pragma HLS UNROLL
                        right_rank_buffer[i] = right_rank_buffer[i-1];
					}
					for(int i = 0; i < NUM_DISPARITY-1; i++)
					{
						#pragma HLS UNROLL
						left_rank_buffer[i] = left_rank_buffer[i+1];
					}
					right_rank_buffer[0] = right_rank;
					left_rank_buffer[NUM_DISPARITY-1] = left_rank;				
                }
				for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> num = 0; num < PARALLEL_DISPARITIES; num++)
				{
					#pragma HLS UNROLL
					DATA_TYPE(RANK_VALUE) left_sub_result = fpABSdiff< DATA_TYPE(RANK_VALUE) >(left_rank,right_rank_buffer[iter*PARALLEL_DISPARITIES+num]);
                    DATA_TYPE(RANK_VALUE) right_sub_result = fpABSdiff< DATA_TYPE(RANK_VALUE) >(right_rank_buffer[NUM_DISPARITY-1],left_rank_buffer[iter*PARALLEL_DISPARITIES+num]);
					if(col<img_width){
						left_cost[num].write(left_sub_result);
					}
					if(col>=(NUM_DISPARITY-1)){
						right_cost[num].write(right_sub_result);	
					}
				}
			}
		}
	}
}

// Matching cost computation: rank transform
template<int BW_INPUT, int ROWS, int COLS, int WINDOW_SIZE, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpComputeRankCost(hls::stream< ap_uint<BW_INPUT> > &src_l, hls::stream< ap_uint<BW_INPUT> > &src_r, 
		hls::stream< DATA_TYPE(RANK_COST(WINDOW_SIZE)) > cost[PARALLEL_DISPARITIES], 
		ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS DATAFLOW
	#pragma HLS ARRAY_PARTITION variable=cost complete dim=1

	hls::stream< DATA_TYPE(RANK_COST(WINDOW_SIZE)) > dst_l;
	hls::stream< DATA_TYPE(RANK_COST(WINDOW_SIZE)) > dst_r;
	fpRankTransformKernel<BW_INPUT,ROWS,COLS,WINDOW_SIZE,RANK_COST(WINDOW_SIZE)>(src_l,dst_l,img_height,img_width);
	fpRankTransformKernel<BW_INPUT,ROWS,COLS,WINDOW_SIZE,RANK_COST(WINDOW_SIZE)>(src_r,dst_r,img_height,img_width);
	fpRankComputationKernel<ROWS,COLS,RANK_COST(WINDOW_SIZE),NUM_DISPARITY,PARALLEL_DISPARITIES>(dst_l,dst_r,cost,img_height,img_width);
}

// Matching cost computation: rank transform for L-R consistency check
template<int BW_INPUT, int ROWS, int COLS, int WINDOW_SIZE, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpLRComputeRankCost(hls::stream< ap_uint<BW_INPUT> > &src_l, hls::stream< ap_uint<BW_INPUT> > &src_r, 
		hls::stream< DATA_TYPE(RANK_COST(WINDOW_SIZE)) > left_cost[PARALLEL_DISPARITIES], 
		hls::stream< DATA_TYPE(RANK_COST(WINDOW_SIZE)) > right_cost[PARALLEL_DISPARITIES],
		ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS DATAFLOW
	#pragma HLS ARRAY_PARTITION variable=left_cost complete dim=1
	#pragma HLS ARRAY_PARTITION variable=right_cost complete dim=1

	hls::stream< DATA_TYPE(RANK_COST(WINDOW_SIZE)) > dst_l;
	hls::stream< DATA_TYPE(RANK_COST(WINDOW_SIZE)) > dst_r;
	fpRankTransformKernel<BW_INPUT,ROWS,COLS,WINDOW_SIZE,RANK_COST(WINDOW_SIZE)>(src_l,dst_l,img_height,img_width);
	fpRankTransformKernel<BW_INPUT,ROWS,COLS,WINDOW_SIZE,RANK_COST(WINDOW_SIZE)>(src_r,dst_r,img_height,img_width);
	fpLRRankComputationKernel<ROWS,COLS,RANK_COST(WINDOW_SIZE),NUM_DISPARITY,PARALLEL_DISPARITIES>(dst_l,dst_r,left_cost,right_cost,img_height,img_width);
}

/*---------------------------------------------------Census Transform-----------------------------------------------------------*/
// Compute census value for each window
template<int BW_INPUT, int WINDOW_SIZE, int CENSUS_VALUE>
ap_uint<CENSUS_VALUE> fpComputeCensus(hls::Window<WINDOW_SIZE, WINDOW_SIZE, ap_uint<BW_INPUT> > window)
{
    #pragma HLS ARRAY_PARTITION variable=window.val complete dim=0
	ap_uint<CENSUS_VALUE> census_value;
	ap_uint<BIT_WIDTH(WINDOW_SIZE)> half_win = WINDOW_SIZE >> 1;
	ap_uint<BW_INPUT> pixel = window.getval(half_win,half_win);
	ap_uint<BIT_WIDTH(CENSUS_VALUE)> index = 0;
	for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
	{
		#pragma HLS UNROLL
		for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_col = 0; win_col < WINDOW_SIZE; win_col++)
		{
			#pragma HLS UNROLL
			ap_uint<BW_INPUT> ref = window.getval(win_row,win_col);
			if ((win_row!=half_win)||(win_col!=half_win)) {
				census_value.range(CENSUS_VALUE-1-index,CENSUS_VALUE-1-index) = (ref<pixel) ? 1 : 0;
				index++;
			}
		}
	}
	return census_value;
}

// Census transform
template<int BW_INPUT, int ROWS, int COLS, int WINDOW_SIZE, int CENSUS_VALUE>
void fpCensusTransformKernel(hls::stream< ap_uint<BW_INPUT> > &src, hls::stream< ap_uint<CENSUS_VALUE> > &dst, 
        ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE off
	//assert(((_window_size == XF_FILTER_3X3) || (_window_size == XF_FILTER_5X5)) && ("Filter width must be either 3 or 5"));
	assert(((img_height <= ROWS ) && (img_width <= COLS)) && "ROWS and COLS should be greater than input image");
	//assert((WINDOW_SIZE <= 1) && ("The window size is too small"));
	//assert(((WINDOW_SIZE >= 16) || (WINDOW_SIZE >= 16)) && ("The window size is too large that it is not efficient"));
	
	hls::Window<WINDOW_SIZE, WINDOW_SIZE, ap_uint<BW_INPUT> > window_buf; //store the window to compute the census.
	hls::LineBuffer<WINDOW_SIZE-1, COLS, ap_uint<BW_INPUT> > line_buf; //store lines of pixels.
	#pragma HLS RESOURCE variable=line_buf.val core=RAM_S2P_BRAM

	ap_uint<BIT_WIDTH(WINDOW_SIZE)> half_win = WINDOW_SIZE >> 1;
	ap_uint<BIT_WIDTH(WINDOW_SIZE)> initial_row;
	ap_uint<BIT_WIDTH(COLS)> col;
	ap_uint<BIT_WIDTH(ROWS)> row;
	
	//Initialize the line buffer
	for( col = 0; col < img_width; col++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
		#pragma HLS pipeline
		for(initial_row = 0; initial_row < half_win; initial_row++)
		{
			#pragma HLS UNROLL
			line_buf.val[initial_row][col] = 0;
		}
		line_buf.val[half_win][col]=src.read();
	}
    ap_uint<BIT_WIDTH(WINDOW_SIZE)> next_row = half_win + 1;
	for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> i = 0; i < half_win-1; i++)
	{
		for( col = 0; col < img_width; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			#pragma HLS pipeline
			line_buf.val[next_row][col]=src.read();
		}
		next_row++;
	}

	//Initialize the window buffer
	for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
	{
		#pragma HLS UNROLL
		for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_col = 0; win_col < WINDOW_SIZE-1; win_col++)
		{
			#pragma HLS UNROLL
			window_buf.val[win_row][win_col] = 0;		
		}
	}
	//Process the image
	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for(col = 0; col < img_width; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			#pragma HLS pipeline
			for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> line_row = 0; line_row < WINDOW_SIZE-1; line_row++)
			{
				#pragma HLS UNROLL
				window_buf.val[line_row][WINDOW_SIZE-1] = line_buf.val[line_row][col];
			}
			ap_uint<BW_INPUT> tmp = 0;
			if(row < img_height-half_win){
				tmp = src.read();
			}
			window_buf.val[WINDOW_SIZE-1][WINDOW_SIZE-1] = tmp;
			line_buf.shift_pixels_up(col);
			line_buf.val[WINDOW_SIZE-2][col] = tmp;

			ap_uint<CENSUS_VALUE> census_value = fpComputeCensus<BW_INPUT,WINDOW_SIZE,CENSUS_VALUE>(window_buf);
			window_buf.shift_pixels_left();
			if (col >= half_win) 
			{
				dst.write(census_value);
			}
		}
		
		for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> col_rborder = 0; col_rborder < half_win; col_rborder++)
		{
			#pragma HLS pipeline
			for(ap_uint<BIT_WIDTH(WINDOW_SIZE)> win_row = 0; win_row < WINDOW_SIZE; win_row++)
			{
				#pragma HLS UNROLL
				window_buf.val[win_row][WINDOW_SIZE-1] = 0;
			}
			ap_uint<CENSUS_VALUE> census_value = fpComputeCensus<BW_INPUT,WINDOW_SIZE,CENSUS_VALUE>(window_buf);
			window_buf.shift_pixels_left();
			dst.write(census_value);
		}
	}	
} 

// Compute hamming distance between census values
template<int ROWS, int COLS, int WINDOW_SIZE, int CENSUS_VALUE, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpHammingDistance(hls::stream< ap_uint<CENSUS_VALUE> > &src_l_census_fifo, hls::stream< ap_uint<CENSUS_VALUE> > &src_r_census_fifo, 
		hls::stream< DATA_TYPE(CENSUS_VALUE) > cost[PARALLEL_DISPARITIES], ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS ARRAY_PARTITION variable=cost complete dim=1
	
	ap_uint<CENSUS_VALUE> left_census;
	ap_uint<CENSUS_VALUE> right_census;
	ap_uint<CENSUS_VALUE> census_buffer[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=census_buffer complete dim=1
	
	//ap_uint<13> col, row;
	ap_uint<BIT_WIDTH(COLS)> col;
	ap_uint<BIT_WIDTH(ROWS)> row;

	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for(int i = 0; i < NUM_DISPARITY; i++)
		{
			#pragma HLS UNROLL
			census_buffer[i] = 0;
		}

		for(col = 0; col < img_width; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			if (NUM_DISPARITY == PARALLEL_DISPARITIES)
			{
				#pragma HLS PIPELINE II=1 //If equal, pipeline the outer loop. 
			}			
			for(ap_uint<BIT_WIDTH(NUM_DISPARITY/PARALLEL_DISPARITIES)> iter = 0; iter < NUM_DISPARITY/PARALLEL_DISPARITIES; iter++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_flatten

				if (iter==0)
				{
					left_census = src_l_census_fifo.read();
					right_census = src_r_census_fifo.read();
					for(int i = NUM_DISPARITY-1; i > 0; i--)
					{
						#pragma HLS UNROLL
						census_buffer[i] = census_buffer[i-1];
					}
					census_buffer[0] = right_census;
				}
				
				for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> num = 0; num < PARALLEL_DISPARITIES; num++)
				{
					#pragma HLS UNROLL
					ap_uint<CENSUS_VALUE> xor_result;
					xor_result = left_census ^ census_buffer[iter*PARALLEL_DISPARITIES+num];
					DATA_TYPE(CENSUS_VALUE) sum = 0;
					for(DATA_TYPE(CENSUS_VALUE) j = 0; j < CENSUS_VALUE; j++)
					{
						#pragma HLS UNROLL
						sum += xor_result.range(j,j);
					}
					cost[num].write(sum);
				}
			}
		}
	}
}

// Compute hamming distance for L-R consistency check (LR2)
template<int ROWS, int COLS, int WINDOW_SIZE, int CENSUS_VALUE, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpLRHammingDistance(hls::stream< ap_uint<CENSUS_VALUE> > &src_l_census_fifo, hls::stream< ap_uint<CENSUS_VALUE> > &src_r_census_fifo, 
        hls::stream< DATA_TYPE(CENSUS_VALUE) > left_cost[PARALLEL_DISPARITIES], hls::stream< DATA_TYPE(CENSUS_VALUE) > right_cost[PARALLEL_DISPARITIES], 
        ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS ARRAY_PARTITION variable=left_cost complete dim=1
	#pragma HLS ARRAY_PARTITION variable=right_cost complete dim=1
	
	ap_uint<CENSUS_VALUE> left_census;
	ap_uint<CENSUS_VALUE> right_census;
	ap_uint<CENSUS_VALUE> left_census_buffer[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=left_census_buffer complete dim=1
	ap_uint<CENSUS_VALUE> right_census_buffer[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=right_census_buffer complete dim=1
	
	ap_uint<BIT_WIDTH(COLS+NUM_DISPARITY-1)> col;
	ap_uint<BIT_WIDTH(ROWS)> row;

	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
		for(int i = 0; i < NUM_DISPARITY; i++)
		{
			#pragma HLS UNROLL
			right_census_buffer[i] = 0;
			left_census_buffer[i] = 0;
		}

		for(col = 0; col < img_width + NUM_DISPARITY-1; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS+NUM_DISPARITY-1 max=COLS+NUM_DISPARITY-1
			if (NUM_DISPARITY == PARALLEL_DISPARITIES)
			{
				#pragma HLS PIPELINE II=1 //If equal, pipeline the outer loop. 
			}			
			for(ap_uint<BIT_WIDTH(NUM_DISPARITY/PARALLEL_DISPARITIES)> iter = 0; iter < NUM_DISPARITY/PARALLEL_DISPARITIES; iter++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_flatten

				if (iter==0)
				{
					if(col<img_width){
						left_census = src_l_census_fifo.read();
						right_census = src_r_census_fifo.read();
					}
					else{
						left_census = 0;
						right_census = 0;
					}
					for(int i = NUM_DISPARITY-1; i > 0; i--)
					{
						#pragma HLS UNROLL
						right_census_buffer[i] = right_census_buffer[i-1];
					}
					for(int i = 0; i < NUM_DISPARITY-1; i++)
					{
						#pragma HLS UNROLL
						left_census_buffer[i] = left_census_buffer[i+1];
					}
					right_census_buffer[0] = right_census;
					left_census_buffer[NUM_DISPARITY-1] = left_census;
				}

				for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> num = 0; num < PARALLEL_DISPARITIES; num++)
				{
					#pragma HLS UNROLL
					ap_uint<CENSUS_VALUE> left_xor_result = left_census ^ right_census_buffer[iter*PARALLEL_DISPARITIES+num];
					DATA_TYPE(CENSUS_VALUE) left_sum = 0;
					ap_uint<CENSUS_VALUE> right_xor_result = right_census_buffer[NUM_DISPARITY-1] ^ left_census_buffer[iter*PARALLEL_DISPARITIES+num];
					DATA_TYPE(CENSUS_VALUE) right_sum = 0;					
					for(DATA_TYPE(CENSUS_VALUE) j = 0; j < CENSUS_VALUE; j++)
					{
						#pragma HLS UNROLL
						left_sum += left_xor_result.range(j,j);
						right_sum += right_xor_result.range(j,j);
					}
					if(col<img_width){
						left_cost[num].write(left_sum);
					}
					if(col>=(NUM_DISPARITY-1)){
						right_cost[num].write(right_sum);	
					}				
				}
			}
		}
	}
}

// Matching cost computation: Census transform 
template<int BW_INPUT, int ROWS, int COLS, int WINDOW_SIZE, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpComputeCensusCost(hls::stream< ap_uint<BW_INPUT> > &src_l, hls::stream< ap_uint<BW_INPUT> > &src_r, 
		hls::stream< DATA_TYPE(CENSUS_COST(WINDOW_SIZE)) > cost[PARALLEL_DISPARITIES], 
		ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS DATAFLOW
	#pragma HLS ARRAY_PARTITION variable=cost complete dim=1

	hls::stream< ap_uint<CENSUS_COST(WINDOW_SIZE)> > src_l_census_fifo;
	hls::stream< ap_uint<CENSUS_COST(WINDOW_SIZE)> > src_r_census_fifo;

	fpCensusTransformKernel<BW_INPUT,ROWS,COLS,WINDOW_SIZE,CENSUS_COST(WINDOW_SIZE)>(src_l,src_l_census_fifo,img_height,img_width);
	fpCensusTransformKernel<BW_INPUT,ROWS,COLS,WINDOW_SIZE,CENSUS_COST(WINDOW_SIZE)>(src_r,src_r_census_fifo,img_height,img_width);
	fpHammingDistance<ROWS,COLS,WINDOW_SIZE,CENSUS_COST(WINDOW_SIZE),NUM_DISPARITY,PARALLEL_DISPARITIES>(src_l_census_fifo,src_r_census_fifo,cost,img_height,img_width);
}

// Matching cost computation: Census transform for L-R consistency check (LR2 method)
template<int BW_INPUT, int ROWS, int COLS, int WINDOW_SIZE, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpLRComputeCensusCost(hls::stream< ap_uint<BW_INPUT> > &src_l, hls::stream< ap_uint<BW_INPUT> > &src_r, 
		hls::stream< DATA_TYPE(CENSUS_COST(WINDOW_SIZE)) > left_cost[PARALLEL_DISPARITIES], 
		hls::stream< DATA_TYPE(CENSUS_COST(WINDOW_SIZE)) > right_cost[PARALLEL_DISPARITIES],
		ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS DATAFLOW
	#pragma HLS ARRAY_PARTITION variable=left_cost complete dim=1
	#pragma HLS ARRAY_PARTITION variable=right_cost complete dim=1

	hls::stream< ap_uint<CENSUS_COST(WINDOW_SIZE)> > src_l_census_fifo;
	hls::stream< ap_uint<CENSUS_COST(WINDOW_SIZE)> > src_r_census_fifo;

	fpCensusTransformKernel<BW_INPUT,ROWS,COLS,WINDOW_SIZE,CENSUS_COST(WINDOW_SIZE)>(src_l,src_l_census_fifo,img_height,img_width);
	fpCensusTransformKernel<BW_INPUT,ROWS,COLS,WINDOW_SIZE,CENSUS_COST(WINDOW_SIZE)>(src_r,src_r_census_fifo,img_height,img_width);
	fpLRHammingDistance<ROWS,COLS,WINDOW_SIZE,CENSUS_COST(WINDOW_SIZE),NUM_DISPARITY,PARALLEL_DISPARITIES>(src_l_census_fifo,src_r_census_fifo,left_cost,right_cost,img_height,img_width);
}

/*------------------------------------------------SHD: Sum of Hamming distances-------------------------------------------------*/
template<int CENSUS_VALUE, int SHD_WINDOW>
DATA_TYPE(SHD_COST(CENSUS_VALUE,SHD_WINDOW)) fpComputeSHD(hls::Window<SHD_WINDOW, SHD_WINDOW, ap_uint<CENSUS_VALUE> > window_l,
        hls::Window<SHD_WINDOW, SHD_WINDOW, ap_uint<CENSUS_VALUE> > window_r)
{
    #pragma HLS ARRAY_PARTITION variable=window_l.val complete dim=0
    #pragma HLS ARRAY_PARTITION variable=window_r.val complete dim=0
	DATA_TYPE(SHD_COST(CENSUS_VALUE,SHD_WINDOW)) sum_HD = 0;
	for(ap_uint<BIT_WIDTH(SHD_WINDOW)> win_row = 0; win_row < SHD_WINDOW; win_row++)
	{
		#pragma HLS UNROLL
		for(ap_uint<BIT_WIDTH(SHD_WINDOW)> win_col = 0; win_col < SHD_WINDOW; win_col++)
		{
			#pragma HLS UNROLL
            ap_uint<CENSUS_VALUE> xor_result = window_l.val[win_row][win_col] ^ window_r.val[win_row][win_col];
            DATA_TYPE(CENSUS_VALUE) sum = 0;
            for(DATA_TYPE(CENSUS_VALUE) j = 0; j < CENSUS_VALUE; j++)
            {
                #pragma HLS UNROLL
                sum += xor_result.range(j,j);
            }
            sum_HD += sum;
		}
	}
	return sum_HD;
}

template<int ROWS, int COLS, int SHD_WINDOW, int CENSUS_VALUE, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpSumHammingDistance(hls::stream< ap_uint<CENSUS_VALUE> > &src_l, hls::stream< ap_uint<CENSUS_VALUE> > &src_r, 
		hls::stream< DATA_TYPE(SHD_COST(CENSUS_VALUE,SHD_WINDOW)) > cost[PARALLEL_DISPARITIES], 
        ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS ARRAY_PARTITION variable=cost complete dim=1

	hls::Window<SHD_WINDOW, SHD_WINDOW, ap_uint<CENSUS_VALUE> > left_window_buf; 
    hls::Window<SHD_WINDOW, SHD_WINDOW, ap_uint<CENSUS_VALUE> > right_window_buf[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=right_window_buf complete dim=1
     
	hls::LineBuffer<SHD_WINDOW-1, COLS, ap_uint<CENSUS_VALUE> > left_line_buf;
    #pragma HLS RESOURCE variable=left_line_buf.val core=RAM_S2P_BRAM 
    hls::LineBuffer<SHD_WINDOW-1, COLS, ap_uint<CENSUS_VALUE> > right_line_buf;
	#pragma HLS RESOURCE variable=right_line_buf.val core=RAM_S2P_BRAM

	ap_uint<BIT_WIDTH(SHD_WINDOW)> half_win = SHD_WINDOW >> 1;
	ap_uint<BIT_WIDTH(SHD_WINDOW)> initial_row;
	ap_uint<BIT_WIDTH(COLS)> col;
	ap_uint<BIT_WIDTH(ROWS)> row;
	const int ITERATION = NUM_DISPARITY/PARALLEL_DISPARITIES;

	//Initialize the line buffer
	for( col = 0; col < img_width; col++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
		#pragma HLS pipeline
		for(initial_row = 0; initial_row < half_win; initial_row++)
		{
			#pragma HLS UNROLL
			left_line_buf.val[initial_row][col] = 0;
            right_line_buf.val[initial_row][col] = 0;
		}
		left_line_buf.val[half_win][col]=src_l.read();
        right_line_buf.val[half_win][col]=src_r.read();
	}
    ap_uint<BIT_WIDTH(SHD_WINDOW)> next_row = half_win + 1;
	for(ap_uint<BIT_WIDTH(SHD_WINDOW)> i = 0; i < half_win-1; i++)
	{
		for( col = 0; col < img_width; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			#pragma HLS pipeline
            left_line_buf.val[next_row][col]=src_l.read();
            right_line_buf.val[next_row][col]=src_r.read();
		}
		next_row++;
	}
	//Initialize the window buffer
    for(ap_uint<BIT_WIDTH(SHD_WINDOW)> win_row = 0; win_row < SHD_WINDOW; win_row++)
	{
		#pragma HLS UNROLL
		for(ap_uint<BIT_WIDTH(SHD_WINDOW)> win_col = 0; win_col < SHD_WINDOW; win_col++)
		{
			#pragma HLS UNROLL
			left_window_buf.val[win_row][win_col] = 0;
		}
	}
	//Process the image
	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        for(int i=0; i<NUM_DISPARITY;i++)
        {
            #pragma HLS UNROLL
            for(ap_uint<BIT_WIDTH(SHD_WINDOW)> win_row = 0; win_row < SHD_WINDOW; win_row++)
            {
                #pragma HLS UNROLL
                for(ap_uint<BIT_WIDTH(SHD_WINDOW)> win_col = 0; win_col < SHD_WINDOW; win_col++)
                {
                    #pragma HLS UNROLL
                    right_window_buf[i].val[win_row][win_col] = 0;		
                }
            }        
        }		
        for(col = 0; col < img_width+half_win; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS+(SHD_WINDOW>>1) max=COLS+(SHD_WINDOW>>1)
			if (NUM_DISPARITY == PARALLEL_DISPARITIES)
			{
				#pragma HLS PIPELINE II=1 //If equal, pipeline the outer loop. 
			}			
			for(ap_uint<BIT_WIDTH(ITERATION)> iter = 0; iter < ITERATION; iter++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_flatten
				if (iter==0)
				{
                    left_window_buf.shift_pixels_left();
                    for(int i=NUM_DISPARITY-1; i > 0; i--)
                    {
                        #pragma HLS UNROLL
                        for(ap_uint<BIT_WIDTH(SHD_WINDOW)> win_row = 0; win_row < SHD_WINDOW; win_row++)
                        {
                            #pragma HLS UNROLL
                            for(ap_uint<BIT_WIDTH(SHD_WINDOW)> win_col = 0; win_col < SHD_WINDOW; win_col++)
                            {
                                #pragma HLS UNROLL
                                right_window_buf[i].val[win_row][win_col] = right_window_buf[i-1].val[win_row][win_col];		
                            }
                        }        
                    }
                    right_window_buf[0].shift_pixels_left();                                         
                    if(col<img_width){
                        for(ap_uint<BIT_WIDTH(SHD_WINDOW)> line_row = 0; line_row < SHD_WINDOW-1; line_row++)
                        {
                            #pragma HLS UNROLL
                            left_window_buf.val[line_row][SHD_WINDOW-1] = left_line_buf.val[line_row][col];
                            right_window_buf[0].val[line_row][SHD_WINDOW-1] = right_line_buf.val[line_row][col];
                        }
                        ap_uint<CENSUS_VALUE> tmp_l = 0;
                        ap_uint<CENSUS_VALUE> tmp_r = 0;
                        if(row < img_height-half_win){
                            tmp_l = src_l.read();
                            tmp_r = src_r.read();
                        }
                        left_window_buf.val[SHD_WINDOW-1][SHD_WINDOW-1] = tmp_l;
                        right_window_buf[0].val[SHD_WINDOW-1][SHD_WINDOW-1] = tmp_r;
                        left_line_buf.shift_pixels_up(col);
                        right_line_buf.shift_pixels_up(col);
                        left_line_buf.val[SHD_WINDOW-2][col] = tmp_l;
                        right_line_buf.val[SHD_WINDOW-2][col] = tmp_r;
                    }
                    else{
                        for(ap_uint<BIT_WIDTH(SHD_WINDOW)> win_row = 0; win_row < SHD_WINDOW; win_row++)
                        {
                            #pragma HLS UNROLL
                            left_window_buf.val[win_row][SHD_WINDOW-1] = 0;
                            right_window_buf[0].val[win_row][SHD_WINDOW-1] = 0;
                        }
                    }
				}				
				for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> num = 0; num < PARALLEL_DISPARITIES; num++)
				{
					#pragma HLS UNROLL
                    DATA_TYPE(SHD_COST(CENSUS_VALUE,SHD_WINDOW)) SHD_value = fpComputeSHD<CENSUS_VALUE,SHD_WINDOW>(left_window_buf,right_window_buf[iter*PARALLEL_DISPARITIES+num]);
                    if (col >= half_win) 
                    {
                        cost[num].write(SHD_value);
                    }
				}
			}
		}
	}	
}

template<int ROWS, int COLS, int SHD_WINDOW, int CENSUS_VALUE, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpLRSumHammingDistance(hls::stream< ap_uint<CENSUS_VALUE> > &src_l, hls::stream< ap_uint<CENSUS_VALUE> > &src_r, 
		hls::stream< DATA_TYPE(SHD_COST(CENSUS_VALUE,SHD_WINDOW)) > left_cost[PARALLEL_DISPARITIES], 
        hls::stream< DATA_TYPE(SHD_COST(CENSUS_VALUE,SHD_WINDOW)) > right_cost[PARALLEL_DISPARITIES],
        ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS ARRAY_PARTITION variable=left_cost complete dim=1
    #pragma HLS ARRAY_PARTITION variable=right_cost complete dim=1

	hls::Window<SHD_WINDOW, SHD_WINDOW, ap_uint<CENSUS_VALUE> > left_window_buf[NUM_DISPARITY];
    #pragma HLS ARRAY_PARTITION variable=left_window_buf complete dim=1  
    hls::Window<SHD_WINDOW, SHD_WINDOW, ap_uint<CENSUS_VALUE> > right_window_buf[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=right_window_buf complete dim=1
     
	hls::LineBuffer<SHD_WINDOW-1, COLS, ap_uint<CENSUS_VALUE> > left_line_buf;
    #pragma HLS RESOURCE variable=left_line_buf.val core=RAM_S2P_BRAM 
    hls::LineBuffer<SHD_WINDOW-1, COLS, ap_uint<CENSUS_VALUE> > right_line_buf;
	#pragma HLS RESOURCE variable=right_line_buf.val core=RAM_S2P_BRAM

	ap_uint<BIT_WIDTH(SHD_WINDOW)> half_win = SHD_WINDOW >> 1;
	ap_uint<BIT_WIDTH(SHD_WINDOW)> initial_row;
	ap_uint<BIT_WIDTH(COLS)> col;
	ap_uint<BIT_WIDTH(ROWS)> row;
	const int ITERATION = NUM_DISPARITY/PARALLEL_DISPARITIES;

	//Initialize the line buffer
	for( col = 0; col < img_width; col++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
		#pragma HLS pipeline
		for(initial_row = 0; initial_row < half_win; initial_row++)
		{
			#pragma HLS UNROLL
			left_line_buf.val[initial_row][col] = 0;
            right_line_buf.val[initial_row][col] = 0;
		}
		left_line_buf.val[half_win][col]=src_l.read();
        right_line_buf.val[half_win][col]=src_r.read();
	}
    ap_uint<BIT_WIDTH(SHD_WINDOW)> next_row = half_win + 1;
	for(ap_uint<BIT_WIDTH(SHD_WINDOW)> i = 0; i < half_win-1; i++)
	{
		for( col = 0; col < img_width; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS max=COLS
			#pragma HLS pipeline
            left_line_buf.val[next_row][col]=src_l.read();
            right_line_buf.val[next_row][col]=src_r.read();
		}
		next_row++;
	}
	//Process the image
	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=ROWS max=ROWS
        for(int i=0; i<NUM_DISPARITY;i++)
        {
            #pragma HLS UNROLL          
            for(ap_uint<BIT_WIDTH(SHD_WINDOW)> win_row = 0; win_row < SHD_WINDOW; win_row++)
            {
                #pragma HLS UNROLL
                for(ap_uint<BIT_WIDTH(SHD_WINDOW)> win_col = 0; win_col < SHD_WINDOW; win_col++)
                {
                    #pragma HLS UNROLL
                    right_window_buf[i].val[win_row][win_col] = 0;	
                    left_window_buf[i].val[win_row][win_col] = 0;		
                }
            }        
        }		
        for(col = 0; col < img_width+half_win+NUM_DISPARITY-1; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=COLS+(SHD_WINDOW>>1)+NUM_DISPARITY-1 max=COLS+(SHD_WINDOW>>1)+NUM_DISPARITY-1
			if (NUM_DISPARITY == PARALLEL_DISPARITIES)
			{
				#pragma HLS PIPELINE II=1 //If equal, pipeline the outer loop. 
			}			
			for(ap_uint<BIT_WIDTH(ITERATION)> iter = 0; iter < ITERATION; iter++)
			{
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_flatten
				if (iter==0)
				{
                    for(int i=0; i < NUM_DISPARITY-1; i++)
                    {
                        #pragma HLS UNROLL
                        for(ap_uint<BIT_WIDTH(SHD_WINDOW)> win_row = 0; win_row < SHD_WINDOW; win_row++)
                        {
                            #pragma HLS UNROLL
                            for(ap_uint<BIT_WIDTH(SHD_WINDOW)> win_col = 0; win_col < SHD_WINDOW; win_col++)
                            {
                                #pragma HLS UNROLL
                                left_window_buf[i].val[win_row][win_col] = left_window_buf[i+1].val[win_row][win_col];		
                            }
                        }        
                    }                    
                    left_window_buf[NUM_DISPARITY-1].shift_pixels_left();
                    for(int i=NUM_DISPARITY-1; i > 0; i--)
                    {
                        #pragma HLS UNROLL
                        for(ap_uint<BIT_WIDTH(SHD_WINDOW)> win_row = 0; win_row < SHD_WINDOW; win_row++)
                        {
                            #pragma HLS UNROLL
                            for(ap_uint<BIT_WIDTH(SHD_WINDOW)> win_col = 0; win_col < SHD_WINDOW; win_col++)
                            {
                                #pragma HLS UNROLL
                                right_window_buf[i].val[win_row][win_col] = right_window_buf[i-1].val[win_row][win_col];		
                            }
                        }        
                    }
                    right_window_buf[0].shift_pixels_left(); 
                    if(col<img_width){
                        for(ap_uint<BIT_WIDTH(SHD_WINDOW)> line_row = 0; line_row < SHD_WINDOW-1; line_row++)
                        {
                            #pragma HLS UNROLL
                            left_window_buf[NUM_DISPARITY-1].val[line_row][SHD_WINDOW-1] = left_line_buf.val[line_row][col];
                            right_window_buf[0].val[line_row][SHD_WINDOW-1] = right_line_buf.val[line_row][col];
                        }
                        ap_uint<CENSUS_VALUE> tmp_l = 0; 
                        ap_uint<CENSUS_VALUE> tmp_r = 0;
                        if(row < img_height-half_win){
                            tmp_l = src_l.read();
                            tmp_r = src_r.read();
                        }
                        left_window_buf[NUM_DISPARITY-1].val[SHD_WINDOW-1][SHD_WINDOW-1] = tmp_l;
                        right_window_buf[0].val[SHD_WINDOW-1][SHD_WINDOW-1] = tmp_r;
                        left_line_buf.shift_pixels_up(col);
                        right_line_buf.shift_pixels_up(col);
                        left_line_buf.val[SHD_WINDOW-2][col] = tmp_l;
                        right_line_buf.val[SHD_WINDOW-2][col] = tmp_r;
                    }
                    else{
                        for(ap_uint<BIT_WIDTH(SHD_WINDOW)> win_row = 0; win_row < SHD_WINDOW; win_row++)
                        {
                            #pragma HLS UNROLL
                            left_window_buf[NUM_DISPARITY-1].val[win_row][SHD_WINDOW-1] = 0;
                            right_window_buf[0].val[win_row][SHD_WINDOW-1] = 0;
                        }
                    } 
				}				
				for(ap_uint<BIT_WIDTH(PARALLEL_DISPARITIES)> num = 0; num < PARALLEL_DISPARITIES; num++)
				{
					#pragma HLS UNROLL
                    DATA_TYPE(SHD_COST(CENSUS_VALUE,SHD_WINDOW)) left_SHD_value = fpComputeSHD<CENSUS_VALUE,SHD_WINDOW>(left_window_buf[NUM_DISPARITY-1], right_window_buf[iter*PARALLEL_DISPARITIES+num]);
                    DATA_TYPE(SHD_COST(CENSUS_VALUE,SHD_WINDOW)) right_SHD_value = fpComputeSHD<CENSUS_VALUE,SHD_WINDOW>(right_window_buf[NUM_DISPARITY-1], left_window_buf[iter*PARALLEL_DISPARITIES+num]);
                    if((col<img_width+half_win)&&(col>=half_win)){
						left_cost[num].write(left_SHD_value);
					}
					if(col>=(half_win+NUM_DISPARITY-1)){
						right_cost[num].write(right_SHD_value);	
					}
				}
			}
		}
	}	
}


template<int BW_INPUT, int ROWS, int COLS, int WINDOW_SIZE, int SHD_WINDOW, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpComputeSHDCost(hls::stream< ap_uint<BW_INPUT> > &src_l, hls::stream< ap_uint<BW_INPUT> > &src_r, 
		hls::stream< DATA_TYPE(SHD_COST(CENSUS_COST(WINDOW_SIZE),SHD_WINDOW)) > cost[PARALLEL_DISPARITIES], 
		ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS DATAFLOW
	#pragma HLS ARRAY_PARTITION variable=cost complete dim=1

	hls::stream< ap_uint<CENSUS_COST(WINDOW_SIZE)> > src_l_census_fifo;
	hls::stream< ap_uint<CENSUS_COST(WINDOW_SIZE)> > src_r_census_fifo;

	fpCensusTransformKernel<BW_INPUT,ROWS,COLS,WINDOW_SIZE,CENSUS_COST(WINDOW_SIZE)>(src_l,src_l_census_fifo,img_height,img_width);
	fpCensusTransformKernel<BW_INPUT,ROWS,COLS,WINDOW_SIZE,CENSUS_COST(WINDOW_SIZE)>(src_r,src_r_census_fifo,img_height,img_width);
	fpSumHammingDistance<ROWS,COLS,SHD_WINDOW,CENSUS_COST(WINDOW_SIZE),NUM_DISPARITY,PARALLEL_DISPARITIES>(src_l_census_fifo,src_r_census_fifo,cost,img_height,img_width);
}

template<int BW_INPUT, int ROWS, int COLS, int WINDOW_SIZE, int SHD_WINDOW, int NUM_DISPARITY, int PARALLEL_DISPARITIES>
void fpLRComputeSHDCost(hls::stream< ap_uint<BW_INPUT> > &src_l, hls::stream< ap_uint<BW_INPUT> > &src_r, 
		hls::stream< DATA_TYPE(SHD_COST(CENSUS_COST(WINDOW_SIZE),SHD_WINDOW)) > left_cost[PARALLEL_DISPARITIES], 
		hls::stream< DATA_TYPE(SHD_COST(CENSUS_COST(WINDOW_SIZE),SHD_WINDOW)) > right_cost[PARALLEL_DISPARITIES],
		ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	#pragma HLS DATAFLOW
	#pragma HLS ARRAY_PARTITION variable=left_cost complete dim=1
	#pragma HLS ARRAY_PARTITION variable=right_cost complete dim=1

	hls::stream< ap_uint<CENSUS_COST(WINDOW_SIZE)> > src_l_census_fifo;
	hls::stream< ap_uint<CENSUS_COST(WINDOW_SIZE)> > src_r_census_fifo;

	fpCensusTransformKernel<BW_INPUT,ROWS,COLS,WINDOW_SIZE,CENSUS_COST(WINDOW_SIZE)>(src_l,src_l_census_fifo,img_height,img_width);
	fpCensusTransformKernel<BW_INPUT,ROWS,COLS,WINDOW_SIZE,CENSUS_COST(WINDOW_SIZE)>(src_r,src_r_census_fifo,img_height,img_width);
	fpLRSumHammingDistance<ROWS,COLS,SHD_WINDOW,CENSUS_COST(WINDOW_SIZE),NUM_DISPARITY,PARALLEL_DISPARITIES>(src_l_census_fifo,src_r_census_fifo,left_cost,right_cost,img_height,img_width);
}


}
#endif

