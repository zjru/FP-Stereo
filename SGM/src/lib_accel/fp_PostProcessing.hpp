/*
 *  FP-Stereo
 *  Copyright (C) 2020  RCSL, HKUST
 *  
 *  GPL-3.0 License
 *
 */

#ifndef _FP_POSTPROCESSING_HPP_
#define _FP_POSTPROCESSING_HPP_

#ifndef __cplusplus
#error C++ is needed to include this header
#endif

//#include "hls_stream.h"
#include "hls_video.h"
#include "common/xf_common.h"
#include "common/xf_utility.h"
#include "lib_accel/fp_common.h"


namespace fp{

template <typename T>
T fpABSdiff(T a, T b)
{
	#pragma HLS INLINE
	return (a>b)?(a-b):(b-a);
}

template<int ROWS, int COLS, int DST_TYPE, int NPC, int NUM_DISPARITY>
void fpLRCheckConsistency(hls::stream< XF_TNAME(DST_TYPE,NPC) > &left_fifo, hls::stream< XF_TNAME(DST_TYPE,NPC) > &right_fifo,
		hls::stream< XF_TNAME(DST_TYPE,NPC) > &dst_fifo, ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
	
	XF_TNAME(DST_TYPE,NPC) right_buffer[NUM_DISPARITY];
	#pragma HLS ARRAY_PARTITION variable=right_buffer complete dim=1
	
	ap_uint<BIT_WIDTH(ROWS)> row;
	ap_uint<BIT_WIDTH(COLS)> col;
	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
		for(int i = 0; i < NUM_DISPARITY; i++)
		{
			#pragma HLS UNROLL
			right_buffer[i] = 0;
		}
		for (col = 0; col < img_width; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
			#pragma HLS PIPELINE II=1
			#pragma HLS loop_flatten

			XF_TNAME(DST_TYPE,NPC) left_disp = left_fifo.read();
			XF_TNAME(DST_TYPE,NPC) right_disp = right_fifo.read();
			for(int i = NUM_DISPARITY-1; i>0; i--){
				#pragma HLS UNROLL
				right_buffer[i] = right_buffer[i-1];
			}
			right_buffer[0] = right_disp;
			XF_TNAME(DST_TYPE,NPC) abs_diff = fpABSdiff<XF_TNAME(DST_TYPE,NPC) >(left_disp,right_buffer[left_disp]);
			
            ap_uint<2> threshold = 1;  //set the threshold to discard invalid disparities.
            if(abs_diff<=threshold){
				dst_fifo.write(left_disp);
			}
			else{
				dst_fifo.write(0);
			}
		}
	}
}


template<int COLS, int DST_TYPE, int NPC, int GAP_THRESHOLD>
XF_TNAME(DST_TYPE,NPC) fpValidatePixel(XF_TNAME(DST_TYPE,NPC) right_disp[GAP_THRESHOLD], XF_TNAME(DST_TYPE,NPC) &left_valid_disp, ap_uint<1> &count)
{
    #pragma HLS INLINE
    #pragma HLS ARRAY_PARTITION variable=right_disp complete dim=1
    XF_TNAME(DST_TYPE,NPC) disp = right_disp[0];
    if(disp==0){
        XF_TNAME(DST_TYPE,NPC) right_valid_disp = 0;
        for (int i = GAP_THRESHOLD-1; i>=1; i--)
        {
            #pragma HLS LOOP_TRIPCOUNT min=GAP_THRESHOLD-1 max=GAP_THRESHOLD-1
            #pragma HLS UNROLL
            if(right_disp[i]!=0){
                right_valid_disp = right_disp[i];
            }
		}
        if(left_valid_disp==0){
            disp=right_valid_disp;
        }
        else if(right_valid_disp==0){
            disp=left_valid_disp;
        }
        else{
            disp=(right_valid_disp>left_valid_disp)?(left_valid_disp):(right_valid_disp);
        }
    }
    else{
        left_valid_disp = disp;
        count=1;
    }
    return disp;
}


template<int ROWS, int COLS, int DST_TYPE, int NPC, int NUM_DISPARITY, int GAP_THRESHOLD>
void fpInterpolation(hls::stream< XF_TNAME(DST_TYPE,NPC) > &src_fifo, hls::stream< XF_TNAME(DST_TYPE,NPC) > &dst_fifo,
        ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
	#pragma HLS INLINE OFF
    XF_TNAME(DST_TYPE,NPC) right_disp[GAP_THRESHOLD];
    #pragma HLS ARRAY_PARTITION variable=right_disp complete dim=1

	ap_uint<BIT_WIDTH(ROWS)> row;
	ap_uint<BIT_WIDTH(COLS+GAP_THRESHOLD-1)> col;
    
    for (ap_uint<BIT_WIDTH(GAP_THRESHOLD)> i = 0; i < GAP_THRESHOLD; i++)
    {
        #pragma HLS UNROLL
        right_disp[i] = 0;
    }

	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
		XF_TNAME(DST_TYPE,NPC) left_valid_disp = 0;
        ap_uint<BIT_WIDTH(COLS)> left_valid_index=0;
        for (col = 0; col < img_width + GAP_THRESHOLD-1; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=1 max=COLS+GAP_THRESHOLD-1
			#pragma HLS PIPELINE II=1
			#pragma HLS loop_flatten
            for (ap_uint<BIT_WIDTH(GAP_THRESHOLD)> i = 0; i < GAP_THRESHOLD-1; i++)
            {
                #pragma HLS UNROLL
                right_disp[i] = right_disp[i+1];
            } 			
            if(col<img_width){
				right_disp[GAP_THRESHOLD-1] = src_fifo.read();
			}
			else{
				right_disp[GAP_THRESHOLD-1] = 0;
			}
			if(col-left_valid_index>GAP_THRESHOLD-1){
				left_valid_disp = 0;
			}
			ap_uint<1> count=0;
            XF_TNAME(DST_TYPE,NPC) disp = fpValidatePixel<DST_TYPE,NPC,GAP_THRESHOLD>(right_disp,left_valid_disp,count);
            if(count==1){
				left_valid_index = col;
			}
			if(col>=GAP_THRESHOLD-1){
				dst_fifo.write(disp);
			}           
        }
    }
}

template<typename T, int FilterWin>
T fpSortMedian(hls::Window<FilterWin, FilterWin, T > window_buf)
{
    #pragma HLS INLINE
    #pragma HLS ARRAY_PARTITION variable=window_buf.val complete dim=0
    T MedianValue = 0;
    T MedianArray[FilterWin*FilterWin];
    #pragma HLS ARRAY_PARTITION variable=MedianArray complete dim=1

    int array_ptr=0;
	for(int win_row = 0; win_row < FilterWin; win_row++)
	{
		#pragma HLS UNROLL
		for(int win_col = 0; win_col < FilterWin; win_col++)
		{
			#pragma HLS UNROLL
			MedianArray[array_ptr] = window_buf.val[win_row][win_col];
            array_ptr++;		
		}
	}
    for(ap_uint<BIT_WIDTH(FilterWin*FilterWin)> i = 0; i < FilterWin*FilterWin; i++)
    {
        if(i.range(0,0)==0)
        {
            for(ap_uint<BIT_WIDTH((FilterWin*FilterWin)>>1)> j = 0; j < (FilterWin*FilterWin>>1); j++)
            {
                #pragma HLS UNROLL
                ap_uint<BIT_WIDTH(FilterWin*FilterWin)> ind = j*2;
                ap_uint<BIT_WIDTH(FilterWin*FilterWin)> ind_next = ind + 1;
                if(MedianArray[ind] < MedianArray[ind_next]){
                    T tmp = MedianArray[ind];
                    MedianArray[ind] = MedianArray[ind_next];
                    MedianArray[ind_next] = tmp;
                }
            }
        }
        else{
            for(ap_uint<BIT_WIDTH((FilterWin*FilterWin)>>1)> j = 0; j < (FilterWin*FilterWin>>1); j++)
            {
                #pragma HLS UNROLL
                ap_uint<BIT_WIDTH(FilterWin*FilterWin)> ind = j*2 + 1;
                ap_uint<BIT_WIDTH(FilterWin*FilterWin)> ind_next = ind + 1;
                if(MedianArray[ind] < MedianArray[ind_next]){
                    T tmp = MedianArray[ind];
                    MedianArray[ind] = MedianArray[ind_next];
                    MedianArray[ind_next] = tmp;
                }
            }
        }
    }
    MedianValue = MedianArray[(FilterWin*FilterWin)>>1];
    return MedianValue;
}

template<int ROWS, int COLS, int DST_TYPE, int NPC, int FilterWin>
void fpMedianFilter(hls::stream< XF_TNAME(DST_TYPE,NPC) > &src_fifo, hls::stream< XF_TNAME(DST_TYPE,NPC) > &dst_fifo,
        ap_uint<BIT_WIDTH(ROWS)> img_height, ap_uint<BIT_WIDTH(COLS)> img_width)
{
    #pragma HLS INLINE OFF
    #pragma HLS DATAFLOW 

    hls::Window<FilterWin, FilterWin, XF_TNAME(DST_TYPE,NPC)> window_buf;
    hls::LineBuffer<FilterWin-1, COLS, XF_TNAME(DST_TYPE,NPC)> line_buf;
	#pragma HLS RESOURCE variable=line_buf.val core=RAM_S2P_BRAM

	ap_uint<BIT_WIDTH(COLS+(FilterWin>>1))> col;
	ap_uint<BIT_WIDTH(ROWS)> row;

	//Initialize the line buffer
	for( col = 0; col < img_width; col++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
		#pragma HLS pipeline
		XF_TNAME(DST_TYPE,NPC) tmp_read = src_fifo.read();
        line_buf.val[FilterWin>>1][col] = tmp_read;
		for(int initial_row = 0; initial_row < FilterWin>>1; initial_row++)
		{
			#pragma HLS UNROLL
			line_buf.val[initial_row][col] = tmp_read; //top border
		}
	}
	for(int next_row = (FilterWin>>1) + 1; next_row < FilterWin-1; next_row++)
	{
		for( col = 0; col < img_width; col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=1 max=COLS
			#pragma HLS pipeline
			line_buf.val[next_row][col] = src_fifo.read();
		}
	}

	//Initialize the window buffer
	for(int win_row = 0; win_row < FilterWin; win_row++)
	{
		#pragma HLS UNROLL
		for(int win_col = 0; win_col < FilterWin-1; win_col++)
		{
			#pragma HLS UNROLL
			window_buf.val[win_row][win_col] = 0;		
		}
	}
    
	//Process the image
	for(row = 0; row < img_height; row++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=1 max=ROWS
		for(col = 0; col < img_width + (FilterWin>>1); col++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=1 max=COLS+(FilterWin>>1)
			#pragma HLS pipeline

            if(col<img_width)
            {
                for(int line_row = 0; line_row < FilterWin-1; line_row++)
                {
                    #pragma HLS UNROLL
                    window_buf.val[line_row][FilterWin-1] = line_buf.val[line_row][col];
                }
                XF_TNAME(DST_TYPE,NPC) tmp;                
                if(row < img_height-(FilterWin>>1)){ //bottom border
                    tmp = src_fifo.read();
                }
                else{
                    tmp = window_buf.val[FilterWin-2][FilterWin-1];
                }
                window_buf.val[FilterWin-1][FilterWin-1] = tmp;
                line_buf.shift_pixels_up(col);
                line_buf.val[FilterWin-2][col] = tmp;
            }
            else{ //right border
                for(int win_row = 0; win_row < FilterWin; win_row++)
                {
                    #pragma HLS UNROLL
                    window_buf.val[win_row][FilterWin-1] = window_buf.val[win_row][FilterWin-2];
                }
            }
            
            XF_TNAME(DST_TYPE,NPC) MedianValue = fpSortMedian<XF_TNAME(DST_TYPE,NPC),FilterWin>(window_buf);
            //Update the window buffer
            for(int win_row = 0; win_row < FilterWin; win_row++)
            {
                #pragma HLS UNROLL
                for(int win_col = 0; win_col < FilterWin-1; win_col++)
                {
                    #pragma HLS UNROLL
                    if(col==0){//left_border
                        window_buf.val[win_row][win_col] = window_buf.val[win_row][FilterWin-1];
                    }
                    else{
                        window_buf.val[win_row][win_col] = window_buf.val[win_row][win_col+1];	
                    }      	
                }
            }
			if (col >= (FilterWin>>1)) 
			{
				dst_fifo.write(MedianValue);
			}
		}
	}	

}


}
#endif

