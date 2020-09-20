/*
 *  FP-Stereo
 *  Copyright (C) 2020  RCSL, HKUST
 *  
 *  GPL-3.0 License
 *
 */
 
#include "fp_sgbm_c.h"
#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define BACKWARD_HAS_UNWIND 1
#define BACKWARD_HAS_DW 1
#include "backward.hpp"

namespace backward {         
  backward::SignalHandling sh; 
}

template <typename T>
T ABSdiff(T a, T b){
    return (a>b)?(a-b):(b-a);
}

/*-----------------------------------SAD: Sum of Absolute Differences-------------------------------*/
int compute_SAD(int *window1, int *window2, int window_size){
    int sad = 0;    
    for(int i=0; i<window_size; i++){
        for(int j=0; j<window_size; j++){
            sad += ABSdiff<int>(window1[i*window_size+j],window2[i*window_size+j]);
        }
    }
    return sad;
}

int compute_SAD_cost(cv::Mat img1, cv::Mat img2, int *cost, int window_size, int max_disp){
    int *window1 = (int*)malloc(window_size*window_size*sizeof(int));
    if (!window1) {
        printf("Memory allocation failed for window1..! \n");
        return -1;
    }
    int *window2 = (int*)malloc(window_size*window_size*sizeof(int));
    if (!window2) {
        printf("Memory allocation failed for window2..! \n");
        return -1;
    }    
    for(int i=0; i<img1.rows; i++){
        for(int j=0; j<img1.cols; j++){
            for (int d=0; d<max_disp; d++){
                int index=0;
                for(int ki=i-window_size/2; ki<=i+window_size/2; ki++){
                    for(int kj=j-window_size/2; kj<=j+window_size/2; kj++){
                        int left_ref;
                        int right_ref;
                        if(ki<0 || ki>img1.rows-1 || kj<0 || kj>img1.cols-1){
                            left_ref=0;
                        }
                        else{
                            left_ref=(int)img1.at<uchar>(ki,kj);
                        }
                        if(ki<0 || ki>img1.rows-1 || kj-d<0 || kj-d>img1.cols-1){
                            right_ref=0;
                        }
                        else{
                            right_ref=(int)img2.at<uchar>(ki,kj-d);
                        }
                        window1[index] = left_ref;
                        window2[index] = right_ref;
                        index++;
                    }
                }        
                cost[(i*img1.cols+j)*max_disp+d] = compute_SAD(window1,window2,window_size);
            }
        }
    }
    free(window1);
    free(window2); 
    return 0;
}

int compute_lr_SAD_cost(cv::Mat img1, cv::Mat img2, int *cost_l, int *cost_r, int window_size, int max_disp){
    int *window_l1 = (int*)malloc(window_size*window_size*sizeof(int));
    if (!window_l1) {
        printf("Memory allocation failed for window_l1..! \n");
        return -1;
    }
    int *window_l2 = (int*)malloc(window_size*window_size*sizeof(int));
    if (!window_l2) {
        printf("Memory allocation failed for window_l2..! \n");
        return -1;
    }
    int *window_r1 = (int*)malloc(window_size*window_size*sizeof(int));
    if (!window_r1) {
        printf("Memory allocation failed for window_r1..! \n");
        return -1;
    }
    int *window_r2 = (int*)malloc(window_size*window_size*sizeof(int));
    if (!window_r2) {
        printf("Memory allocation failed for window_r2..! \n");
        return -1;
    }             
    for(int i=0; i<img1.rows; i++){
        for(int j=0; j<img1.cols; j++){
            for (int d=0; d<max_disp; d++){
                int index=0;
                for(int ki=i-window_size/2; ki<=i+window_size/2; ki++){
                    for(int kj=j-window_size/2; kj<=j+window_size/2; kj++){
                        int ref_l1, ref_l2, ref_r1, ref_r2;
                        if(ki<0 || ki>img1.rows-1 || kj<0 || kj>img1.cols-1){
                            ref_l1=0;
                            ref_r1=0;
                        }
                        else{
                            ref_l1=(int)img1.at<uchar>(ki,kj);
                            ref_r1=(int)img2.at<uchar>(ki,kj);
                        }
                        if(ki<0 || ki>img1.rows-1 || kj-d<0 || kj-d>img1.cols-1){
                            ref_l2=0;
                        }
                        else{
                            ref_l2=(int)img2.at<uchar>(ki,kj-d);
                        }
                        if(ki<0 || ki>img1.rows-1 || kj+d<0 || kj+d>img1.cols-1){
                            ref_r2=0;
                        }
                        else{
                            ref_r2=(int)img1.at<uchar>(ki,kj+d);
                        }                        
                        window_l1[index] = ref_l1;
                        window_l2[index] = ref_l2;
                        window_r1[index] = ref_r1;
                        window_r2[index] = ref_r2;                        
                        index++;
                    }
                }        
                cost_l[(i*img1.cols+j)*max_disp+d] = compute_SAD(window_l1,window_l2,window_size);  
                cost_r[(i*img1.cols+j)*max_disp+d] = compute_SAD(window_r1,window_r2,window_size);              
            }
        }
    }
    free(window_l1);
    free(window_l2); 
    free(window_r1);
    free(window_r2);
    return 0;    
}


/*----------------------------ZSAD: Zero-Mean Sum of Absolute Differences---------------------------*/
float compute_diff_mean(int *window1, int *window2, int window_size){
    float diff_mean = 0.0;
    int sum1 = 0;
    int sum2 = 0;     
    for(int i=0; i<window_size; i++){
        for(int j=0; j<window_size; j++){
            sum1 += window1[i*window_size+j];
            sum2 += window2[i*window_size+j];
        }
    }
    diff_mean = float(sum1-sum2)/float(window_size*window_size);
    return diff_mean;
}

int compute_ZSAD(int *window1, int *window2, int window_size){
    int zsad = 0;
    float diff_mean = compute_diff_mean(window1,window2,window_size);
    for(int i=0; i<window_size; i++){
        for(int j=0; j<window_size; j++){
            float diff_val = window1[i*window_size+j] - window2[i*window_size+j];
            float diff = ABSdiff<float>(diff_val,diff_mean);
            zsad += (int)(round(diff));
        }
    }
    return zsad;    
}

int compute_ZSAD_cost(cv::Mat img1, cv::Mat img2, int *cost, int window_size, int max_disp){
    int *window1 = (int*)malloc(window_size*window_size*sizeof(int));
    if (!window1) {
        printf("Memory allocation failed for window1..! \n");
        return -1;
    }
    int *window2 = (int*)malloc(window_size*window_size*sizeof(int));
    if (!window2) {
        printf("Memory allocation failed for window2..! \n");
        return -1;
    }    
    for(int i=0; i<img1.rows; i++){
        for(int j=0; j<img1.cols; j++){
            for (int d=0; d<max_disp; d++){
                int index=0;
                for(int ki=i-window_size/2; ki<=i+window_size/2; ki++){
                    for(int kj=j-window_size/2; kj<=j+window_size/2; kj++){
                        int left_ref;
                        int right_ref;
                        if(ki<0 || ki>img1.rows-1 || kj<0 || kj>img1.cols-1){
                            left_ref=0;
                        }
                        else{
                            left_ref=(int)img1.at<uchar>(ki,kj);
                        }
                        if(ki<0 || ki>img1.rows-1 || kj-d<0 || kj-d>img1.cols-1){
                            right_ref=0;
                        }
                        else{
                            right_ref=(int)img2.at<uchar>(ki,kj-d);
                        }
                        window1[index] = left_ref;
                        window2[index] = right_ref;
                        index++;
                    }
                }        
                cost[(i*img1.cols+j)*max_disp+d] = compute_ZSAD(window1,window2,window_size);
            }
        }
    }
    free(window1);
    free(window2); 
    return 0;
}

int compute_lr_ZSAD_cost(cv::Mat img1, cv::Mat img2, int *cost_l, int *cost_r, int window_size, int max_disp){
    int *window_l1 = (int*)malloc(window_size*window_size*sizeof(int));
    if (!window_l1) {
        printf("Memory allocation failed for window_l1..! \n");
        return -1;
    }
    int *window_l2 = (int*)malloc(window_size*window_size*sizeof(int));
    if (!window_l2) {
        printf("Memory allocation failed for window_l2..! \n");
        return -1;
    }
    int *window_r1 = (int*)malloc(window_size*window_size*sizeof(int));
    if (!window_r1) {
        printf("Memory allocation failed for window_r1..! \n");
        return -1;
    }
    int *window_r2 = (int*)malloc(window_size*window_size*sizeof(int));
    if (!window_r2) {
        printf("Memory allocation failed for window_r2..! \n");
        return -1;
    }             
    for(int i=0; i<img1.rows; i++){
        for(int j=0; j<img1.cols; j++){
            for (int d=0; d<max_disp; d++){
                int index=0;
                for(int ki=i-window_size/2; ki<=i+window_size/2; ki++){
                    for(int kj=j-window_size/2; kj<=j+window_size/2; kj++){
                        int ref_l1, ref_l2, ref_r1, ref_r2;
                        if(ki<0 || ki>img1.rows-1 || kj<0 || kj>img1.cols-1){
                            ref_l1=0;
                            ref_r1=0;
                        }
                        else{
                            ref_l1=(int)img1.at<uchar>(ki,kj);
                            ref_r1=(int)img2.at<uchar>(ki,kj);
                        }
                        if(ki<0 || ki>img1.rows-1 || kj-d<0 || kj-d>img1.cols-1){
                            ref_l2=0;
                        }
                        else{
                            ref_l2=(int)img2.at<uchar>(ki,kj-d);
                        }
                        if(ki<0 || ki>img1.rows-1 || kj+d<0 || kj+d>img1.cols-1){
                            ref_r2=0;
                        }
                        else{
                            ref_r2=(int)img1.at<uchar>(ki,kj+d);
                        }                        
                        window_l1[index] = ref_l1;
                        window_l2[index] = ref_l2;
                        window_r1[index] = ref_r1;
                        window_r2[index] = ref_r2;                        
                        index++;
                    }
                }        
                cost_l[(i*img1.cols+j)*max_disp+d] = compute_ZSAD(window_l1,window_l2,window_size);  
                cost_r[(i*img1.cols+j)*max_disp+d] = compute_ZSAD(window_r1,window_r2,window_size);              
            }
        }
    }
    free(window_l1);
    free(window_l2); 
    free(window_r1);
    free(window_r2);  
    return 0;  
}

/*-------------------------------------------Rank Transform-----------------------------------------*/
void compute_rank_transform(cv::Mat img, int *rank, int window_size){
    for(int i=0; i<img.rows; i++){
        for(int j=0; j<img.cols; j++){
            int rank_val = 0;
            for(int ki=i-window_size/2; ki<=i+window_size/2; ki++){
                for(int kj=j-window_size/2; kj<=j+window_size/2; kj++){
                    unsigned char ref;
                    if(ki<0 || ki>img.rows-1 || kj<0 || kj>img.cols-1){
                        ref=0;
                    }
                    else{
                        ref=img.at<unsigned char>(ki,kj);
                    }
                    if(ki!=i||kj!=j){
                        if(ref < img.at<unsigned char>(i,j)){
                            rank_val += 1;
                        }
                    }
                }
            }
            rank[i*img.cols+j] = rank_val;
        }
    }
}

void compute_rank_cost(int *rank1, int *rank2, int *cost, int rows, int cols, int max_disp){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            for (int d=0; d<max_disp; d++){
                if(j-d>=0){
                    int dist = ABSdiff<int>(rank1[i*cols+j],rank2[i*cols+j-d]);
                    cost[(i*cols+j)*max_disp+d] = dist;
                }
                else{
                    int dist = ABSdiff<int>(rank1[i*cols+j],0);
                    cost[(i*cols+j)*max_disp+d] = dist;
                }
            }
        }
    }
}

void compute_lr_rank_cost(int *rank1, int *rank2, int *cost_l, int *cost_r, int rows, int cols, int max_disp){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            for (int d=0; d<max_disp; d++){
                if(j-d>=0){
                    int dist = ABSdiff<int>(rank1[i*cols+j],rank2[i*cols+j-d]);
                    cost_l[(i*cols+j)*max_disp+d] = dist;
                }
                else{
                    int dist = ABSdiff<int>(rank1[i*cols+j],0);
                    cost_l[(i*cols+j)*max_disp+d] = dist;
                }
                if(j+d<cols){
                    int dist = ABSdiff<int>(rank2[i*cols+j],rank1[i*cols+j+d]);
                    cost_r[(i*cols+j)*max_disp+d] = dist;
                }
                else{
                    int dist = ABSdiff<int>(rank2[i*cols+j],0);
                    cost_r[(i*cols+j)*max_disp+d] = dist;
                }
            }
        }
    }
}

/*-------------------------------------------Census Transform-----------------------------------------*/
void compute_census_transform(cv::Mat img, __int128_t *census, int window_size){
    for(int i=0; i<img.rows; i++){
        for(int j=0; j<img.cols; j++){
            __int128_t census_val = 0;
            for(int ki=i-window_size/2; ki<=i+window_size/2; ki++){
                for(int kj=j-window_size/2; kj<=j+window_size/2; kj++){
                    unsigned char ref;
                    if(ki<0 || ki>img.rows-1 || kj<0 || kj>img.cols-1){
                        ref=0;
                    }
                    else{
                        ref=img.at<unsigned char>(ki,kj);
                    }
                    if(ki!=i||kj!=j){
                        census_val = census_val<<1;
                        if(ref < img.at<unsigned char>(i,j)){
                            census_val += 1;
                        }
                    }
                }
            }           
            census[i*img.cols+j] = census_val;
        }
    }
}

int compute_hamming_distance (__int128_t a, __int128_t b) {
	__int128_t tmp = a ^ b;
	int sum = 0;
	while (tmp>0) {
		short int c = tmp & 0x1;
		sum += c;
		tmp >>= 1;
	}
	return sum;
} // end compute_hamming_distance()

void compute_census_cost(__int128_t *census1, __int128_t *census2, int *cost, int rows, int cols, int max_disp){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            for (int d=0; d<max_disp; d++){
                if(j-d>=0){
                    int dist = compute_hamming_distance(census1[i*cols+j],census2[i*cols+j-d]);
                    cost[(i*cols+j)*max_disp+d] = dist;
                }
                else{
                    int dist = compute_hamming_distance(census1[i*cols+j],0);//census2[i*cols]);
                    cost[(i*cols+j)*max_disp+d] = dist;
                }
            }
        }
    }
}

void compute_lr_census_cost(__int128_t *census1, __int128_t *census2, int *cost_l, int *cost_r, int rows, int cols, int max_disp){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            for (int d=0; d<max_disp; d++){
                if(j-d>=0){
                    int dist = compute_hamming_distance(census1[i*cols+j],census2[i*cols+j-d]);
                    cost_l[(i*cols+j)*max_disp+d] = dist;
                }
                else{
                    int dist = compute_hamming_distance(census1[i*cols+j],0);//census2[i*cols]);
                    cost_l[(i*cols+j)*max_disp+d] = dist;
                }
                if(j+d<cols){
                    int dist = compute_hamming_distance(census2[i*cols+j],census1[i*cols+j+d]);
                    cost_r[(i*cols+j)*max_disp+d] = dist;
                }
                else{
                    int dist = compute_hamming_distance(census2[i*cols+j],0);//census2[i*cols+cols-1]);
                    cost_r[(i*cols+j)*max_disp+d] = dist;
                }
            }
        }
    }
}

/*-------------------------------------------SHD: sum of Hamming Distance-----------------------------------------*/
int compute_SHD(long int *window1, long int *window2, int window_size){
    int sad = 0;    
    for(int i=0; i<window_size; i++){
        for(int j=0; j<window_size; j++){
            sad += compute_hamming_distance(window1[i*window_size+j],window2[i*window_size+j]);
        }
    }
    return sad;
}

int compute_SHD_cost(cv::Mat img1, cv::Mat img2, int *cost, int window_size, int shd_window, int max_disp){
    __int128_t *ct1 = (__int128_t*)malloc(img1.rows*img1.cols*sizeof(__int128_t));
    if (!ct1) {
        printf("Memory allocation failed for ct1..! \n");
        return -1;
    }
    __int128_t *ct2 = (__int128_t*)malloc(img1.rows*img1.cols*sizeof(__int128_t));
    if (!ct2) {
        printf("Memory allocation failed for ct2..! \n");
        return -1;
    }
    compute_census_transform(img1, ct1, window_size);
    compute_census_transform(img2, ct2, window_size);
    long int *window1 = (long int*)malloc(shd_window*shd_window*sizeof(long int));
    if (!window1) {
        printf("Memory allocation failed for window1..! \n");
        return -1;
    }
    long int *window2 = (long int*)malloc(shd_window*shd_window*sizeof(long int));
    if (!window2) {
        printf("Memory allocation failed for window2..! \n");
        return -1;
    }     
    for(int i=0; i<img1.rows; i++){
        for(int j=0; j<img1.cols; j++){
            for (int d=0; d<max_disp; d++){
                int index=0;
                for(int ki=i-shd_window/2; ki<=i+shd_window/2; ki++){
                    for(int kj=j-shd_window/2; kj<=j+shd_window/2; kj++){
                        long int left_ref,right_ref;
                        if(ki<0 || ki>img1.rows-1 || kj<0 || kj>img1.cols-1){
                            left_ref=0;
                        }
                        else{
                            left_ref=ct1[ki*img1.cols+kj];
                        }
                        if(ki<0 || ki>img1.rows-1 || kj-d<0 || kj-d>img1.cols-1){
                            right_ref=0;
                        }
                        else{
                            right_ref=ct2[ki*img1.cols+kj-d];
                        }
                        window1[index] = left_ref;
                        window2[index] = right_ref;
                        index++;
                    }
                }        
                cost[(i*img1.cols+j)*max_disp+d] = compute_SHD(window1,window2,shd_window);
            }
        }
    }     
    free(ct1);
    free(ct2);
    free(window1);
    free(window2);
    return 0;
}

int compute_lr_SHD_cost(cv::Mat img1, cv::Mat img2, int *cost_l, int *cost_r, int window_size, int shd_window, int max_disp){
    __int128_t *ct1 = (__int128_t*)malloc(img1.rows*img1.cols*sizeof(__int128_t));
    if (!ct1) {
        printf("Memory allocation failed for ct1..! \n");
        return -1;
    }
    __int128_t *ct2 = (__int128_t*)malloc(img1.rows*img1.cols*sizeof(__int128_t));
    if (!ct2) {
        printf("Memory allocation failed for ct2..! \n");
        return -1;
    }
    compute_census_transform(img1, ct1, window_size);
    compute_census_transform(img2, ct2, window_size);    
    long int *window_l1 = (long int*)malloc(shd_window*shd_window*sizeof(long int));
    if (!window_l1) {
        printf("Memory allocation failed for window_l1..! \n");
        return -1;
    }
    long int *window_l2 = (long int*)malloc(shd_window*shd_window*sizeof(long int));
    if (!window_l2) {
        printf("Memory allocation failed for window_l2..! \n");
        return -1;
    }
    long int *window_r1 = (long int*)malloc(shd_window*shd_window*sizeof(long int));
    if (!window_r1) {
        printf("Memory allocation failed for window_r1..! \n");
        return -1;
    }
    long int *window_r2 = (long int*)malloc(shd_window*shd_window*sizeof(long int));
    if (!window_r2) {
        printf("Memory allocation failed for window_r2..! \n");
        return -1;
    }             
    for(int i=0; i<img1.rows; i++){
        for(int j=0; j<img1.cols; j++){
            for (int d=0; d<max_disp; d++){
                int index=0;
                for(int ki=i-shd_window/2; ki<=i+shd_window/2; ki++){
                    for(int kj=j-shd_window/2; kj<=j+shd_window/2; kj++){
                        long int ref_l1, ref_l2, ref_r1, ref_r2;
                        if(ki<0 || ki>img1.rows-1 || kj<0 || kj>img1.cols-1){
                            ref_l1=0;
                            ref_r1=0;
                        }
                        else{
                            ref_l1=ct1[ki*img1.cols+kj];
                            ref_r1=ct2[ki*img1.cols+kj];
                        }
                        if(ki<0 || ki>img1.rows-1 || kj-d<0 || kj-d>img1.cols-1){
                            ref_l2=0;
                        }
                        else{
                            ref_l2=ct2[ki*img1.cols+kj-d];
                        }
                        if(ki<0 || ki>img1.rows-1 || kj+d<0 || kj+d>img1.cols-1){
                            ref_r2=0;
                        }
                        else{
                            ref_r2=ct1[ki*img1.cols+kj+d];;
                        }                        
                        window_l1[index] = ref_l1;
                        window_l2[index] = ref_l2;
                        window_r1[index] = ref_r1;
                        window_r2[index] = ref_r2;                        
                        index++;
                    }
                }        
                cost_l[(i*img1.cols+j)*max_disp+d] = compute_SHD(window_l1,window_l2, shd_window);  
                cost_r[(i*img1.cols+j)*max_disp+d] = compute_SHD(window_r1,window_r2, shd_window);              
            }
        }
    }
    free(ct1);
    free(ct2);    
    free(window_l1);
    free(window_l2); 
    free(window_r1);
    free(window_r2); 
    return 0;   
}

/*-------------------------------------------Compute Initial Costs-----------------------------------------*/
int compute_initial_cost(cv::Mat img1, cv::Mat img2, int *cost, int function_type, int window_size, int shd_window, int max_disp){
    if(function_type==0){
        __int128_t *ct1 = (__int128_t*)malloc(img1.rows*img1.cols*sizeof(__int128_t));
        if (!ct1) {
            printf("Memory allocation failed for ct1..! \n");
            return -1;
        }
        __int128_t *ct2 = (__int128_t*)malloc(img1.rows*img1.cols*sizeof(__int128_t));
        if (!ct2) {
            printf("Memory allocation failed for ct2..! \n");
            return -1;
        }
        // Compute census transform
        compute_census_transform(img1, ct1, window_size);
        compute_census_transform(img2, ct2, window_size);
        compute_census_cost(ct1,ct2,cost,img1.rows,img1.cols,max_disp);
        free(ct1);
	    free(ct2);
    }
    else if(function_type==1){
        int *rt1 = (int*)malloc(img1.rows*img1.cols*sizeof(int));
        if (!rt1) {
            printf("Memory allocation failed for rt1..! \n");
            return -1;
        }
        int *rt2 = (int*)malloc(img1.rows*img1.cols*sizeof(int));
        if (!rt2) {
            printf("Memory allocation failed for rt2..! \n");
            return -1;
        }
        compute_rank_transform(img1,rt1,window_size);
        compute_rank_transform(img2,rt2,window_size);
        compute_rank_cost(rt1,rt2,cost,img1.rows,img1.cols,max_disp);
        free(rt1);
        free(rt2);
    }
    else if(function_type == 2){
        int sad = compute_SAD_cost(img1,img2,cost,window_size,max_disp);
        return sad;
    }
    else if(function_type == 3){
        int zsad = compute_ZSAD_cost(img1,img2,cost,window_size,max_disp);
        return zsad;
    }
    else if(function_type == 4){
        int shd = compute_SHD_cost(img1,img2,cost,window_size,shd_window,max_disp);
        return shd;
    }
    return 0;
}

int compute_lr_initial_cost(cv::Mat img1, cv::Mat img2, int *cost_l, int *cost_r, int function_type, int window_size, int shd_window, int max_disp){
    if(function_type==0){
        __int128_t *ct1 = (__int128_t*)malloc(img1.rows*img1.cols*sizeof(__int128_t));
        if (!ct1) {
            printf("Memory allocation failed for ct1..! \n");
            return -1;
        }
        __int128_t *ct2 = (__int128_t*)malloc(img1.rows*img1.cols*sizeof(__int128_t));
        if (!ct2) {
            printf("Memory allocation failed for ct2..! \n");
            return -1;
        }
        // Compute census transform
        compute_census_transform(img1, ct1, window_size);
        compute_census_transform(img2, ct2, window_size);
        compute_lr_census_cost(ct1,ct2,cost_l,cost_r,img1.rows,img1.cols,max_disp);
        free(ct1);
	    free(ct2);
    }
    else if(function_type==1){
        int *rt1 = (int*)malloc(img1.rows*img1.cols*sizeof(int));
        if (!rt1) {
            printf("Memory allocation failed for rt1..! \n");
            return -1;
        }
        int *rt2 = (int*)malloc(img1.rows*img1.cols*sizeof(int));
        if (!rt2) {
            printf("Memory allocation failed for rt2..! \n");
            return -1;
        }
        compute_rank_transform(img1,rt1,window_size);
        compute_rank_transform(img2,rt2,window_size);
        compute_lr_rank_cost(rt1,rt2,cost_l,cost_r,img1.rows,img1.cols,max_disp);
        free(rt1);
        free(rt2);
    }
    else if(function_type == 2){
        int sad = compute_lr_SAD_cost(img1,img2,cost_l,cost_r,window_size,max_disp);
        return sad;
    }
    else if(function_type == 3){
        int zsad = compute_lr_ZSAD_cost(img1,img2,cost_l,cost_r,window_size,max_disp);
        return zsad;
    }
    else if(function_type == 4){
        int shd = compute_lr_SHD_cost(img1,img2,cost_l,cost_r,window_size,shd_window,max_disp);
        return shd;
    }
    return 0;
}


/*-------------------------------------------Cost Aggregation-----------------------------------------*/
void init_Lr(int *Lr, int *cost, int sizeOfCpd, int dir) {
	for (int r=0; r<dir; r++) {
		for (int i=0; i<sizeOfCpd; i++) {
			Lr[r*sizeOfCpd+i] = cost[i];
		}
	}
} // end init_Lr()

int find_minLri(int *Lrpr, int d, int ndisparity) {
	int minLri = INT_MAX;
	for (int i=0; i<d-1; i++) {
		if (minLri > Lrpr[i]) {
			minLri = Lrpr[i];
		}
	}
	for (int i=d+2; i<ndisparity; i++) {
		if (minLri > Lrpr[i]) {
			minLri = Lrpr[i];
		}
	}
	return minLri;
} // end find_minLri()

int find_min(int a, int b, int c, int d) {

	int minimum = a;
	if (minimum > b)
		minimum = b;
	if (minimum > c)
		minimum = c;
	if (minimum > d)
		minimum = d;
	return minimum;
} // end find_min()

void cost_computation(int *Lr, int *cost, int rows, int cols, int numDir, int ndisparity, int P1, int P2) {
	// Computing cost. (i,j-1) (i-1,j-1) (i-1,j) (i-1,j+1) (i,j+1) (i+1,j+1) (i+1,j) (i+1,j-1)
	int iDisp = 0, jDisp = 0;
	for (int r=0; r<numDir; r++) {
		if (r==0) {
			iDisp = 0; jDisp = -1;
		}
		else if (r==1) {
			iDisp = -1; jDisp = -1;
		}
		else if (r==2) {
			iDisp = -1; jDisp = 0;
		}
		else if (r==3) {
			iDisp = -1; jDisp = 1;
		}
		else if (r==4) {
			iDisp = 0; jDisp = 1;
		}
        else if (r==5) {
            iDisp = 1; jDisp = 1;
        }
        else if (r==6) {
            iDisp = 1; jDisp = 0;            
        }
        else if (r==7) {
            iDisp = 1; jDisp = -1;            
        }       

		// Changed the indices of the loop below to accommodate for number of directions = 8.
		if(r<4){
            for (int i=0; i<rows; i++) {
                for (int j=0; j<cols; j++) {
                    // Compute p-r
                    int iNorm = i + iDisp;
                    int jNorm = j + jDisp;

                    int *Lrpr = Lr+((r*rows+iNorm)*cols+jNorm)*ndisparity;

                    for (int d=0; d<ndisparity; d++) {
                        int Cpd = cost[(i*cols+j)*ndisparity+d];

                        int tmp;
                        if ( (((r==0)||(r==1))&&(j==0)) || (((r==1)||(r==2)||(r==3))&&(i==0)) || ((r==3)&&(j==cols-1)) )
                            tmp = Cpd;
                        else {
                            // Find min_i{Lr(p-r,i)}
                            int minLri = find_minLri(Lrpr, d, ndisparity);
                            int Lrpdm1, Lrpdp1;
                            if (d==0)
                                Lrpdm1 = INT_MAX-P1;
                            else
                                Lrpdm1 = Lrpr[d-1];
                            if (d==ndisparity-1)
                                Lrpdp1 = INT_MAX-P1;
                            else
                                Lrpdp1 = Lrpr[d+1];

                            int v2 = std::min(std::min(std::min(minLri,Lrpdp1),Lrpdm1),Lrpr[d]);
                            int v1 = find_min(Lrpr[d], Lrpdm1+P1, Lrpdp1+P1, v2+P2);

                            tmp = Cpd + v1 - v2;
                        }
                        Lr[((r*rows+i)*cols+j)*ndisparity+d] = tmp;
                    }
                }
            }
        }
        else{
            for (int i=rows-1; i>=0; i--) {
                for (int j=cols-1; j>=0; j--) {
                    // Compute p-r
                    int iNorm = i + iDisp;
                    int jNorm = j + jDisp;

                    int *Lrpr = Lr+((r*rows+iNorm)*cols+jNorm)*ndisparity;

                    for (int d=0; d<ndisparity; d++) {
                        int Cpd = cost[(i*cols+j)*ndisparity+d];

                        int tmp;
                        if ( ((r==7)&&(j==0)) || (((r==4)||(r==5))&&(j==cols-1)) || (((r==5)||(r==6)||(r==7))&&(i==rows-1)) )
                            tmp = Cpd;
                        else {
                            // Find min_i{Lr(p-r,i)}
                            int minLri = find_minLri(Lrpr, d, ndisparity);
                            int Lrpdm1, Lrpdp1;
                            if (d==0)
                                Lrpdm1 = INT_MAX-P1;
                            else
                                Lrpdm1 = Lrpr[d-1];
                            if (d==ndisparity-1)
                                Lrpdp1 = INT_MAX-P1;
                            else
                                Lrpdp1 = Lrpr[d+1];

                            int v2 = std::min(std::min(std::min(minLri,Lrpdp1),Lrpdm1),Lrpr[d]);
                            int v1 = find_min(Lrpr[d], Lrpdm1+P1, Lrpdp1+P1, v2+P2);

                            tmp = Cpd + v1 - v2;
                        }
                        Lr[((r*rows+i)*cols+j)*ndisparity+d] = tmp;
                    }
                }
            }
        }
	}
} // end cost_computation()

void cost_aggregation(int *aggregatedCost, int *Lr, int rows, int cols, int ndir, int ndisparity) {
	for (int i=0; i<rows; i++) {
		for (int j=0; j<cols; j++) {
			for (int d=0; d<ndisparity; d++) {
				int *ptr = aggregatedCost + (i*cols+j)*ndisparity+d;
				ptr[0] = 0;
				for (int r=0; r<ndir; r++) {
					ptr[0] += Lr[((r*rows+i)*cols+j)*ndisparity+d];
				}
			}
		}
	}
}// end cost_aggregation()


/*-------------------------------------------Post Processing-----------------------------------------*/
void compute_disparity(float *disparity, int *aggregatedCost, int rows, int cols, int ndisparity) {
	for (int i=0; i<rows; i++) {
		for (int j=0; j<cols; j++) {
			int *costPtr = aggregatedCost + (i*cols+j)*ndisparity;
			int minCost = costPtr[0];
			int mind = 0;
			for (int d=1; d<ndisparity; d++) {
				if (costPtr[d] < minCost) {
					minCost = costPtr[d];
					mind = d;
				}
			}
			disparity[i*cols+j] = mind;//out_disp;
		}
	}
}

void compute_lr_disparity(float *disparity_l, float *disparity_r, int *aggregatedCost, int rows, int cols, int ndisparity) {
	for (int i=0; i<rows; i++) {
		for (int j=0; j<cols; j++) {
			int *costPtr = aggregatedCost + (i*cols+j)*ndisparity;
			int minCost = costPtr[0];
			int mind = 0;
            int minCost_r = costPtr[0];
            int mind_r = 0;
			for (int d=1; d<ndisparity; d++) {
				if (costPtr[d] < minCost) {
					minCost = costPtr[d];
					mind = d;
				}
                if(j+d<cols){
                    if (aggregatedCost[(i*cols+j+d)*ndisparity+d] < minCost_r){
                        minCost_r = aggregatedCost[(i*cols+j+d)*ndisparity+d];
                        mind_r = d;
                    }
                }
			}
			disparity_l[i*cols+j] = mind;
            disparity_r[i*cols+j] = mind_r;
		}
	}
}

void sort_array(int *array, int *min_value, int *min_d, int ndisparity){
    min_value[0] = INT_MAX;
    min_value[1] = INT_MAX;
    min_value[2] = INT_MAX;
    min_d[0] = 0;
    min_d[1] = 0;    
    for(int i=0; i<ndisparity; i++){
        if(array[i]<min_value[0]){
            min_value[2] = min_value[1];
            min_value[1] = min_value[0];
            min_value[0] = array[i];
            min_d[1] = min_d[0];
            min_d[0] = i;
        }
        else if(array[i]<min_value[1]){
            min_value[2] = min_value[1];
            min_value[1] = array[i];
            min_d[1] = i; 
        }
        else if(array[i]<min_value[2]){
            min_value[2] = array[i];
        }
    }
}

void compute_disparity_uniqueness(float *disparity, int *aggregatedCost, int rows, int cols, int ndisparity) {
	for (int i=0; i<rows; i++) {
		for (int j=0; j<cols; j++) {
			int *costPtr = aggregatedCost + (i*cols+j)*ndisparity;
            int min_value[3];
            int min_d[2];
            sort_array(costPtr,min_value,min_d,ndisparity);
            int abs_diff = ABSdiff<int>(min_d[1],min_d[0]);
            int mind = min_d[0];
            int min0 = min_value[0]*20;
            int min1 = min_value[1]*19;
            int min2 = min_value[2]*19;
            if(abs_diff>1 && min0>min1){
                mind = 0;
            }
            else if(min0>min2){
                mind = 0;
            }
			disparity[i*cols+j] = mind;
		}
	}
}

void compute_lr_disparity_uniqueness(float *disparity_l, float *disparity_r, int *aggregatedCost, int rows, int cols, int ndisparity) {
	for (int i=0; i<rows; i++) {
		for (int j=0; j<cols; j++) {
			int *costPtr = aggregatedCost + (i*cols+j)*ndisparity;
            int minCost_r = costPtr[0];
            int mind_r = 0;            
            int min_value[3];
            int min_d[2];
            sort_array(costPtr,min_value,min_d,ndisparity);
            int abs_diff = ABSdiff<int>(min_d[1],min_d[0]);
            int mind = min_d[0];
            int min0 = min_value[0]*20;
            int min1 = min_value[1]*19;
            int min2 = min_value[2]*19;
            if(abs_diff>1 && min0>min1){
                mind = 0;
            }
            else if(min0>min2){
                mind = 0;
            }
			disparity_l[i*cols+j] = mind;
			
            for (int d=1; d<ndisparity; d++) {
                if(j+d<cols){
                    if (aggregatedCost[(i*cols+j+d)*ndisparity+d] < minCost_r){
                        minCost_r = aggregatedCost[(i*cols+j+d)*ndisparity+d];
                        mind_r = d;
                    }
                }
			}
            disparity_r[i*cols+j] = mind_r;
		}
	}  
}

void check_consistency(float *disparity_l, float *disparity_r, float *disparity, int rows, int cols){
	for (int i=0; i<rows; i++) {
		for (int j=0; j<cols; j++) {
            int left_disp = disparity_l[i*cols+j];
            int right_disp = 0;
            if(j-left_disp>=0){
                right_disp = disparity_r[i*cols+j-left_disp];
            }
            else{
                right_disp = 0;
            }
            int diff = ABSdiff<int>(left_disp,right_disp);
            int threshold = 1;
            int disp = 0;
            if(diff<=threshold){
                disp = left_disp;
            }
            disparity[i*cols+j] = disp;
		}
	}
}

template <typename T>
T compute_median(T *window, int size){
    T median = 0;
    for(int i=0; i<size; i++){
        for(int j=0; j<size/2; j++){
            int ind = j*2+i%2;
            int ind_next = ind+1;
            if(window[ind]<window[ind_next]){
               T tmp = window[ind];
               window[ind] = window[ind_next];
               window[ind_next] = tmp;
            } 
        }
    }
    median = window[size/2];
    return median;
}

int median_filter(float *disparity_src, float *disparity_dst, int rows, int cols, int filter_win){
    float *window = (float*)malloc(filter_win*filter_win*sizeof(float));
    if (!window) {
        printf("Memory allocation failed for window..! \n");
        return -1;
    }	
    for (int i=0; i<rows; i++) {
		for (int j=0; j<cols; j++) {
            int index = 0;
            for(int ki=i-filter_win/2; ki<=i+filter_win/2; ki++){
                for(int kj=j-filter_win/2; kj<=j+filter_win/2; kj++){
                    if(ki<0){
                        if(kj<0){
                            window[index] = disparity_src[0];
                        }
                        else if(kj>cols-1){
                            window[index] = disparity_src[cols-1];
                        }
                        else{
                            window[index] = disparity_src[kj];
                        }                                                
                    }
                    else if(ki>rows-1){
                        if(kj<0){
                            window[index] = disparity_src[(rows-1)*cols];
                        }
                        else if(kj>cols-1){
                            window[index] = disparity_src[(rows-1)*cols+cols-1];
                        }
                        else{
                            window[index] = disparity_src[(rows-1)*cols+kj];
                        }
                    }
                    else{
                        if(kj<0){
                            window[index] = disparity_src[ki*cols];
                        }
                        else if(kj>cols-1){
                            window[index] = disparity_src[ki*cols+cols-1];
                        }
                        else{
                            window[index] = disparity_src[ki*cols+kj];
                        }
                    }                  
                    index++;
                }
            }
            disparity_dst[i*cols+j] = compute_median<float>(window, filter_win*filter_win);
		}
	}
    return 0;   
}

int interpolateDisp (cv::Mat &disp)
{
	int32_t height_ = disp.rows;
	int32_t width_ = disp.cols;
	// for each row do
    // compared to FPGA implementation, no GAP_threshold is set for interpolation.
	for (int32_t v=0; v<height_; v++)
	{
		// init counter
		int32_t count = 0;
		// for each pixel do
		for (int32_t u=0; u<width_; u++)
		{
			// if disparity valid
			if (disp.at<uchar>(v,u) > 0)
			{
				// at least one pixel requires interpolation
				if (count>=1)
				{
					// first and last value for interpolation
					int32_t u1 = u-count;
					int32_t u2 = u-1;
					// set pixel to min disparity
					if (u1>0 && u2<width_-1)
					{
						uchar d_ipol = std::min(disp.at<uchar>(v,u1-1),disp.at<uchar>(v,u2+1));
						for (int32_t u_curr=u1; u_curr<=u2; u_curr++)
							disp.at<uchar>(v,u_curr) = d_ipol;
					}
				}
				// reset counter
				count = 0;
			}
			else
			{
				count++;
			}
		}

		// extrapolate to the left
		for (int32_t u=0; u<width_; u++)
		{
			if (disp.at<uchar>(v,u) > 0)
			{
				for (int32_t u2=0; u2<u; u2++)
					disp.at<uchar>(v,u2) = disp.at<uchar>(v,u);
				break;
			}
		}

		// extrapolate to the right
		for (int32_t u=width_-1; u>=0; u--)
		{
			if (disp.at<uchar>(v,u) > 0)
			{
				for (int32_t u2=u+1; u2<=width_-1; u2++)
					disp.at<uchar>(v,u2) = disp.at<uchar>(v,u);
				break;
			}
		}
	}

    // compared to FPGA implementation, add interpolation on the column direction.
	// for each column do
	for (int32_t u=0; u<width_; u++)
	{
		// extrapolate to the top
		for (int32_t v=0; v<height_; v++)
		{
			if (disp.at<uchar>(v,u) > 0)
			{
				for (int32_t v2=0; v2<v; v2++)
					disp.at<uchar>(v2,u) = disp.at<uchar>(v,u);
				break;
			}
		}

		// extrapolate to the bottom
		for (int32_t v=height_-1; v>=0; v--)
		{
			if (disp.at<uchar>(v,u) > 0)
			{
				for (int32_t v2=v+1; v2<=height_-1; v2++)
					disp.at<uchar>(v2,u) = disp.at<uchar>(v,u);
				break;
			}
		}
	}

	return 0;
}

/*-----------------------------------------------SGBM---------------------------------------------*/
// input images are 1 channel grayscale images.
int compute_SGM(cv::Mat img1, cv::Mat img2, float *disparity,int dir,int max_disp,int p1,int p2,int cost_type,int window_size,int filter_win, int shd_window)
{
	// Memory to store cost of size height x width x number of disparities
	int *cost = (int*)malloc(img1.rows*img1.cols*max_disp*sizeof(int));
	if (!cost) {
		printf("Memory allocation failed for accumulatedCost..! \n");
		return -1;
	}
    compute_initial_cost(img1,img2,cost,cost_type,window_size,shd_window,max_disp);
    //Create array for L(r,p,d)
	int *Lr = (int*)malloc(dir*img1.rows*img1.cols*max_disp*sizeof(int));
	if (!Lr) {
		printf("Memory allocation failed for Lr..! \n");
		return -1;
	}
	// Initialize Lr(p,d) to C(p,d)
	init_Lr(Lr, cost, img1.rows*img1.cols*max_disp,dir);

	// Compute cost along different directions
	cost_computation(Lr, cost, img1.rows, img1.cols, dir, max_disp, p1, p2);

	// Array for aggregated cost
	int *aggregatedCost = (int*)malloc(img1.rows*img1.cols*max_disp*sizeof(int));
	if (!aggregatedCost) {
		printf("Memory allocation failed for aggregatedCost..! \n");
		return -1;
	}
	cost_aggregation(aggregatedCost, Lr, img1.rows, img1.cols, dir, max_disp);

	// Disparity computation
    float *disparity_src = (float*)malloc(img1.rows*img1.cols*sizeof(float));
	compute_disparity(disparity, aggregatedCost, img1.rows, img1.cols, max_disp);

    //median_filter(disparity_src,disparity,img1.rows, img1.cols, filter_win);

	free(cost);
	free(Lr);
	free(aggregatedCost);
    free(disparity_src);

	return 0;
}

int compute_SGM_lr(cv::Mat img1, cv::Mat img2, float *disparity,int dir,int max_disp,int p1,int p2,int cost_type,int window_size,int filter_win,int shd_window)
{
	// Memory to store cost of size height x width x number of disparities
	int *cost_l = (int*)malloc(img1.rows*img1.cols*max_disp*sizeof(int));
	if (!cost_l) {
		printf("Memory allocation failed for accumulatedCost..! \n");
		return -1;
	}
	int *cost_r = (int*)malloc(img1.rows*img1.cols*max_disp*sizeof(int));
	if (!cost_r) {
		printf("Memory allocation failed for accumulatedCost..! \n");
		return -1;
	}   
    compute_lr_initial_cost(img1,img2,cost_l,cost_r,cost_type,window_size,shd_window,max_disp);
    //Create array for L(r,p,d)
	int *Lr_l = (int*)malloc(dir*img1.rows*img1.cols*max_disp*sizeof(int));
    int *Lr_r = (int*)malloc(dir*img1.rows*img1.cols*max_disp*sizeof(int));
	if (!Lr_l||!Lr_r) {
		printf("Memory allocation failed for Lr..! \n");
		return -1;
	}
	// Initialize Lr(p,d) to C(p,d)
	init_Lr(Lr_l, cost_l, img1.rows*img1.cols*max_disp, dir);
	init_Lr(Lr_r, cost_r, img1.rows*img1.cols*max_disp, dir);

	// Compute cost along different directions
	cost_computation(Lr_l, cost_l, img1.rows, img1.cols, dir, max_disp, p1, p2);
    cost_computation(Lr_r, cost_r, img1.rows, img1.cols, dir, max_disp, p1, p2);

	// Array for aggregated cost
	int *aggregatedCost_l = (int*)malloc(img1.rows*img1.cols*max_disp*sizeof(int));
    int *aggregatedCost_r = (int*)malloc(img1.rows*img1.cols*max_disp*sizeof(int));
	if (!aggregatedCost_l||!aggregatedCost_r) {
		printf("Memory allocation failed for aggregatedCost..! \n");
		return -1;
	}
	cost_aggregation(aggregatedCost_l, Lr_l, img1.rows, img1.cols, dir, max_disp);
    cost_aggregation(aggregatedCost_r, Lr_r, img1.rows, img1.cols, dir, max_disp);

	// Disparity computation
    float *disparity_src_l = (float*)malloc(img1.rows*img1.cols*sizeof(float));
    float *disparity_src_r = (float*)malloc(img1.rows*img1.cols*sizeof(float));
	compute_disparity(disparity_src_l, aggregatedCost_l, img1.rows, img1.cols, max_disp);
    compute_disparity(disparity_src_r, aggregatedCost_r, img1.rows, img1.cols, max_disp);
    
    float *disparity_dst_l = (float*)malloc(img1.rows*img1.cols*sizeof(float));
    float *disparity_dst_r = (float*)malloc(img1.rows*img1.cols*sizeof(float));    
    median_filter(disparity_src_l,disparity_dst_l,img1.rows, img1.cols, filter_win);
    median_filter(disparity_src_r,disparity_dst_r,img1.rows, img1.cols, filter_win);
    check_consistency(disparity_dst_l,disparity_dst_r,disparity,img1.rows,img1.cols);

	free(cost_l);
    free(cost_r);
	free(Lr_l);
    free(Lr_r);
	free(aggregatedCost_l);
    free(aggregatedCost_r);
    free(disparity_src_l);
    free(disparity_src_r);
    free(disparity_dst_l);
    free(disparity_dst_r);

	return 0;
}

void saveDisparityMap(float *disparity, int rows, int cols, int ndisparity, char* outputFile) {
	cv::Mat disparityMap(rows, cols, CV_8U);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			disparityMap.at<unsigned char>(i, j) = (unsigned char) (disparity[i*cols+j]);
		}
	}
	imwrite(outputFile, disparityMap);
} // end saveDisparityMap


int compute_disparity_errors(cv::Mat original_disp, cv::Mat interpolate_disp, cv::Mat gt_disp, cv::Mat obj_map, float *errors)
{
    if(original_disp.rows!=gt_disp.rows || original_disp.cols!=gt_disp.cols){
        fprintf(stderr,"Wrong Image Size\n");
        return 0;
    }
    int width = gt_disp.rows;
    int height = gt_disp.cols;
    // init errors
    int num_errors_bg = 0;
    int num_pixels_bg = 0;
    int num_errors_bg_result = 0;
    int num_pixels_bg_result = 0;
    int num_errors_fg = 0;
    int num_pixels_fg = 0;
    int num_errors_fg_result = 0;
    int num_pixels_fg_result = 0;
    int num_errors_all = 0;
    int num_pixels_all = 0;
    int num_errors_all_result = 0;
    int num_pixels_all_result = 0;

    for(int i=0; i<width; i++){
        for(int j=0; j<height; j++){
            unsigned short gt_disp_val = gt_disp.at<unsigned short>(i,j);
            if(gt_disp_val>0){
                float d_gt = ((float)gt_disp_val)/256.0;
                float d_est = (float)interpolate_disp.at<uchar>(i,j);
                bool d_err = fabsf(d_gt-d_est)>ABS_THRESH && fabsf(d_gt-d_est)/fabsf(d_gt)>REL_THRESH;
                // load object map (0:background, >0:foreground)
                if(obj_map.at<uchar>(i,j)==0){
                    if(d_err){
                        num_errors_bg++;
                    }
                    num_pixels_bg++;
                    if(original_disp.at<uchar>(i,j)>0){
                        if(d_err){
                            num_errors_bg_result++;
                        }
                        num_pixels_bg_result++;
                    }
                }
                else{
                    if(d_err){
                        num_errors_fg++;
                    }
                    num_pixels_fg++;
                    if(original_disp.at<unsigned char>(i,j)>0){
                        if(d_err){
                            num_errors_fg_result++;
                        }
                        num_pixels_fg_result++;
                    }
                }
                if(d_err){
                    num_errors_all++;
                }
                num_pixels_all++;
                if(original_disp.at<unsigned char>(i,j)>0){
                    if(d_err){
                        num_errors_all_result++;
                    }
                    num_pixels_all_result++;
                }
            }
        }
    }

    errors[0] = num_errors_bg;
    errors[1] = num_pixels_bg;
    errors[2] = num_errors_bg_result;
    errors[3] = num_pixels_bg_result;
    errors[4] = num_errors_fg;
    errors[5] = num_pixels_fg;
    errors[6] = num_errors_fg_result;
    errors[7] = num_pixels_fg_result;
    errors[8] = num_errors_all;
    errors[9] = num_pixels_all;
    errors[10] = num_errors_all_result;
    errors[11] = num_pixels_all_result;    

    errors[12] = (float)num_pixels_all_result/std::max((float)num_pixels_all,1.0f);

    return 0; 
}

int write_error_map(cv::Mat interpolate_disp, cv::Mat noc_gt_disp, cv::Mat occ_gt_disp, cv::Mat error_mat)
{
    int height = occ_gt_disp.rows;
    int width = occ_gt_disp.cols;

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            unsigned short gt_disp_val = occ_gt_disp.at<unsigned short>(i,j);
            if(gt_disp_val>0){
                float d_gt = ((float)gt_disp_val)/256.0;
                float d_est = (float)interpolate_disp.at<uchar>(i,j);
                float d_err = fabsf(d_gt-d_est);
                if(noc_gt_disp.at<unsigned short>(i,j)==0){
                    d_err *= 0.5;
                }
                error_mat.at<uchar>(i,j) = (uchar)(std::round(d_err));
            }
            else{
                error_mat.at<uchar>(i,j) = 0;
            }
        }
    }
    return 0; 
}

void get_gt_disp(cv::Mat gt_disp, cv::Mat &actual_disp)
{
    int height = gt_disp.rows;
    int width = gt_disp.cols;

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            unsigned short gt_disp_val = gt_disp.at<unsigned short>(i,j);
            float d_gt = ((float)gt_disp_val)/256.0;
            actual_disp.at<uchar>(i,j) = (uchar)(std::round(d_gt));
        }
    }
}


int main(int argc, char** argv)
{
	if (argc != 10)
	{
		fprintf(stderr,"Invalid Number of Arguments!\nUsage:\n");
		fprintf(stderr,"<Executable Name> <Dataset folder path> <MAX_DISPARITY> <NUM_DIR> <P1> <P2> <COST_TYPE> <COST_WINDOW> <FILTER_WINDOW> <SHD_WINDOW> \n");
		return -1;
	}

    std::string ImageFolderDir = argv[1];    

    int max_disp = std::atoi(argv[2]);
    int dir = std::atoi(argv[3]);
    int p1 = std::atoi(argv[4]); 
    int p2 = std::atoi(argv[5]);
    int cost_type = std::atoi(argv[6]);
    int window_size = std::atoi(argv[7]);
    int filter_win = std::atoi(argv[8]);
    int shd_window = std::atoi(argv[9]);

    if(p1>=p2){
        fprintf(stderr,"P1 should be smaller than P2\n");
        return -1;
    }

    std::string option = std::string(argv[2]) + "_" + std::string(argv[3]) + "_" + std::string(argv[4]) + "_" + std::string(argv[5]) + "_" + std::string(argv[6]) + "_" + std::string(argv[7]) + "_" + std::string(argv[8]) + "_" + std::string(argv[9]);
    std::string ResultsDir = ImageFolderDir + "/results/" + option;

    int succeed = std::system(("mkdir " + ResultsDir).c_str());
    int succeed1 = std::system(("mkdir " + ResultsDir + "/results_disp").c_str());
    int succeed2 = std::system(("mkdir " + ResultsDir + "/errors_disp_occ_0").c_str());
    int succeed3 = std::system(("mkdir " + ResultsDir + "/inter_disp_occ_0").c_str());

    if(!succeed && !succeed1 && !succeed2 && !succeed3){
        fprintf(stderr,"Successfully construct new directories.\n");
    }

    FILE *stats_noc_file = fopen((ResultsDir + "/stats_disp_noc_0.txt").c_str(),"w");
    FILE *stats_occ_file = fopen((ResultsDir + "/stats_disp_occ_0.txt").c_str(),"w");

    // accumulators
    float errors_disp_noc_0[3*4] = {0,0,0,0,0,0,0,0,0,0,0,0};
    float errors_disp_occ_0[3*4] = {0,0,0,0,0,0,0,0,0,0,0,0};       
   
    printf("Start runing SGM on image pairs...\n");    
    //clock_t timer_start=clock();   
    for(int i=0; i<NUM_TEST_IMAGES; i++){
        char prefix[256];
        sprintf(prefix,"%06d_10",i);
        std::string leftImageName = ImageFolderDir + "/image_2/" + prefix + ".png";
        std::string rightImageName = ImageFolderDir + "/image_3/" + prefix + ".png";

        cv::Mat in_imgL_ori = cv::imread(leftImageName);
        cv::Mat in_imgR_ori = cv::imread(rightImageName);
        if (in_imgL_ori.data == NULL || in_imgR_ori.data == NULL)
        {
            fprintf(stderr,"Cannot open image at %s or %s\n",leftImageName.c_str(),rightImageName.c_str());
            return 0;
        }

        int crop_height = in_imgL_ori.rows;
        int crop_width = in_imgL_ori.cols;
        cv::Rect roi(0, 0, crop_width, crop_height);

        cv::Mat in_imgL = in_imgL_ori(roi);
        cv::Mat in_imgR = in_imgR_ori(roi);
        cv::Mat in_imgL_gray, in_imgR_gray;

        unsigned short height  = in_imgL.rows;
        unsigned short width  = in_imgL.cols;
        
        cv::cvtColor(in_imgL, in_imgL_gray, CV_BGR2GRAY);
        cv::cvtColor(in_imgR, in_imgR_gray, CV_BGR2GRAY);

        // Array to store disparity
        float *disparity = (float*)malloc(height*width*sizeof(float));
        if (!disparity) {
            printf("Memory allocation failed for disparity..! \n");
            return -1;
        }

        compute_SGM(in_imgL_gray,in_imgR_gray,disparity,dir,max_disp,p1,p2,cost_type,window_size,filter_win,shd_window);
        //compute_SGM_lr(in_imgL_gray,in_imgR_gray,disparity,dir,max_disp,p1,p2,cost_type,window_size,filter_win,shd_window);
        
        // Write disparity to file
        cv::Mat original_disp(height,width,CV_8UC1);
        cv::Mat interpolate_disp(height,width,CV_8UC1);
        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                original_disp.at<unsigned char>(r,c) = (unsigned char) (disparity[r*width+c]);
                interpolate_disp.at<unsigned char>(r,c) = (unsigned char) (disparity[r*width+c]);
            }
        }

        free(disparity);
    
    // clock_t timer_end=clock();
    // double time_consume=(double)(timer_end-timer_start)/CLOCKS_PER_SEC;
    // printf("Total time consumed for computing disparity maps: %f\n",time_consume);    
   
        std::string DisparityImage = ResultsDir + "/results_disp/" + prefix + ".png";      
        std::string GTDispNocImage = ImageFolderDir + "/disp_noc_0/" + prefix + ".png";
        std::string GTDispOccImage = ImageFolderDir + "/disp_occ_0/" + prefix + ".png";
        std::string ObjectMap = ImageFolderDir + "/obj_map/" + prefix + ".png";
    
        if (interpolate_disp.empty() == true)
        {
            fprintf(stderr,"Cannot open disparity image\n");
            return 0;
        }

        interpolateDisp(interpolate_disp);

        cv::Mat gt_disp_noc_roi = cv::imread(GTDispNocImage,cv::IMREAD_UNCHANGED);
        cv::Mat gt_disp_occ_roi = cv::imread(GTDispOccImage,cv::IMREAD_UNCHANGED);

        cv::Mat gt_disp_noc = gt_disp_noc_roi(roi);
        cv::Mat gt_disp_occ = gt_disp_occ_roi(roi);        

        cv::Mat obj_map_roi;
        obj_map_roi = cv::imread(ObjectMap,0);
        cv::Mat obj_map = obj_map_roi(roi);

        float noc_errors[13] = {0,0,0,0,0,0,0,0,0,0,0,0,0};
        float occ_errors[13] = {0,0,0,0,0,0,0,0,0,0,0,0,0};

        compute_disparity_errors(original_disp,interpolate_disp,gt_disp_noc,obj_map,noc_errors);
        compute_disparity_errors(original_disp,interpolate_disp,gt_disp_occ,obj_map,occ_errors);
        
        for(int num=0; num<12; num++){
            errors_disp_noc_0[num] += noc_errors[num];
            errors_disp_occ_0[num] += occ_errors[num];
        }

        if(i<NUM_ERROR_IMAGES){
            fprintf(stats_noc_file,"%s: ",prefix);
            for(int j=0; j<12; j+=2){
                fprintf(stats_noc_file,"%f ",noc_errors[j]/std::max(noc_errors[j+1],1.0f));
            }
            fprintf(stats_noc_file,"%f ",noc_errors[12]);
            fprintf(stats_noc_file,"\n");
            
            fprintf(stats_occ_file,"%s: ",prefix);
            for(int j=0; j<12; j+=2){
                fprintf(stats_occ_file,"%f ",occ_errors[j]/std::max(occ_errors[j+1],1.0f));
            }
            fprintf(stats_occ_file,"%f ",occ_errors[12]);
            fprintf(stats_occ_file,"\n");
            
            //cv::Mat noc_error_mat(original_disp.rows,original_disp.cols,CV_8UC1,0);
            cv::Mat occ_error_mat(original_disp.rows,original_disp.cols,CV_8UC1);
            write_error_map(interpolate_disp,gt_disp_noc,gt_disp_occ,occ_error_mat); 
            double min = 0;
            double max = 10;
            cv::Mat adjMap;
            occ_error_mat.convertTo(adjMap,CV_8UC1, 255/(max-min), min);
            cv::Mat falseColorMap;
            cv::applyColorMap(adjMap, falseColorMap, cv::COLORMAP_JET);
            cv::imwrite(ResultsDir + "/errors_disp_occ_0/" + prefix + ".png", falseColorMap); 
            
            // cv::Mat actual_disp(original_disp.rows,original_disp.cols,CV_8UC1);
            // cv::Mat falseColorMap_gt;
            // get_gt_disp(gt_disp_occ, actual_disp);
            // cv::applyColorMap(actual_disp, falseColorMap_gt, cv::COLORMAP_JET);
            // cv::imwrite(ResultsDir + "/gt_disp_occ_0/" + prefix + ".png", falseColorMap_gt);                  
            
            cv::Mat falseColorMap_disp;
            cv::applyColorMap(interpolate_disp, falseColorMap_disp, cv::COLORMAP_JET);
            cv::imwrite(ResultsDir + "/inter_disp_occ_0/" + prefix + ".png", falseColorMap_disp);
            cv::imwrite(DisparityImage, original_disp);        
        }
    }
    
    fprintf(stats_noc_file,"%s: ","Average");
    for(int i=0; i<12; i+=2){
        fprintf(stats_noc_file,"%f ",errors_disp_noc_0[i]/std::max(errors_disp_noc_0[i+1],1.0f));
    }
    fprintf(stats_noc_file,"%f ",errors_disp_noc_0[11]/std::max(errors_disp_noc_0[9],1.0f));
    fprintf(stats_noc_file,"\n");
    fclose(stats_noc_file);
    
    fprintf(stats_occ_file,"%s: ","Average");
    for(int i=0; i<12; i+=2){
        fprintf(stats_occ_file,"%f ",errors_disp_occ_0[i]/std::max(errors_disp_occ_0[i+1],1.0f));
    }
    fprintf(stats_occ_file,"%f ",errors_disp_occ_0[11]/std::max(errors_disp_occ_0[9],1.0f));
    fprintf(stats_occ_file,"\n");
    fclose(stats_occ_file);             

    printf("%f ",errors_disp_noc_0[8]/std::max(errors_disp_noc_0[9],1.0f));
    printf("%f ",errors_disp_occ_0[8]/std::max(errors_disp_occ_0[9],1.0f));

    printf("Finish successfully!\n");
	//clock_t timer_end1=clock();
    //double time_consume=(double)(timer_end-timer_start)/CLOCKS_PER_SEC;
    //printf("Total time consumed for computing disparity maps: %f\n",time_consume); 
    return 0;
}
