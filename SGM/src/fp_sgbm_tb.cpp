/*
 *  FP-Stereo
 *  Copyright (C) 2020  RCSL, HKUST
 *  
 *  GPL-3.0 License
 *
 */

#include "fp_headers.h"
#include "fp_sgbm_accel.h"

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
void compute_census_transform(cv::Mat img, long int *census, int window_size){
    for(int i=0; i<img.rows; i++){
        for(int j=0; j<img.cols; j++){
            long int census_val = 0;
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

int compute_hamming_distance (long int a, long int b) {
	long int tmp = a ^ b;
	int sum = 0;
	while (tmp>0) {
		short int c = tmp & 0x1;
		sum += c;
		tmp >>= 1;
	}
	return sum;
} // end compute_hamming_distance()

void compute_census_cost(long int *census1, long int *census2, int *cost, int rows, int cols, int max_disp){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            for (int d=0; d<max_disp; d++){
                if(j-d>=0){
                    int dist = compute_hamming_distance(census1[i*cols+j],census2[i*cols+j-d]);
                    cost[(i*cols+j)*max_disp+d] = dist;
                }
                else{
                    int dist = compute_hamming_distance(census1[i*cols+j],0);
                    cost[(i*cols+j)*max_disp+d] = dist;
                }
            }
        }
    }
}

void compute_lr_census_cost(long int *census1, long int *census2, int *cost_l, int *cost_r, int rows, int cols, int max_disp){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            for (int d=0; d<max_disp; d++){
                if(j-d>=0){
                    int dist = compute_hamming_distance(census1[i*cols+j],census2[i*cols+j-d]);
                    cost_l[(i*cols+j)*max_disp+d] = dist;
                }
                else{
                    int dist = compute_hamming_distance(census1[i*cols+j],0);
                    cost_l[(i*cols+j)*max_disp+d] = dist;
                }
                if(j+d<cols){
                    int dist = compute_hamming_distance(census2[i*cols+j],census1[i*cols+j+d]);
                    cost_r[(i*cols+j)*max_disp+d] = dist;
                }
                else{
                    int dist = compute_hamming_distance(census2[i*cols+j],0);
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
    long int *ct1 = (long int*)malloc(img1.rows*img1.cols*sizeof(long int));
    if (!ct1) {
        printf("Memory allocation failed for ct1..! \n");
        return -1;
    }
    long int *ct2 = (long int*)malloc(img1.rows*img1.cols*sizeof(long int));
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
    long int *ct1 = (long int*)malloc(img1.rows*img1.cols*sizeof(long int));
    if (!ct1) {
        printf("Memory allocation failed for ct1..! \n");
        return -1;
    }
    long int *ct2 = (long int*)malloc(img1.rows*img1.cols*sizeof(long int));
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
        long int *ct1 = (long int*)malloc(img1.rows*img1.cols*sizeof(long int));
        if (!ct1) {
            printf("Memory allocation failed for ct1..! \n");
            return -1;
        }
        long int *ct2 = (long int*)malloc(img1.rows*img1.cols*sizeof(long int));
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
        long int *ct1 = (long int*)malloc(img1.rows*img1.cols*sizeof(long int));
        if (!ct1) {
            printf("Memory allocation failed for ct1..! \n");
            return -1;
        }
        long int *ct2 = (long int*)malloc(img1.rows*img1.cols*sizeof(long int));
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
			disparity[i*cols+j] = mind;
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

int compute_SGM(cv::Mat img1, cv::Mat img2, float *disparity,int dir,int max_disp,int p1,int p2,int cost_type,int window_size,int filter_win, int shd_window, int post_option)
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
    if(post_option == 0){
        float *disparity_src = (float*)malloc(img1.rows*img1.cols*sizeof(float));
        compute_disparity(disparity_src, aggregatedCost, img1.rows, img1.cols, max_disp);
        median_filter(disparity_src,disparity,img1.rows, img1.cols, filter_win);
        free(disparity_src);
    }
    else if(post_option == 1){
        float *disparity_src_l = (float*)malloc(img1.rows*img1.cols*sizeof(float));
        float *disparity_src_r = (float*)malloc(img1.rows*img1.cols*sizeof(float));
        compute_lr_disparity(disparity_src_l, disparity_src_r, aggregatedCost, img1.rows, img1.cols, max_disp);
        float *disparity_dst_l = (float*)malloc(img1.rows*img1.cols*sizeof(float));
        float *disparity_dst_r = (float*)malloc(img1.rows*img1.cols*sizeof(float));    
        median_filter(disparity_src_l,disparity_dst_l,img1.rows, img1.cols, filter_win);
        median_filter(disparity_src_r,disparity_dst_r,img1.rows, img1.cols, filter_win);
        check_consistency(disparity_dst_l,disparity_dst_r,disparity,img1.rows,img1.cols);
        free(disparity_src_l);
        free(disparity_src_r);
        free(disparity_dst_l);
        free(disparity_dst_r);
    }
    else if(post_option == 2){
        float *disparity_src = (float*)malloc(img1.rows*img1.cols*sizeof(float));
        compute_disparity_uniqueness(disparity_src, aggregatedCost, img1.rows, img1.cols, max_disp);
        median_filter(disparity_src,disparity,img1.rows, img1.cols, filter_win);
        free(disparity_src);        
    }
    else if(post_option == 3){
        float *disparity_src_l = (float*)malloc(img1.rows*img1.cols*sizeof(float));
        float *disparity_src_r = (float*)malloc(img1.rows*img1.cols*sizeof(float));
    	compute_lr_disparity_uniqueness(disparity_src_l, disparity_src_r, aggregatedCost, img1.rows, img1.cols, max_disp);
        float *disparity_dst_l = (float*)malloc(img1.rows*img1.cols*sizeof(float));
        float *disparity_dst_r = (float*)malloc(img1.rows*img1.cols*sizeof(float));    
        median_filter(disparity_src_l,disparity_dst_l,img1.rows, img1.cols, filter_win);
        median_filter(disparity_src_r,disparity_dst_r,img1.rows, img1.cols, filter_win);
        check_consistency(disparity_dst_l,disparity_dst_r,disparity,img1.rows,img1.cols);   
        free(disparity_src_l);
        free(disparity_src_r);
        free(disparity_dst_l);
        free(disparity_dst_r);             
    }
	free(cost);
	free(Lr);
	free(aggregatedCost);
	return 0;
}

int compute_SGM_lr(cv::Mat img1, cv::Mat img2, float *disparity,int dir,int max_disp,int p1,int p2,int cost_type,int window_size,int filter_win,int shd_window, int post_option)
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

    float *disparity_dst_l = (float*)malloc(img1.rows*img1.cols*sizeof(float));
    float *disparity_dst_r = (float*)malloc(img1.rows*img1.cols*sizeof(float));

    if(post_option == 4){
        compute_disparity(disparity_src_l, aggregatedCost_l, img1.rows, img1.cols, max_disp);
        compute_disparity(disparity_src_r, aggregatedCost_r, img1.rows, img1.cols, max_disp);
    
        median_filter(disparity_src_l,disparity_dst_l,img1.rows, img1.cols, filter_win);
        median_filter(disparity_src_r,disparity_dst_r,img1.rows, img1.cols, filter_win);

        check_consistency(disparity_dst_l,disparity_dst_r,disparity,img1.rows,img1.cols);
    }
    else if(post_option == 5){
	    compute_disparity_uniqueness(disparity_src_l, aggregatedCost_l, img1.rows, img1.cols, max_disp);
        compute_disparity_uniqueness(disparity_src_r, aggregatedCost_r, img1.rows, img1.cols, max_disp);        
    
        median_filter(disparity_src_l,disparity_dst_l,img1.rows, img1.cols, filter_win);
        median_filter(disparity_src_r,disparity_dst_r,img1.rows, img1.cols, filter_win);
        
        check_consistency(disparity_dst_l,disparity_dst_r,disparity,img1.rows,img1.cols);    
    }

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
	float factor = 256.0 / ndisparity;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			disparityMap.at<unsigned char>(i, j) = (unsigned char) (disparity[i*cols+j] /** factor*/);
		}
	}
	imwrite(outputFile, disparityMap);
} // end saveDisparityMap


int main(int argc, char** argv)
{
	if (argc != 3)
	{
		fprintf(stderr,"Invalid Number of Arguments!\nUsage:\n");
		fprintf(stderr,"<Executable Name> <left image path> <right image path> \n");
		return -1;
	}

	cv::Mat in_imgL, in_imgR;

	in_imgL = cv::imread(argv[1],0);
	in_imgR = cv::imread(argv[2],0);

	if (in_imgL.data == NULL)
	{
		fprintf(stderr,"Cannot open image at %s\n",argv[1]);
		return 0;
	}
	if (in_imgR.data == NULL)
	{
		fprintf(stderr,"Cannot open image at %s\n",argv[2]);
		return 0;
	}

	unsigned short height  = in_imgL.rows;
	unsigned short width  = in_imgL.cols;

	static xf::Mat<IN_T, HEIGHT, WIDTH, XF_NPPC1> imgInputL(height,width);
	static xf::Mat<IN_T, HEIGHT, WIDTH, XF_NPPC1> imgInputR(height,width);
	static xf::Mat<IN_T, HEIGHT, WIDTH, XF_NPPC1> copy_imgInputL(height,width);
	static xf::Mat<IN_T, HEIGHT, WIDTH, XF_NPPC1> copy_imgInputR(height,width);

	static xf::Mat<OUT_T, HEIGHT, WIDTH, XF_NPPC1> imgOutput(height,width);

	imgInputL = xf::imread<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(argv[1], 0);
	imgInputR = xf::imread<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(argv[2], 0);


#if __SDSCC__
	perf_counter hw_ctr;
	hw_ctr.start();
#endif

	/* For 4 paths aggregation */
	semiglobalbm_accel(imgInputL,imgInputR,imgOutput);

#if __SDSCC__
	hw_ctr.stop();
	uint64_t hw_cycles = hw_ctr.avg_cpu_cycles();
#endif

	xf::imwrite("hls_out.png", imgOutput);

	// reference code
	// Array to store disparity
	float *disparity = (float*)malloc(height*width*sizeof(float));
	if (!disparity) {
		printf("Memory allocation failed for disparity..! \n");
		return -1;
	}

	if(UNIQ==0&&LR_CHECK==0){
		compute_SGM(in_imgL,in_imgR,disparity,NUM_DIR,NUM_DISPARITY,SMALL_PENALTY,LARGE_PENALTY,COST_FUNCTION,WINDOW_SIZE,FilterWin,SHD_WINDOW,0);
	}
	else if(UNIQ==0&&LR_CHECK==1){
		compute_SGM(in_imgL,in_imgR,disparity,NUM_DIR,NUM_DISPARITY,SMALL_PENALTY,LARGE_PENALTY,COST_FUNCTION,WINDOW_SIZE,FilterWin,SHD_WINDOW,1);
	}
	else if(UNIQ==1&&LR_CHECK==0){
		compute_SGM(in_imgL,in_imgR,disparity,NUM_DIR,NUM_DISPARITY,SMALL_PENALTY,LARGE_PENALTY,COST_FUNCTION,WINDOW_SIZE,FilterWin,SHD_WINDOW,2);
	}
	else if(UNIQ==1&&LR_CHECK==1){
		compute_SGM(in_imgL,in_imgR,disparity,NUM_DIR,NUM_DISPARITY,SMALL_PENALTY,LARGE_PENALTY,COST_FUNCTION,WINDOW_SIZE,FilterWin,SHD_WINDOW,3);
	}
	else if(UNIQ==0&&LR_CHECK==2){
		compute_SGM_lr(in_imgL,in_imgR,disparity,NUM_DIR,NUM_DISPARITY,SMALL_PENALTY,LARGE_PENALTY,COST_FUNCTION,WINDOW_SIZE,FilterWin,SHD_WINDOW,4);
	}
	else if(UNIQ==1&&LR_CHECK==2){
		compute_SGM_lr(in_imgL,in_imgR,disparity,NUM_DIR,NUM_DISPARITY,SMALL_PENALTY,LARGE_PENALTY,COST_FUNCTION,WINDOW_SIZE,FilterWin,SHD_WINDOW,5);
	}

	// Write disparity to file
	saveDisparityMap(disparity, height, width, NUM_DISPARITY, "disp_map.png");

	cv::Mat disp_mat(height,width,CV_8UC1);
	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			disp_mat.at<unsigned char>(r,c) = (unsigned char) (disparity[r*width+c]);
		}
	}
	free(disparity);

	cv::Mat diff;
	diff.create(height,width,CV_8UC1);
	int cnt = 0;
	for (int i=0; i<height; i++)
	{
		for (int j=0; j<width; j++)
		{
			int d_val = (unsigned char)imgOutput.data[i*width+j] - disp_mat.at<unsigned char>(i,j);

			if (d_val > 0)
			{
				diff.at<unsigned char>(i,j) = 255;
				cnt++;
			}
			else
				diff.at<unsigned char>(i,j) = 0;
		}
	}

	cv::imwrite("diff.png",diff);
	std::cout<<"Number of erroneous pixels:"<<cnt<<std::endl;
	std::cout<<"run success!"<<std::endl;

	return 0;
}
