/*
 *  FP-Stereo
 *  Copyright (C) 2020  RCSL, HKUST
 *  
 *  GPL-3.0 License
 *
 */

#include "fp_headers.h"
#include "fp_sgbm_accel.h"


// This is the host code for measuring the run-time power consumption. 
// The hardware accelerator is invoked 1000 times to process 200 images (each image is processed for five times). Then we can record the power data at different time stamps in a longer period (minutes).

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		fprintf(stderr,"Invalid Number of Arguments!\nUsage:\n");
		fprintf(stderr,"<Executable Name> <left image path> <right image path> \n");
		return -1;
	}

    std::string ImageFolderDir = argv[1];   

	unsigned short height  = 1242;
	unsigned short width  = 374;

	static xf::Mat<IN_T, HEIGHT, WIDTH, XF_NPPC1> imgInputL[200];
	static xf::Mat<IN_T, HEIGHT, WIDTH, XF_NPPC1> imgInputR[200];
	static xf::Mat<IN_T, HEIGHT, WIDTH, XF_NPPC1> copy_imgInputL[200];
	static xf::Mat<IN_T, HEIGHT, WIDTH, XF_NPPC1> copy_imgInputR[200];

	static xf::Mat<OUT_T, HEIGHT, WIDTH, XF_NPPC1> imgOutput[200];

    for(int i=0; i<200; i++){
        char prefix[256];
        sprintf(prefix,"%06d_10",i);
        std::string leftImageName = ImageFolderDir + "/image_2/" + prefix + ".png";
        std::string rightImageName = ImageFolderDir + "/image_3/" + prefix + ".png";
        char *leftImage = new char[leftImageName.length()+1];
        std::strcpy(leftImage,leftImageName.c_str());
        char *rightImage = new char[rightImageName.length()+1];
        std::strcpy(rightImage,rightImageName.c_str());        
        imgInputL[i] = xf::imread<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(leftImage, 0);
        imgInputR[i] = xf::imread<XF_8UC1, HEIGHT, WIDTH, XF_NPPC1>(rightImage, 0);

        delete [] leftImage;
        delete [] rightImage;
    }


#if __SDSCC__
	perf_counter hw_ctr;
	hw_ctr.start();
#endif

for(int j=0; j<5; j++){
	for(int i=0; i<200; i++){
		/* For 4 paths aggregation */
		semiglobalbm_accel(imgInputL[i],imgInputR[i],imgOutput[i]);
	}
}

#if __SDSCC__
	hw_ctr.stop();
	uint64_t hw_cycles = hw_ctr.avg_cpu_cycles();
#endif

    for(int i=0; i<200; i++){
        char prefix[256];
        sprintf(prefix,"%06d_10",i);
        std::string OutputImageName = ImageFolderDir + "/disparity/" + prefix + ".png";        
        xf::imwrite(OutputImageName.c_str(), imgOutput[i]);
    }
	std::cout<<"run success!"<<std::endl;

	return 0;
}
