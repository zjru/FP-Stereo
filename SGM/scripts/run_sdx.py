#  FP-Stereo
#  Copyright (C) 2020  RCSL, HKUST
#  GPL-3.0 License

import subprocess
import os
import shutil
import tempfile
import sys

# set the paths to folders of source code and results  
FP_Stereo = #PATH_TO_FP_STEREO_SRC_FOLDER
DSE_FOLDER = #PATH_TO_HLS_RESULTS_FOLDER 

# set the environmental variable
os.environ["SYSROOT"] = "PATH_TO_ZCU102_REVISION_LOCATION/zcu102-rv-min-2018-3/zcu102_rv_min/sw/a53_linux/a53_linux/sysroot/aarch64-xilinx-linux"

workspace = DSE_FOLDER + "sdx_workspace"
subprocess.call(["mkdir", "-p", workspace])

# height = 374
# width = 1242
# cost_function = 0
# window_size = 5
# max_disp = 64
# parallel_disp = 16
# num_dir = 4
# uniqueness = 0
# lr_check = 0
# filter_win = 5
# shd_window = 3
# p1 = 7
# p2 = 86

height = sys.argv[1]
width = sys.argv[2]
cost_function = sys.argv[3]
window_size = sys.argv[4]
max_disp = sys.argv[5]
parallel_disp = sys.argv[6]
num_dir = sys.argv[7]
uniqueness = sys.argv[8]
lr_check = sys.argv[9]
filter_win = sys.argv[10]
shd_window = sys.argv[11]
p1 = sys.argv[12]
p2 = sys.argv[13]

configuration = workspace + "/" + str(height) + "_" + str(width) + "_" + str(max_disp) + "_" + str(parallel_disp) + "_" + str(num_dir) + "_" + str(p1) + "_" + str(p2) + "_" + str(cost_function) + "_" + str(window_size) + "_" + str(filter_win) + "_" + str(shd_window) + "_" + str(uniqueness) + "_" + str(lr_check)
subprocess.call(["mkdir", "-p", configuration])

# copy source code to HLS project
src = configuration + "/" + "src"
subprocess.call(["mkdir", "-p", src])
shutil.copy(FP_Stereo+'fp_headers.h',src)
shutil.copy(FP_Stereo+'fp_sgbm_accel.h',src)
shutil.copy(FP_Stereo+'fp_sgbm_accel.cpp',src)
shutil.copy(FP_Stereo+'fp_sgbm_tb.cpp',src)
shutil.copy(FP_Stereo+'fp_config_params.h',src)
shutil.copy(FP_Stereo+'fp_config_arch.h',src)

KEYWORDS = ["HEIGHT", "WIDTH", "NUM_DISPARITY", "SMALL_PENALTY", "LARGE_PENALTY", "WINDOW_SIZE", "SHD_WINDOW", "FilterWin", "PARALLEL_DISPARITIES"]
VALUES = [height, width, max_disp, p1, p2, window_size, shd_window, filter_win, parallel_disp]

ARCH_KEYWORDS = ["NUM_DIR", "COST_FUNCTION", "UNIQ", "LR_CHECK", "MAX_PORT_BW", "PARALLELISM", "PENALTY2", "COST_WIN", "SHD_WIN"]
ARCH_VALUES = [num_dir, cost_function, uniqueness, lr_check, 128, parallel_disp, p2, window_size, shd_window]

# update parameter setting
def update_configuration(file_path, keywords, values):
    temp, abs_path = tempfile.mkstemp()
    with open(abs_path,'w') as update_file, open(file_path) as orig_file:
        for line in orig_file:
            words = line.split()
            if line.startswith('#define ') and words[1] in keywords:
                for i in range(len(keywords)):
                    if keywords[i] == words[1]:
                        keyword = keywords[i]
                        value = values[i]
                        update_file.write('#define ' + keyword + ' ' + str(value) + '\n')            
            else:
                update_file.write(line)

    os.remove(file_path)
    shutil.move(abs_path,file_path)

update_configuration(src+'/fp_config_params.h',KEYWORDS,VALUES)
update_configuration(src+'/fp_config_arch.h',ARCH_KEYWORDS,ARCH_VALUES)

lib_accel = src + "/" + "lib_accel"
subprocess.call(["mkdir", "-p", lib_accel])
shutil.copy(FP_Stereo+'lib_accel/fp_AggregateCost.hpp',lib_accel)
shutil.copy(FP_Stereo+'lib_accel/fp_common.h',lib_accel)
shutil.copy(FP_Stereo+'lib_accel/fp_ComputeCost.hpp',lib_accel)
shutil.copy(FP_Stereo+'lib_accel/fp_ComputeDisparity.hpp',lib_accel)
shutil.copy(FP_Stereo+'lib_accel/fp_PostProcessing.hpp',lib_accel)
shutil.copy(FP_Stereo+'lib_accel/fp_sgbm.hpp',lib_accel)

build_folder = configuration + "/" + "build"
subprocess.call(["mkdir", "-p", build_folder])
os.chdir(build_folder)
shutil.copy(FP_Stereo+'Makefile',build_folder)

# invoke HLS tool for synthesis
subprocess.call(["make", "NUM_DIR="+str(num_dir), "WINDOW_SIZE="+str(window_size), "SHD_WINDOW="+str(shd_window), "NUM_DISPARITY="+str(max_disp), "PARALLEL_DISPARITIES="+str(parallel_disp), "FilterWin="+str(filter_win), "HEIGHT="+str(height), "WIDTH="+str(width), "SMALL_PENALTY="+str(p1), "LARGE_PENALTY="+str(p2)])
