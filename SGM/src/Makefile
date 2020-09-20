APPSOURCES = fp_sgbm_accel.cpp fp_sgbm_tb.cpp 
EXECUTABLE = fpstereo.elf

PLATFORM = #PATH_TO_ZCU_REVISION_PLATFORM/zcu102-rv-min-2018-3/zcu102_rv_min

HW_FUNC = "fp::SemiGlobalBM<${WINDOW_SIZE},${SHD_WINDOW},${NUM_DISPARITY},${PARALLEL_DISPARITIES},${FilterWin},0,0,${HEIGHT},${WIDTH},1,${SMALL_PENALTY},${LARGE_PENALTY}>"


SDSFLAGS = -sds-pf ${PLATFORM} -sds-sys-config a53_linux -sds-proc a53_linux -sds-hw ${HW_FUNC} ../src/fp_sgbm_accel.cpp -files ../src/lib_accel/fp_sgbm.hpp -clkid 4 -sds-end -dmclkid 4

CC = sds++ ${SDSFLAGS} -perf-root main -perf-est swdata.xml -perf-funcs SemiGlobalBM
#CC = sds++ ${SDSFLAGS} -impl-strategy Performance_ExplorePostRoutePhysOpt -maxjobs 8 -maxthreads 8

# Set SDS++ Compiler 
IDIRS = -I../src -I${SYSROOT}/usr/include -I${SYSROOT}/../../inc/xfopencv
CFLAGS = -O3 -Wall -c ${IDIRS} -DHLS_NO_XIL_FPO_LIB -mstrict-align -hls-target 1

# Set SDS++ Linker
LDIRS = --sysroot=${SYSROOT} -L=/lib -L=/usr/lib -Wl,-rpath-link=${SYSROOT}/lib,-rpath-link=${SYSROOT}/usr/lib
LLIBS = -lopencv_imgcodecs -lopencv_core -lopencv_imgproc -lopencv_calib3d -lopencv_features2d -lopencv_flann -llzma -ltiff -lpng16 -lz -ljpeg -ldl -lrt -lwebp
LFLAGS = ${LDIRS} ${LLIBS}


SOURCES := $(patsubst %,../src/%,$(APPSOURCES))
OBJECTS := $(APPSOURCES:.cpp=.o)
DEPS := $(OBJECTS:.o=.d)
.PHONY: all
all: ${EXECUTABLE}
${EXECUTABLE}: ${OBJECTS}
	${CC} ${OBJECTS} ${LFLAGS} -o $@
-include ${DEPS}
%.o: ../src/%.cpp
	${CC} ${CFLAGS} $< -o $@
clean:
	${RM} ${EXECUTABLE} ${OBJECTS}
ultraclean: clean
	${RM} -rf _sds sd_card ${EXECUTABLE}.bit ${EXECUTABLE} ${OBJECTS}

 

