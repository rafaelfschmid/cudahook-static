CXX=g++
CUDAPATH?=/usr/local/cuda-8.0
arch=-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60


UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
OPENCL_LIB = -framework OpenCL
OPENCL_INC =
all: libcudahook.so libclhook.so
endif

ifeq ($(UNAME), Linux)
.PHONY: .check-env
.check-env:
	@if [ ! -d "$(CLDIR)" ]; then \
		echo "ERROR: set CLDIR variable."; exit 1; \
	fi
OPENCL_LIB = -L$(CLDIR)/lib -lOpenCL
OPENCL_INC = -I $(CLDIR)/include
all: libcudahook.so main.exe #libclhook.so
endif

COMMONFLAGS=-Wall -fPIC -shared -ldl

libcudahook.so: cudahook.cpp
	$(CXX) -I$(CUDAPATH)/include $(COMMONFLAGS) -o libcudahook.so cudahook.cpp -std=c++11

libclhook.so: clhook.cpp
	$(CXX) $(OPENCL_INC) $(OPENCL_LIB) $(COMMONFLAGS) -o libclhook.so clhook.cpp

main.exe: main.cu
	nvcc $(arch) $+ -o $@ -I"../lib" -std=c++11 --expt-extended-lambda -lcudart -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)	

libclhook.dylib: clhook.cpp
	$(CXX) $(OPENCL_INC) $(OPENCL_LIB) -Wall -dynamiclib -o libclhook.dylib clhook.cpp

clean:
	-rm libcudahook.so main.exe scheduler.o
