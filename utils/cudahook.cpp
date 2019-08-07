#include <thread>
//#include <algorithm>
#include <chrono>

#include "cudahook.h"

#define DEBUG 0


typedef cudaError_t (*cudaEventCreate_t)(cudaEvent_t*);
static cudaEventCreate_t realCudaEventCreate = NULL;

extern "C" cudaError_t cudaEventCreate(cudaEvent_t *event) {
	if (realCudaEventCreate == NULL)
		realCudaEventCreate = (cudaEventCreate_t) dlsym(RTLD_NEXT,
				"cudaEventCreate");

	assert(realCudaEventCreate != NULL && "cudaEventCreate is null");

	return realCudaEventCreate(event);
}

typedef cudaError_t (*cudaEventRecord_t)(cudaEvent_t);
static cudaEventRecord_t realCudaEventRecord = NULL;

extern "C" cudaError_t cudaEventRecord(cudaEvent_t event) {
	if (realCudaEventRecord == NULL)
		realCudaEventRecord = (cudaEventRecord_t) dlsym(RTLD_NEXT,
				"cudaEventRecord");

	assert(realCudaEventRecord != NULL && "cudaEventRecord is null");

	return realCudaEventRecord(event);
}

typedef cudaError_t (*cudaEventSynchronize_t)(cudaEvent_t);
static cudaEventSynchronize_t realCudaEventSynchronize = NULL;

extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t event) {
	if (realCudaEventSynchronize == NULL)
		realCudaEventSynchronize = (cudaEventSynchronize_t) dlsym(RTLD_NEXT,
				"cudaEventSynchronize");

	assert(realCudaEventSynchronize != NULL && "cudaEventSynchronize is null");

	return realCudaEventSynchronize(event);
}

typedef cudaError_t (*cudaEventElapsedTime_t)(float *, cudaEvent_t, cudaEvent_t);
static cudaEventElapsedTime_t realCudaEventElapsedTime = NULL;

extern "C" cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
	if (realCudaEventElapsedTime == NULL)
		realCudaEventElapsedTime = (cudaEventElapsedTime_t) dlsym(RTLD_NEXT,
				"cudaEventElapsedTime");

	assert(realCudaEventElapsedTime != NULL && "cudaEventElapsedTime is null");

	return realCudaEventElapsedTime(ms, start, end);
}



typedef cudaError_t (*cudaFuncGetAttributes_t)(struct cudaFuncAttributes *,	const void *);
static cudaFuncGetAttributes_t realCudaFuncGetAttributes = NULL;

extern "C" cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func) {

	if (realCudaFuncGetAttributes == NULL)
		realCudaFuncGetAttributes = (cudaFuncGetAttributes_t) dlsym(RTLD_NEXT,
				"cudaFuncGetAttributes");

	assert(realCudaFuncGetAttributes != NULL && "cudaFuncGetAttributes is null");

	return realCudaFuncGetAttributes(attr, func);
}

typedef cudaError_t (*cudaGetDeviceProperties_t)(struct cudaDeviceProp *prop, int device);
static cudaGetDeviceProperties_t realCudaGetDeviceProperties = NULL;

extern "C" cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop,	int device) {

	if (realCudaGetDeviceProperties == NULL)
		realCudaGetDeviceProperties = (cudaGetDeviceProperties_t) dlsym(RTLD_NEXT, "cudaGetDeviceProperties");

	assert(realCudaGetDeviceProperties != NULL && "cudaGetDeviceProperties is null");

	auto ret = realCudaGetDeviceProperties(prop, device);

	deviceInfo().numOfSMs = prop->multiProcessorCount;
	deviceInfo().numOfRegister = prop->regsPerMultiprocessor;
	deviceInfo().sharedMemory = prop->sharedMemPerMultiprocessor;
	deviceInfo().maxThreads = prop->maxThreadsPerMultiProcessor;
	devices().push_back(deviceInfo());

	return ret;
}

typedef cudaError_t (*cudaStreamCreate_t)(cudaStream_t *pStream);
static cudaStreamCreate_t realCudaStreamCreate = NULL;

extern "C" cudaError_t cudaStreamCreate(cudaStream_t *pStream) {

	if (realCudaStreamCreate == NULL)
		realCudaStreamCreate = (cudaStreamCreate_t) dlsym(RTLD_NEXT, "cudaStreamCreate");

	assert(realCudaStreamCreate != NULL && "cudaStreamCreate is null");

	return realCudaStreamCreate(pStream);
}

typedef cudaError_t (*cudaFree_t)(void *devPtr);
static cudaFree_t realCudaFree = NULL;

extern "C" cudaError_t cudaFree(void *devPtr) {

	if (realCudaFree == NULL)
		realCudaFree = (cudaFree_t) dlsym(RTLD_NEXT, "cudaFree");

	assert(realCudaFree != NULL && "cudaFree is null");

	return realCudaFree(devPtr);
}

void printDevices() {
	for(auto d : devices()) {
		printf("##################################################\n");
		printf("numOfSMs=%d\n", d.numOfSMs);
		printf("numOfRegister=%d\n", d.numOfRegister);
		printf("sharedMemory=%d\n", d.sharedMemory);
		printf("maxThreads=%d\n", d.maxThreads);
		printf("##################################################\n");
	}
}

/*void printKernels() {
	for(auto k : kernels()) {
		printf("##################################################\n");
		//printf("entry=%d\n", k.entry);
		printf("numOfBlocks=%d\n", k.numOfBlocks);
		printf("numOfThreads=%d\n", k.numOfThreads);
		printf("numOfRegisters=%d\n", k.numOfRegisters);
		printf("sharedMemory=%d\n", k.sharedDynamicMemory);
		printf("sharedMemory=%d\n", k.sharedStaticMemory);
		//printf("computationalTime=%d\n", k.computationalTime);
		printf("##################################################\n");
	}
}*/

/*extern "C" bool scheduleKernels(int num_streams) {
	bip::managed_shared_memory segment(bip::open_only, "shared_memory");

	SharedMap* kernels = segment.find<SharedMap>("Kernels").first;

	cudaStream_t* streams = new cudaStream_t[num_streams];
	for (int i = 0; i < num_streams; i++) {
		cudaStreamCreate(&streams[i]);
	}

	int s = 0;
	int count = 0;
	while(count < kernels->size()) {
		std::vector<MapKey> resp;
		//std::vector<std::string> resp;

		//schedule(kernels, resp);

		for(MapKey& i : resp) {
			kernels->at(i).stream = streams[s];
			s = (s+1) % num_streams;
			count++;
		}

		s=0;
	}

	cudaFree(streams);
	return true;
}*/

typedef cudaError_t (*cudaConfigureCall_t)(dim3, dim3, size_t, cudaStream_t);
static cudaConfigureCall_t realCudaConfigureCall = NULL;

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0) {
	if(DEBUG)
		printf("TESTE 1\n");

	kernelInfo().sharedDynamicMemory = sharedMem;
	kernelInfo().numOfThreads = blockDim.x * blockDim.y * blockDim.z;
	kernelInfo().numOfBlocks = gridDim.x * gridDim.y * gridDim.z;

	//std::this_thread::sleep_for(std::chrono::seconds(2));
	if (realCudaConfigureCall == NULL)
		realCudaConfigureCall = (cudaConfigureCall_t) dlsym(RTLD_NEXT, "cudaConfigureCall");

	assert(realCudaConfigureCall != NULL && "cudaConfigureCall is null");
	return realCudaConfigureCall(gridDim, blockDim, sharedMem, stream);

}

typedef cudaError_t (*cudaLaunch_t)(const char *);
static cudaLaunch_t realCudaLaunch = NULL;

extern "C" cudaError_t cudaLaunch(const char *entry) {

	if(DEBUG)
		printf("TESTE 3\n");

	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, (void*) entry);

	bip::managed_shared_memory segment(bip::open_only, "shared_memory");

	SharedMap* 			kernels = segment.find<SharedMap>("Kernels").first;
	int* 				index 	= segment.find<int>("Index").first;

	kernelInfo_t k;
	k.sharedDynamicMemory = kernelInfo().sharedDynamicMemory;
	k.numOfThreads = kernelInfo().numOfThreads;
	k.numOfBlocks = kernelInfo().numOfBlocks;
	k.numOfRegisters = attr.numRegs;
	k.sharedStaticMemory = attr.sharedSizeBytes;
	k.start = false;
	k.id = *index = (*index) + 1;

	if (realCudaLaunch == NULL) {
		realCudaLaunch = (cudaLaunch_t) dlsym(RTLD_NEXT, "cudaLaunch");
	}
	assert(realCudaLaunch != NULL && "cudaLaunch is null");

    auto start = std::chrono::high_resolution_clock::now();
    cudaError_t ret = realCudaLaunch(entry);
	auto finish = std::chrono::high_resolution_clock::now();
	auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(finish-start);
	//printf("microseconds=%d µs\n", microseconds.count());
	//printf("microseconds=%d\n", microseconds.count());

	k.microseconds = (float)microseconds.count();

	CharAllocator alloc(segment.get_allocator<char>());
	std::string s(kernelsMap()[entry] + std::to_string(k.id));
	ShmemString str(s.data(), alloc);
	kernels->insert(ValueType(str, k));

	return ret;
}

typedef void (*cudaRegisterFunction_t)(void **, const char *, char *,
		const char *, int, uint3 *, uint3 *, dim3 *, dim3 *, int *);
static cudaRegisterFunction_t realCudaRegisterFunction = NULL;

extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
		const char *hostFun, char *deviceFun, const char *deviceName,
		int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim,
		int *wSize) {
	if(DEBUG)
		printf("TESTE 0n");

	kernelsMap()[hostFun] = deviceFun;

	if (realCudaRegisterFunction == NULL) {
		realCudaRegisterFunction = (cudaRegisterFunction_t) dlsym(RTLD_NEXT,
				"__cudaRegisterFunction");
	}
	assert(realCudaRegisterFunction != NULL && "cudaRegisterFunction is null");

	realCudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName,
			thread_limit, tid, bid, bDim, gDim, wSize);
}

typedef cudaError_t (*cudaSetupArgument_t)(const void *, size_t, size_t);
static cudaSetupArgument_t realCudaSetupArgument = NULL;

extern "C" cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
	if(DEBUG)
		printf("TESTE 2\n");

	//kernelInfo().args.push_back(const_cast<void *>(arg));
	if (realCudaSetupArgument == NULL) {
		realCudaSetupArgument = (cudaSetupArgument_t) dlsym(RTLD_NEXT,
				"cudaSetupArgument");
	}
	assert(realCudaSetupArgument != NULL);
	return realCudaSetupArgument(arg, size, offset);
}

