#include <thread>
//#include <algorithm>
#include <chrono>

#include "cudahook.h"





#define DEBUG 0

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

//void knapsack(std::vector<std::vector<int>>& tab, int itens, int pesoTotal){
void knapsack(int ** tab, SharedMap* kernels, int itens, int pesoTotal){

	int item = 1;
	for(SharedMap::iterator iter = kernels->begin(); iter != kernels->end(); iter++)
	{
		for(int peso = 1; peso <= pesoTotal; peso++) {
			if(iter->second.start == true) {
				tab[item][peso] = tab[item-1][peso];
			}
			else {
				int pesoi = iter->second.numOfThreads;
				if(pesoi <= peso) {
					if(pesoi + tab[item-1][peso-pesoi] > tab[item-1][peso]) {
						tab[item][peso] = pesoi + tab[item-1][peso-pesoi];
					}
					else
						tab[item][peso] = tab[item-1][peso];
				}
				else {
					tab[item][peso] = tab[item-1][peso];
				}
			}
		}
		item++;

	}
}

void fill(int **tab, SharedMap* kernels, int itens, int pesoTotal, std::vector<int>& resp){

	SharedMap::iterator iter = kernels->end();
	iter--;

	// se já calculamos esse estado da dp, retornamos o resultado salvo
	while(itens > 0 && pesoTotal > 0) {
		if(tab[itens][pesoTotal] != tab[itens-1][pesoTotal])
		{
			pesoTotal = pesoTotal - iter->second.numOfThreads;
			//printf("iter->first=%d\n", iter->first);
			resp.push_back(iter->first);
		}
		iter--;
		itens--;
	}

}

void schedule(SharedMap* kernels, std::vector<int>& resp) {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	int peso = prop.maxThreadsPerMultiProcessor;
	int itens = kernels->size();

	int **tab = new int*[itens+1];
	for(int i = 0; i <= itens; i++) {
		tab[i] = new int[peso+1];
	}

	for(int i = 0; i <= itens; i++) {
		tab[i][0] = 0;
	}

	for(int j = 0; j <= peso; j++) {
		tab[0][j] = 0;
	}

	knapsack(tab, kernels, itens, peso);

	/*for(int i = 0; i <= itens; i++) {
		for(int j = 0; j <= peso; j++) {
			printf("%d ", tab[i][j]);
		}
		printf("\n");
	}*/

	fill(tab, kernels, itens, peso, resp);

	for(int j = 0; j <= itens; j++) {
		delete[] tab[j];
	}
	delete[] tab;
}

extern "C" bool scheduleKernels(int n, int num_streams) {
	bip::managed_shared_memory segment(bip::open_only, "shared_memory");

	SharedMap* kernels = segment.find<SharedMap>("Kernels").first;

	sem_t *sem_1 = sem_open("semaforo1", O_CREAT, S_IRUSR);
	while(kernels->size() != n)
		sem_wait(sem_1);

	cudaStream_t* streams = new cudaStream_t[num_streams];
	for (int i = 0; i < num_streams; i++) {
		cudaStreamCreate(&streams[i]);
	}

	//std::vector<int> respAll;
	while(true) {
		sem_t *sem_1 = sem_open("semaforo1", O_CREAT, S_IRUSR | S_IWUSR);
		std::vector<int> resp;
		{
			//sem_t *sem_1 = sem_open("semaforo1", O_CREAT, S_IRUSR);
			schedule(kernels, resp);
			//sem_post(sem_1);
		}

		int s = 0;
		for(int i : resp) {
			printf("sem_2.1 --- id=%d\n", i);
			//sem_t *sem_2 = sem_open("semaforo2", O_CREAT, S_IRUSR | S_IWUSR);
			printf("sem_2.2 --- id=%d\n", i);
			kernels->at(i).stream = streams[s];
			s = (s+1) % num_streams;
			kernels->at(i).start = true;
			//sem_post(sem_2);
		}

		for(int i : resp) {
			printf("sem_3.1 --- id=%d\n", i);
			//sem_t *sem_3 = sem_open("semaforo3", O_CREAT, S_IRUSR);
			printf("sem_3.2 --- id=%d\n", i);
			while(!kernels->at(i).finished);
				//sem_wait(sem_3);
			//kernels->erase(i);
			//sem_post(sem_3);
		}

		for(int i : resp) {
			printf("sem_4.1 --- id=%d\n", i);
			//sem_t *sem_3 = sem_open("semaforo3", O_CREAT, S_IRUSR | S_IWUSR);
			kernels->erase(i);
			printf("sem_4.2 --- id=%d\n", i);
			//sem_post(sem_3);
		}
		sem_post(sem_1);
	}

	cudaFree(streams);
	return true;
}

typedef cudaError_t (*cudaConfigureCall_t)(dim3, dim3, size_t, cudaStream_t);
static cudaConfigureCall_t realCudaConfigureCall = NULL;

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0) {
	if(DEBUG)
		printf("TESTE 1\n");

	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, (void*) kernelInfo().entry);

	bip::managed_shared_memory segment(bip::open_only, "shared_memory");

	SharedMap* 			kernels = segment.find<SharedMap>("Kernels").first;
	int* 				index 	= segment.find<int>("Index").first;

	kernelInfo_t k;
	k.sharedDynamicMemory = sharedMem;
	k.numOfThreads = blockDim.x * blockDim.y * blockDim.z;
	k.numOfBlocks = gridDim.x * gridDim.y * gridDim.z;
	k.numOfRegisters = attr.numRegs;
	k.sharedStaticMemory = attr.sharedSizeBytes;
	k.start = false;

	pid_t pid = getpid();
	printf("pidProcess = %d\n", pid);
	{
		printf("cuda --- sem_1.1 --- pid=%d\n", pid);
		sem_t *sem_1 = sem_open("semaforo1", O_CREAT, S_IRUSR | S_IWUSR);
		printf("cuda --- sem_1.2 --- pid=%d\n", pid);
		k.id = *index = (*index) + 1;
		k.pid = pid;
		kernels->insert(std::pair<const int, kernelInfo_t>(k.id, k));
		printf("cuda --- sem_1.3 --- id=%d\n", k.id);
		sem_post(sem_1);
	}

	{
		printf("Waiting... \n");
		//sem_t *sem_2 = sem_open("semaforo2", O_CREAT, S_IRUSR);
		printf("cuda -- sem_2 started ==> id=%d\n", k.id);
		while(!kernels->at(k.id).start);
			//sem_wait(sem_2);
		printf("cuda -- sem_2 finished ==> id=%d\n", k.id);
	}
	{


		//printf("%d...finished waiting.\n", k.id);
		//sem_t *sem_3 = sem_open("semaforo3", O_CREAT, S_IRUSR | S_IWUSR);
		printf("cuda -- sem_3 started ==> id=%d\n", k.id);
		//sem_t *sem_3 = sem_open("semaforo3", O_CREAT, S_IRUSR | S_IWUSR);
		kernels->at(k.id).finished = true;
		//sem_post(sem_3);
		printf("cuda -- sem_3 finished ==> id=%d\n", k.id);
	}

	//std::this_thread::sleep_for(std::chrono::seconds(2));
	if (realCudaConfigureCall == NULL)
		realCudaConfigureCall = (cudaConfigureCall_t) dlsym(RTLD_NEXT, "cudaConfigureCall");

	assert(realCudaConfigureCall != NULL && "cudaConfigureCall is null");
	return realCudaConfigureCall(gridDim, blockDim, sharedMem, stream);

}

typedef cudaError_t (*cudaLaunch_t)(const char *);
static cudaLaunch_t realCudaLaunch = NULL;

extern "C" cudaError_t cudaLaunch(const char *entry) {

	if (realCudaLaunch == NULL) {
		realCudaLaunch = (cudaLaunch_t) dlsym(RTLD_NEXT, "cudaLaunch");
	}
	assert(realCudaLaunch != NULL && "cudaLaunch is null");

	return realCudaLaunch(entry);
}

typedef void (*cudaRegisterFunction_t)(void **, const char *, char *,
		const char *, int, uint3 *, uint3 *, dim3 *, dim3 *, int *);
static cudaRegisterFunction_t realCudaRegisterFunction = NULL;

extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
		const char *hostFun, char *deviceFun, const char *deviceName,
		int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim,
		int *wSize) {

	kernelInfo().entry = hostFun;

	if(DEBUG)
		printf("TESTE 0\n");

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


/*{
	sem_1 = sem_open("semaforo1", O_CREAT, S_IRUSR | S_IWUSR);
	//if(kernels->size() > 0) {
		//printf("kernelsize=%d\n", kernels->size());
		/*for(int j = 0; j < kernels.size(); j++) {

			kill(kernels->at(j)->second.pid, 0);
			if(errno == ESRCH) {
				if(iter->first != 0 && iter->second.pid != 0){
				printf("ID %d finalizado\n", iter->second.id);
				printf("processo %d não existe\n", iter->second.pid);
				printf("kernels->size()=%d\n", kernels->size());
				}
				kernels->erase(iter->first);
			}
		}*/
