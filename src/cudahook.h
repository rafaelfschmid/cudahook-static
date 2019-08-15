
#include <stdio.h>
#include <stdlib.h>  /* exit */

#include <list>
#include <cuda.h>
#include <vector_types.h>
#include <vector>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/unordered_map.hpp>

#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable

#include <dlfcn.h>
#include <cassert>

#include <semaphore.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <unistd.h>  /* _exit, fork */
#include <errno.h>   /* errno */
#include <signal.h>
#include <string.h>

namespace bip = boost::interprocess;

typedef struct {
	char* entry;
	int id = -1;
	//dim3 gridDim;
	//dim3 blockDim;
	int numOfBlocks;
	int numOfThreads;
	int numOfRegisters;
	int sharedDynamicMemory;
	int sharedStaticMemory;
	cudaStream_t stream;
	float microseconds;
	bool start = false;
} kernelInfo_t;

kernelInfo_t &kernelInfo() {
	static kernelInfo_t _kernelInfo;
	return _kernelInfo;
}

std::map<const char *, char *> &kernelsMap() {
  static std::map<const char*, char*> _kernels;
  return _kernels;
}

/*std::vector<kernelInfo_t> &kernels() {
	static std::vector<kernelInfo_t> _kernels;
	return _kernels;
}*/

typedef struct {
	int numOfSMs;
	int numOfRegister; // register per SM
	int maxThreads;    // max threads per SM
	int sharedMemory;  // sharedMemory per SM
} deviceInfo_t;

deviceInfo_t &deviceInfo() {
	static deviceInfo_t _deviceInfo;
	return _deviceInfo;
}

std::vector<deviceInfo_t> &devices() {
	static std::vector<deviceInfo_t> _devices;
	return _devices;
}

typedef bip::allocator<char, bip::managed_shared_memory::segment_manager> CharAllocator;
typedef bip::basic_string<char, std::char_traits<char>, CharAllocator> ShmemString;

//typedef bip::allocator<cudaStream_t, bip::managed_shared_memory::segment_manager> StreamAllocator;
//typedef bip::basic_string<char, std::char_traits<char>, CharAllocator> ShmemString;

typedef ShmemString MapKey;
typedef cudaStream_t MapValue;

typedef std::pair< MapKey, MapValue> ValueType;

//allocator of for the map.
typedef bip::allocator<ValueType, bip::managed_shared_memory::segment_manager> ShMemAllocator;
typedef boost::unordered_map< MapKey, MapValue, boost::hash<MapKey>, std::equal_to<MapKey>, ShMemAllocator > SharedMap;

