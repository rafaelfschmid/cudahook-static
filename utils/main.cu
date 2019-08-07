#include <stdio.h>
#include <stdlib.h>
//#include <cuda.h>
//#include <cuda_profiler_api.h>
#include <iostream>
#include <fstream>

#include <vector>
#include <thread>
#include <future>
#include <string.h>

#include <unistd.h>
#include <dlfcn.h>
#include <signal.h>

#include "cudahook.h"

typedef void* my_lib_t;

my_lib_t MyLoadLib(const char* szMyLib) {
	return dlopen(szMyLib, RTLD_LAZY);
}

void MyUnloadLib(my_lib_t hMyLib) {
	dlclose(hMyLib);
}

void* MyLoadProc(my_lib_t hMyLib, const char* szMyProc) {
	return dlsym(hMyLib, szMyProc);
}

typedef bool (*scheduleKernels_t)(int);
my_lib_t hMyLib = NULL;
scheduleKernels_t scheduleKernels = NULL;

void callcudahook(int streams) {
  if (!(hMyLib = MyLoadLib("/home/rafael/cuda-workspace/cudahook-static/libcudahook.so"))) { /*error*/ }
  if (!(scheduleKernels = (scheduleKernels_t)MyLoadProc(hMyLib, "scheduleKernels"))) { /*error*/ }

  bool ret = scheduleKernels(streams);

  MyUnloadLib(hMyLib);
}


void exec(const char* s){
	system(s);
}

int main(int argc, char **argv) {

	bip::shared_memory_object::remove("shared_memory");
	bip::managed_shared_memory segment(boost::interprocess::create_only, "shared_memory", 65536);

	// Index of threads
	int *id = segment.construct<int>("Index")(-1);
	// Shared map of kernels
	//SharedMap *kernels =  segment.construct<SharedMap>("Kernels") (std::less<MapKey>() ,segment.get_segment_manager());


	SharedMap *kernels = segment.construct<SharedMap>("Kernels")( 3, boost::hash<ShmemString>(), std::equal_to<ShmemString>()
	        , segment.get_allocator<SharedMap>());


	std::ofstream f_out;
	f_out.open("kernels.txt", std::ios::app);

	std::string line = "";
	std::getline(std::cin, line);
	f_out << line.data() << "\n";
	exec(line.data());

	f_out << kernels->size() << "\n";
	for(SharedMap::iterator iter = kernels->begin(); iter != kernels->end(); iter++)
	{
		//printf("%d %s %f\n", iter->second.id, iter->first.data(), iter->second.microseconds);
		f_out << iter->second.id << " " << iter->first.data() << " " << iter->second.microseconds << "\n";
	}
	f_out << "\n";
	f_out.close();
	//callcudahook(2);

	/*
	 * Concurrent execution
	 */
	/*std::vector<std::future<void>> vec;
	std::getline (std::cin, line1);
	vec.push_back(std::async(std::launch::async,exec,line1.data()));

	std::getline (std::cin, line2);
	vec.push_back(std::async(std::launch::async,exec,line2.data()));*/

/*	printf("começoooouuuu\n");
	bool test = callcudahook(2, 2);
	printf("acaboooouuuu\n");

	vec[0].get();
	vec[1].get();
	vec[2].get();
*/


	bip::shared_memory_object::remove("shared_memory");

	return 0;
}


/*std::vector<std::future<void>> vec;

	std::string line1 = "";
	std::string line2 = "";
	std::string line3 = "";
	std::string line4 = "";
	std::string line5 = "";
	std::string line6 = "";
	std::string line7 = "";
	std::string line8 = "";

	std::getline (std::cin, line1);
	std::getline (std::cin, line2);
	std::getline (std::cin, line3);
	std::getline (std::cin, line4);


	std::vector<char*> commandVector;
	commandVector.push_back(const_cast<char*>(line2.data()));
	commandVector.push_back(const_cast<char*>(line3.data()));
	commandVector.push_back(const_cast<char*>(line4.data()));
	commandVector.push_back(NULL);
	//const int status = execvp(commandVector[0], &commandVector[0]);
	//exec(commandVector[0], commandVector);
	myclass a(commandVector[0], commandVector);

	std::vector<char*> commandVector2;
	//commandVector2.push_back(const_cast<char*>(line1.data()));
	std::getline (std::cin, line1);
	std::getline (std::cin, line2);
	std::getline (std::cin, line3);
	std::getline (std::cin, line4);
	std::getline (std::cin, line5);
	std::getline (std::cin, line6);
	std::getline (std::cin, line7);
	std::getline (std::cin, line8);

	commandVector2.push_back(const_cast<char*>(line2.data()));
	commandVector2.push_back(const_cast<char*>(line3.data()));
	commandVector2.push_back(const_cast<char*>(line4.data()));
	commandVector2.push_back(const_cast<char*>(line5.data()));
	commandVector2.push_back(const_cast<char*>(line6.data()));
	commandVector2.push_back(const_cast<char*>(line7.data()));
	commandVector2.push_back(const_cast<char*>(line8.data()));
	commandVector2.push_back(NULL);
	//const int status = execvp(commandVector[0], &commandVector[0]);
	//exec(commandVector2[0], commandVector2);
	myclass b(commandVector2[0], commandVector2);*/
