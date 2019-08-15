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
#include "Scheduler.h"

#include <omp.h>
#include <mpi.h>

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
  if (!(hMyLib = MyLoadLib("/home/rafael/cuda-workspace/cudahook-static/src/libcudahook.so"))) { /*error*/ }
  if (!(scheduleKernels = (scheduleKernels_t)MyLoadProc(hMyLib, "scheduleKernels"))) { /*error*/ }

  bool ret = scheduleKernels(streams);

  MyUnloadLib(hMyLib);
}

/*std::map<std::string, cudaStream_t> &streams() {
  static std::map<std::string, cudaStream_t> _streams;
  return _streams;
}*/

struct Application {
	std::string command = "";
	std::map<std::string, int> _kernels;
	int time = 0;
};


void exec(const char* s){
	system(s);
}

int main(int argc, char **argv) {

	bip::shared_memory_object::remove("shared_memory");
	bip::managed_shared_memory segment(boost::interprocess::create_only, "shared_memory", 65536);

	// Index of threads
	int *id = segment.construct<int>("Index")(-1);
	SharedMap *kernels = segment.construct<SharedMap>("Kernels")( 3, boost::hash<ShmemString>(), std::equal_to<ShmemString>()
	        , segment.get_allocator<SharedMap>());

	int numOfStreams = 1;
	//cudaStream_t* streams = segment.construct<cudaStream_t>("Streams")[numOfStreams]();//;
	cudaStream_t* streams = segment.construct<cudaStream_t>("Streams")();//;

	//callcudahook(numOfStreams);

	std::ifstream f_in;
	f_in.open("../utils/kernel.txt");

	if (!f_in) {
	    std::cout << "Unable to open file ";
	    exit(1);   // call system to stop
	}

	std::vector<Application> commands;

	std::string line = " ";
	while(std::getline(f_in, line)) {
		//std::string command = line;
		Application app;
		app.command = line;

		//std::cout << command << "\n"; //printf teste
		f_in >> line;
		int count = std::stoi(line);
		for(int i = 0; i < count; i++) {
			f_in >> line;
			int id = std::stoi(line);
			//std::cout << id << " "; //printf teste
			f_in >> line;
			std::string kernel = line;
			//std::cout << kernel << " "; //printf teste
			f_in >> line;
			int time = std::stoi(line);
			//std::cout << time << "\n"; //printf teste

			app._kernels.insert(std::make_pair(kernel, time));
			app.time += time;
		}

		commands.push_back(app);

		std::getline(f_in, line);
	}


/*	for(Application app : commands){
		std::cout << app.command << "\n";

		for(auto& t : app._kernels)
		{
			std::cout << t.first << " " << t.second << "\n";
		}
		std::cout << "Tempo total = " << app.time << "\n";
	}
*/

//	streams = segment.find<cudaStream_t>("Streams").first;
	kernels = segment.find<SharedMap>("Kernels").first;

/*	for (int i = 0; i < numOfStreams; i++) {
		cudaStreamCreate(&streams[i]);
	}



	for(Application app : commands){
		int streamId = 0;
		for(auto& t : app._kernels)
		{
			CharAllocator alloc(segment.get_allocator<char>());
			std::string s(t.first);
			ShmemString str(s.data(), alloc);
			//kernels->insert(ValueType(str, &streams[streamId]));
			kernels->insert(ValueType(str, streams[streamId]));
		}

		streamId = (streamId+1) % numOfStreams;
	}*/

	/*for(SharedMap::iterator iter = kernels->begin(); iter != kernels->end(); iter++)
	{
		printf("%s --- %s\n", iter->first.data(), iter->second);
	}*/

	char message[20];
	int myrank, tag=99;
	MPI_Status status;

	/* Initialize the MPI library */
	MPI_Init(&argc, &argv);
	/* Determine unique id of the calling process of all processes participating
	   in this MPI program. This id is usually called MPI rank. */
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	if (myrank == 0) {
		strcpy(message, "Hello, there");
		/* Send the message "Hello, there" from the process with rank 0 to the
		   process with rank 1. */
		MPI_Send(message, strlen(message)+1, MPI_CHAR, 1, tag, MPI_COMM_WORLD);
	} else {
		/* Receive a message with a maximum length of 20 characters from process
		   with rank 0. */
		MPI_Recv(message, 20, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);
		printf("received %s\n", message);
	}

	/* Finalize the MPI library to free resources acquired by it. */
	MPI_Finalize();


	/*int nstreams = 4;
	omp_set_num_threads(nstreams);
	#pragma omp parallel
	{
		uint id = omp_get_thread_num(); //cpu_thread_id
		for (int i = 0; i < commands.size(); i+=nstreams) {
			uint k = i + id;
			std::cout << commands[k].command << "\n";
			exec(commands[k].command.data());
		}
	}*/

		//vec.push_back(std::async(std::launch::async,exec,commands[k].command.data()));
	//}

	for(auto& k : vec){
		//k.get();
	}


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
