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

#include <omp.h>
#include <mpi.h>

struct Application {
	std::string command = "";
	//std::map<std::string, int> _kernels;
	int time = 0;
	int stream;
};

void min_min(std::vector<Application>& commands, std::vector<std::vector<int> >& machines) {

	float *completion_times = (float *) malloc(sizeof(float) * (machines.size()));
	for (int j = 0; j < machines.size(); j++) {
		completion_times[j] = 0;
	}

	uint jmin = 0;
	float min_value = 0;

	for (int i = 0; i < commands.size(); i++) {

		min_value = std::numeric_limits<float>::max();

		for (int j = 0; j < machines.size(); j++) {
			if (completion_times[j] + commands[i].time < min_value) {
			//	printf("j=%d\n", j);
				jmin = j;
				min_value = completion_times[jmin] + commands[i].time;
			}
		}
		//printf("taskid=%d scheduled in machine %d\n", i, jmin);
		commands[i].stream = jmin;
		completion_times[jmin] = min_value;
		machines[jmin].push_back(i);
	}
}

void exec(const char* s){
	system(s);
}

int main(int argc, char **argv) {

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

			//app._kernels.insert(std::make_pair(kernel, time));
			app.time += time;
		}

		commands.push_back(app);

		std::getline(f_in, line);
	}

	int n = commands.size();
	for (int i = 0; i < n; i++) {
		commands.push_back(commands[i]);
	}

	char message[20];
	int myrank, nstream, tag=99;
	MPI_Status status;

	/* Initialize the MPI library */
	MPI_Init(&argc, &argv);
	/* Determine unique id of the calling process of all processes participating
	   in this MPI program. This id is usually called MPI rank. */
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &nstream);

	std::vector<std::vector<int> > machines(nstream);
	//printf("nstream=%d\n", nstream);
	if(myrank == 0) {
		min_min(commands, machines);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	for (int i = 0; i < machines[myrank].size(); i++) {
		printf("myrank=%d ---- task=%d\n\n", myrank, machines[myrank][i]);
		exec(commands[machines[myrank][i]].command.data());
	}

/*	for (int i = 0; i < commands.size(); i++) {
		//printf("myrank=%d ---- stream=%d\n\n", myrank, commands[i].stream);
		if(myrank == commands[i].stream) {
			printf("myrank=%d\n", myrank);
			std::cout << commands[i].command << "\n";
			exec(commands[i].command.data());
		}

	}*/

	/* Finalize the MPI library to free resources acquired by it. */
	MPI_Finalize();

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
