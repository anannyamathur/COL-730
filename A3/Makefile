all: part1 part2 part3

part1: mr-pr-cpp.cpp 
	g++ -Wall -std=c++11 -I include/detail -I include -o mr-pr-cpp.o mr-pr-cpp.cpp -lboost_system -lpthread -lboost_iostreams -lboost_filesystem

part2: mr-pr-mpi.cpp 
	mpic++ -std=c++11 mr-pr-mpi.cpp -o mr-pr-mpi.o

part3: mr-pr-mpi-base.cpp 
	mpic++ -std=c++11  -I src -o mr-pr-mpi-base.o  mr-pr-mpi-base.cpp src/libmrmpi_mpicc.a

