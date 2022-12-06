## To compile the executables

```
$ make

``` 

## To Run The Code

```
$ ./mr-pr-cpp.o ${filename}.txt -o ${filename}-pr-cpp.txt
$ ./mr-pr-mpi.o ${filename}.txt -o ${filename}-pr-mpi.txt
$ ./mr-pr-mpi-base.o ${filename}.txt -o ${filename}-pr-mpi-base.txt

For example,
Name of the file= barabasi-20000

Run the following:
$ ./mr-pr-cpp.o barabasi-20000.txt -o barabasi-20000-pr-cpp.txt
``` 

Or

```
$ bash build.sh [file name] 

For example,
Name of the file= barabasi-20000

Run the following:
$ bash build.sh barabasi-20000


``` 