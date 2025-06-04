## To Run The Code

```
$ bash build.sh [file name for train data] [file name for test data]

Or

make all train=[file name for train data] test=[file name for test data]

For example,
Train set file name= gene.train
Test set file name: gene.test

Run the following:
$ bash build.sh gene.train gene.test

Or

make all train=gene.train test=gene.test
``` 

>> Wrote CUDA implementation of the important functions like feed-forward run, delta and MSE computations, back-propagation run, weight update for [FANN](https://github.com/libfann/fann) to show the effect of each optimisation on performance; compared with the serial/OpenMP implementation to check that the functionality is correct; used the datasets given in its git repo https://github.com/libfann/fann/tree/master/datasets as the benchmarks. 
