all: fann; ./fann $(train) $(test)
    
fann: fann.cu; nvcc fann.cu -o fann

