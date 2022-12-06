#include <vector>
#include <set>
#include <map>
#include <string>
#include <list>
#include <math.h>
#include <cstring>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
# include <time.h>

#include "mpi.h"
#include "mapreduce.h"
#include "keyvalue.h"

using namespace std;
using namespace MAPREDUCE_NS;

const double alpha = 0.85;
const double convergence = 0.000001;

vector<int> num_outgoing; // number of outgoing links per column
vector< vector<int> > rows; // the number of rows of the hyperlink matrix

vector<double> pr; // the pagerank table
size_t num_rows;
int n;
vector<double> old_pr;
int num_processors;

double one_Av=0.0;
double one_Iv=0.0; 

double diff;
double main_sum=0.0;

vector<double> h_;

void add_edge(int from, int to)
{
    size_t maximum_dimension=max(from,to);
    if (rows.size()<=maximum_dimension)
    {
        maximum_dimension+=1;
        rows.resize(maximum_dimension);
        if (num_outgoing.size()<maximum_dimension)
        {
            num_outgoing.resize(maximum_dimension);
        }
    }

    rows[to].push_back(from);
    num_outgoing[from]+=1;
    
}


void create_graph_from_inputfile(char* input)
{
    int from;
    int to;
    ifstream input_file(input);
    string line;
    
    while(input_file)
    {   
        input_file >> from >> to;
        
        add_edge(from,to);

    }   
    n=rows.size();
}

void maptask(int rank, KeyValue *kv, void *ptr)
{   
   
    int number_of_proc=(num_rows)/(num_processors); // number of processes in each processor
    
    for (int key=rank*number_of_proc; key<(rank+1)*number_of_proc; key++)
    {   
        int len=rows[key].size()*sizeof(int);
        kv->add((char*) &key, sizeof(int),(char*) &rows[key], len);
    }
}

void gather(char *key, int keybytes, char *multivalue, int nvalues, int *valuebytes, KeyValue *kv, void *ptr)
{   
   
    double h = 0.0;
    double difference=0.0;
    int index=*(int*)key;
    double h_duplicate=0.0;

    for (auto ci = rows[index].begin(); ci != rows[index].end(); ci++) 
    {   
            double h_v = 0.0;
            if (num_outgoing[*ci]>0.0)
            {
                h_v=1.0/num_outgoing[*ci];
            }
            h += h_v * old_pr[*ci];
        
    }
    h=h*alpha;
    h_[index] = h;
    h_duplicate=h_[index];
    
    kv->add(key,keybytes,(char*)&h,sizeof(double));

}

void output(uint64_t itask, char *key, int keybytes, char *value,
	    int valuebytes, KeyValue *kv, void *ptr)
{
  
  pr[*(int *)key] = *(double *)value + one_Av + one_Iv;
  
}

void pagerank_calculation() {
    
    diff = 1.0;
    double dangling_pr; // sum of current pagerank vector elements for dangling nodes
    double sum_pr; // sum of current pagerank vector elements
    int num_iterations = 0;
    
    num_rows = rows.size();
    
    if (num_rows == 0) {
        return;
    }

    pr.resize(num_rows,1.0);
    h_.resize(num_rows,0.0);
    old_pr=pr;

   
    int rank;
    int size;

    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    num_processors=size;
   
    MapReduce *mr=new MapReduce(MPI_COMM_WORLD);
    while (diff > convergence) 
    {   sum_pr = 0.0;
        dangling_pr = 0.0;
        diff=0.0;
        main_sum=0.0;
       
        if (rank==0)
        {   
            for (size_t to = 0; to < pr.size(); to++) 
            {
                sum_pr += pr[to];
                if (num_outgoing[to] == 0) 
                {
                    dangling_pr += pr[to];
                }

            }
        
            if (num_iterations>=1)
            {
                for (size_t to = 0; to < pr.size(); to++) 
                {
                    old_pr[to] = pr[to] / sum_pr;
                }
            }
        
            /* An element of the A x I vector; all elements are identical */
            one_Av = alpha * dangling_pr/num_rows;

            /* An element of the 1 x I vector; all elements are identical */
            one_Iv = (1 - alpha) * 1.0 / num_rows;
            
             

        }
         
    
        mr->map(size,maptask,NULL);
        mr->convert();
        mr->reduce(gather,NULL);
        mr->gather(1);
        mr->map(mr,output,NULL);
        
        for (int i=0; i<num_rows;i++)
        {
            diff+=pr[i]-old_pr[i];
            main_sum+=pr[i];
            
        }

        num_iterations=num_iterations+1;
       
    }

    delete mr;
    MPI_Finalize();
}

void output_pagerank_values(char* output)
{
    ofstream output_file(output);
    double sum = 0.0;
    size_t i;
    size_t num_rows = pr.size();

    output_file.precision(numeric_limits<double>::digits10);

    for (i = 0; i < num_rows; i++) {
        pr[i]/=main_sum;
        output_file << i << " = " << pr[i]<< endl;
        sum += pr[i];
    }
    
    output_file << "s = " << sum;
}

int main(int argc, char *argv[])
{   
    
    create_graph_from_inputfile(argv[1]);
    clock_t timer;
    timer=clock();
    pagerank_calculation();
    timer=clock()-timer;
    double time_elapsed=timer/CLOCKS_PER_SEC; // in seconds

    printf("Time elapsed (%f)",time_elapsed);
    output_pagerank_values(argv[3]);
    return 0;
}