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
#include <time.h>
#include "mpi.h"

using namespace std;

const double alpha = 0.85;
const double convergence = 0.000001;

vector<int> num_outgoing; // number of outgoing links per column
vector< vector<int> > rows; // the number of rows of the hyperlink matrix

vector<double> pr; // the pagerank table
int n;
vector<double> old_pr;

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

void pagerank_calculation() {
    
    double diff = 1.0;
    double dangling_pr; // sum of current pagerank vector elements for dangling nodes
    double sum_pr; // sum of current pagerank vector elements
    int num_iterations = 0;
    double * difference;

    int * keys;
    int* keys_scattered;
    double* d;

    size_t num_rows = rows.size();
    
    if (num_rows == 0) {
        return;
    }

   keys=(int*)malloc(sizeof(int)*num_rows);
   d=(double*)malloc(sizeof(double)*num_rows);

    pr.resize(num_rows,1.0);
    old_pr=pr;
    
    

    int n_per_proc;
    int rank;
    int size;
    
    double one_Av=0.0;
    double one_Iv=0.0; 
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    while (diff > convergence) 
    {   
        sum_pr = 0.0;
        dangling_pr = 0.0;
        diff=0.0;
       
        
        
        if (rank==0)
        {
            for (size_t to = 0; to < pr.size(); to++) 
            {
                sum_pr += pr[to];
                if (num_outgoing[to] == 0) 
                {
                    dangling_pr += pr[to];
                }

                keys[to]=to;
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
        
        n_per_proc = num_rows/size;

        keys_scattered=(int*)malloc(sizeof(int)*n_per_proc);
        difference=(double*)malloc(sizeof(double)*n_per_proc);
        
        MPI_Scatter(keys,n_per_proc,MPI_INT,keys_scattered,n_per_proc,MPI_INT,0,MPI_COMM_WORLD);
        
        for (int i=0; i<n_per_proc; i++)
        {
            /* The corresponding element of the H multiplication */
            double h = 0.0;
            for (auto ci = rows[keys_scattered[i]].begin(); ci != rows[keys_scattered[i]].end(); ci++) {
                /* The current element of the H vector */
                double h_v = 0.0;
                if (num_outgoing[*ci]>0.0)
                {
                    h_v=1.0/num_outgoing[*ci];
                }
                h += h_v * old_pr[*ci];
            }
            h = h*alpha;
            pr[keys_scattered[i]] = h + one_Av + one_Iv;
            difference[i]= abs(pr[keys_scattered[i]] - old_pr[keys_scattered[i]]);
        }
        
        MPI_Gather(difference,n_per_proc, MPI_DOUBLE,d,n_per_proc,MPI_DOUBLE,0, MPI_COMM_WORLD);

        if (rank==0)
        {
            for (size_t i=0;i<num_rows;i++)
            {
                diff+=d[i];
            }
            
        }

          
        num_iterations=num_iterations+1;
    }
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

        output_file << i << " = " << pr[i] << endl;
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