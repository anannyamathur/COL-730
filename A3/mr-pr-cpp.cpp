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

#include <boost/config.hpp>
#include "mapreduce.hpp"

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


namespace computation_of_pagerank
{
    template<typename MapTask>
class datasource : mapreduce::detail::noncopyable
{
  public:
    datasource() : sequence_(0)
    {
    }

    bool const setup_key(typename MapTask::key_type &key)
    {
        key = sequence_++;
        return key < n;
    }

    bool const get_data(typename MapTask::key_type const &key, typename MapTask::value_type &value)
    {
        value=rows[key];
        return true;
    }

  private:
    int sequence_;
};

struct map_task : public mapreduce::map_task<int, std::vector<int> >
{
    template<typename Runtime>
    void operator()(Runtime &runtime, key_type const &key, value_type const &value) const
    {   
        runtime.emit_intermediate(key, 0.0);
    
        for (auto ci = value.begin(); ci != value.end(); ci++)
        {   
            if (num_outgoing[*ci]>0)
            {
                runtime.emit_intermediate(key, (1.0/num_outgoing[*ci]*old_pr[*ci]));
            }
            else
            {
                runtime.emit_intermediate(key, 0.0);
            }
            
        }
           
    }
};

struct reduce_task : public mapreduce::reduce_task<int, double>
{
    template<typename Runtime, typename It>
    void operator()(Runtime &runtime, key_type const &key, It it, It ite) const
    {   
        reduce_task::value_type h=0.0;
        for (It it1=++it; it1!=ite; ++it1)
        {
            h=h + *it1;
        }
        runtime.emit(key,h);

    }
};

typedef mapreduce::job<computation_of_pagerank::map_task,computation_of_pagerank::reduce_task,mapreduce::null_combiner,
computation_of_pagerank::datasource<computation_of_pagerank::map_task> > job;

}


void pagerank_calculation() {
    
    double diff = 1.0;
    double dangling_pr; // sum of current pagerank vector elements for dangling nodes
    double sum_pr; // sum of current pagerank vector elements
    int num_iterations = 0;
    
    size_t num_rows = rows.size();
    
    if (num_rows == 0) {
        return;
    }
    
    pr.resize(num_rows,1.0);
    old_pr=pr;


    mapreduce::specification spec;
    mapreduce::results result;
    spec.reduce_tasks = std::max(1U, std::thread::hardware_concurrency());
    
    while (diff > convergence) 
    {
        sum_pr = 0.0;
        dangling_pr = 0.0;
        
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
        double one_Av = alpha * dangling_pr/num_rows;

        /* An element of the 1 x I vector; all elements are identical */
        double one_Iv = (1 - alpha) * 1.0 / num_rows;

        /* The difference to be checked for convergence */
        diff = 0.0;

        computation_of_pagerank::job::datasource_type datasource;
        computation_of_pagerank::job job(datasource, spec);
        job.run<mapreduce::schedule_policy::cpu_parallel<computation_of_pagerank::job> >(result);
        

        for (auto it=job.begin_results(); it!=job.end_results(); ++it)
        {
            pr[it->first] = alpha*(it->second) + one_Av + one_Iv;
            diff += abs(pr[it->first] - old_pr[it->first]);
        }

        num_iterations=num_iterations+1;
        
    }
     
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