# include <stdarg.h>
#include <stdlib.h>
# include <time.h>
#include <stdio.h>

# include <math.h>
# include <limits.h> 

# include <string>

# include <pthread.h>

#ifndef _WIN32
#define set_random drand48()*100
#else
#define set_random (double(rand())/RAND_MAX)
#endif

void print_to_file(double** values, const char* filename, int n)
{
    FILE *f=fopen(filename,"w");
    for (int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            fprintf(f,"%f",values[i][j]);
            fprintf(f," ");
        }
        fprintf(f,"\n");
    }
    fclose(f);
}

double** initialise(int n, int flag, int init) // to allocate space to n by n matrix of double precision
{
double **M= (double **)calloc(n,sizeof(double*));

for (int i=0; i<n; i++)
{
 M[i]=(double *)calloc(n,sizeof(double));
 
 if (init==1)
 {
    for( int j=0; j<n; j++)
        {   if (flag==0)                    // initialise A 
            {   M[i][j]=set_random;
            }
            else if (flag==1)              // upper triangular matrix
            {
                if (i<=j)
                {
                   M[i][j]=set_random;
                }
                
            }
            else                          // lower triangular matrix
            {
                if (i==j)
                {
                    M[i][j]=1.0;
                }

                else if (i>j)
                {
                    M[i][j]=set_random;
                }
                
            }
        }
 }
 
}
return M;
}

double** allocate_space(int n)
{
    double** A= (double **)calloc(n,sizeof(double*));
    
for (int i=0; i<n; i++)
{
 A[i]=(double *)calloc(n,sizeof(double));
 
}

return A;

}

void initialise_and_copy(double **A, double **copy, int n)
{
for (int i=0; i<n; i++)
{
 for(int j=0; j<n; j++)
 {
    A[i][j]=set_random;
    copy[i][j]=A[i][j];
 }
 
 
}


}

double verify(double** A, double** P, double** L, double** U, int n)
{
    double sum=0.0;
    double norm=0.0;
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<n; j++)
        {
            norm=0.0;
            for (int k=0; k<n; k++)
            {
                norm=norm + P[i][k]*A[k][j]-L[i][k]*U[k][j];
            }
            sum=sum+pow(norm,2);
        }
    }
    return sum;
}

struct values_for_each_thread
{
    int rank;
    int num_of_elements;
    int begin;
    double** a;
    double** l;
    double** u;
    int remaining;
   
};

int N;

void* lu_computation_in_each_thread (void* values_for_thread)
{
    int rank= ((struct values_for_each_thread*)values_for_thread)->rank;
    int num_of_elements=((struct values_for_each_thread*)values_for_thread)->num_of_elements;
    int begin=((struct values_for_each_thread*)values_for_thread)->begin;
    double** a=((struct values_for_each_thread*)values_for_thread)->a;
    double** l=((struct values_for_each_thread*)values_for_thread)->l;
    double** u=((struct values_for_each_thread*)values_for_thread)->u;
    
    int begin_in_thread=begin+num_of_elements*rank;
    int last_in_thread=begin_in_thread+num_of_elements+((struct values_for_each_thread*)values_for_thread)->remaining;

    for(int i=begin_in_thread; i<last_in_thread; i++)
        {
            for (int j=begin; j<N; j++)
            {   
                a[i][j]=a[i][j]-l[i][begin-1]*u[begin-1][j];
            }
        }
    
    return NULL;
}

void LU_Decomposition(int n, int threads, double** a, double** copy)
{   
    pthread_t* pthreads=(pthread_t *)malloc(threads*sizeof(pthread_t));

    int* pi= (int*)calloc(n,sizeof(int));

    double** u= initialise(n,1,1);
    double** l=initialise(n,2,1);
    double** p=initialise(n,0,0);

    double threshold=pow(10,-16);

    clock_t timer;
    timer=clock();

    for (int i=0; i< n; i++)
    {
        pi[i]=i;
    }
    
    for(int k=0; k<n; k++)
    {
        double max=0.0;
        int index=0; // k' that represents index of the max value observed
        for (int i=k; i<n; i++)
        {
            if (max< abs(a[i][k]))
            {
                max=abs(a[i][k]);
                index=i;
            }
        }

        if (max==0.0)
        {
            printf("singular matrix");
        }
        
        int temp=pi[k];
        pi[k]=pi[index];
        pi[index]=temp;
        
        for(int j=0; j<n;j++)
        {
        double a_temp=a[k][j];
        a[k][j]=a[index][j];
        a[index][j]=a_temp;

        }
        
        for (int j=0; j<k; j++)
        {
            double l_temp=l[k][j];
            l[k][j]=l[index][j];
            l[index][j]=l_temp;
        }
        
        u[k][k]=a[k][k];

        for(int i=k+1; i<n; i++)
        {
            l[i][k]=a[i][k]/(u[k][k]+threshold);
            u[k][i]=a[k][i];
        }
        
        for(int i=0; i<threads; i++)
        {   struct values_for_each_thread* thread=(struct values_for_each_thread*)malloc(sizeof(struct values_for_each_thread));
            thread->a=a;
            thread->rank=i;
            thread->u=u;
            thread->l=l;
            thread->num_of_elements=(n-k-1)/threads;
            thread->begin=(k+1);
            thread->remaining=0;

            if (i==threads-1)
            {
                int num=thread->num_of_elements;
                int remaining=(n-k-1)-num*threads;
                thread->remaining=remaining;
            }
           
            pthread_create(&pthreads[i], NULL,lu_computation_in_each_thread, (void*) thread); 

        }

        for (int i=0; i<threads; i++)
        {
            pthread_join(pthreads[i], NULL);
        }

    }

    for (int i=0; i<n;i++)
    {
        p[i][pi[i]]=1.0;
    }

    timer=clock()-timer;
    double time_elapsed=timer/CLOCKS_PER_SEC; // in seconds

    printf("Time elapsed (%f)",time_elapsed);
    
    print_to_file(p,"P",n);
    print_to_file(u,"U",n);
    print_to_file(l,"L",n);
    print_to_file(copy,"A",n);

    double error=verify(copy,p,l,u,n);

    printf("error magnitude (%f)", error);
    
    free(pthreads);
    
    for ( int i=0; i<n; i++)
    {
        free(u[i]);
        free(l[i]);
        free(p[i]);
        free(copy[i]);
    }
    free(u);
    free(l);
    free(p);
    free(copy);

    free(pi);

}

int main(int argc, char* argv[])
{
    time_t t=time(NULL);
    
    #ifndef _WIN32
    srand48((unsigned int) t);
    #else
    srand((unsigned int) t);
    #endif
    
    N=atoi(argv[1]);
    int threads= atoi(argv[2]);

    double **a=allocate_space(N);
    double **copy=allocate_space(N);
    
    initialise_and_copy(a,copy,N);

    LU_Decomposition(N,threads,a,copy);
    
    return 0;

}