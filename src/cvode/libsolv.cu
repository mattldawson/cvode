/* Copyright (C) 2020 Christian Guzman and Guillermo Oyarzun
 * Licensed under the GNU General Public License version 1 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Basic GPU functions
 *
 */

#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>

#include "libsolv.h"

//#include<cublas.h> //todo fix cublas not compiling fine
//#include<cublas_v2.h>

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError(cudaError_t err,
                        const char *file,
                        int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err),
           file, line);
    exit(EXIT_FAILURE);
  }
}

using namespace std;

void cvcudaGetLastErrorC(){
     cudaError_t error;
     error=cudaGetLastError();
     if(error!= cudaSuccess)
     {
       cout<<" ERROR INSIDE A CUDA FUNCTION: "<<error<<" "<<cudaGetErrorString(error)<<endl;
       exit(0);
     }
}

__global__ void cvcudamatScaleAddI(int nrows, double* dA, int* djA, int* diA, double alpha)
{
	int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows)
  {
    int jstart = diA[row];
    int jend   = diA[row+1];
    for(int j=jstart; j<jend; j++)
    {
      if(djA[j]==row)
      {
        dA[j] = 1.0 + alpha*dA[j];
      }
      else{
        dA[j] = alpha*dA[j];
      }
    }
  }
}

// A = I - gamma*J
// Based on CSR format, works on CSC too
// dA  : Matrix values (nnz size)
// djA : Matrix columns (nnz size)
// diA : Matrix rows (nrows+1 size)
// alpha : Scale factor
void gpu_matScaleAddI(int nrows, double* dA, int* djA, int* diA, double alpha, int blocks, int threads)
{

   blocks = (nrows+threads-1)/threads;

   dim3 dimGrid(blocks,1,1);
   dim3 dimBlock(threads,1,1);

  cvcudamatScaleAddI<<<dimGrid,dimBlock>>>(nrows, dA, djA, diA, alpha);
}

__global__
void libsolvcheck_input_gpud(double *x, int len, int var_id)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  printf("%d[%d]=%-le\n",var_id,i,x[i]);

}

// Diagonal precond
//todo same name not works, like some conflict happens
__global__ void cvcudadiagprecond(int nrows, double* dA, int* djA, int* diA, double* ddiag)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;

  if(row < nrows){
    int jstart=diA[row];
    int jend  =diA[row+1];
    for(int j=jstart;j<jend;j++){
      if(djA[j]==row){
        if(dA[j]!=0.0)
          ddiag[row]= 1.0/dA[j];
        else{
          ddiag[row]= 1.0;
        }
      }
    }
  }

}

void gpu_diagprecond(int nrows, double* dA, int* djA, int* diA, double* ddiag, int blocks, int threads)
{

  blocks = (nrows+threads-1)/threads;

  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);

  //printf("HOLA0 %d %d %d\n",blocks,threads,nrows);
  cvcudadiagprecond<<<dimGrid,dimBlock>>>(nrows, dA, djA, diA, ddiag);
  //cvcudaGetLastErrorC();
  //cvcheck_input_gpud<< < 1, 5>> >(ddiag,nrows,0);
}

// y = constant
__global__ void cvcudasetconst(double* dy,double constant,int nrows)
{
	int row= threadIdx.x + blockDim.x*blockIdx.x;
   	if(row < nrows){
		dy[row]=constant;
	}
}

void gpu_yequalsconst(double *dy, double constant, int nrows, int blocks, int threads)
{
   dim3 dimGrid(blocks,1,1);
   dim3 dimBlock(threads,1,1);

   cvcudasetconst<<<dimGrid,dimBlock>>>(dy,constant,nrows);

}


// x=A*b
__global__ void cvcudaSpmvCSR(double* dx, double* db, int nrows, double* dA, int* djA, int* diA)
{
	int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows)
  {
    int jstart = diA[row];
    int jend   = diA[row+1];
    double sum = 0.0;
    for(int j=jstart; j<jend; j++)
    {
      sum+= db[djA[j]]*dA[j];
    }
    dx[row]=sum;
	}

}

__global__ void cvcudaSpmvCSC(double* dx, double* db, int nrows, double* dA, int* djA, int* diA)
{
	double mult;
	int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows)
  {
    int jstart = diA[row];
    int jend   = diA[row+1];
    for(int j=jstart; j<jend; j++)
    {
      mult = db[row]*dA[j];
      atomicAdd(&(dx[djA[j]]),mult);
    }
	}
}

void gpu_spmv(double* dx ,double* db, int nrows, double* dA, int *djA,int *diA,int mattype,int blocks,int threads)
{
   dim3 dimGrid(blocks,1,1);
   dim3 dimBlock(threads,1,1);

   if(mattype==0)
   {
     cvcudaSpmvCSR<<<dimGrid,dimBlock>>>(dx, db, nrows, dA, djA, diA);
   }
   else
   {
	    cvcudasetconst<<<dimGrid,dimBlock>>>(dx, 0.0, nrows);
	    cvcudaSpmvCSC<<<dimGrid,dimBlock>>>(dx, db, nrows, dA, djA, diA);
   }
}

// y= a*x+ b*y
__global__ void cvcudaaxpby(double* dy,double* dx, double a, double b, int nrows)
{
	int row= threadIdx.x + blockDim.x*blockIdx.x;
   	if(row < nrows){
		dy[row]= a*dx[row] + b*dy[row];
	}
}

void gpu_axpby(double* dy ,double* dx, double a, double b, int nrows, int blocks, int threads)
{

   dim3 dimGrid(blocks,1,1);
   dim3 dimBlock(threads,1,1);

   cvcudaaxpby<<<dimGrid,dimBlock>>>(dy,dx,a,b,nrows);
}

// y = x
__global__ void cvcudayequalsx(double* dy,double* dx,int nrows)
{
	int row= threadIdx.x + blockDim.x*blockIdx.x;
   	if(row < nrows){
		dy[row]=dx[row];
	}
}

void gpu_yequalsx(double *dy, double* dx, int nrows, int blocks, int threads)
{
   dim3 dimGrid(blocks,1,1);
   dim3 dimBlock(threads,1,1);

   cvcudayequalsx<<<dimGrid,dimBlock>>>(dy,dx,nrows);

}

__global__ void cvcudadotxy(double *g_idata1, double *g_idata2, double *g_odata, unsigned int n)
{
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;//*2 because init blocks is half
  //unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;//*2 because init blocks is half

  double mySum = (i < n) ? g_idata1[i]*g_idata2[i] : 0;

  if (i + blockDim.x < n)
    mySum += g_idata1[i+blockDim.x]*g_idata2[i+blockDim.x];

  sdata[tid] = mySum;
  __syncthreads();

  //for (unsigned int s=(blockDim.x+1)/2; s>0; s>>=1)
  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
  {
    if (tid < s)
      sdata[tid] = mySum = mySum + sdata[tid + s];

    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void cvcudareducey(double *g_odata, unsigned int n)
{
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;

  double mySum =  (tid < n) ? g_odata[tid] : 0;

  sdata[tid] = mySum;
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
  {
    if (tid < s)
      sdata[tid] = mySum = mySum + sdata[tid + s];

    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

//threads need to be pow of 2 //todo remove h_temp since not needed now
double gpu_dotxy(double* vec1, double* vec2, double* h_temp, double* d_temp, int nrows, int blocks,int threads)
{
  double sum;
  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);

  //threads*sizeof(double)
  cvcudadotxy<<<dimGrid,dimBlock,threads*sizeof(double)>>>(vec1,vec2,d_temp,nrows);
  cudaMemcpy(&sum, d_temp, sizeof(double), cudaMemcpyDeviceToHost);
  //printf("rho1 %f", sum);

  int redsize= sqrt(blocks) +1;
  redsize=pow(2,redsize);

  dim3 dimGrid2(1,1,1);
  dim3 dimBlock2(redsize,1,1);

  cvcudareducey<<<dimGrid2,dimBlock2,redsize*sizeof(double)>>>(d_temp,blocks);
  cudaMemcpy(&sum, d_temp, sizeof(double), cudaMemcpyDeviceToHost);

  return sum;

/*
  cudaMemcpy(h_temp, d_temp, blocks * sizeof(double), cudaMemcpyDeviceToHost);
  double sum=0;
  for(int i=0;i<blocks;i++)
  {
    sum+=h_temp[i];
  }
  return sum;
*/
  /*dim3 dimGrid2(1,1,1);
  dim3 dimBlock2(blocks,1,1);

  //cvcuda only sum kernel call
  //cvcudareducey<<<dimGrid2,dimBlock2,blocks*sizeof(double)>>>(d_temp,blocks); //Takes quasi WAY MORE than cpu calc

  cudaMemcpy(h_temp, d_temp, sizeof(double), cudaMemcpyDeviceToHost);
  return h_temp[0];*/
}

// z= a*z + x + b*y
__global__ void cvcudazaxpbypc(double* dz, double* dx,double* dy, double a, double b, int nrows)
{
	int row= threadIdx.x + blockDim.x*blockIdx.x;
   	if(row < nrows){
		dz[row]=a*dz[row]  + dx[row] + b*dy[row];
	}
}

void gpu_zaxpbypc(double* dz, double* dx ,double* dy, double a, double b, int nrows, int blocks, int threads)
{

   dim3 dimGrid(blocks,1,1);
   dim3 dimBlock(threads,1,1);

   cvcudazaxpbypc<<<dimGrid,dimBlock>>>(dz,dx,dy,a,b,nrows);
}

// z= x*y
__global__ void cvcudamultxy(double* dz, double* dx,double* dy, int nrows)
{
	int row= threadIdx.x + blockDim.x*blockIdx.x;
   	if(row < nrows){
		dz[row]=dx[row]*dy[row];
	}
}

void gpu_multxy(double* dz, double* dx ,double* dy, int nrows, int blocks, int threads)
{

   dim3 dimGrid(blocks,1,1);
   dim3 dimBlock(threads,1,1);

   cvcudamultxy<<<dimGrid,dimBlock>>>(dz,dx,dy,nrows);
}

// z= a*x + b*y
//__global__ void cvcudazaxpby(double* dz, double* dx,double* dy, double a, double b, int nrows)
__global__ void cvcudazaxpby(double a, double* dx, double b, double* dy, double* dz, int nrows)
{
	int row= threadIdx.x + blockDim.x*blockIdx.x;
   	if(row < nrows){
		dz[row]=a*dx[row] + b*dy[row];
	}
}

void gpu_zaxpby(double a, double* dx, double b, double* dy, double* dz, int nrows, int blocks, int threads)
{

   dim3 dimGrid(blocks,1,1);
   dim3 dimBlock(threads,1,1);

  cvcudazaxpby<<<dimGrid,dimBlock>>>(a,dx,b,dy,dz,nrows);
}

// y= a*x + y
__global__ void cvcudaaxpy(double* dy,double* dx, double a, int nrows)
{
	int row= threadIdx.x + blockDim.x*blockIdx.x;
   	if(row < nrows){
		dy[row]=a*dx[row] + dy[row];
	}
}

void gpu_axpy(double* dy, double* dx ,double a, int nrows, int blocks, int threads)
{

   dim3 dimGrid(blocks,1,1);
   dim3 dimBlock(threads,1,1);

   cvcudaaxpy<<<dimGrid,dimBlock>>>(dy,dx,a,nrows);
}

// sqrt(sum ( (x_i*y_i)^2)/n)
__global__ void cvcudaDVWRMS_Norm(double *g_idata1, double *g_idata2, double *g_odata, unsigned int n)
{
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

  double mySum = (i < n) ? g_idata1[i]*g_idata1[i]*g_idata2[i]*g_idata2[i] : 0;

  if (i + blockDim.x < n)
    mySum += g_idata1[i+blockDim.x]*g_idata1[i+blockDim.x]*g_idata2[i+blockDim.x]*g_idata2[i+blockDim.x];

  sdata[tid] = mySum;
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
  {
    if (tid < s)
      sdata[tid] = mySum = mySum + sdata[tid + s];

    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

double gpu_VWRMS_Norm(int n, double* vec1,double* vec2,double* h_temp,double* d_temp, int blocks,int threads)
{
  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);

  cvcudaDVWRMS_Norm<<<dimGrid,dimBlock,threads*sizeof(double)>>>(vec1,vec2,d_temp,n);

  //cudaMemcpy(h_temp, d_temp, blocks * sizeof(double), cudaMemcpyDeviceToHost);

  int redsize= sqrt(blocks) +1;
  redsize=pow(2,redsize);

  dim3 dimGrid2(1,1,1);
  dim3 dimBlock2(redsize,1,1);

  cvcudareducey<<<dimGrid2,dimBlock2,redsize*sizeof(double)>>>(d_temp,blocks);

  double sum;
  cudaMemcpy(&sum, d_temp, sizeof(double), cudaMemcpyDeviceToHost);

  return sqrt(sum/n);

/*
  double sum=0;
  for(int i=0;i<blocks;i++)
  {
    sum+=h_temp[i];
  }
  return sqrt(sum/n);
  */
}

// y=alpha*y
__global__ void cvcudascaley(double* dy, double a, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dy[row]=a*dy[row];
  }
}

void gpu_scaley(double* dy, double a, int nrows, int blocks, int threads)
{
  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);

  cvcudascaley<<<dimGrid,dimBlock>>>(dy,a,nrows);
}




// Device functions (equivalent to global functions but in device to allow calls from gpu)
__device__ void cudaDevicematScaleAddI(int nrows, double* dA, int* djA, int* diA, double alpha)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows)
  {
    int jstart = diA[row];
    int jend   = diA[row+1];
    for(int j=jstart; j<jend; j++)
    {
      if(djA[j]==row)
      {
        dA[j] = 1.0 + alpha*dA[j];
      }
      else{
        dA[j] = alpha*dA[j];
      }
    }
  }
}

// Diagonal precond
__device__ void cudaDevicediagprecond(int nrows, double* dA, int* djA, int* diA, double* ddiag)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    int jstart=diA[row];
    int jend  =diA[row+1];
    for(int j=jstart;j<jend;j++){
      if(djA[j]==row){
        if(dA[j]!=0.0)
          ddiag[row]= 1.0/dA[j];
        else{
          ddiag[row]= 1.0;
        }
      }
    }
  }

}

// y = constant
__device__ void cudaDevicesetconst(double* dy,double constant,int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dy[row]=constant;
  }
}

// x=A*b
__device__ void cudaDeviceSpmvCSR(double* dx, double* db, int nrows, double* dA, int* djA, int* diA)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows)
  {
    int jstart = diA[row];
    int jend   = diA[row+1];
    double sum = 0.0;
    for(int j=jstart; j<jend; j++)
    {
      sum+= db[djA[j]]*dA[j];
    }
    dx[row]=sum;
  }

}

__device__ void cudaDeviceSpmvCSC_block(double* dx, double* db, int nrows, double* dA, int* djA, int* diA)
{
  double mult;
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows)
  {
    dx[row]=0.0;
    __syncthreads(); //Multiple threads can save to the same row
    int jstart = diA[row];
    int jend   = diA[row+1];
    for(int j=jstart; j<jend; j++)
    {
      mult = db[row]*dA[j];
      //atomicAdd_block(&(dx[djA[j]]),mult);
      //todo eliminate atomicAdd
      atomicAdd_block(&(dx[djA[j]]),mult);
//		dx[djA[j]]+= db[row]*dA[j];
    }
  }
}

__device__ void cudaDeviceSpmvCSC(double* dx, double* db, int nrows, double* dA, int* djA, int* diA)
{
  double mult;
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows)
  {
    int jstart = diA[row];
    int jend   = diA[row+1];
    for(int j=jstart; j<jend; j++)
    {
      mult = db[row]*dA[j];
      //atomicAdd(&(dx[djA[j]]),mult);
      atomicAdd(&(dx[djA[j]]),mult);
//		dx[djA[j]]+= db[row]*dA[j];
    }
  }
}

// y= a*x+ b*y
__device__ void cudaDeviceaxpby(double* dy,double* dx, double a, double b, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dy[row]= a*dx[row] + b*dy[row];
  }
}

// y = x
__device__ void cudaDeviceyequalsx(double* dy,double* dx,int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dy[row]=dx[row];
  }
}

/* //unused
__device__ void cudaDevicereducey(double *g_odata, unsigned int n, int n_shr_empty)
{
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  //int id = blockIdx.x * blockDim.x + threadIdx.x;

  double mySum =  (tid < n) ? g_odata[tid] : 0;

#ifdef DEV_DEVICEDOTXY
  //under development, fix returning deriv=0 and slower
  if(tid<blockDim.x/2)
    for (int j=0; j<2; j++)
      sdata[j*blockDim.x/2 + tid] = 0;
#else
  //Last thread assign 0 to empty shr values
  if (tid == 0)//one thread
  {
    //todo fix, returning 0 sometimes on mock_monarch cells=1000
    //speedup when active, but why? and sometimes returns deriv=0
    for (int j=0; j<n_shr_empty; j++)
      sdata[blockDim.x+j] = 0; //Assign 0 to non interesting sdata
  }
#endif

  sdata[tid] = mySum;
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
  {
    if (tid < s)
      sdata[tid] = mySum = mySum + sdata[tid + s];

    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
*/
/*
__device__ void cudaDevicedotxy_old(double *g_idata1, double *g_idata2,
                                double *g_odata, unsigned int n, int n_shr_empty)
{
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  //unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  //Used to ensure last block has 0 values for non-zero cases (Last block can have less cells than previous blocks)
  //todo condition not needed with tid<active_threads
  double mySum = (i < n) ? g_idata1[i]*g_idata2[i] : 0.;

#ifndef DEV_DEVICEDOTXY
  //under development, fix returning deriv=0 and slower


  for( int j = threadIdx.x; j < n_shr_empty+blockDim.x; j+=blockDim.x)
    sdata[j]=0.0;

#else
    //Last thread assign 0 to empty shr values
  //todo: it's needed?
  if (tid == 0)//one thread
  {
    //todo fix, returning 0 sometimes on mock_monarch cells=1000 (bug appears after <=7 attemps)
    //speedup when active, probably cause if no active some threads are not
    // doing anything so it takes more time to converge, but then sometimes returns deriv=0(fix)
    //needed or diff results on n_cells=100 3 species
    for (int j=0; j<n_shr_empty; j++)
      sdata[blockDim.x+j] = 0.; //Assign 0 to remaining sdata (cases sdata_id>=threads_block)
  }
#endif

  //Set shr_memory to local values
  sdata[tid] = mySum;
  __syncthreads();

  //todo ensure that n_shr_empty is less than half of the max_threads to have enough threads
  for (unsigned int s=(blockDim.x+n_shr_empty)/2; s>0; s>>=1)
  {
    if (tid < s)
      sdata[tid] = mySum = mySum + sdata[tid + s];

    __syncthreads();
  }

  //dont need to access global memory on block-cells
  //if (tid == 0) g_odata[blockIdx.x] = sdata[0];
  *g_odata = sdata[0];
}

*/

__device__ void cudaDevicedotxy(double *g_idata1, double *g_idata2,
                                 double *g_odata, unsigned int n, int n_shr_empty)
{
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  //unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

#ifdef BCG_ALL_THREADS

  double mySum = (i < n) ? g_idata1[i]*g_idata2[i] : 0.;

  sdata[tid] = mySum;

  __syncthreads();

  for (unsigned int s=(blockDim.x+n_shr_empty)/2; s>0; s>>=1)
  {
    if (tid < s)
      sdata[tid] = mySum = mySum + sdata[tid + s];

    __syncthreads();
  }

  //if (tid==0) *g_odata = sdata[0];
  *g_odata = sdata[0];
  //*g_odata = sdata[0]+0.1*tid;
  __syncthreads();

#else

  if (tid == 0){
    for (int j=0; j<blockDim.x+n_shr_empty; j++)
      sdata[j] = 0.;
  }

/*
  for (unsigned int s=(blockDim.x+n_shr_empty)/2; s>0; s>>=1)
  {
    if (tid < s){
      sdata[tid] = 0.;
      sdata[tid+s] = 0.;
    }
  }
*/

  __syncthreads();


  sdata[tid] = g_idata1[i]*g_idata2[i];

  __syncthreads();

  for (unsigned int s=(blockDim.x+n_shr_empty)/2; s>0; s>>=1)
  {
    if (tid < s)
      sdata[tid] += sdata[tid + s];

    __syncthreads();
  }

  //if (tid==0) *g_odata = sdata[0];
  *g_odata = sdata[0];
  //*g_odata = sdata[0]+0.1*tid;
  __syncthreads();

#endif

}

//n_shr_empty its a different implementation from cuda reduce extended samples ( https://docs.nvidia.com/cuda/cuda-samples/index.html)
// since n_threads_blocks isnotpowerof2
// while these samples only takes into account n=notpowerof2, also we need active_threads able to be < max_threads
// because other operations must work only with this number of threads to ensure work only with complete cells

// z= a*z + x + b*y
__device__ void cudaDevicezaxpbypc(double* dz, double* dx,double* dy, double a, double b, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dz[row]=a*dz[row]  + dx[row] + b*dy[row];
  }
}

// z= x*y
__device__ void cudaDevicemultxy(double* dz, double* dx,double* dy, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dz[row]=dx[row]*dy[row];
  }
}

// z= a*x + b*y
__device__ void cudaDevicezaxpby(double a, double* dx, double b, double* dy, double* dz, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dz[row]=a*dx[row] + b*dy[row];
  }
}

// y= a*x + y
__device__ void cudaDeviceaxpy(double* dy,double* dx, double a, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dy[row]=a*dx[row] + dy[row];
  }
}

// sqrt(sum ( (x_i*y_i)^2)/n)
__device__ void cudaDeviceDVWRMS_Norm(double *g_idata1, double *g_idata2, double *g_odata, unsigned int n)
{
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

  double mySum = (i < n) ? g_idata1[i]*g_idata1[i]*g_idata2[i]*g_idata2[i] : 0;

  if (i + blockDim.x < n)
    mySum += g_idata1[i+blockDim.x]*g_idata1[i+blockDim.x]*g_idata2[i+blockDim.x]*g_idata2[i+blockDim.x];

  sdata[tid] = mySum;
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
  {
    if (tid < s)
      sdata[tid] = mySum = mySum + sdata[tid + s];

    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// y=alpha*y
__device__ void cudaDevicescaley(double* dy, double a, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dy[row]=a*dy[row];
  }
}

