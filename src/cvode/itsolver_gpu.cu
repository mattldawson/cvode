#include "itsolver_gpu.h"
#include "cuda_structs.h"

#ifdef __cplusplus
extern "C" {
#endif

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

void createSolver(itsolver *bicg)
{

  printf("createSolver\n");

  //Init variables ("public")
  bicg->mattype=1; //CSC
  bicg->maxIt=1000;
  bicg->tolmax=1e-12; //cv_mem->cv_reltol CAMP selected accuracy (1e-8) //1e-10;//1e-6

  int device=0;//Select GPU
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  bicg->threads=prop.maxThreadsPerBlock;
  //bicg->threads=1024;
  bicg->blocks=(bicg->nrows+bicg->threads-1)/bicg->threads;

  //Auxiliary vectors ("private")
  double ** dr0 = &bicg->dr0;
  double ** dr0h = &bicg->dr0h;
  double ** dn0 = &bicg->dn0;
  double ** dp0 = &bicg->dp0;
  double ** dt = &bicg->dt;
  double ** ds = &bicg->ds;
  double ** dAx2 = &bicg->dAx2;
  double ** dy = &bicg->dy;
  double ** dz = &bicg->dz;
  double ** daux = &bicg->daux;
  double ** ddiag = &bicg->ddiag;

  //Allocate
  int nrows = bicg->nrows;
  int blocks = bicg->blocks;
  cudaMalloc(dr0,nrows*sizeof(double));
  cudaMalloc(dr0h,nrows*sizeof(double));
  cudaMalloc(dn0,nrows*sizeof(double));
  cudaMalloc(dp0,nrows*sizeof(double));
  cudaMalloc(dt,nrows*sizeof(double));
  cudaMalloc(ds,nrows*sizeof(double));
  cudaMalloc(dAx2,nrows*sizeof(double));
  cudaMalloc(dy,nrows*sizeof(double));
  cudaMalloc(dz,nrows*sizeof(double));
  cudaMalloc(ddiag,nrows*sizeof(double));
  HANDLE_ERROR(cudaMalloc(daux,nrows*sizeof(double)));
  bicg->aux=(double*)malloc(sizeof(double)*blocks);

  /*

  // Allocating matrix data to the GPU
  //ModelDataGPU *mGPU = &sd->mGPU;
  //bicg->dA=mGPU->J;//set itsolver gpu pointer to jac pointer initialized at camp
  cudaMalloc((void**)&bicg->dA,bicg->nnz*sizeof(double));
  //bicg->dftemp=mGPU->deriv_data; //deriv is gpu pointer
  cudaMalloc((void**)&bicg->dftemp,bicg->nrows*sizeof(double));

  cudaMalloc((void**)&bicg->djA,bicg->nnz*sizeof(int));
  cudaMalloc((void**)&bicg->diA,(bicg->nrows+1)*sizeof(int));
  cudaMemcpy(bicg->djA,bicg->jA,bicg->nnz*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(bicg->diA,bicg->iA,(bicg->nrows+1)*sizeof(int),cudaMemcpyHostToDevice);

  //ODE concs arrays
  cudaMalloc((void**)&bicg->dcv_y,bicg->nrows*sizeof(double));
  cudaMalloc((void**)&bicg->dx,bicg->nrows*sizeof(double));

  cudaMalloc((void**)&bicg->dewt,bicg->nrows*sizeof(double));
  cudaMalloc((void**)&bicg->dacor,bicg->nrows*sizeof(double));
  cudaMalloc((void**)&bicg->dtempv,bicg->nrows*sizeof(double));

  */

  /*
  //ODE aux variables

  cudaMalloc((void**)&bicg->dzn,bicg->nrows*(cv_mem->cv_qmax+1)*sizeof(double));
  double *ewt = N_VGetArrayPointer(cv_mem->cv_ewt);
  double *tempv = N_VGetArrayPointer(cv_mem->cv_tempv);

  cudaMemcpy(bicg->dewt,ewt,bicg->nrows*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(bicg->dacor,ewt,bicg->nrows*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(bicg->dftemp,ewt,bicg->nrows*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(bicg->dx,tempv,bicg->nnz*sizeof(double),cudaMemcpyHostToDevice);

   */

#ifndef PMC_DEBUG_GPU
  bicg->counterprecvStep=0;
  bicg->counterNewtonIt=0;
  bicg->counterLinSolSetup=0;
  bicg->counterLinSolSolve=0;
  bicg->countercvStep=0;
  bicg->counterDerivNewton=0;
  bicg->counterBiConjGrad=0;
  bicg->counterBiConjGradInternal=0;
  cudaMalloc((void**)&bicg->counterBiConjGradInternalGPU,sizeof(int));
  bicg->counterDerivSolve=0;
  bicg->counterJac=0;

  int min_double=1.0e-30;

  bicg->timeprecvStep=min_double;
  bicg->timeNewtonIt=min_double;
  bicg->timeLinSolSetup=min_double;
  bicg->timeLinSolSolve=min_double;
  bicg->timecvStep=min_double;
  bicg->timeDerivNewton=min_double;
  bicg->timeBiConjGrad=min_double;
  bicg->timeDerivSolve=min_double;
  bicg->timeJac=min_double;

  cudaEventCreate(&bicg->startDerivNewton);
  cudaEventCreate(&bicg->startDerivSolve);
  cudaEventCreate(&bicg->startLinSolSetup);
  cudaEventCreate(&bicg->startLinSolSolve);
  cudaEventCreate(&bicg->startNewtonIt);
  cudaEventCreate(&bicg->startcvStep);
  cudaEventCreate(&bicg->startBiConjGrad);
  cudaEventCreate(&bicg->startJac);

  cudaEventCreate(&bicg->stopDerivNewton);
  cudaEventCreate(&bicg->stopDerivSolve);
  cudaEventCreate(&bicg->stopLinSolSetup);
  cudaEventCreate(&bicg->stopLinSolSolve);
  cudaEventCreate(&bicg->stopNewtonIt);
  cudaEventCreate(&bicg->stopcvStep);
  cudaEventCreate(&bicg->stopBiConjGrad);
  cudaEventCreate(&bicg->stopJac);
#endif

  //printf("createSolver\n");

}

#ifndef COMMENT

int nextPowerOfTwo(int v){

  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  //printf("nextPowerOfTwo %d", v);

  return v;
}

//todo instead sending all in one kernel, divide in 2 or 4 kernels with streams and check if
//cuda reassigns better the resources
//todo profiling del dot y ver cuanta occupancy me esta dando de shared memory porque me limita
//el numero de bloques que se ejecutan a la vez(solo se ejecutan a la vez en toda la function
// los bloques que "quepan" con la shared memory available: solution use cudastreams and launch instead
//of only 1 kernel use 2 or 4 to cubrir huecos (de memoria y eso), y tmb reducir la shared
//con una implementacion hibrida del dotxy

//todo add debug variables in some way (maybe pass always it pointer or something like that)
__global__
void cvsolveBcgCuda(
        double *dA, int *djA, int *diA, double *dx, double *dtempv //Input data
        ,int nrows, int blocks, int n_shr_empty, int maxIt, int mattype
        ,int n_cells, double tolmax, double *ddiag //Init variables
        ,double *dr0, double *dr0h, double *dn0, double *dp0
        ,double *dt, double *ds, double *dAx2, double *dy, double *dz
        ,double *daux // Auxiliary vectors
#ifndef PMC_DEBUG_GPU
        ,int *it_pointer //debug
#endif
        //,double *aux_params
        //double *alpha, double *rho0, double* omega0, double *beta,
        //double *rho1, double *temp1, double *temp2 //Auxiliary parameters
)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;
  int active_threads = nrows;

#ifdef BCG_ALL_THREADS
  if(1){
#else
  if(i<active_threads){
#endif

    double alpha,rho0,omega0,beta,rho1,temp1,temp2;
    alpha=rho0=omega0=beta=rho1=temp1=temp2=1.0;

    /*alpha  = 1.0;
    rho0   = 1.0;
    omega0 = 1.0;*/

    //gpu_yequalsconst(dn0,0.0,nrows,blocks,threads);  //n0=0.0 //memset???
    //gpu_yequalsconst(dp0,0.0,nrows,blocks,threads);  //p0=0.0
    /*cudaDevicesetconst(dn0, 0.0, nrows);
    cudaDevicesetconst(dp0, 0.0, nrows);
    cudaDevicesetconst(dt, 0.0, nrows);*/

    cudaDevicesetconst(dr0, 0.0, nrows);
    cudaDevicesetconst(dr0h, 0.0, nrows);
    cudaDevicesetconst(dn0, 0.0, nrows);
    cudaDevicesetconst(dp0, 0.0, nrows);
    cudaDevicesetconst(dt, 0.0, nrows);
    cudaDevicesetconst(ds, 0.0, nrows);
    cudaDevicesetconst(dAx2, 0.0, nrows);
    cudaDevicesetconst(dy, 0.0, nrows);
    cudaDevicesetconst(dz, 0.0, nrows);

#ifdef BASIC_SPMV
    cudaDevicesetconst(dr0, 0.0, nrows);
    __syncthreads();
    cudaDeviceSpmvCSC(dr0,dx,nrows,dA,djA,diA); //y=A*x
#else
    cudaDeviceSpmvCSC_block(dr0,dx,nrows,dA,djA,diA); //y=A*x
#endif

    //gpu_axpby(dr0,dtempv,1.0,-1.0,nrows,blocks,threads); // r0=1.0*rhs+-1.0r0 //y=ax+by
    cudaDeviceaxpby(dr0,dtempv,1.0,-1.0,nrows);

    //gpu_yequalsx(dr0h,dr0,nrows,blocks,threads);  //r0h=r0
    cudaDeviceyequalsx(dr0h,dr0,nrows);

#ifndef PMC_DEBUG_GPU
    //int it=*it_pointer;
    int it=0;
#else
    int it=0;
#endif

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

    if(i==0){
      //printf("%d dr0[%d] %-le\n",it,i,dr0[i]);
      printf("%d %d rho1 %-le\n",it,i,rho1);
    }

#endif

    do
    {

      //rho1=gpu_dotxy(dr0, dr0h, aux, daux, nrows,(blocks + 1) / 2, threads);
      __syncthreads();

      cudaDevicedotxy(dr0, dr0h, &rho1, nrows, n_shr_empty);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

      if(i==0){
      //printf("%d dr0[%d] %-le\n",it,i,dr0[i]);
      printf("%d %d rho1 %-le\n",it,i,rho1);
    }

#endif

      __syncthreads();//necessary to reduce accuracy error
      beta = (rho1 / rho0) * (alpha / omega0);

      //gpu_zaxpbypc(dp0,dr0,dn0,beta,-1.0*omega0*beta,nrows,blocks,threads);   //z = ax + by + c
      cudaDevicezaxpbypc(dp0, dr0, dn0, beta, -1.0 * omega0 * beta, nrows);   //z = ax + by + c

      //gpu_multxy(dy,ddiag,dp0,nrows,blocks,threads);  // precond y= p0*diag
      cudaDevicemultxy(dy, ddiag, dp0, nrows);

      cudaDevicesetconst(dn0, 0.0, nrows);
      //gpu_spmv(dn0,dy,nrows,dA,djA,diA,mattype,blocks,threads);  // n0= A*y
#ifdef BASIC_SPMV
      cudaDevicesetconst(dn0, 0.0, nrows);
      __syncthreads();
      cudaDeviceSpmvCSC(dn0, dy, nrows, dA, djA, diA);
#else
      //cudaDeviceSpmvCSC_block(dn0, dy, nrows, dA, djA, diA);
      __syncthreads();
      cudaDeviceSpmvCSC_block(dn0, dy, nrows, dA, djA, diA);
#endif

      __syncthreads();
      //temp1=gpu_dotxy(dr0h, dn0, aux, daux, nrows,(blocks + 1) / 2, threads);

      cudaDevicedotxy(dr0h, dn0, &temp1, nrows, n_shr_empty);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

      if(i==0){
        printf("%d %d temp1 %-le\n",it,i,temp1);
        //printf("%d %d &temp1 %p\n",it,i,&temp1);
        //printf("%d %d &test %p\n",it,i,&test);
        //printf("%d %d &i %p\n",it,i,&i);
      }

#endif

      __syncthreads();
      alpha = rho1 / temp1;

      //gpu_zaxpby(1.0,dr0,-1.0*alpha,dn0,ds,nrows,blocks,threads); // a*x + b*y = z
      cudaDevicezaxpby(1.0, dr0, -1.0 * alpha, dn0, ds, nrows);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

      if(i==0){
        printf("%d ds[%d] %-le\n",it,i,ds[i]);
      }

#endif

      //gpu_multxy(dz,ddiag,ds,nrows,blocks,threads); // precond z=diag*s
      cudaDevicemultxy(dz, ddiag, ds, nrows); // precond z=diag*s

      //gpu_spmv(dt,dz,nrows,dA,djA,diA,mattype,blocks,threads);
#ifdef BASIC_SPMV
      cudaDevicesetconst(dt, 0.0, nrows);
      __syncthreads();
      cudaDeviceSpmvCSC(dt, dz, nrows, dA, djA, diA);
#else
      __syncthreads();
      cudaDeviceSpmvCSC_block(dt, dz, nrows, dA, djA, diA);
#endif

      __syncthreads();///todo find why are needed
      //gpu_multxy(dAx2,ddiag,dt,nrows,blocks,threads);
      cudaDevicemultxy(dAx2, ddiag, dt, nrows);

      __syncthreads();
      //temp1=gpu_dotxy(dz, dAx2, aux, daux, nrows,(blocks + 1) / 2, threads);

      cudaDevicedotxy(dz, dAx2, &temp1, nrows, n_shr_empty);
      __syncthreads();

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

      if(i>=0){
        //printf("%d ddiag[%d] %-le\n",it,i,ddiag[i]);
        //printf("%d dt[%d] %-le\n",it,i,dt[i]);
        //printf("%d dAx2[%d] %-le\n",it,i,dAx2[i]);
        //printf("%d dz[%d] %-le\n",it,i,dz[i]);
      }

      if(i==0){
        printf("%d %d temp1 %-le\n",it,i,temp1);
      }

#endif

      __syncthreads();
      //temp2=gpu_dotxy(dAx2, dAx2, aux, daux, nrows,(blocks + 1) / 2, threads);

      cudaDevicedotxy(dAx2, dAx2, &temp2, nrows, n_shr_empty);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

      if(i==0){
        printf("%d %d temp2 %-le\n",it,i,temp2);
      }

#endif

      __syncthreads();
      omega0 = temp1 / temp2;
      //gpu_axpy(dx,dy,alpha,nrows,blocks,threads); // x=alpha*y +x
      cudaDeviceaxpy(dx, dy, alpha, nrows); // x=alpha*y +x

      __syncthreads();
      //gpu_axpy(dx,dz,omega0,nrows,blocks,threads);
      cudaDeviceaxpy(dx, dz, omega0, nrows);

      __syncthreads();
      //gpu_zaxpby(1.0,ds,-1.0*omega0,dt,dr0,nrows,blocks,threads);
      cudaDevicezaxpby(1.0, ds, -1.0 * omega0, dt, dr0, nrows);
      cudaDevicesetconst(dt, 0.0, nrows);

      __syncthreads();
      //temp1=gpu_dotxy(dr0, dr0, aux, daux, nrows,(blocks + 1) / 2, threads);

      cudaDevicedotxy(dr0, dr0, &temp1, nrows, n_shr_empty);

      temp1 = sqrt(temp1);

      rho0 = rho1;
      /**/
      __syncthreads();
      /**/

      //if (tid==0) it++;
      it++;
    } while(it<maxIt && temp1>tolmax);//while(it<maxIt && temp1>tolmax);//while(0);

    //todo itpointer should be an array of n_blocks size, and in cpu reduce to max number
    // (since the max its supposed to be the last to exit)
#ifndef PMC_DEBUG_GPU
    *it_pointer = it;
#endif

  }

/*
if (id == 0) //return aux variables if debugging
{
  aux_params[0]=alpha;
  aux_params[1]=rho0;
  aux_params[2]=omega0;
  aux_params[3]=beta;//0.01;
  aux_params[4]=rho1;//rho1
  aux_params[5]=temp1;
  aux_params[6]=temp2;
}
*/

}

//solveGPU_block: Each block will compute only a cell/group of cells
//Algorithm: Biconjugate gradient
void solveGPU_block(itsolver *bicg, double *dA, int *djA, int *diA, double *dx, double *dtempv)
{
  //Init variables ("public")
  int nrows = bicg->nrows;
  int threads = bicg->threads;
  int maxIt = bicg->maxIt;
  int mattype = bicg->mattype;
  int n_cells = bicg->n_cells;
  double tolmax = bicg->tolmax;
  double *ddiag = bicg->ddiag;

  // Auxiliary vectors ("private")
  double *dr0 = bicg->dr0;
  double *dr0h = bicg->dr0h;
  double *dn0 = bicg->dn0;
  double *dp0 = bicg->dp0;
  double *dt = bicg->dt;
  double *ds = bicg->ds;
  double *dAx2 = bicg->dAx2;
  double *dy = bicg->dy;
  double *dz = bicg->dz;
  double *daux = bicg->daux;

#ifndef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad==0) {
    printf("solveGPUBlock\n");
  }
#endif

//todo eliminate atomicadd in spmv through using CSR or something like that
  //gpu_spmv(dr0,dx,nrows,dA,djA,diA,mattype,bicg->blocks,threads);  // r0= A*x
/*
  gpu_axpby(dr0,dtempv,1.0,-1.0,nrows,blocks,threads); // r0=1.0*rhs+-1.0r0 //y=ax+by

  gpu_yequalsx(dr0h,dr0,nrows,blocks,threads);  //r0h=r0

  gpu_yequalsconst(dn0,0.0,nrows,blocks,threads);  //n0=0.0 //memset???
  gpu_yequalsconst(dp0,0.0,nrows,blocks,threads);  //p0=0.0

  alpha  = 1.0;
  rho0   = 1.0;
  omega0 = 1.0;
*/
  /*int n_aux_params=7;
  double *aux_params;
  aux_params=(double*)malloc(n_aux_params*sizeof(double));
  double *daux_params;
  cudaMalloc(&daux_params,n_aux_params*sizeof(double));*/
  //cudaMemcpy(bicg->djA,bicg->jA,7*sizeof(double),cudaMemcpyHostToDevice);

  int size_cell = nrows/n_cells;

#ifdef INDEPENDENCY_CELLS

  int max_threads_block = nextPowerOfTwo(size_cell);//bicg->threads;
  //int n_shr_empty = max_threads-size_cell;//nextPowerOfTwo(size_cell)-size_cell;
  //int threads_block = max_threads_block - n_shr_empty; //last multiple of size_cell before max_threads

#else

  int max_threads_block = bicg->threads;//bicg->threads; 128;

  //int n_shr_empty = max_threads%size_cell; //Wrong
  //int n_shr_empty = max_threads-nrows;

#endif

#ifdef BCG_ALL_THREADS

  int threads_block = max_threads_block;
  int n_shr_empty = 0;
  int blocks = (nrows+threads_block-1)/threads_block;

#else
  int n_cells_block =  max_threads_block/size_cell;
  int threads_block = n_cells_block*size_cell;
  int n_shr_empty = max_threads_block-threads_block;
  int blocks = (nrows+threads_block-1)/threads_block;
#endif

#ifndef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad==0) {
    printf("size_cell %d nrows %d max_threads_block %d blocks %d threads_block %d n_shr_empty %d\n",
           size_cell,nrows,max_threads_block,blocks,threads_block,n_shr_empty);
  }
#endif


  /*aux_params[0] = alpha;
  aux_params[1] = rho0;
  aux_params[2] = omega0;
  aux_params[3] = beta;
  aux_params[4] = rho1;
  aux_params[5] = temp1;
  aux_params[6] = temp2;
  cudaMemcpy(daux_params, aux_params, n_aux_params * sizeof(double), cudaMemcpyHostToDevice);*/

#ifndef PMC_DEBUG_GPU
  int it = 0;
  int *dit_ptr=bicg->counterBiConjGradInternalGPU;
  //int *dit_ptr;
  //cudaMalloc((void**)&dit_ptr,sizeof(int));
  //cudaMemcpy(dit_ptr, &it, sizeof(int), cudaMemcpyHostToDevice);
#endif


  cvsolveBcgCuda << < blocks, threads_block, max_threads_block * sizeof(double) >> >
                                           //cvsolveBcgCuda << < blocks, threads_block, threads_block * sizeof(double) >> >
                                           (dA, djA, diA, dx, dtempv, nrows, blocks, n_shr_empty, maxIt, mattype, n_cells
                                                   ,tolmax, ddiag, dr0, dr0h, dn0, dp0, dt, ds, dAx2, dy, dz, daux
#ifndef PMC_DEBUG_GPU
                                                   ,dit_ptr
#endif
                                                   //,daux_params
                                           );

#ifndef PMC_DEBUG_GPU
  cudaMemcpy(&it,dit_ptr,sizeof(int),cudaMemcpyDeviceToHost);
  bicg->counterBiConjGradInternal += it;

#ifndef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad==0) {
    printf("counterBiConjGradInternal %d\n",
           bicg->counterBiConjGradInternal);
  }
#endif

#endif

  /*cudaDeviceSynchronize();
  cudaMemcpy(aux_params, daux_params, n_aux_params * sizeof(double), cudaMemcpyDeviceToHost);

  alpha = aux_params[0];
  rho0 = aux_params[1];
  omega0 = aux_params[2];
  beta = aux_params[3];
  rho1 = aux_params[4];
  temp1 = aux_params[5];
  temp2 = aux_params[6];*/
  //printf("temp1 %-le", temp1);
  //printf("rho1 %f", rho1);

  //cudaFreeMem(daux_params);

}

void check_inputd(double *dx, int len, int var_id){

  double *x=(double*)malloc(len*sizeof(double));

  cudaMemcpy(x, dx, len*sizeof(double),cudaMemcpyDeviceToHost);

  int n_zeros=0;
  for (int i=0; i<5; i++){
    if(x[i]==0.0)
      n_zeros++;
    printf("%d[%d]=%-le\n",var_id,i,x[i]);
  }
  if(n_zeros==len)
    printf("%d is all zeros\n",var_id);

  free(x);
}

void check_inputi(int *dx, int len, int var_id){

  int *x=(int*)malloc(len*sizeof(int));

  cudaMemcpy(x, dx, len*sizeof(int),cudaMemcpyDeviceToHost);

  int n_zeros=0;
  for (int i=0; i<5; i++){
    if(x[i]==0.0)
      n_zeros++;
    printf("%d[%d]=%d\n",var_id,i,x[i]);
  }
  if(n_zeros==len)
    printf("%d is all zeros\n",var_id);

  free(x);
}

/*
void check_inputd2(double *x, int len, int var_id){

  int n_zeros=0;
  for (int i=0; i<5; i++){
    if(x[i]==0.0)
      n_zeros++;
    printf("%d[%d]=%-le\n",var_id,i,x[i]);
  }
  if(n_zeros==len)
    printf("%d is all zeros\n",var_id);

}
*/
__global__
void cvcheck_input_gpud(double *x, int len, int var_id)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<5)
  {
    printf("%d[%d]=%-le\n",var_id,i,x[i]);
  }
}

__global__
void cvcheck_input_gpui(int *x, int len, int var_id)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<5)
  {
    printf("%d[%d]=%d\n",var_id,i,x[i]);
  }
}

void check_input_solvegpu(itsolver *bicg, double *dA, int *djA, int *diA, double *dx, double *dtempv) {
  //Init variables ("public")
  int nrows = bicg->nrows;
  int blocks = bicg->blocks;
  int threads = bicg->threads;
  int maxIt = bicg->maxIt;
  int mattype = bicg->mattype;
  double tolmax = bicg->tolmax;
  double *ddiag = bicg->ddiag;

  // Auxiliary vectors ("private")
  double *dr0 = bicg->dr0;
  double *dr0h = bicg->dr0h;
  double *dn0 = bicg->dn0;
  double *dp0 = bicg->dp0;
  double *dt = bicg->dt;
  double *ds = bicg->ds;
  double *dAx2 = bicg->dAx2;
  double *dy = bicg->dy;
  double *dz = bicg->dz;
  double *aux = bicg->aux;
  double *daux = bicg->daux;

  int i=0;
  //check_inputd2(bicg->A,bicg->nnz,i++);
  //check_input(bicg->dA,bicg->nnz,i++);
  //check_input(bicg->dtempv,bicg->nrows,i++);
  //check_inputd(bicg->ddiag,bicg->nrows,i++);
  //check_input_gpu<< < 1, 5>> >(bicg->dA,bicg->nnz,i++);

  //check_input(bicg->djA,bicg->nnz*sizeof(int),i++);
  //check_input(bicg->diA,(bicg->nrows+1)*sizeof(int),i++);
  //check_input(dx,bicg->nrows*sizeof(double),i++);

  /*
  check_input(bicg->dA,bicg->nnz*sizeof(double),i++);
  check_input(bicg->dA,bicg->nnz*sizeof(double),i++);
  check_input(bicg->dA,bicg->nnz*sizeof(double),i++);

*/


}


//Algorithm: Biconjugate gradient
//dx[in/out]=last x solution (at start is 0)
//dtempv[in]=current x input
void solveGPU(itsolver *bicg, double *dA, int *djA, int *diA, double *dx, double *dtempv)
{
  //Init variables ("public")
  int nrows = bicg->nrows;
  int blocks = bicg->blocks;
  int threads = bicg->threads;
  int maxIt = bicg->maxIt;
  int mattype = bicg->mattype;
  double tolmax = bicg->tolmax;
  double *ddiag = bicg->ddiag;

  // Auxiliary vectors ("private")
  double *dr0 = bicg->dr0;
  double *dr0h = bicg->dr0h;
  double *dn0 = bicg->dn0;
  double *dp0 = bicg->dp0;
  double *dt = bicg->dt;
  double *ds = bicg->ds;
  double *dAx2 = bicg->dAx2;
  double *dy = bicg->dy;
  double *dz = bicg->dz;
  double *aux = bicg->aux;
  double *daux = bicg->daux;

#ifndef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad==0) {
    printf("solveGPU\n");
  }
#endif

#ifdef CHECK_SOLVEGPU

  if(bicg->counterBiConjGrad==0)
    check_input_solvegpu(bicg, dA, djA, diA, dx, dtempv);

#endif

  //Function private variables
  double alpha,rho0,omega0,beta,rho1,temp1,temp2;

  gpu_spmv(dr0,dx,nrows,dA,djA,diA,mattype,blocks,threads);  // r0= A*x

#ifdef CHECK_SOLVEGPU

  int k=0;
  //cvcheck_input_gpud<< < 1, 5>> >(dr0,bicg->nrows,k++);
  cvcheck_input_gpud<< < 1, 5>> >(dx,bicg->nrows,k++);
  cvcheck_input_gpud<< < 1, 5>> >(dtempv,bicg->nrows,k++);
  //cvcheck_input_gpud<< < 1, 5>> >(bicg->dA,bicg->nnz,k++);
  //cvcheck_input_gpui<< < 1, 5>> >(bicg->djA,bicg->nnz,k++);
  //cvcheck_input_gpui<< < 1, 5>> >(bicg->diA,(bicg->nrows+1),k++);

#endif

  gpu_axpby(dr0,dtempv,1.0,-1.0,nrows,blocks,threads); // r0=1.0*rhs+-1.0r0 //y=ax+by

  gpu_yequalsx(dr0h,dr0,nrows,blocks,threads);  //r0h=r0

  gpu_yequalsconst(dn0,0.0,nrows,blocks,threads);  //n0=0.0 //memset???
  gpu_yequalsconst(dp0,0.0,nrows,blocks,threads);  //p0=0.0

  alpha  = 1.0;
  rho0   = 1.0;
  omega0 = 1.0;

  //printf("temp1 %-le", temp1);
  //printf("rho1 %f", rho1);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

  double *aux_x1;
  aux_x1=(double*)malloc(bicg->nrows*sizeof(double));

#endif

  //for(int it=0;it<maxIt;it++){
  int it=0;
  do {

    rho1=gpu_dotxy(dr0, dr0h, aux, daux, nrows,(blocks + 1) / 2, threads);//rho1 =<r0,r0h>
    //rho1=gpu_dotxy(dr0, dr0h, aux, daux, nrows,blocks, threads);//rho1 =<r0,r0h>

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    //good here first iter
    printf("%d rho1 %-le\n",it,rho1);
#endif

    beta=(rho1/rho0)*(alpha/omega0);

    //    cout<<"rho1 "<<rho1<<" beta "<<beta<<endl;

    gpu_zaxpbypc(dp0,dr0,dn0,beta,-1.0*omega0*beta,nrows,blocks,threads);   //z = ax + by + c

    gpu_multxy(dy,ddiag,dp0,nrows,blocks,threads);  // precond y= p0*diag

    gpu_spmv(dn0,dy,nrows,dA,djA,diA,mattype,blocks,threads);  // n0= A*y

    temp1=gpu_dotxy(dr0h, dn0, aux, daux, nrows,(blocks + 1) / 2, threads);
    //temp1=gpu_dotxy(dr0h, dn0, aux, daux, nrows, blocks, threads);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    printf("%d temp1 %-le\n",it,temp1);
#endif

    alpha=rho1/temp1;

    //       cout<<"temp1 "<<temp1<<" alpha "<<alpha<<endl;

    gpu_zaxpby(1.0,dr0,-1.0*alpha,dn0,ds,nrows,blocks,threads);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    cudaMemcpy(aux_x1,ds,bicg->nrows*sizeof(double),cudaMemcpyDeviceToHost);

    printf("%d ds[0] %-le\n",it,aux_x1[0]);

#endif

    gpu_multxy(dz,ddiag,ds,nrows,blocks,threads); // precond z=diag*s

    gpu_spmv(dt,dz,nrows,dA,djA,diA,mattype,blocks,threads);

    gpu_multxy(dAx2,ddiag,dt,nrows,blocks,threads);

    temp1=gpu_dotxy(dz, dAx2, aux, daux, nrows,(blocks + 1) / 2, threads);
    //temp1=gpu_dotxy(dz, dAx2, aux, daux, nrows,blocks, threads);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    printf("%d temp1 %-le\n",it,temp1);
#endif

    temp2=gpu_dotxy(dAx2, dAx2, aux, daux, nrows,(blocks + 1) / 2, threads);
    //temp2=gpu_dotxy(dAx2, dAx2, aux, daux, nrows,blocks, threads);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    printf("%d temp2 %-le\n",it,temp2);
#endif

    omega0= temp1/temp2;

    gpu_axpy(dx,dy,alpha,nrows,blocks,threads); // x=alpha*y +x

    gpu_axpy(dx,dz,omega0,nrows,blocks,threads);

    gpu_zaxpby(1.0,ds,-1.0*omega0,dt,dr0,nrows,blocks,threads);

    temp1=gpu_dotxy(dr0, dr0, aux, daux, nrows,(blocks + 1) / 2, threads);
    //temp1=gpu_dotxy(dr0, dr0, aux, daux, nrows,blocks, threads);
    temp1=sqrt(temp1);

    //cout<<it<<": "<<temp1<<endl;

    rho0=rho1;

    //if(temp1<tolmax){
    //  break;}}

    it++;
  }while(it<maxIt && temp1>tolmax);

#ifdef CHECK_SOLVEGPU

  printf("output dx:\n");

  //int k=0;
  //cvcheck_input_gpud<< < 1, 5>> >(dr0,bicg->nrows,k++);

  //cvcheck_input_gpud<< < 1, 5>> >(dtempv,bicg->nrows,k++);
  /*cvcheck_input_gpud<< < 1, 5>> >(bicg->dA,bicg->nrows,k++);
  cvcheck_input_gpud<< < 1, 5>> >(bicg->dA,bicg->nrows,k++);
  cvcheck_input_gpud<< < 1, 5>> >(bicg->dA,bicg->nrows,k++);
  cvcheck_input_gpud<< < 1, 5>> >(bicg->dA,bicg->nrows,k++);*/

  //bicg->dx = dx;
/*
  cvcheck_input_gpud<< < 1, 5>> >(dx,bicg->nrows,k++);
  check_inputd(bicg->dx,bicg->nrows,k++);
  check_inputd(dx,bicg->nrows,k++);*/


#endif

#ifndef PMC_DEBUG_GPU
  bicg->counterBiConjGradInternal += it;
#endif

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
  free(aux_x1);
#endif

}

void free_itsolver(itsolver *bicg)
{
  //Auxiliary vectors ("private")
  double ** dr0 = &bicg->dr0;
  double ** dr0h = &bicg->dr0h;
  double ** dn0 = &bicg->dn0;
  double ** dp0 = &bicg->dp0;
  double ** dt = &bicg->dt;
  double ** ds = &bicg->ds;
  double ** dAx2 = &bicg->dAx2;
  double ** dy = &bicg->dy;
  double ** dz = &bicg->dz;
  double ** daux = &bicg->daux;
  double ** ddiag = &bicg->ddiag;

  cudaFree(dr0);
  cudaFree(dr0h);
  cudaFree(dn0);
  cudaFree(dp0);
  cudaFree(dt);
  cudaFree(ds);
  cudaFree(dAx2);
  cudaFree(dy);
  cudaFree(dz);
  cudaFree(ddiag);
  cudaFree(daux);
  free(bicg->aux);

}



#endif



#ifdef __cplusplus
}
#endif

/*
void setUpSolver(itsolver *bicg, double reltol, double *ewt, int tnrows,int tnnz,double *tA, int *tjA, int *tiA, int tmattype, int qmax, double *dACamp, double *dftempCamp);
{

 bicg.tolmax=reltol;

}
*/

