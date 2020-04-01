/* Copyright (C) 2020 Christian Guzman and Guillermo Oyarzun
 * Licensed under the GNU General Public License version 1 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Iterative GPU solver
 *
 */

//todo Reorder class elements, GPU variables like dA has to be accessed during all the
//GPU ODE solving, so convert it to a more general class and move setup & solve to a
//library file (separate from libsolv.cu, since libsolv.cu is only basic functions and
//is used by itsolver)

#ifndef ITSOLVERGPU_H
#define ITSOLVERGPU_H

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<iostream>
#include"libsolv.h"

#define NUM_TESTS    5     /* number of error test quantities     */

using namespace std;

class itsolver{

   public:
	   itsolver(){};
	   itsolver(int, int, double*, int*, int*,int);
	   ~itsolver(){};

      // matrix data
      double* A;
      int*    jA;
      int*    iA;
      int     nrows;
      int     nnz;

      //GPU pointers
      double* dA;
      int*    djA;
      int*    diA;
      double* dx;
      double* drhs;
      double* ddiag;
      double* aux;
      double* daux;

      // Allocate ewt, acor, tempv, ftemp
      double* dewt;
      double* dacor;
      double* dacor_init;
      double* dtempv;
      double* dftemp;
      double* dzn;
      double* dcv_y;
      //double* dcv_tq[NUM_TESTS+1]; /* array of test quantities indexed from 1 to NUM_TESTS(=5) */

      int threads,blocks;

      // Auxiliary scalars
      double tolmax;
      int maxIt;
      int mattype;

      int totits;
      double totres;
      // Auxiliary vectors
      double * dr0;
      double * dr0h;
      double * dn0;
      double * dp0;
      double * dt;
      double * ds;
      double * dAx;
      double * dAx2;
      double * dy;
      double * dz;
      double * diag;

      //subroutines
      void setUpSolver(double *,int ,int ,double *, int *, int *, int);
      void matScaleAddI(double);
      void setUpGPU();
      void solveGPU(double*, double*);
      void yequalsx(double*, double*);

      /*void spmv(double*, double*);
      void axpby(double*,double*,double,double);
      //void yequalsx(double*, double*);
      void yequalsconst(double*, double);
      double dotxy(double*, double*);
      void zaxpbypc(double *, double* , double*, double , double );
      void multxy(double*,double*, double*);
      void zaxpby(double *, double*, double*, double , double );
      void axpy(double* , double*, double );
      void transferXandRHSToGPU(double *, double *);
      void transferSolution(double *);*/



};


itsolver::itsolver(int tnrows,int tnnz,double *tA, int *tjA, int *tiA, int tmattype)
{
  /*
  //Linking Matrix data, later this data must be allocated in GPU
  nrows=tnrows;
  nnz=tnnz;
  mattype=tmattype;
  A=tA;
  jA=tjA;
  iA=tiA;

  threads=1024; //128
  blocks=(nrows+threads-1)/threads;

  // allocating matrix data to the GPU
  cudaMallocDouble(dA,nnz);
  cudaMallocInt(djA,nnz);
  cudaMallocInt(diA,(nrows+1));

  cudaMallocDouble(dx,nrows);
  cudaMallocDouble(drhs,nrows);
  // moving matrix data to the GPU

  cudaMemcpyDToGpu(A,dA,nnz);
  cudaMemcpyIToGpu(jA,djA,nnz);
  cudaMemcpyIToGpu(iA,diA,(nrows+1));

  cudaMallocDouble(dr0,nrows);
  cudaMallocDouble(dr0h,nrows);
  cudaMallocDouble(dn0,nrows);
  cudaMallocDouble(dp0,nrows);
  cudaMallocDouble(dt,nrows);
  cudaMallocDouble(ds,nrows);
  cudaMallocDouble(dAx,nrows);
  cudaMallocDouble(dAx2,nrows);
  cudaMallocDouble(dy,nrows);
  cudaMallocDouble(dz,nrows);
  cudaMallocDouble(ddiag,nrows);

  diag=(double*)malloc(sizeof(double)*nrows);

  maxIt=100;
  tolmax=1e-6;//1e-6
*/
}

void itsolver::setUpSolver(double *ewt, int tnrows,int tnnz,double *tA, int *tjA, int *tiA, int tmattype){

  //Linking Matrix data, later this data must be allocated in GPU
  nrows=tnrows;
  nnz=tnnz;
  mattype=tmattype;
  A=tA;
  jA=tjA;
  iA=tiA;

  threads=1024; //128 //todo set at max gpu (maybe at late when gpu ode solving finish)
  blocks=(nrows+threads-1)/threads;

  // allocating matrix data to the GPU
  cudaMallocDouble(dA,nnz);
  cudaMallocInt(djA,nnz);
  cudaMallocInt(diA,(nrows+1));

  cudaMallocDouble(dx,nrows);
  cudaMallocDouble(drhs,nrows);
  // moving matrix data to the GPU

  cudaMemcpyDToGpu(A,dA,nnz);
  cudaMemcpyIToGpu(jA,djA,nnz);
  cudaMemcpyIToGpu(iA,diA,(nrows+1));

  cudaMallocDouble(dr0,nrows);
  cudaMallocDouble(dr0h,nrows);
  cudaMallocDouble(dn0,nrows);
  cudaMallocDouble(dp0,nrows);
  cudaMallocDouble(dt,nrows);
  cudaMallocDouble(ds,nrows);
  cudaMallocDouble(dAx,nrows);
  cudaMallocDouble(dAx2,nrows);
  cudaMallocDouble(dy,nrows);
  cudaMallocDouble(dz,nrows);
  cudaMallocDouble(ddiag,nrows);

  aux=(double*)malloc(sizeof(double)*blocks);
  cudaMallocDouble(daux,nrows);

  // Allocate ewt, acor, tempv, ftemp
  cudaMallocDouble(dewt,nrows);
  cudaMallocDouble(dacor,nrows);
  //cudaMallocDouble(dacor_init,nrows);
  cudaMallocDouble(dtempv,nrows);
  cudaMallocDouble(dftemp,nrows);
  cudaMallocDouble(dzn,nrows*NUM_TESTS);//*NUM_TESTS
  cudaMallocDouble(dcv_y,nrows);//if anything, set to zero to avoid seg fault errors

  cudaMemcpyDToGpu(ewt,dewt,nnz);
  cudaMemcpyDToGpu(ewt,dacor,nnz);
  cudaMemcpyDToGpu(ewt,dtempv,nnz);
  cudaMemcpyDToGpu(ewt,dftemp,nnz);

  diag=(double*)malloc(sizeof(double)*nrows);

  maxIt=100;
  tolmax=1e-6;//1e-6
//todo set tolmax to same accuracy than CVODE (I think is the default, but better receive it from camp)
}

void itsolver::setUpGPU()
{

  gpu_diagprecond(nrows,dA,djA,diA,ddiag,blocks,threads);

	//todo if indices has changed copy again
  //cudaMemcpyDToGpu(A,dA,nnz);
  //cudaMemcpyIToGpu(jA,djA,nnz);
  //cudaMemcpyIToGpu(iA,diA,(nrows+1));

}

//Biconugate gradient
//todo check communications cpu/gpu (e.g. cublas in dotxy is creating and destroying a handle each time)
//solution: cudastreams for blas reduce moments
//void itsolver::solveGPU(double *rhs, double *x)
void itsolver::solveGPU(double *rhs, double *x)
{
  double alpha,rho0,omega0,beta,rho1,temp1,temp2;

  cudaMemcpyDToGpu(rhs,drhs,nrows);
  cudaMemcpyDToGpu(x,dx,nrows);

  gpu_spmv(dr0,dx,nrows,dA,djA,diA,mattype,blocks,threads);  // r0= A*x

  gpu_axpby(dr0,drhs,1.0,-1.0,nrows,blocks,threads); // r0=1.0*rhs+-1.0r0        //y=ax+by

  gpu_yequalsx(dr0h,dr0,nrows,blocks,threads);  //r0h=r0

  alpha  = 1.0;
  rho0   = 1.0;
  omega0 = 1.0;

  gpu_yequalsconst(dn0,0.0,nrows,blocks,threads);  //n0=0.0
  gpu_yequalsconst(dp0,0.0,nrows,blocks,threads);  //p0=0.0

  for(int it=0;it<maxIt;it++){

    rho1=gpu_dotxy(dr0,dr0h,nrows); //rho1 =<r0,r0h>

    //  cout<<"rho1 "<<rho1<<endl;
    beta=(rho1/rho0)*(alpha/omega0);

    //    cout<<"rho1 "<<rho1<<" beta "<<beta<<endl;

    gpu_zaxpbypc(dp0,dr0,dn0,beta,-1.0*omega0*beta,nrows,blocks,threads);   //z = ax + by + c

    gpu_multxy(dy,ddiag,dp0,nrows,blocks,threads);  // precond y= p0*diag

    gpu_spmv(dn0,dy,nrows,dA,djA,diA,mattype,blocks,threads);  // n0= A*y

    temp1=gpu_dotxy(dr0h,dn0,nrows);

    alpha=rho1/temp1;

    //       cout<<"temp1 "<<temp1<<" alpha "<<alpha<<endl;

    gpu_zaxpby(1.0,dr0,-1.0*alpha,dn0,ds,nrows,blocks,threads);
    //gpu_zaxpby(double a, double* dx ,double b, double* dy, double* dz, int nrows, int blocks, int threads);

    //gpu_zaxpby(ds,dr0,dn0,1.0,-1.0*alpha,nrows,blocks,threads);
    //gpu_zaxpby(double* dz, double* dx ,double* dy, double a, double b, int nrows, int blocks, int threads);

    gpu_multxy(dz,ddiag,ds,nrows,blocks,threads); // precond z=diag*s

    gpu_spmv(dt,dz,nrows,dA,djA,diA,mattype,blocks,threads);

    gpu_multxy(dAx2,ddiag,dt,nrows,blocks,threads);

    temp1=gpu_dotxy(dz,dAx2,nrows);

    temp2=gpu_dotxy(dAx2,dAx2,nrows);

    omega0= temp1/temp2;

    gpu_axpy(dx,dy,alpha,nrows,blocks,threads); // x=alpha*y +x

    gpu_axpy(dx,dz,omega0,nrows,blocks,threads);

    gpu_zaxpby(1.0,ds,-1.0*omega0,dt,dr0,nrows,blocks,threads);
    //gpu_zaxpby(double a, double* dx ,double b, double* dy, double* dz, int nrows, int blocks, int threads);

    //gpu_zaxpby(dr0,ds,dt,1.0,-1.0*omega0,nrows,blocks,threads);
    //gpu_zaxpby(double* dz, double* dx ,double* dy, double a, double b, int nrows, int blocks, int threads);

    temp1=gpu_dotxy(dr0,dr0,nrows);
    temp1=sqrt(temp1);

    cout<<it<<": "<<temp1<<endl;

    if(temp1<tolmax){
      totits=it;
      totres=temp1;
      break;
    }
    rho0=rho1;
  }

   //cudaMemcpyDToCpu(x,dx,nrows);
}

#endif