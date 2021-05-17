/*
 * -----------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ SMU
 *                Radu Serban @ LLNL
 * -----------------------------------------------------------------
 * LLNS/SMU Copyright Start
 * Copyright (c) 2017, Southern Methodist University and
 * Lawrence Livermore National Security
 *
 * This work was performed under the auspices of the U.S. Department
 * of Energy by Southern Methodist University and Lawrence Livermore
 * National Laboratory under Contract DE-AC52-07NA27344.
 * Produced at Southern Methodist University and the Lawrence
 * Livermore National Laboratory.
 *
 * All rights reserved.
 * For details, see the LICENSE file.
 * LLNS/SMU Copyright End
 * -----------------------------------------------------------------
 * This is the implementation file for the CVDLS linear solver
 * interface
 * -----------------------------------------------------------------
 */

/*=================================================================
  IMPORTED HEADER FILES
  =================================================================*/

#include <stdio.h>
#include <stdlib.h>

#include "cvode_impl.h"
#include "cvode_direct_impl.h"
#include <sundials/sundials_math.h>
#include <sunmatrix/sunmatrix_band.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunmatrix/sunmatrix_sparse.h>

//#include "test_cpu.h"

#include "itsolver_gpu.h"
#include "libsolv.h"

#ifdef PMC_USE_GPU

#endif

#ifdef PMC_PROFILING
#include <mpi.h>
#endif

/*=================================================================
  FUNCTION SPECIFIC CONSTANTS
  =================================================================*/

/* Constant for DQ Jacobian approximation */
#define MIN_INC_MULT RCONST(1000.0)

#define ZERO         RCONST(0.0)
#define ONE          RCONST(1.0)
#define TWO          RCONST(2.0)

#include <time.h>

/*=================================================================
  EXPORTED FUNCTIONS -- REQUIRED
  =================================================================*/

#ifdef PMC_USE_GPU

//#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void alloc_solver_gpu_cvode(void *cvode_mem)
{
  CVodeMem cv_mem = (CVodeMem) cvode_mem;
  itsolver *bicg = &cv_mem->bicg;
  CVDlsMem cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;
  SUNMatrix J = cvdls_mem->A;

  //Init GPU ODE solver variables
  //Linking Matrix data, later this data must be allocated in GPU
  //bicg->n_cells=1;//md->n_cells;

  bicg->n_cells=10;//md->n_cells;

  bicg->nnz=SM_NNZ_S(J);
  bicg->nrows=SM_NP_S(J);
  bicg->A=(double*)SM_DATA_S(J);

  //Using int per default as sundindextype gives wrong results in CPU, so translate from int64 to int
  bicg->jA=(int*)malloc(sizeof(int)*SM_NNZ_S(J));
  bicg->iA=(int*)malloc(sizeof(int)*(SM_NP_S(J)+1));
  for(int i=0;i<SM_NNZ_S(J);i++)
    bicg->jA[i]=SM_INDEXVALS_S(J)[i];
  for(int i=0;i<=SM_NP_S(J);i++){
    bicg->iA[i]=SM_INDEXPTRS_S(J)[i];
    //printf("bicg->iA %d[%d]",bicg->iA[i],i);
  }

  // Allocating matrix data to the GPU
  cudaMalloc((void**)&bicg->dA,bicg->nnz*sizeof(double));
  cudaMalloc((void**)&bicg->djA,bicg->nnz*sizeof(int));
  cudaMalloc((void**)&bicg->diA,(bicg->nrows+1)*sizeof(int));

  //ODE aux variables
  cudaMalloc((void**)&bicg->dewt,bicg->nrows*sizeof(double));
  cudaMalloc((void**)&bicg->dacor,bicg->nrows*sizeof(double));
  cudaMalloc((void**)&bicg->dtempv,bicg->nrows*sizeof(double));
  //cudaMalloc((void**)&bicg->dzn,bicg->nrows*(cv_mem->cv_qmax+1)*sizeof(double));

  //ODE concs arrays
  cudaMalloc((void**)&bicg->dcv_y,bicg->nrows*sizeof(double));
  cudaMalloc((void**)&bicg->dx,bicg->nrows*sizeof(double));

  double *ewt = N_VGetArrayPointer(cv_mem->cv_ewt);
  double *tempv = N_VGetArrayPointer(cv_mem->cv_tempv);
  cudaMemcpy(bicg->djA,bicg->jA,bicg->nnz*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(bicg->diA,bicg->iA,(bicg->nrows+1)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(bicg->dewt,ewt,bicg->nrows*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(bicg->dacor,ewt,bicg->nrows*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(bicg->dftemp,ewt,bicg->nrows*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemset(bicg->dx, 0.0, bicg->nrows*sizeof(double));
  cudaMemset(bicg->dtempv, 0.0, bicg->nrows*sizeof(double));
  //cudaMemcpy(bicg->dx,tempv,bicg->nnz*sizeof(double),cudaMemcpyHostToDevice);

  cudaMemcpy(bicg->dA,bicg->A,bicg->nnz*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(bicg->djA,bicg->jA,bicg->nnz*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(bicg->diA,bicg->iA,(bicg->nrows+1)*sizeof(int),cudaMemcpyHostToDevice);

  //Init Linear Solver variables
  createSolver(bicg);

  //gpu_diagprecond(bicg->nrows,bicg->dA,bicg->djA,bicg->diA,bicg->ddiag,bicg->blocks,bicg->threads); //Setup linear solver






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

  //test_print2();

}

void print_counterBiConjGrad(CVodeMem cv_mem)
{

  itsolver *bicg = &cv_mem->bicg;

  printf("counterBiConjGrad2 %d\n", bicg->counterBiConjGrad);

}

#endif

/*---------------------------------------------------------------
 CVDlsSetLinearSolver specifies the direct linear solver.
---------------------------------------------------------------*/
int CVDlsSetLinearSolver(void *cvode_mem, SUNLinearSolver LS,
                         SUNMatrix A)
{
  CVodeMem cv_mem;
  CVDlsMem cvdls_mem;

  /* Return immediately if any input is NULL */
  if (cvode_mem == NULL) {
    cvProcessError(NULL, CVDLS_MEM_NULL, "CVDLS",
                   "CVDlsSetLinearSolver", MSGD_CVMEM_NULL);
    return(CVDLS_MEM_NULL);
  }
  if ( (LS == NULL)  || (A == NULL) ) {
    cvProcessError(NULL, CVDLS_ILL_INPUT, "CVDLS",
                   "CVDlsSetLinearSolver",
                    "Both LS and A must be non-NULL");
    return(CVDLS_ILL_INPUT);
  }
  cv_mem = (CVodeMem) cvode_mem;

  /* Test if solver and vector are compatible with DLS */
  if (SUNLinSolGetType(LS) != SUNLINEARSOLVER_DIRECT) {
    cvProcessError(cv_mem, CVDLS_ILL_INPUT, "CVDLS",
                   "CVDlsSetLinearSolver",
                   "Non-direct LS supplied to CVDls interface");
    return(CVDLS_ILL_INPUT);
  }
  if (cv_mem->cv_tempv->ops->nvgetarraypointer == NULL ||
      cv_mem->cv_tempv->ops->nvsetarraypointer == NULL) {
    cvProcessError(cv_mem, CVDLS_ILL_INPUT, "CVDLS",
                   "CVDlsSetLinearSolver", MSGD_BAD_NVECTOR);
    return(CVDLS_ILL_INPUT);
  }

  /* free any existing system solver attached to CVode */
  if (cv_mem->cv_lfree)  cv_mem->cv_lfree(cv_mem);

  /* Set four main system linear solver function fields in cv_mem */
  cv_mem->cv_linit  = cvDlsInitialize;
  cv_mem->cv_lsetup = cvDlsSetup;
  cv_mem->cv_lsolve = cvDlsSolve;
  cv_mem->cv_lfree  = cvDlsFree;

  /* Get memory for CVDlsMemRec */
  cvdls_mem = NULL;
  cvdls_mem = (CVDlsMem) malloc(sizeof(struct CVDlsMemRec));
  if (cvdls_mem == NULL) {
    cvProcessError(cv_mem, CVDLS_MEM_FAIL, "CVDLS",
                    "CVDlsSetLinearSolver", MSGD_MEM_FAIL);
    return(CVDLS_MEM_FAIL);
  }

  /* set SUNLinearSolver pointer */
  cvdls_mem->LS = LS;

  /* Initialize Jacobian-related data */
  cvdls_mem->jacDQ = SUNTRUE;
  cvdls_mem->jac = cvDlsDQJac;
  cvdls_mem->J_data = cv_mem;
  cvdls_mem->last_flag = CVDLS_SUCCESS;

  /* Initialize counters */
  cvDlsInitializeCounters(cvdls_mem);

  /* Store pointer to A and create saved_J */
  cvdls_mem->A = A;
  cvdls_mem->savedJ = SUNMatClone(A);
  if (cvdls_mem->savedJ == NULL) {
    cvProcessError(cv_mem, CVDLS_MEM_FAIL, "CVDLS",
                    "CVDlsSetLinearSolver", MSGD_MEM_FAIL);
    free(cvdls_mem); cvdls_mem = NULL;
    return(CVDLS_MEM_FAIL);
  }

  /* Allocate memory for x */
  cvdls_mem->x = N_VClone(cv_mem->cv_tempv);
  if (cvdls_mem->x == NULL) {
    cvProcessError(cv_mem, CVDLS_MEM_FAIL, "CVDLS",
                    "CVDlsSetLinearSolver", MSGD_MEM_FAIL);
    SUNMatDestroy(cvdls_mem->savedJ);
    free(cvdls_mem); cvdls_mem = NULL;
    return(CVDLS_MEM_FAIL);
  }
  /* Attach linear solver memory to integrator memory */
  cv_mem->cv_lmem = cvdls_mem;

#ifdef PMC_USE_GPU
  alloc_solver_gpu_cvode(cv_mem);
  //print_counterBiConjGrad(cv_mem);
#endif

  return(CVDLS_SUCCESS);
}


/*
 * =================================================================
 * EXPORTED FUNCTIONS -- OPTIONAL
 * =================================================================
 */

/* CVDlsSetJacFn specifies the Jacobian function. */
int CVDlsSetJacFn(void *cvode_mem, CVDlsJacFn jac)
{
  CVodeMem cv_mem;
  CVDlsMem cvdls_mem;

  /* Return immediately if cvode_mem or cv_mem->cv_lmem are NULL */
  if (cvode_mem == NULL) {
    cvProcessError(NULL, CVDLS_MEM_NULL, "CVDLS",
                   "CVDlsSetJacFn", MSGD_CVMEM_NULL);
    return(CVDLS_MEM_NULL);
  }
  cv_mem = (CVodeMem) cvode_mem;
  if (cv_mem->cv_lmem == NULL) {
    cvProcessError(cv_mem, CVDLS_LMEM_NULL, "CVDLS",
                   "CVDlsSetJacFn", MSGD_LMEM_NULL);
    return(CVDLS_LMEM_NULL);
  }
  cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;

  if (jac != NULL) {
    cvdls_mem->jacDQ  = SUNFALSE;
    cvdls_mem->jac    = jac;
    cvdls_mem->J_data = cv_mem->cv_user_data;
  } else {
    cvdls_mem->jacDQ  = SUNTRUE;
    cvdls_mem->jac    = cvDlsDQJac;
    cvdls_mem->J_data = cv_mem;
  }

  return(CVDLS_SUCCESS);
}


/* CVDlsGetWorkSpace returns the length of workspace allocated for the
   CVDLS linear solver. */
int CVDlsGetWorkSpace(void *cvode_mem, long int *lenrwLS,
                      long int *leniwLS)
{
  CVodeMem cv_mem;
  CVDlsMem cvdls_mem;
  sunindextype lrw1, liw1;
  long int lrw, liw;

  /* Return immediately if cvode_mem or cv_mem->cv_lmem are NULL */
  if (cvode_mem == NULL) {
    cvProcessError(NULL, CVDLS_MEM_NULL, "CVDLS",
                   "CVDlsGetWorkSpace", MSGD_CVMEM_NULL);
    return(CVDLS_MEM_NULL);
  }
  cv_mem = (CVodeMem) cvode_mem;
  if (cv_mem->cv_lmem == NULL) {
    cvProcessError(cv_mem, CVDLS_LMEM_NULL, "CVDLS",
                   "CVDlsGetWorkSpace", MSGD_LMEM_NULL);
    return(CVDLS_LMEM_NULL);
  }
  cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;

  /* initialize outputs with requirements from CVDlsMem structure */
  *lenrwLS = 0;
  *leniwLS = 4;

  /* add NVector size */
  if (cvdls_mem->x->ops->nvspace) {
    N_VSpace(cvdls_mem->x, &lrw1, &liw1);
    *lenrwLS = lrw1;
    *leniwLS = liw1;
  }

  /* add SUNMatrix size (only account for the one owned by Dls interface) */
  if (cvdls_mem->savedJ->ops->space) {
    (void) SUNMatSpace(cvdls_mem->savedJ, &lrw, &liw);
    *lenrwLS += lrw;
    *leniwLS += liw;
  }

  /* add LS sizes */
  if (cvdls_mem->LS->ops->space) {
    (void) SUNLinSolSpace(cvdls_mem->LS, &lrw, &liw);
    *lenrwLS += lrw;
    *leniwLS += liw;
  }

  return(CVDLS_SUCCESS);
}


/* CVDlsGetNumJacEvals returns the number of Jacobian evaluations. */
int CVDlsGetNumJacEvals(void *cvode_mem, long int *njevals)
{
  CVodeMem cv_mem;
  CVDlsMem cvdls_mem;

  /* Return immediately if cvode_mem or cv_mem->cv_lmem are NULL */
  if (cvode_mem == NULL) {
    cvProcessError(NULL, CVDLS_MEM_NULL, "CVDLS",
                   "CVDlsGetNumJacEvals", MSGD_CVMEM_NULL);
    return(CVDLS_MEM_NULL);
  }
  cv_mem = (CVodeMem) cvode_mem;
  if (cv_mem->cv_lmem == NULL) {
    cvProcessError(cv_mem, CVDLS_LMEM_NULL, "CVDLS",
                   "CVDlsGetNumJacEvals", MSGD_LMEM_NULL);
    return(CVDLS_LMEM_NULL);
  }
  cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;

  *njevals = cvdls_mem->nje;

  return(CVDLS_SUCCESS);
}


/* CVDlsGetNumRhsEvals returns the number of calls to the ODE function
   needed for the DQ Jacobian approximation. */
int CVDlsGetNumRhsEvals(void *cvode_mem, long int *nfevalsLS)
{
  CVodeMem cv_mem;
  CVDlsMem cvdls_mem;

  /* Return immediately if cvode_mem or cv_mem->cv_lmem are NULL */
  if (cvode_mem == NULL) {
    cvProcessError(NULL, CVDLS_MEM_NULL, "CVDLS",
                   "CVDlsGetNumRhsEvals", MSGD_CVMEM_NULL);
    return(CVDLS_MEM_NULL);
  }
  cv_mem = (CVodeMem) cvode_mem;
  if (cv_mem->cv_lmem == NULL) {
    cvProcessError(cv_mem, CVDLS_LMEM_NULL, "CVDLS",
                   "CVDlsGetNumRhsEvals", MSGD_LMEM_NULL);
    return(CVDLS_LMEM_NULL);
  }
  cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;

  *nfevalsLS = cvdls_mem->nfeDQ;

  return(CVDLS_SUCCESS);
}


/* CVDlsGetReturnFlagName returns the name associated with a CVDLS
   return value. */
char *CVDlsGetReturnFlagName(long int flag)
{
  char *name;

  name = (char *)malloc(30*sizeof(char));

  switch(flag) {
  case CVDLS_SUCCESS:
    sprintf(name,"CVDLS_SUCCESS");
    break;
  case CVDLS_MEM_NULL:
    sprintf(name,"CVDLS_MEM_NULL");
    break;
  case CVDLS_LMEM_NULL:
    sprintf(name,"CVDLS_LMEM_NULL");
    break;
  case CVDLS_ILL_INPUT:
    sprintf(name,"CVDLS_ILL_INPUT");
    break;
  case CVDLS_MEM_FAIL:
    sprintf(name,"CVDLS_MEM_FAIL");
    break;
  case CVDLS_JACFUNC_UNRECVR:
    sprintf(name,"CVDLS_JACFUNC_UNRECVR");
    break;
  case CVDLS_JACFUNC_RECVR:
    sprintf(name,"CVDLS_JACFUNC_RECVR");
    break;
  case CVDLS_SUNMAT_FAIL:
    sprintf(name,"CVDLS_SUNMAT_FAIL");
    break;
  default:
    sprintf(name,"NONE");
  }

  return(name);
}


/* CVDlsGetLastFlag returns the last flag set in a CVDLS function. */
int CVDlsGetLastFlag(void *cvode_mem, long int *flag)
{
  CVodeMem cv_mem;
  CVDlsMem cvdls_mem;

  /* Return immediately if cvode_mem or cv_mem->cv_lmem are NULL */
  if (cvode_mem == NULL) {
    cvProcessError(NULL, CVDLS_MEM_NULL, "CVDLS",
                   "CVDlsGetLastFlag", MSGD_CVMEM_NULL);
    return(CVDLS_MEM_NULL);
  }
  cv_mem = (CVodeMem) cvode_mem;
  if (cv_mem->cv_lmem == NULL) {
    cvProcessError(cv_mem, CVDLS_LMEM_NULL, "CVDLS",
                   "CVDlsGetLastFlag", MSGD_LMEM_NULL);
    return(CVDLS_LMEM_NULL);
  }
  cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;

  *flag = cvdls_mem->last_flag;

  return(CVDLS_SUCCESS);
}


/*=================================================================
  CVDLS PRIVATE FUNCTIONS
  =================================================================*/


/*-----------------------------------------------------------------
  cvDlsDQJac
  -----------------------------------------------------------------
  This routine is a wrapper for the Dense and Band
  implementations of the difference quotient Jacobian
  approximation routines.
  ---------------------------------------------------------------*/
int cvDlsDQJac(realtype t, N_Vector y, N_Vector fy,
               SUNMatrix Jac, void *cvode_mem, N_Vector tmp1,
               N_Vector tmp2, N_Vector tmp3)
{
  int retval;
  CVodeMem cv_mem;
  cv_mem = (CVodeMem) cvode_mem;

  /* verify that Jac is non-NULL */
  if (Jac == NULL) {
    cvProcessError(cv_mem, CVDLS_LMEM_NULL, "CVDLS",
		    "cvDlsDQJac", MSGD_LMEM_NULL);
    return(CVDLS_LMEM_NULL);
  }

  if (SUNMatGetID(Jac) == SUNMATRIX_DENSE) {
    retval = cvDlsDenseDQJac(t, y, fy, Jac, cv_mem, tmp1);
  } else if (SUNMatGetID(Jac) == SUNMATRIX_BAND) {
    retval = cvDlsBandDQJac(t, y, fy, Jac, cv_mem, tmp1, tmp2);
  } else if (SUNMatGetID(Jac) == SUNMATRIX_SPARSE) {
    cvProcessError(cv_mem, CV_ILL_INPUT, "CVDLS",
                   "cvDlsDQJac",
                   "cvDlsDQJac not implemented for SUNMATRIX_SPARSE");
    retval = CV_ILL_INPUT;
  } else {
    cvProcessError(cv_mem, CV_ILL_INPUT, "CVDLS",
                   "cvDlsDQJac",
                   "unrecognized matrix type for cvDlsDQJac");
    retval = CV_ILL_INPUT;
  }
  return(retval);
}


/*-----------------------------------------------------------------
  cvDlsDenseDQJac
  -----------------------------------------------------------------
  This routine generates a dense difference quotient approximation
  to the Jacobian of f(t,y). It assumes that a dense SUNMatrix is
  stored column-wise, and that elements within each column are
  contiguous. The address of the jth column of J is obtained via
  the accessor function SUNDenseMatrix_Column, and this pointer
  is associated with an N_Vector using the N_VSetArrayPointer
  function.  Finally, the actual computation of the jth column of
  the Jacobian is done with a call to N_VLinearSum.
  -----------------------------------------------------------------*/
int cvDlsDenseDQJac(realtype t, N_Vector y, N_Vector fy,
                    SUNMatrix Jac, CVodeMem cv_mem, N_Vector tmp1)
{
  realtype fnorm, minInc, inc, inc_inv, yjsaved, srur;
  realtype *y_data, *ewt_data;
  N_Vector ftemp, jthCol;
  sunindextype j, N;
  int retval = 0;
  CVDlsMem cvdls_mem;

  /* access DlsMem interface structure */
  cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;

  /* access matrix dimension */
  N = SUNDenseMatrix_Rows(Jac);

  /* Rename work vector for readibility */
  ftemp = tmp1;

  /* Create an empty vector for matrix column calculations */
  jthCol = N_VCloneEmpty(tmp1);

  /* Obtain pointers to the data for ewt, y */
  ewt_data = N_VGetArrayPointer(cv_mem->cv_ewt);
  y_data   = N_VGetArrayPointer(y);

  /* Set minimum increment based on uround and norm of f */
  srur = SUNRsqrt(cv_mem->cv_uround);
  fnorm = N_VWrmsNorm(fy, cv_mem->cv_ewt);
  minInc = (fnorm != ZERO) ?
    (MIN_INC_MULT * SUNRabs(cv_mem->cv_h) * cv_mem->cv_uround * N * fnorm) : ONE;

  for (j = 0; j < N; j++) {

    /* Generate the jth col of J(tn,y) */

    N_VSetArrayPointer(SUNDenseMatrix_Column(Jac,j), jthCol);

    yjsaved = y_data[j];
    inc = SUNMAX(srur*SUNRabs(yjsaved), minInc/ewt_data[j]);
    y_data[j] += inc;

    retval = cv_mem->cv_f(t, y, ftemp, cv_mem->cv_user_data);
    cvdls_mem->nfeDQ++;
    if (retval != 0) break;

    y_data[j] = yjsaved;

    inc_inv = ONE/inc;
    N_VLinearSum(inc_inv, ftemp, -inc_inv, fy, jthCol);

    /* DENSE_COL(Jac,j) = N_VGetArrayPointer(jthCol);   /\*UNNECESSARY?? *\/ */
  }

  /* Destroy jthCol vector */
  N_VSetArrayPointer(NULL, jthCol);  /* SHOULDN'T BE NEEDED */
  N_VDestroy(jthCol);

  return(retval);
}


/*-----------------------------------------------------------------
  cvDlsBandDQJac
  -----------------------------------------------------------------
  This routine generates a banded difference quotient approximation
  to the Jacobian of f(t,y).  It assumes that a band SUNMatrix is
  stored column-wise, and that elements within each column are
  contiguous. This makes it possible to get the address of a column
  of J via the accessor function SUNBandMatrix_Column, and to write
  a simple for loop to set each of the elements of a column in
  succession.
  -----------------------------------------------------------------*/
int cvDlsBandDQJac(realtype t, N_Vector y, N_Vector fy, SUNMatrix Jac,
                   CVodeMem cv_mem, N_Vector tmp1, N_Vector tmp2)
{
  N_Vector ftemp, ytemp;
  realtype fnorm, minInc, inc, inc_inv, srur;
  realtype *col_j, *ewt_data, *fy_data, *ftemp_data, *y_data, *ytemp_data;
  sunindextype group, i, j, width, ngroups, i1, i2;
  int retval = 0;
  sunindextype N, mupper, mlower;
  CVDlsMem cvdls_mem;

  /* access DlsMem interface structure */
  cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;

  /* access matrix dimensions */
  N = SUNBandMatrix_Columns(Jac);
  mupper = SUNBandMatrix_UpperBandwidth(Jac);
  mlower = SUNBandMatrix_LowerBandwidth(Jac);

  /* Rename work vectors for use as temporary values of y and f */
  ftemp = tmp1;
  ytemp = tmp2;

  /* Obtain pointers to the data for ewt, fy, ftemp, y, ytemp */
  ewt_data   = N_VGetArrayPointer(cv_mem->cv_ewt);
  fy_data    = N_VGetArrayPointer(fy);
  ftemp_data = N_VGetArrayPointer(ftemp);
  y_data     = N_VGetArrayPointer(y);
  ytemp_data = N_VGetArrayPointer(ytemp);

  /* Load ytemp with y = predicted y vector */
  N_VScale(ONE, y, ytemp);

  /* Set minimum increment based on uround and norm of f */
  srur = SUNRsqrt(cv_mem->cv_uround);
  fnorm = N_VWrmsNorm(fy, cv_mem->cv_ewt);
  minInc = (fnorm != ZERO) ?
    (MIN_INC_MULT * SUNRabs(cv_mem->cv_h) * cv_mem->cv_uround * N * fnorm) : ONE;

  /* Set bandwidth and number of column groups for band differencing */
  width = mlower + mupper + 1;
  ngroups = SUNMIN(width, N);

  /* Loop over column groups. */
  for (group=1; group <= ngroups; group++) {

    /* Increment all y_j in group */
    for(j=group-1; j < N; j+=width) {
      inc = SUNMAX(srur*SUNRabs(y_data[j]), minInc/ewt_data[j]);
      ytemp_data[j] += inc;
    }

    /* Evaluate f with incremented y */
    retval = cv_mem->cv_f(cv_mem->cv_tn, ytemp, ftemp, cv_mem->cv_user_data);
    cvdls_mem->nfeDQ++;
    if (retval != 0) break;

    /* Restore ytemp, then form and load difference quotients */
    for (j=group-1; j < N; j+=width) {
      ytemp_data[j] = y_data[j];
      col_j = SUNBandMatrix_Column(Jac, j);
      inc = SUNMAX(srur*SUNRabs(y_data[j]), minInc/ewt_data[j]);
      inc_inv = ONE/inc;
      i1 = SUNMAX(0, j-mupper);
      i2 = SUNMIN(j+mlower, N-1);
      for (i=i1; i <= i2; i++)
        SM_COLUMN_ELEMENT_B(col_j,i,j) = inc_inv * (ftemp_data[i] - fy_data[i]);
    }
  }

  return(retval);
}


/*-----------------------------------------------------------------
  cvDlsInitialize
  -----------------------------------------------------------------
  This routine performs remaining initializations specific
  to the direct linear solver interface (and solver itself)
  -----------------------------------------------------------------*/
int cvDlsInitialize(CVodeMem cv_mem)
{
  CVDlsMem cvdls_mem;

  /* Return immediately if cv_mem or cv_mem->cv_lmem are NULL */
  if (cv_mem == NULL) {
    cvProcessError(NULL, CVDLS_MEM_NULL, "CVDLS",
                    "cvDlsInitialize", MSGD_CVMEM_NULL);
    return(CVDLS_MEM_NULL);
  }
  if (cv_mem->cv_lmem == NULL) {
    cvProcessError(cv_mem, CVDLS_LMEM_NULL, "CVDLS",
                    "cvDlsInitialize", MSGD_LMEM_NULL);
    return(CVDLS_LMEM_NULL);
  }
  cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;

  cvDlsInitializeCounters(cvdls_mem);

  /* Set Jacobian function and data, depending on jacDQ (in case
     it has changed based on user input) */
  if (cvdls_mem->jacDQ) {
    cvdls_mem->jac    = cvDlsDQJac;
    cvdls_mem->J_data = cv_mem;
  } else {
    cvdls_mem->J_data = cv_mem->cv_user_data;
  }

  /* Call LS initialize routine */
  cvdls_mem->last_flag = SUNLinSolInitialize(cvdls_mem->LS);
  return(cvdls_mem->last_flag);
}

/*

void cvdcheck_inputd(double *x, int len, int var_id){

  int n_zeros=0;
  for (int i=0; i<5; i++){
    if(x[i]==0.0)
      n_zeros++;
    printf("%d[%d]=%-le\n",var_id,i,x[i]);
  }
  if(n_zeros==len)
    printf("%d is all zeros\n",var_id);


}

void cvdcheck_inputi(int *x, int len, int var_id){

  int n_zeros=0;
  for (int i=0; i<5; i++){
    if(x[i]==0.0)
      n_zeros++;
    printf("%d[%d]=%d\n",var_id,i,x[i]);
  }
  if(n_zeros==len)
    printf("%d is all zeros\n",var_id);

}
 */

/*-----------------------------------------------------------------
  cvDlsSetup
  -----------------------------------------------------------------
  This routine determines whether to update a Jacobian matrix (or
  use a stored version), based on heuristics regarding previous
  convergence issues, the number of time steps since it was last
  updated, etc.; it then creates the system matrix from this, the
  'gamma' factor and the identity matrix,
    A = I-gamma*J.
  This routine then calls the LS 'setup' routine with A.
  -----------------------------------------------------------------*/
int cvDlsSetup(CVodeMem cv_mem, int convfail, N_Vector ypred,
               N_Vector fpred, booleantype *jcurPtr, N_Vector vtemp1,
               N_Vector vtemp2, N_Vector vtemp3)
{
  booleantype jbad, jok;
  realtype dgamma;
  CVDlsMem cvdls_mem;
  int retval;

  /* Return immediately if cv_mem or cv_mem->cv_lmem are NULL */
  if (cv_mem == NULL) {
    cvProcessError(NULL, CVDLS_MEM_NULL, "CVDLS",
                    "cvDlsSetup", MSGD_CVMEM_NULL);
    return(CVDLS_MEM_NULL);
  }
  if (cv_mem->cv_lmem == NULL) {
    cvProcessError(cv_mem, CVDLS_LMEM_NULL, "CVDLS",
                    "cvDlsSetup", MSGD_LMEM_NULL);
    return(CVDLS_LMEM_NULL);
  }
  cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;

  /* Use nst, gamma/gammap, and convfail to set J eval. flag jok */
  dgamma = SUNRabs((cv_mem->cv_gamma/cv_mem->cv_gammap) - ONE);
  jbad = (cv_mem->cv_nst == 0) ||
    (cv_mem->cv_nst > cvdls_mem->nstlj + CVD_MSBJ) ||
    ((convfail == CV_FAIL_BAD_J) && (dgamma < CVD_DGMAX)) ||
    (convfail == CV_FAIL_OTHER);
  jok = !jbad;

  /* If jok = SUNTRUE, use saved copy of J */
  if (jok) {
    *jcurPtr = SUNFALSE;
    retval = SUNMatCopy(cvdls_mem->savedJ, cvdls_mem->A);
    if (retval) {
      cvProcessError(cv_mem, CVDLS_SUNMAT_FAIL, "CVDLS",
                      "cvDlsSetup",  MSGD_MATCOPY_FAILED);
      cvdls_mem->last_flag = CVDLS_SUNMAT_FAIL;
      return(-1);
    }

  /* If jok = SUNFALSE, call jac routine for new J value */
  } else {
    cvdls_mem->nje++;
    cvdls_mem->nstlj = cv_mem->cv_nst;
    *jcurPtr = SUNTRUE;
    retval = SUNMatZero(cvdls_mem->A);
    if (retval) {
      cvProcessError(cv_mem, CVDLS_SUNMAT_FAIL, "CVDLS",
                      "cvDlsSetup",  MSGD_MATZERO_FAILED);
      cvdls_mem->last_flag = CVDLS_SUNMAT_FAIL;
      return(-1);
    }

#ifdef PMC_PROFILING
    double startJac=MPI_Wtime();
#endif

    //not working cvdls_mem->J_data->gamma = cv_mem->cv_gamma;
    retval = cvdls_mem->jac(cv_mem->cv_tn, ypred,
                            fpred, cvdls_mem->A,
                            cvdls_mem->J_data, vtemp1, vtemp2, vtemp3);
#ifdef PMC_PROFILING
    cv_mem->timeJac+= MPI_Wtime() - startJac;
    cv_mem->counterJac++;
#endif

   if (retval < 0) {
      cvProcessError(cv_mem, CVDLS_JACFUNC_UNRECVR, "CVDLS",
                      "cvDlsSetup",  MSGD_JACFUNC_FAILED);
      cvdls_mem->last_flag = CVDLS_JACFUNC_UNRECVR;
      return(-1);
    }
    if (retval > 0) {
      cvdls_mem->last_flag = CVDLS_JACFUNC_RECVR;
      return(1);
    }

    retval = SUNMatCopy(cvdls_mem->A, cvdls_mem->savedJ);
    if (retval) {
      cvProcessError(cv_mem, CVDLS_SUNMAT_FAIL, "CVDLS",
                      "cvDlsSetup",  MSGD_MATCOPY_FAILED);
      cvdls_mem->last_flag = CVDLS_SUNMAT_FAIL;
      return(-1);
    }

  }

  // Scale and add I to get A = I - gamma*J //
  retval = SUNMatScaleAddI(-cv_mem->cv_gamma, cvdls_mem->A);

  //cudaMemcpy(bicg->A, bicg->dA, md->jac_size, cudaMemcpyDeviceToHost);

  if (retval) {
    cvProcessError(cv_mem, CVDLS_SUNMAT_FAIL, "CVDLS",
                   "cvDlsSetup",  MSGD_MATSCALEADDI_FAILED);
    cvdls_mem->last_flag = CVDLS_SUNMAT_FAIL;
    return(-1);
  }

  // Call generic linear solver 'setup' with this system matrix, and
  //  return success/failure flag

#ifdef PMC_USE_GPU

#ifdef PMC_PROFILING
  double startKLUSparseSetup = MPI_Wtime();
#endif

  itsolver *bicg = &cv_mem->bicg;


//  HANDLE_ERROR(cudaMemcpy(bicg->djA,bicg->jA,bicg->nnz*sizeof(int),cudaMemcpyHostToDevice));


  //printf("bicg->nrows %d",bicg->nrows);
  //printf("SM_NP_S(cvdls_mem->A) %d",SM_NP_S(cvdls_mem->A));


  //cudaMemcpy(bicg->diA,bicg->iA,(bicg->nrows+1)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(bicg->diA,bicg->iA,(bicg->nrows+1)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(bicg->djA,bicg->jA,bicg->nnz*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(bicg->dA,bicg->A,bicg->nnz*sizeof(double),cudaMemcpyHostToDevice);

  /*
  int i=0;
  check_inputd(bicg->A,bicg->nnz,i++);
  check_inputi(bicg->jA,bicg->nnz,i++);
  check_inputi(bicg->iA,(bicg->nrows+1),i++);
   */

  gpu_diagprecond(bicg->nrows,bicg->dA,bicg->djA,bicg->diA,bicg->ddiag,bicg->blocks,bicg->threads); //Setup linear solver

  cvdls_mem->last_flag = SUNLinSolSetup(cvdls_mem->LS, cvdls_mem->A);

#ifdef PMC_PROFILING
  cv_mem->timeKLUSparseSetup+= MPI_Wtime() - startKLUSparseSetup;
  cv_mem->counterKLUSparseSetup++;
#endif

#else

#ifdef PMC_PROFILING
  double startKLUSparseSetup = MPI_Wtime();
#endif

  cvdls_mem->last_flag = SUNLinSolSetup(cvdls_mem->LS, cvdls_mem->A);

#ifdef PMC_PROFILING
  cv_mem->timeKLUSparseSetup+= MPI_Wtime() - startKLUSparseSetup;
  cv_mem->counterKLUSparseSetup++;
#endif

#endif

return(cvdls_mem->last_flag);
}


/*-----------------------------------------------------------------
  cvDlsSolve
  -----------------------------------------------------------------
  This routine interfaces between CVode and the generic
  SUNLinearSolver object LS, by calling the solver and scaling
  the solution appropriately when gamrat != 1.
  -----------------------------------------------------------------*/
int cvDlsSolve(CVodeMem cv_mem, N_Vector b, N_Vector weight,
               N_Vector ycur, N_Vector fcur)
{
  int retval;
  CVDlsMem cvdls_mem;

  /* Return immediately if cv_mem or cv_mem->cv_lmem are NULL */
  if (cv_mem == NULL) {
    cvProcessError(NULL, CVDLS_MEM_NULL, "CVDLS",
		    "cvDlsSolve", MSGD_CVMEM_NULL);
    return(CVDLS_MEM_NULL);
  }
  if (cv_mem->cv_lmem == NULL) {
    cvProcessError(cv_mem, CVDLS_LMEM_NULL, "CVDLS",
		    "cvDlsSolve", MSGD_LMEM_NULL);
    return(CVDLS_LMEM_NULL);
  }
  cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;

#ifdef PMC_USE_GPU

  itsolver *bicg = &cv_mem->bicg;

  cudaMemcpy(bicg->dA,bicg->A,bicg->nnz*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(bicg->djA,bicg->jA,bicg->nnz*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(bicg->diA,bicg->iA,(bicg->nrows+1)*sizeof(int),cudaMemcpyHostToDevice);
  double *b_ptr = N_VGetArrayPointer(b);//tempv

  /*
  for(int i=0; i<bicg->nrows; i++){
    if(b_ptr[i]==0.0) b_ptr[i]=1.0E-60;
  }
*/

  cudaMemcpy(bicg->dtempv,b_ptr,bicg->nrows*sizeof(double),cudaMemcpyHostToDevice);
  double* x_ptr = N_VGetArrayPointer(cvdls_mem->x);
  retval=0;

  //cudaMemcpy(bicg->dx,x_ptr,bicg->nrows*sizeof(double),cudaMemcpyHostToDevice);
  //cudaMemset(bicg->dx, 0.0, bicg->nrows*sizeof(double));


#ifdef PMC_PROFILING
  double startKLUSparseSolve = MPI_Wtime();
#endif

#ifdef PMC_PROFILING
  cv_mem->timeKLUSparseSolve+= MPI_Wtime() - startKLUSparseSolve;
  cv_mem->counterKLUSparseSolve++;
#endif

#ifndef PMC_DEBUG_GPU
#ifdef CHECK_LS_GPU_CPU

    //cudaMemcpy(x,bicg->dx,bicg->nrows*sizeof(double),cudaMemcpyDeviceToHost);
    //printf("HI\n");
    //printf("bicg->counterBiConjGrad %d\n",bicg->counterBiConjGrad);
    //if(bicg->counterBiConjGrad<=100){
    if(bicg->counterBiConjGrad<1){

      printf("CHECK_LS_GPU_CPU\n");

      /*
      int i=0;
      check_inputd(bicg->A,bicg->nnz,i++);
      check_inputi(bicg->jA,bicg->nnz,i++);
      check_inputi(bicg->iA,bicg->nnz,i++);
       */

      double *dx_init;
      cudaMalloc((void**)&dx_init,bicg->nrows*sizeof(double));
      gpu_yequalsx(dx_init, bicg->dx, bicg->nrows, bicg->blocks, bicg->threads);

      //Compute case 1
      // call the generic linear system solver, and copy b to x
      retval = SUNLinSolSolve(cvdls_mem->LS, cvdls_mem->A, cv_mem->cv_tempv2, b, ZERO);
      double *aux_x1=N_VGetArrayPointer(cv_mem->cv_tempv2);

      //retval = SUNLinSolSolve(cvdls_mem->LS, cvdls_mem->A, cvdls_mem->x, b, ZERO);
      //double *aux_x1=N_VGetArrayPointer(cvdls_mem->x);

      //Compute case 2
      double *b_ptr2 = N_VGetArrayPointer(b);//tempv
      cudaMemcpy(bicg->dtempv,b_ptr2,bicg->nrows*sizeof(double),cudaMemcpyHostToDevice);

      solveGPU_block(bicg,bicg->dA,bicg->djA,bicg->diA,bicg->dx,bicg->dtempv);
      //solveGPU(bicg,bicg->dA,bicg->djA,bicg->diA,bicg->dx,bicg->dtempv);

      //Save result
      double *aux_x2=(double*)malloc(bicg->nrows*sizeof(double));
      cudaMemcpy(aux_x2,bicg->dx,bicg->nrows*sizeof(double),cudaMemcpyDeviceToHost);
      gpu_yequalsx(bicg->dx, dx_init, bicg->nrows, bicg->blocks, bicg->threads);

      //Print accuracy
      printf("aux_x1[0] aux_x2[0] %-le %-le\n", aux_x1[0], aux_x2[0]);
      double error;
      double max_error = aux_x1[0]- aux_x2[0];
      int max_error_i = 0;
      double aux1_max_error = 0.0;
      double aux2_max_error = 0.0;
      for (int i=0; i<bicg->nrows; i++){
        error = fabs(aux_x1[i]-aux_x2[i]);
        //printf("%d %-le %-le\n", i, aux_x1[i], aux_x2[i]);
        if (error > max_error){
          max_error = error;
          max_error_i = i;
          aux1_max_error = aux_x1[i];
          aux2_max_error = aux_x2[i];
        }
      }
      //Local max error
      //printf("Max Error linsolver dx %-le at position %d\n",max_error, max_error_i);
      printf("[%d] max_error %-le aux1_max_error aux2_max_error %-le %-le  \n",
              max_error_i, max_error, aux1_max_error, aux2_max_error);

      //Iter linsolve
      cudaFree(dx_init);
      free(aux_x2);

    }

#endif

#endif

  //printf("AAA\n");

  //solveGPU(bicg,bicg->dA,bicg->djA,bicg->diA,bicg->dx,bicg->dtempv);
  solveGPU_block(bicg,bicg->dA,bicg->djA,bicg->diA,bicg->dx,bicg->dtempv);

  cudaMemcpy(x_ptr,bicg->dx,bicg->nrows*sizeof(double),cudaMemcpyDeviceToHost);

  /*
  for(int i=0; i<bicg->nrows; i++){
    if(x_ptr[i]==0.0) x_ptr[i]=1.0E-60;
  }
*/

  bicg->counterBiConjGrad++;

  // call the generic linear system solver, and copy b to x
  //retval = SUNLinSolSolve(cvdls_mem->LS, cvdls_mem->A, cvdls_mem->x, b, ZERO);

#else

#ifdef PMC_PROFILING
  double startKLUSparseSolve = MPI_Wtime();
#endif

  // call the generic linear system solver, and copy b to x
  retval = SUNLinSolSolve(cvdls_mem->LS, cvdls_mem->A, cvdls_mem->x, b, ZERO);

#ifdef PMC_PROFILING
  cv_mem->timeKLUSparseSolve+= MPI_Wtime() - startKLUSparseSolve;
  cv_mem->counterKLUSparseSolve++;
#endif

#endif

  //copy x into b
  N_VScale(ONE, cvdls_mem->x, b);
  //scale the correction to account for change in gamma
  if ((cv_mem->cv_lmm == CV_BDF) && (cv_mem->cv_gamrat != ONE))
    N_VScale(TWO/(ONE + cv_mem->cv_gamrat), b, b);

  // store solver return value and return
  cvdls_mem->last_flag = retval;
  return(retval);
}


/*-----------------------------------------------------------------
  cvDlsFree
  -----------------------------------------------------------------
  This routine frees memory associates with the CVDls solver
  interface.
  -----------------------------------------------------------------*/
int cvDlsFree(CVodeMem cv_mem)
{
  CVDlsMem cvdls_mem;

  /* Return immediately if cv_mem or cv_mem->cv_lmem are NULL */
  if (cv_mem == NULL)  return (CVDLS_SUCCESS);
  if (cv_mem->cv_lmem == NULL)  return(CVDLS_SUCCESS);
  cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;

  /* Free x vector */
  if (cvdls_mem->x) {
    N_VDestroy(cvdls_mem->x);
    cvdls_mem->x = NULL;
  }

  /* Free savedJ memory */
  if (cvdls_mem->savedJ) {
    SUNMatDestroy(cvdls_mem->savedJ);
    cvdls_mem->savedJ = NULL;
  }

  /* Nullify other SUNMatrix pointer */
  cvdls_mem->A = NULL;

  /* free CVDls interface structure */
  free(cv_mem->cv_lmem);

  return(CVDLS_SUCCESS);
}


/*-----------------------------------------------------------------
  cvDlsInitializeCounters
  -----------------------------------------------------------------
  This routine resets the counters inside the CVDlsMem object.
  -----------------------------------------------------------------*/
int cvDlsInitializeCounters(CVDlsMem cvdls_mem)
{
  cvdls_mem->nje   = 0;
  cvdls_mem->nfeDQ = 0;
  cvdls_mem->nstlj = 0;
  return(0);
}
