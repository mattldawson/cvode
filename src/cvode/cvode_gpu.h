#ifndef CVODE_gpu_SOLVER_H_
#define CVODE_gpu_SOLVER_H_

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <sunmatrix/sunmatrix_sparse.h>
#include <sundials/sundials_nvector.h>

#include <nvector/nvector_serial.h>

//#include <nvector/nvector_cuda.h> //can't :(

#include <sundials/sundials_math.h>
#include <sundials/sundials_direct.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_linearsolver.h>

#include <cvode/cvode.h>
#include "cvode_impl.h"
#include "cvode_direct_impl.h"

//#include<cublas.h>
//#include<cublas_v2.h>

//#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int CVode_gpu(void *cvode_mem, realtype tout, N_Vector yout,
              realtype *tret, int itask);
int CVodeGetDky_gpu(void *cvode_mem, realtype t, int k, N_Vector dky);
void CVodeFree_gpu(void **cvode_mem);
booleantype cvCheckNvector_gpu(N_Vector tmpl);
booleantype cvAllocVectors_gpu(CVodeMem cv_mem, N_Vector tmpl);
void cvFreeVectors_gpu(CVodeMem cv_mem);
int cvInitialSetup_gpu(CVodeMem cv_mem);
int cvHin_gpu(CVodeMem cv_mem, realtype tout);
realtype cvUpperBoundH0_gpu(CVodeMem cv_mem, realtype tdist);
int cvYddNorm_gpu(CVodeMem cv_mem, realtype hg, realtype *yddnrm);
int cvRcheck1_gpu(CVodeMem cv_mem);
int cvRcheck2_gpu(CVodeMem cv_mem);
int cvRcheck3_gpu(CVodeMem cv_mem);
int cvRootfind_gpu(CVodeMem cv_mem);


void set_data_gpu(CVodeMem cv_mem);
int cvStep_gpu(CVodeMem cv_mem);
void cvAdjustParams_gpu(CVodeMem cv_mem);
void cvIncreaseBDF_gpu(CVodeMem cv_mem);
void cvDecreaseBDF_gpu(CVodeMem cv_mem);
void cvRescale_gpu(CVodeMem cv_mem);
void cvPredict_gpu(CVodeMem cv_mem);
void cvSet_gpu(CVodeMem cv_mem);
void cvSetBDF_gpu(CVodeMem cv_mem);
void cvSetTqBDF_gpu(CVodeMem cv_mem, realtype hsum, realtype alpha0,
                           realtype alpha0_hat, realtype xi_inv, realtype xistar_inv);
int cvHandleNFlag_gpu(CVodeMem cv_mem, int *nflagPtr, realtype saved_t,
                             int *ncfPtr);
void cvRestore_gpu(CVodeMem cv_mem, realtype saved_t);
booleantype cvDoErrorTest_gpu(CVodeMem cv_mem, int *nflagPtr,
                                     realtype saved_t, int *nefPtr, realtype *dsmPtr);
void cvCompleteStep_gpu(CVodeMem cv_mem);
void cvPrepareNextStep_gpu(CVodeMem cv_mem, realtype dsm);
void cvSetEta_gpu(CVodeMem cv_mem);
void cvChooseEta_gpu(CVodeMem cv_mem);
void cvBDFStab_gpu(CVodeMem cv_mem);
int cvSLdet_gpu(CVodeMem cv_mem);
int cvEwtSetSV_gpu(CVodeMem cv_mem, N_Vector cv_ewt, N_Vector weight);
int cvNlsNewton_gpu(CVodeMem cv_mem, int nflag);
//int linsolsetup_gpu(CVodeMem cv_mem);
int linsolsetup_gpu(CVodeMem cv_mem,int convfail,N_Vector vtemp1,N_Vector vtemp2,N_Vector vtemp3);
void free_gpu(CVodeMem cv_mem);
int linsolsolve_gpu(CVodeMem cv_mem);
//int linsolsolve_gpu(int *m, double *del, double *delp, double *dcon, SUNMatrix J, CVodeMem cv_mem, double *x, double *b);

int cvHandleFailure_gpu(CVodeMem cv_mem, int flag);

#endif
