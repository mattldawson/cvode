# ---------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ SMU
#                 Radu Serban @ LLNL
# ---------------------------------------------------------------
# LLNS/SMU Copyright Start
# Copyright (c) 2017, Southern Methodist University and 
# Lawrence Livermore National Security
#
# This work was performed under the auspices of the U.S. Department 
# of Energy by Southern Methodist University and Lawrence Livermore 
# National Laboratory under Contract DE-AC52-07NA27344.
# Produced at Southern Methodist University and the Lawrence 
# Livermore National Laboratory.
#
# All rights reserved.
# For details, see the LICENSE file.
# LLNS/SMU Copyright End
# ---------------------------------------------------------------
# CMakeLists.txt file for the CVODE library

INSTALL(CODE "MESSAGE(\"\nInstall CVODE\n\")")

# Add variable cvode_SOURCES with the sources for the CVODE library
SET(cvode_SOURCES
  cvode.c
  cvode_bandpre.c
  cvode_bbdpre.c
  cvode_diag.c        
  cvode_direct.c        
  cvode_io.c
  cvode_spils.c
  )

# Add variable shared_SOURCES with the common SUNDIALS sources which will
# also be included in the CVODE library
SET(shared_SOURCES
  ${sundials_SOURCE_DIR}/src/sundials/sundials_nvector.c
  ${sundials_SOURCE_DIR}/src/sundials/sundials_matrix.c
  ${sundials_SOURCE_DIR}/src/sundials/sundials_linearsolver.c
  ${sundials_SOURCE_DIR}/src/sundials/sundials_math.c
  ${sundials_SOURCE_DIR}/src/sundials/sundials_band.c
  ${sundials_SOURCE_DIR}/src/sundials/sundials_dense.c
  ${sundials_SOURCE_DIR}/src/sundials/sundials_direct.c
  ${sundials_SOURCE_DIR}/src/sundials/sundials_iterative.c
  ${sundials_SOURCE_DIR}/src/sundials/sundials_version.c
  ${sundials_SOURCE_DIR}/src/nvec_ser/nvector_serial.c
  )

# Add variable sunmatrix_SOURCES with the common SUNMatrix sources which will
# also be included in the CVODE library
SET(sunmatrix_SOURCES
  ${sundials_SOURCE_DIR}/src/sunmat_band/sunmatrix_band.c
  ${sundials_SOURCE_DIR}/src/sunmat_dense/sunmatrix_dense.c
  ${sundials_SOURCE_DIR}/src/sunmat_sparse/sunmatrix_sparse.c
  )

# Add variable sunlinsol_SOURCES with the common SUNLinearSolver sources which will
# also be included in the CVODE library
SET(sunlinsol_SOURCES
  ${sundials_SOURCE_DIR}/src/sunlinsol_band/sunlinsol_band.c
  ${sundials_SOURCE_DIR}/src/sunlinsol_dense/sunlinsol_dense.c
  ${sundials_SOURCE_DIR}/src/sunlinsol_spbcgs/sunlinsol_spbcgs.c
  ${sundials_SOURCE_DIR}/src/sunlinsol_spfgmr/sunlinsol_spfgmr.c
  ${sundials_SOURCE_DIR}/src/sunlinsol_spgmr/sunlinsol_spgmr.c
  ${sundials_SOURCE_DIR}/src/sunlinsol_sptfqmr/sunlinsol_sptfqmr.c
  ${sundials_SOURCE_DIR}/src/sunlinsol_pcg/sunlinsol_pcg.c
  )

# Add variable cvode_HEADERS with the exported CVODE header files
SET(cvode_HEADERS
  cvode.h
  cvode_bandpre.h
  cvode_bbdpre.h
  cvode_diag.h
  cvode_direct.h
  cvode_spils.h
  )

# Add prefix with complete path to the CVODE header files
ADD_PREFIX(${sundials_SOURCE_DIR}/include/cvode/ cvode_HEADERS)

# Add source directories to include directories for access to
# implementation only header files.
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(../sundials)

# Define C preprocessor flag -DBUILD_SUNDIALS_LIBRARY 
ADD_DEFINITIONS(-DBUILD_SUNDIALS_LIBRARY)

# Add debugging functions if indicated
IF(SUNDIALS_DEBUG)
  ADD_DEFINITIONS(-DSUNDIALS_DEBUG)
ENDIF(SUNDIALS_DEBUG)

# Build the static library
IF(BUILD_STATIC_LIBS)

  # Add the build target for the static CVODE library
  ADD_LIBRARY(sundials_cvode_static STATIC 
    ${cvode_SOURCES} ${shared_SOURCES} ${sunmatrix_SOURCES} ${sunlinsol_SOURCES})

  # Set the library name and make sure it is not deleted
  SET_TARGET_PROPERTIES(sundials_cvode_static
    PROPERTIES OUTPUT_NAME sundials_cvode CLEAN_DIRECT_OUTPUT 1)

  set_target_properties(sundials_cvode_static PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  #set_target_properties(sundials_cvode_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  # Install the CVODE library
  INSTALL(TARGETS sundials_cvode_static DESTINATION lib)

ENDIF(BUILD_STATIC_LIBS)

# Build the shared library
IF(BUILD_SHARED_LIBS)

  # Add the build target for the CVODE library
  ADD_LIBRARY(sundials_cvode_shared SHARED 
    ${cvode_SOURCES} ${shared_SOURCES} ${sunmatrix_SOURCES} ${sunlinsol_SOURCES})

  IF(UNIX)
    TARGET_LINK_LIBRARIES(sundials_cvode_shared m)
  ENDIF()

  # Set the library name and make sure it is not deleted
  SET_TARGET_PROPERTIES(sundials_cvode_shared
    PROPERTIES OUTPUT_NAME sundials_cvode CLEAN_DIRECT_OUTPUT 1)

  # Set VERSION and SOVERSION for shared libraries
  SET_TARGET_PROPERTIES(sundials_cvode_shared
    PROPERTIES VERSION ${cvodelib_VERSION} SOVERSION ${cvodelib_SOVERSION})

  set_target_properties(sundials_cvode_shared PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  #set_target_properties(sundials_cvode_shared PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  # Install the CVODE library
  INSTALL(TARGETS sundials_cvode_shared DESTINATION lib)

ENDIF(BUILD_SHARED_LIBS)

# Install the CVODE header files
INSTALL(FILES ${cvode_HEADERS} DESTINATION include/cvode)

# Install the CVODE implementation header file
INSTALL(FILES cvode_impl.h DESTINATION include/cvode)

# Install the CVODE direct implementation header file
INSTALL(FILES cvode_direct_impl.h DESTINATION include/cvode)

#INSTALL(FILES itsolver_gpu.h DESTINATION include/cvode)
#INSTALL(FILES cuda_structs.h DESTINATION include/cvode)

#
MESSAGE(STATUS "Added CVODE module")
