# -*- mode: cmake -*-
# -----------------------------------------------------------------
# Programmer: Radu Serban @ LLNL
# -----------------------------------------------------------------
# LLNS Copyright Start
# Copyright (c) 2014, Lawrence Livermore National Security
# This work was performed under the auspices of the U.S. Department 
# of Energy by Lawrence Livermore National Laboratory in part under 
# Contract W-7405-Eng-48 and in part under Contract DE-AC52-07NA27344.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see the LICENSE file.
# LLNS Copyright End
# -----------------------------------------------------------------
# CMakeLists.txt for  parallel examples
#
# This file is generated from a template using  various variables
# set at configuration time. It can be used as a template for
# other user CMakeLists configuration files.
# -----------------------------------------------------------------

cmake_minimum_required(VERSION 2.8)

# Specify project name
PROJECT(_parallel_examples C)

MARK_AS_ADVANCED(EXECUTABLE_OUTPUT_PATH LIBRARY_OUTPUT_PATH)

# Set the names of the examples to be built
SET(examples  test_nvector_mpi)

# Set names of examples specific dependency source files
SET(examples_dependencies  test_nvector sundials_nvector)

# Set name of MPI compiler to be used (CC or MPICC)
SET(MPI_MPICC /usr/local/bin/mpicc CACHE STRING "MPICC compiler script")
SET(CMAKE_C_COMPILER ${MPI_MPICC})

# Specify path to SUNDIALS header files
SET(SUNDIALS_INC_DIR
  /Users/mattdawson/Documents/recherche_lite/bsc_research/cacm_mpmpo/partmc-chem/cvode-1.0-alpha/install/include
  CACHE STRING
  "Location of SUNDIALS header files")

# Add path to SUNDIALS header files
INCLUDE_DIRECTORIES(${SUNDIALS_INC_DIR})

# Set search path for SUNDIALS libraries 
SET(SUNDIALS_LIB_DIR /Users/mattdawson/Documents/recherche_lite/bsc_research/cacm_mpmpo/partmc-chem/cvode-1.0-alpha/install/lib)

# Find the SUNDIALS solver's library
FIND_LIBRARY(SUNDIALS_SOLVER_LIB
  sundials_nvecparallel ${SUNDIALS_LIB_DIR}
  DOC " library")

# Find the NVECTOR library
FIND_LIBRARY(SUNDIALS_NVEC_LIB
  sundials_nvecparallel ${SUNDIALS_LIB_DIR}
  DOC "NVECTOR library")

# Set additional libraries
SET(SUNDIALS_EXTRA_LIB  -lm CACHE STRING "Additional libraries")

# List of all libraries
SET(SUNDIALS_LIBS ${SUNDIALS_SOLVER_LIB} ${SUNDIALS_NVEC_LIB} ${SUNDIALS_EXTRA_LIB})

# Build each example one by one
FOREACH(example ${examples})
  
  # example source files
  ADD_EXECUTABLE(${example} ${example}.c ${examples_dependencies})

  # libraries to link against
  TARGET_LINK_LIBRARIES(${example} ${SUNDIALS_LIBS})

ENDFOREACH(example ${examples})
