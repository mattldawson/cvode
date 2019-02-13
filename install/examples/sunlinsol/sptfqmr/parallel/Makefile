# -*- mode: makefile -*-
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
# Makefile for  parallel examples
#
# This file is generated from a template using various variables
# set at configuration time. It can be used as a template for
# other user Makefiles.
# -----------------------------------------------------------------

SHELL = sh

prefix       = /Users/mattdawson/Documents/recherche_lite/bsc_research/cacm_mpmpo/partmc-chem/cvode-1.0-alpha/install
exec_prefix  = /Users/mattdawson/Documents/recherche_lite/bsc_research/cacm_mpmpo/partmc-chem/cvode-1.0-alpha/install
includedir   = /Users/mattdawson/Documents/recherche_lite/bsc_research/cacm_mpmpo/partmc-chem/cvode-1.0-alpha/install/include
libdir       = /Users/mattdawson/Documents/recherche_lite/bsc_research/cacm_mpmpo/partmc-chem/cvode-1.0-alpha/install/lib

CPP         = /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
CPPFLAGS    = -O3 -DNDEBUG
CC          = /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
CFLAGS      = -O3 -DNDEBUG
LDFLAGS     = 
LIBS        =  -lm

LINKFLAGS = -Wl,-rpath,/Users/mattdawson/Documents/recherche_lite/bsc_research/cacm_mpmpo/partmc-chem/cvode-1.0-alpha/install/lib

MPICC       = /usr/local/bin/mpicc
MPI_INC_DIR = .
MPI_LIB_DIR = .
MPI_LIBS    = 
MPI_FLAGS   = 

TMP_INCS  = ${includedir} ${MPI_INC_DIR}
INCLUDES  = $(addprefix -I, ${TMP_INCS})
LIBRARIES = -lsundials_sunlinsolsptfqmr -lsundials_nvecparallel ${LIBS}

EXAMPLES =  test_sunlinsol_sptfqmr_parallel
EXAMPLES_DEPENDENCIES =  test_sunlinsol sundials_nvector sundials_linearsolver

OBJECTS = ${EXAMPLES:=.o}
OBJECTS_DEPENDENCIES = ${EXAMPLES_DEPENDENCIES:=.o}

# -----------------------------------------------------------------------------------------

.SUFFIXES : .o .c

.c.o :
	${MPICC} ${CPPFLAGS} ${CFLAGS} ${MPI_FLAGS} ${INCLUDES} -c $<

# -----------------------------------------------------------------------------------------

all: ${OBJECTS}
	@for i in ${EXAMPLES} ; do \
	  echo "${MPICC} -o $${i} $${i}.o ${OBJECTS_DEPENDENCIES} ${MPI_FLAGS} ${CFLAGS} ${LDFLAGS} ${INCLUDES} -L${libdir} ${LIBRARIES} -L${MPI_LIB_DIR} ${MPI_LIBS} ${LINKFLAGS}" ; \
	  ${MPICC} -o $${i} $${i}.o ${OBJECTS_DEPENDENCIES} ${MPI_FLAGS} ${CFLAGS} ${LDFLAGS} ${INCLUDES} -L${libdir} ${LIBRARIES} -L${MPI_LIB_DIR} ${MPI_LIBS} ${LINKFLAGS} ; \
	done

${OBJECTS}: ${OBJECTS_DEPENDENCIES}

clean:
	rm -f ${OBJECTS_DEPENDENCIES}
	rm -f ${OBJECTS}
	rm -f ${EXAMPLES}

# -----------------------------------------------------------------------------------------
