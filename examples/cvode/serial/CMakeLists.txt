# ---------------------------------------------------------------
# Programmer(s): Radu Serban @ LLNL
#                David J. Gardner @ LLNL
# ---------------------------------------------------------------
# LLNS Copyright Start
# Copyright (c) 2014, Lawrence Livermore National Security
# This work was performed under the auspices of the U.S. Department
# of Energy by Lawrence Livermore National Laboratory in part under
# Contract W-7405-Eng-48 and in part under Contract DE-AC52-07NA27344.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see the LICENSE file.
# LLNS Copyright End
# ---------------------------------------------------------------
# CMakeLists.txt file for CVODE serial examples
# ---------------------------------------------------------------

# Example lists are tuples "name\;type" where the type is
# 'develop' for examples excluded from 'make test' in releases

# Examples using SUNDIALS linear solvers
SET(CVODE_examples
  #"cvAdvDiff_bnd\;develop"
  #"cvDirectDemo_ls\;develop"
  #"cvDiurnal_kry_bp\;develop"
  #"cvDiurnal_kry\;develop"
  "cvDisc_dns\;develop"
  #"cvKrylovDemo_ls\;develop"
  #"cvKrylovDemo_prec\;develop"
  #"cvRoberts_dns\;"
  #"cvRoberts_dns_uw\;develop"
  #"cvRoberts_dns_negsol\;develop"
  #cvAdvDiffReac_kry\;develop"
  )

# Examples using LAPACK linear solvers
SET(CVODE_examples_BL
  "cvAdvDiff_bndL\;develop"
  "cvRoberts_dnsL\;develop"
  )

# Examples using KLU linear solver
SET(CVODE_examples_KLU
  "cvRoberts_klu\;develop"
  )

# Examples using SuperLU_MT linear solver
SET(CVODE_examples_SUPERLUMT
  "cvRoberts_sps\;develop"
  )

# Specify libraries to link against (through the target that was used to
# generate them) based on the value of the variable LINK_LIBRARY_TYPE
IF(LINK_LIBRARY_TYPE MATCHES "static")
  SET(CVODE_LIB sundials_cvode_static)
  SET(NVECS_LIB sundials_nvecserial_static)
ELSE()
  SET(CVODE_LIB sundials_cvode_shared)
  SET(NVECS_LIB sundials_nvecserial_shared)
ENDIF()

# Set-up linker flags and link libraries
SET(SUNDIALS_LIBS ${CVODE_LIB} ${NVECS_LIB} ${EXTRA_LINK_LIBS})


# Add the build and install targets for each example
FOREACH(example_tuple ${CVODE_examples})

  # parse the example tuple
  LIST(GET example_tuple 0 example)
  LIST(GET example_tuple 1 example_type)

  # example source files
  ADD_EXECUTABLE(${example} ${example}.c)

  SET_TARGET_PROPERTIES(${example} PROPERTIES FOLDER "Examples")

  # add example to regression tests
  SUNDIALS_ADD_TEST(${example} ${example}
    ANSWER_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    ANSWER_FILE ${example}.out
    EXAMPLE_TYPE ${example_type})

  # libraries to link against
  TARGET_LINK_LIBRARIES(${example} ${SUNDIALS_LIBS})

  # install example source and out files
  IF(EXAMPLES_INSTALL)
    INSTALL(FILES ${example}.c ${example}.out
      DESTINATION ${EXAMPLES_INSTALL_PATH}/cvode/serial)
  ENDIF()

ENDFOREACH(example_tuple ${CVODE_examples})


# Add the build and install targets for each LAPACK example (if needed)
IF(LAPACK_FOUND)

  # Sundials LAPACK linear solver modules
  IF(LINK_LIBRARY_TYPE MATCHES "static")
    SET(SUNLINSOLLAPACK_LIBS
      sundials_sunlinsollapackband_static
      sundials_sunlinsollapackdense_static)
  ELSE()
    SET(SUNLINSOLLAPACK_LIBS
      sundials_sunlinsollapackband_shared
      sundials_sunlinsollapackdense_shared)
  ENDIF()

  # LAPACK libraries
  LIST(APPEND SUNLINSOLLAPACK_LIBS ${LAPACK_LIBRARIES})

  # BLAS libraries
  IF(BLAS_FOUND)
    LIST(APPEND SUNLINSOLLAPACK_LIBS ${BLAS_LIBRARIES})
  ENDIF(BLAS_FOUND)

  FOREACH(example_tuple ${CVODE_examples_BL})

    # parse the example tuple
    LIST(GET example_tuple 0 example)
    LIST(GET example_tuple 1 example_type)

    # example source files
    ADD_EXECUTABLE(${example} ${example}.c)

    SET_TARGET_PROPERTIES(${example} PROPERTIES FOLDER "Examples")

    # add example to regression tests
    SUNDIALS_ADD_TEST(${example} ${example}
      ANSWER_DIR ${CMAKE_CURRENT_SOURCE_DIR}
      ANSWER_FILE ${example}.out
      EXAMPLE_TYPE ${example_type})

    # libraries to link against
    TARGET_LINK_LIBRARIES(${example} ${SUNDIALS_LIBS} ${SUNLINSOLLAPACK_LIBS})

    # install example source and out files
    IF(EXAMPLES_INSTALL)
      INSTALL(FILES ${example}.c ${example}.out
        DESTINATION ${EXAMPLES_INSTALL_PATH}/cvode/serial)
    ENDIF()

  ENDFOREACH(example_tuple ${CVODE_examples_BL})

ENDIF(LAPACK_FOUND)


# Add the build and install targets for each KLU example (if needed)
IF(KLU_FOUND)

  # Sundials KLU linear solver module
  IF(LINK_LIBRARY_TYPE MATCHES "static")
    SET(SUNLINSOLKLU_LIBS sundials_sunlinsolklu_static)
  ELSE()
    SET(SUNLINSOLKLU_LIBS sundials_sunlinsolklu_shared)
  ENDIF()

  # KLU libraries
  LIST(APPEND SUNLINSOLKLU_LIBS ${KLU_LIBRARIES})

  FOREACH(example_tuple ${CVODE_examples_KLU})

    # parse the example tuple
    LIST(GET example_tuple 0 example)
    LIST(GET example_tuple 1 example_type)

    # example source files
    ADD_EXECUTABLE(${example} ${example}.c)

    SET_TARGET_PROPERTIES(${example} PROPERTIES FOLDER "Examples")

    # add example to regression tests
    SUNDIALS_ADD_TEST(${example} ${example}
      ANSWER_DIR ${CMAKE_CURRENT_SOURCE_DIR}
      ANSWER_FILE ${example}.out
      EXAMPLE_TYPE ${example_type})

    # libraries to link against
    TARGET_LINK_LIBRARIES(${example} ${SUNDIALS_LIBS} ${SUNLINSOLKLU_LIBS})

    # install example source and out files
    IF(EXAMPLES_INSTALL)
      INSTALL(FILES ${example}.c ${example}.out
        DESTINATION ${EXAMPLES_INSTALL_PATH}/cvode/serial)
    ENDIF()

  ENDFOREACH(example_tuple ${CVODE_examples_KLU})

ENDIF(KLU_FOUND)


# Add the build and install targets for each SuperLU_MT example (if needed)
IF(SUPERLUMT_FOUND)

  # Sundials SuperLU_MT linear solver module
  IF(LINK_LIBRARY_TYPE MATCHES "static")
    SET(SUNLINSOLSLUMT_LIBS sundials_sunlinsolsuperlumt_static)
  ELSE()
    SET(SUNLINSOLSLUMT_LIBS sundials_sunlinsolsuperlumt_shared)
  ENDIF()

  # SuperLU_MT libraries
  LIST(APPEND SUNLINSOLSLUMT_LIBS ${SUPERLUMT_LIBRARIES})

  # BLAS libraries
  IF(BLAS_FOUND)
    LIST(APPEND SUNLINSOLSLUMT_LIBS ${BLAS_LIBRARIES})
  ENDIF(BLAS_FOUND)

  FOREACH(example_tuple ${CVODE_examples_SUPERLUMT})

    # parse the example tuple
    LIST(GET example_tuple 0 example)
    LIST(GET example_tuple 1 example_type)

    # example source files
    ADD_EXECUTABLE(${example} ${example}.c)

    SET_TARGET_PROPERTIES(${example} PROPERTIES FOLDER "Examples")

    # add example to regression tests
    SUNDIALS_ADD_TEST(${example} ${example}
      ANSWER_DIR ${CMAKE_CURRENT_SOURCE_DIR}
      ANSWER_FILE ${example}.out
      EXAMPLE_TYPE ${example_type})

    # libraries to link against
    TARGET_LINK_LIBRARIES(${example} ${SUNDIALS_LIBS} ${SUNLINSOLSLUMT_LIBS})

    # install example source and out files
    IF(EXAMPLES_INSTALL)
      INSTALL(FILES ${example}.c ${example}.out
        DESTINATION ${EXAMPLES_INSTALL_PATH}/cvode/serial)
    ENDIF()

  ENDFOREACH(example_tuple ${CVODE_examples_SUPERLUMT})

ENDIF(SUPERLUMT_FOUND)


# create Makfile and CMakeLists.txt for examples
IF(EXAMPLES_INSTALL)

  # Install the README file
  INSTALL(FILES README DESTINATION ${EXAMPLES_INSTALL_PATH}/cvode/serial)

  # Prepare substitution variables for Makefile and/or CMakeLists templates
  SET(SOLVER "CVODE")
  SET(SOLVER_LIB "sundials_cvode")

  EXAMPLES2STRING(CVODE_examples EXAMPLES)

  IF(LAPACK_FOUND)
    EXAMPLES2STRING(CVODE_examples_BL EXAMPLES_BL)
  ELSE()
    SET(EXAMPLES_BL "")
  ENDIF()

  IF(KLU_FOUND)
    EXAMPLES2STRING(CVODE_examples_KLU EXAMPLES_KLU)
  ELSE()
    SET(EXAMPLES_KLU "")
  ENDIF()

  IF(SUPERLUMT_FOUND)
    EXAMPLES2STRING(CVODE_examples_SUPERLUMT EXAMPLES_SLUMT)
  ELSE()
    SET(EXAMPLES_SLUMT "")
  ENDIF()

  # Regardless of the platform we're on, we will generate and install
  # CMakeLists.txt file for building the examples. This file  can then
  # be used as a template for the user's own programs.

  # generate CMakelists.txt in the binary directory
  CONFIGURE_FILE(
    ${PROJECT_SOURCE_DIR}/examples/templates/cmakelists_serial_C_ex.in
    ${PROJECT_BINARY_DIR}/examples/cvode/serial/CMakeLists.txt
    @ONLY
    )

  # install CMakelists.txt
  INSTALL(
    FILES ${PROJECT_BINARY_DIR}/examples/cvode/serial/CMakeLists.txt
    DESTINATION ${EXAMPLES_INSTALL_PATH}/cvode/serial
    )

  # On UNIX-type platforms, we also  generate and install a makefile for
  # building the examples. This makefile can then be used as a template
  # for the user's own programs.

  IF(UNIX)
    # generate Makefile and place it in the binary dir
    CONFIGURE_FILE(
      ${PROJECT_SOURCE_DIR}/examples/templates/makefile_serial_C_ex.in
      ${PROJECT_BINARY_DIR}/examples/cvode/serial/Makefile_ex
      @ONLY
      )
    # install the configured Makefile_ex as Makefile
    INSTALL(
      FILES ${PROJECT_BINARY_DIR}/examples/cvode/serial/Makefile_ex
      DESTINATION ${EXAMPLES_INSTALL_PATH}/cvode/serial
      RENAME Makefile
      )
  ENDIF(UNIX)

  # add test_install target
  SUNDIALS_ADD_TEST_INSTALL(cvode cvRoberts_dns
    EXAMPLE_DIR ${EXAMPLES_INSTALL_PATH}/cvode/serial)

ENDIF(EXAMPLES_INSTALL)
