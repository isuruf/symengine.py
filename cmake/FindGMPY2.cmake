execute_process(
	COMMAND ${PYTHON_BIN} -c "import gmpy2; print(gmpy2.__file__)"
	OUTPUT_VARIABLE GMPY2_LIBRARY
    ERROR_VARIABLE GMPY2_ERROR
	)

string(STRIP "${GMPY2_ERROR}" GMPY2_ERROR)
string(STRIP "${GMPY2_LIBRARY}" GMPY2_LIBRARY)

if ("${GMPY2_ERROR}" STREQUAL "")
    get_filename_component(GMP2_DIR ${GMPY2_LIBRARY} DIRECTORY)
    get_filename_component(GMP2_DIR ${GMP2_DIR} DIRECTORY)
    get_filename_component(GMP2_DIR ${GMP2_DIR} DIRECTORY)
    get_filename_component(GMP2_DIR ${GMP2_DIR} DIRECTORY)
    FIND_PATH(GMPY2_INCLUDE_DIR gmpy2/gmpy2_mpz.h
        PATHS ${GMP2_DIR}/include/python${PYTHON_VERSION}/ ${GMP2_DIR}/include/python${PYTHON_VERSION_WITHOUT_DOTS}/
        )
    if (GMPY2_INCLUDE_DIR)
        add_library(gmpy2 UNKNOWN IMPORTED)
        set_property(TARGET gmpy2 PROPERTY IMPORTED_LOCATION "${GMPY2_LIBRARY}")
    endif()
endif()
