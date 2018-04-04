set(PXD_FILES
    symengine.pxd
    matrices.pxd
    core.pxd
)

set(PY_PATH ${PYTHON_INSTALL_PATH}/symengine/lib)

configure_file(config.pxi.in config.pxi)

include_directories(BEFORE ${python_wrapper_BINARY_DIR}/symengine/lib)

macro(add_symengine_cython_library name)
    cython_add_module_pyx(${name} ${PXD_FILES})
    add_python_library(${name} ${name}.cpp ${ARGN})
    target_link_libraries(${name} ${SYMENGINE_LIBRARIES})

    if (CMAKE_CXX_COMPILER_ID MATCHES GNU|Clang)
      # Must suppress strict aliasing for this file
      set_source_files_properties(
        ${name}.cpp
        PROPERTIES COMPILE_FLAGS "-fno-strict-aliasing -Wno-unused-function"
        )
    endif()

    install(TARGETS ${name}
        RUNTIME DESTINATION ${PY_PATH}
        ARCHIVE DESTINATION ${PY_PATH}
        LIBRARY DESTINATION ${PY_PATH}
    )
endmacro()

add_symengine_cython_library(core pywrapper.cpp)
add_symengine_cython_library(matrices)
add_symengine_cython_library(lambdify)
add_symengine_cython_library(functions)

install(FILES __init__.py
    ${CMAKE_CURRENT_BINARY_DIR}/config.pxi
    ${PXD_FILES}
    pywrapper.h
    symengine_wrapper.py
    DESTINATION ${PY_PATH}
    )
