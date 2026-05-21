set(_SB_BINARY_DIR "${SB_BINARY_DIR}/pypopsift")

# Pypopsift
find_package(CUDA 7.0)

if(CUDA_FOUND)
    message(STATUS "CUDA found for PyPopsift: ${CUDA_VERSION} (${CUDA_TOOLKIT_ROOT_DIR})")
    ExternalProject_Add(pypopsift
        DEPENDS
        PREFIX            ${_SB_BINARY_DIR}
        TMP_DIR           ${_SB_BINARY_DIR}/tmp
        STAMP_DIR         ${_SB_BINARY_DIR}/stamp
        #--Download step--------------
        DOWNLOAD_DIR      ${SB_DOWNLOAD_DIR}
        GIT_REPOSITORY    https://github.com/OpenDroneMap/pypopsift
        GIT_TAG           fe2d1ccc63877ba315e65f34d2adeadd838b3ac3
        #--Update/Patch step----------
        UPDATE_COMMAND    ""
        #--Configure step-------------
        SOURCE_DIR        ${SB_SOURCE_DIR}/pypopsift
        CMAKE_ARGS
            -DOUTPUT_DIR=${SB_INSTALL_DIR}/bin/opensfm/opensfm
            -DCMAKE_INSTALL_PREFIX=${SB_INSTALL_DIR}
            ${WIN32_CMAKE_ARGS}
            ${ARM64_CMAKE_ARGS}
        #--Build step-----------------
        BINARY_DIR        ${_SB_BINARY_DIR}
        #--Install step---------------
        INSTALL_DIR       ${SB_INSTALL_DIR}
        #--Output logging-------------
        LOG_DOWNLOAD      OFF
        LOG_CONFIGURE     OFF
        LOG_BUILD         OFF
        )
else()
    if(DEFINED ENV{GPU_INSTALL} AND NOT "$ENV{GPU_INSTALL}" STREQUAL "")
        message(FATAL_ERROR "GPU_INSTALL is set, but CUDA >= 7.0 was not found; cannot build PyPopsift GPU SIFT support.")
    else()
        message(WARNING "Could not find CUDA >= 7.0")
    endif()
endif()
