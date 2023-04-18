function(download_boost_histogram)
    set(options)
    set(oneValueArgs BOOST_TAG)
    set(multiValueArgs)
    cmake_parse_arguments(DOWNLOAD_BH "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED DOWNLOAD_BH_BOOST_TAG)
      message(STATUS "Using default version")
      set(DOWNLOAD_BH_BOOST_TAG 1.80.0)
    endif()
    # Validation
    # Add `boost-` to tag unless looking for develop to ensure the tag is (usually) formatted correctly
    if (NOT BOOST_TAG STREQUAL "develop")
      set(DOWNLOAD_BH_BOOST_TAG boost-${DOWNLOAD_BH_BOOST_TAG})
    endif()

    message(STATUS "Boost tag: ${DOWNLOAD_BH_BOOST_TAG}")

    message(STATUS "Fetching BoostFetch.cmake")
    file(DOWNLOAD
      "https://raw.githubusercontent.com/boostorg/cmake/develop/include/BoostFetch.cmake"
      "${CMAKE_BINARY_DIR}/BoostFetch.cmake"
    )
    include("${CMAKE_BINARY_DIR}/BoostFetch.cmake")
    boost_fetch(boostorg/cmake TAG ${DOWNLOAD_BH_BOOST_TAG} NO_ADD_SUBDIR)

    FetchContent_GetProperties(boostorg_cmake)
    list(APPEND CMAKE_MODULE_PATH ${boostorg_cmake_SOURCE_DIR}/include)
    set(BH_INCLUDE_DIRS "")

    foreach(boost_ext assert config core mp11 throw_exception variant2 histogram)
        boost_fetch(boostorg/${boost_ext} TAG ${DOWNLOAD_BH_BOOST_TAG} EXCLUDE_FROM_ALL)
        FetchContent_GetProperties(boostorg_${boost_ext})
        list(APPEND BH_INCLUDE_DIRS ${boostorg_${boost_ext}_SOURCE_DIR}/include)
        #target_include_directories(${target_name) PUBLIC ${boostorg_${boost_ext}_SOURCE_DIR}/include)
    endforeach(boost_ext)
    set(BH_INCLUDE_DIRS ${BH_INCLUDE_DIRS} PARENT_SCOPE)
endfunction()
