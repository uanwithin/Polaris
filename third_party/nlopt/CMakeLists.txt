find_package(NLopt CONFIG REQUIRED)

add_library(polaris_nlopt INTERFACE)

if(TARGET NLopt::nlopt_cxx)
    target_link_libraries(polaris_nlopt
        INTERFACE NLopt::nlopt_cxx
    )
elseif(TARGET NLopt::nlopt)
    target_link_libraries(polaris_nlopt
        INTERFACE NLopt::nlopt
    )
else()
    message(FATAL_ERROR "No valid NLopt target found")
endif()

add_library(polaris::nlopt ALIAS polaris_nlopt)