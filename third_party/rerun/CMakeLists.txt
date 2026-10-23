find_package(
    rerun_sdk
    CONFIG
    REQUIRED
    PATHS /opt/sdk/rerun
)
add_library(polaris_rerun INTERFACE)
target_link_libraries(polaris_rerun INTERFACE rerun_sdk)
add_library(polaris::rerun ALIAS polaris_rerun)
