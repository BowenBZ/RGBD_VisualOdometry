add_library( myslam SHARED
    frame.cpp
    mappoint.cpp
    camera.cpp
    config.cpp
    frontend.cpp
    backend.cpp
    mapmanager.cpp
    viewer.cpp
)

target_link_libraries( myslam
    ${THIRD_PARTY_LIBS}
)
