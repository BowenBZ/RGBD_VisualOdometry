add_library( myslam SHARED
    frame.cpp
    mappoint.cpp
    map.cpp
    camera.cpp
    config.cpp
    front_end.cpp
    viewer.cpp
)

target_link_libraries( myslam
    ${THIRD_PARTY_LIBS}
)
