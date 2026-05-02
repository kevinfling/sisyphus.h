set(EXAMPLES
    example_denoise_1d
    example_denoise_2d
    example_spline_fit
    example_scattered
    example_fractional
    example_barycentric
    example_nufct
)

foreach(e ${EXAMPLES})
    add_executable(${e} c/${e}.c)
    target_link_libraries(${e} PRIVATE sisyphus::spectral)
    target_compile_options(${e} PRIVATE -Wall -Wextra -O2)
endforeach()

# C++ wrapper examples
set(CPP_EXAMPLES example_cpp_spline_fit)
foreach(e ${CPP_EXAMPLES})
    add_executable(${e} cpp/${e}.cpp)
    target_link_libraries(${e} PRIVATE sisyphus::spectral)
    target_compile_options(${e} PRIVATE -Wall -Wextra -O2)
endforeach()
