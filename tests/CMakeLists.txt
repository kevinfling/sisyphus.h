set(TESTS
    test_dct
    test_denoise
    test_derivative
    test_pepin
    test_tensor
    test_tps
    test_smoothing
    test_jacobian
    test_fractional
    test_barycentric
    test_nufct
)

foreach(t ${TESTS})
    add_executable(${t} ${t}.c)
    target_link_libraries(${t} PRIVATE sisyphus::spectral)
    target_compile_options(${t} PRIVATE -Wall -Wextra -Wpedantic -O2)
    add_test(NAME ${t} COMMAND ${t})
endforeach()

# C++ wrapper tests
set(CPP_TESTS test_cpp_wrapper)
foreach(t ${CPP_TESTS})
    add_executable(${t} ${t}.cpp)
    target_link_libraries(${t} PRIVATE sisyphus::spectral)
    target_compile_options(${t} PRIVATE -Wall -Wextra -Wpedantic -O2)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
