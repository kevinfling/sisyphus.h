set(BENCHMARKS
    bench_dct
    bench_pepin
    bench_tensor
    bench_tps
    bench_fractional
    bench_barycentric
    bench_nufct
    bench_pepin_convergence
)

foreach(b ${BENCHMARKS})
    add_executable(${b} ${b}.c)
    target_link_libraries(${b} PRIVATE sisyphus::spectral)
    target_compile_options(${b} PRIVATE -Wall -Wextra -O3 -march=native)
endforeach()

# C++ wrapper benchmarks
set(CPP_BENCHMARKS bench_cpp_wrapper)
foreach(b ${CPP_BENCHMARKS})
    add_executable(${b} ${b}.cpp)
    target_link_libraries(${b} PRIVATE sisyphus::spectral)
    target_compile_options(${b} PRIVATE -Wall -Wextra -O3 -march=native)
endforeach()
