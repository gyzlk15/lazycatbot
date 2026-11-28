cmake_minimum_required(VERSION 3.16)
# Fathom's tbprobe.c needs C11 atomics.  Set this before project() so both
# the Ninja and Visual Studio generators pick it up; the VS generator drops
# CMAKE_C_FLAGS, so the MSVC-specific flag is attached to the source file.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
project(ChessBot VERSION 2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

set(ENGINE_SOURCES
    src/bitboards.cpp
    src/position.cpp
    src/eval.cpp
    src/nnue.cpp
    src/search.cpp
    src/syzygy.cpp
    src/nnue_weights.cpp
    third_party/fathom/tbprobe.c
)

# Fathom's tbprobe.c uses C11 atomics.  Set a source property so the flag
# survives both the Ninja and Visual Studio generators (the VS generator
# ignores target-level COMPILE_LANGUAGE expressions and CMAKE_C_FLAGS).
set_source_files_properties(third_party/fathom/tbprobe.c PROPERTIES
    COMPILE_OPTIONS "$<$<C_COMPILER_ID:MSVC>:/experimental:c11atomics>"
    LANGUAGE C
)

add_executable(chessbot src/uci.cpp ${ENGINE_SOURCES})
target_include_directories(chessbot PRIVATE third_party/fathom)
target_link_libraries(chessbot PRIVATE Threads::Threads)

add_executable(trainer src/trainer.cpp ${ENGINE_SOURCES})
target_include_directories(trainer PRIVATE third_party/fathom)
target_link_libraries(trainer PRIVATE Threads::Threads)

# Batch classical-evaluation labeler for distillation datasets.
add_executable(label_classical tools/label_classical.cpp ${ENGINE_SOURCES})
target_include_directories(label_classical PRIVATE src third_party/fathom)
target_link_libraries(label_classical PRIVATE Threads::Threads)

# Standalone loader/accumulator/inference tests (no search).
add_executable(nnue_tests
    tests/nnue_test.cpp
    src/bitboards.cpp
    src/position.cpp
    src/eval.cpp
    src/nnue.cpp
    src/nnue_weights.cpp
)
target_include_directories(nnue_tests PRIVATE src third_party/fathom)
target_link_libraries(nnue_tests PRIVATE Threads::Threads)

add_executable(nnue_bench
    tests/nnue_bench.cpp
    src/bitboards.cpp
    src/position.cpp
    src/eval.cpp
    src/nnue.cpp
    src/nnue_weights.cpp
)
target_include_directories(nnue_bench PRIVATE src third_party/fathom)
target_link_libraries(nnue_bench PRIVATE Threads::Threads)

if(MSVC)
    target_compile_options(chessbot PRIVATE /O2 /arch:AVX2 /W3)
    target_compile_options(trainer PRIVATE /O2 /arch:AVX2 /W3)
    target_compile_options(label_classical PRIVATE /O2 /arch:AVX2 /W3)
    target_compile_options(nnue_tests PRIVATE /O2 /arch:AVX2 /W3)
    target_compile_options(nnue_bench PRIVATE /O2 /arch:AVX2 /W3)
    set_target_properties(chessbot PROPERTIES
        C_STANDARD 11 C_STANDARD_REQUIRED ON)
    set_target_properties(trainer PROPERTIES
        C_STANDARD 11 C_STANDARD_REQUIRED ON)
    set_target_properties(label_classical PROPERTIES
        C_STANDARD 11 C_STANDARD_REQUIRED ON)
    target_compile_definitions(chessbot PRIVATE CHESSBOT_ENABLE_AVX2=1)
    target_compile_definitions(trainer PRIVATE CHESSBOT_ENABLE_AVX2=1)
    target_compile_definitions(label_classical PRIVATE CHESSBOT_ENABLE_AVX2=1)
    target_compile_definitions(nnue_tests PRIVATE CHESSBOT_ENABLE_AVX2=1)
    target_compile_definitions(nnue_bench PRIVATE CHESSBOT_ENABLE_AVX2=1)
    target_link_options(chessbot PRIVATE /STACK:33554432)
    target_link_options(trainer PRIVATE /STACK:33554432)
    target_link_options(label_classical PRIVATE /STACK:33554432)
    target_link_options(nnue_tests PRIVATE /STACK:33554432)
    target_link_options(nnue_bench PRIVATE /STACK:33554432)
else()
    # Portable release: AVX2 (with the scalar path compiled in) instead of
    # -march=native, per docs/NNUE_PILOT.md section 4.
    target_compile_options(chessbot PRIVATE -O3 -mavx2 -mfma -flto -Wall -Wextra)
    target_compile_options(trainer PRIVATE -O3 -mavx2 -mfma -flto -Wall -Wextra)
    target_compile_options(label_classical PRIVATE -O3 -mavx2 -mfma -flto -Wall -Wextra)
    target_compile_options(nnue_tests PRIVATE -O3 -mavx2 -mfma -Wall -Wextra)
    target_compile_options(nnue_bench PRIVATE -O3 -mavx2 -mfma -Wall -Wextra)
    target_link_options(chessbot PRIVATE -flto)
    target_link_options(trainer PRIVATE -flto)
    target_link_options(label_classical PRIVATE -flto)
    if(MINGW)
        # GNU ld option; only valid for the MinGW toolchain. Other linkers
        # (ld64, lld on ELF) reject it. The default thread stack is already
        # sufficient for MAX_PLY=128; this is extra headroom.
        target_link_options(chessbot PRIVATE
            -static-libgcc -static-libstdc++ -Wl,--stack,33554432)
        target_link_options(trainer PRIVATE
            -static-libgcc -static-libstdc++ -Wl,--stack,33554432)
        target_link_options(label_classical PRIVATE
            -static-libgcc -static-libstdc++ -Wl,--stack,33554432)
        target_link_options(nnue_tests PRIVATE
            -static-libgcc -static-libstdc++ -Wl,--stack,33554432)
        target_link_options(nnue_bench PRIVATE
            -static-libgcc -static-libstdc++ -Wl,--stack,33554432)
    endif()
endif()
