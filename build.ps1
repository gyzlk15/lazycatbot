# Build ChessBot with MinGW g++ (default on this machine).
$ErrorActionPreference = "Stop"

$sources = @(
    "src/bitboards.cpp",
    "src/position.cpp",
    "src/eval.cpp",
    "src/nnue.cpp",
    "src/search.cpp",
    "src/syzygy.cpp",
    "src/nnue_weights.cpp",
    "third_party/fathom/tbprobe.c"
)

$flags = @("-std=c++20", "-O3", "-mavx2", "-mfma", "-flto", "-DNDEBUG",
           "-Wall", "-Wextra", "-pthread", "-Ithird_party/fathom", "-Isrc")
$linkFlags = @("-flto", "-static-libgcc", "-static-libstdc++", "-Wl,--stack,33554432")

Write-Host "Building chessbot.exe ..."
& g++ @flags -o chessbot.exe src/uci.cpp @sources @linkFlags
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Building trainer.exe ..."
& g++ @flags -o trainer.exe src/trainer.cpp @sources @linkFlags
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Building nnue_tests.exe ..."
& g++ @flags -o nnue_tests.exe tests/nnue_test.cpp `
    src/bitboards.cpp src/position.cpp src/eval.cpp src/nnue.cpp `
    src/nnue_weights.cpp @linkFlags
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Building nnue_bench.exe ..."
& g++ @flags -o nnue_bench.exe tests/nnue_bench.cpp `
    src/bitboards.cpp src/position.cpp src/eval.cpp src/nnue.cpp `
    src/nnue_weights.cpp @linkFlags
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Building label_classical.exe ..."
& g++ @flags -o label_classical.exe tools/label_classical.cpp @sources @linkFlags
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Done. Run .\chessbot.exe for UCI mode."
