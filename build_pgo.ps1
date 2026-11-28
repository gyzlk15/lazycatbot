# Build a PGO-optimized ChessBot with MinGW g++.
#
# The profile runtime on MinGW cannot handle absolute paths with a drive
# letter (it tries to create "C:" inside the profile directory), so the build
# is staged in an ASCII temp directory and compiled with relative paths.
#
# Usage:
#   .\build_pgo.ps1
#   .\build_pgo.ps1 -BenchDepth 14 -Runs 2
param(
    [int]$BenchDepth = 12,
    [int]$Runs = 2
)

$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$stage = Join-Path $env:TEMP "opencode\chessbot_pgo_build"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force $stage | Out-Null
Copy-Item (Join-Path $root "src\*") $stage -Force
Copy-Item (Join-Path $root "third_party") $stage -Recurse -Force

$sources = @(
    "bitboards.cpp",
    "position.cpp",
    "eval.cpp",
    "nnue.cpp",
    "search.cpp",
    "syzygy.cpp",
    "nnue_weights.cpp",
    "third_party/fathom/tbprobe.c"
)

$common = @("-std=c++20", "-O3", "-march=native", "-DNDEBUG",
            "-Wall", "-Wextra", "-pthread", "-Ithird_party/fathom")
$link = @("-flto", "-static-libgcc", "-static-libstdc++", "-Wl,--stack,33554432")

Push-Location $stage
try {
    Write-Host "Stage 1/2: instrumented build ..."
    & g++ @common -flto "-fprofile-generate=prof" `
        -o chessbot_pgo.exe uci.cpp @sources @link
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "Stage 1/2: collecting profiles (bench $BenchDepth x $Runs) ..."
    for ($i = 0; $i -lt $Runs; ++$i) {
        "bench $BenchDepth`nquit`n" | & .\chessbot_pgo.exe | Select-Object -Last 1
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    $gcda = @(Get-ChildItem prof -Recurse -Filter *.gcda -ErrorAction SilentlyContinue)
    if ($gcda.Count -eq 0) {
        Write-Error "no profile data in $stage\prof; PGO build aborted"
        exit 1
    }
    Write-Host "profiles: $($gcda.Count) .gcda files"

    Write-Host "Stage 2/2: profile-use build ..."
    & g++ @common -flto "-fprofile-use=prof" -fprofile-correction `
        -o chessbot_pgo.exe uci.cpp @sources @link
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Copy-Item chessbot_pgo.exe (Join-Path $root "chessbot_pgo.exe") -Force
    Copy-Item chessbot_pgo.exe (Join-Path $root "chessbot.exe") -Force
} finally {
    Pop-Location
}

Write-Host "Done: chessbot.exe + chessbot_pgo.exe (PGO, bench $BenchDepth)"
