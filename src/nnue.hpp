// HalfKP v1 neural-network evaluator: loader, accumulator, inference.
//
// The on-disk format is fixed by docs/NNUE_PILOT.md section 3.3.  The legacy
// 768->64 residual network that used to live in eval.cpp/nnue_weights.cpp is
// kept as a compatibility path (see engine.hpp) but is not used when a v1
// .nnue file is loaded.
//
// Threading: after loadFile() the network is immutable and shared read-only by
// every search thread.  Each Position owns its own accumulator arrays, so the
// hot path performs no allocation and no synchronisation beyond one atomic
// pointer load for the network itself.
#pragma once

#include "engine.hpp"

#include <cstdint>
#include <string>

namespace chess {

// Fixed-size snug storage.  Positions always reserve MAX_NNUE_HIDDEN lanes
// (declared in engine.hpp) so the network can be swapped without changing
// Position's layout.
namespace NNUE {

constexpr int HALFKP_INPUTS = 64 * 12 * 64;  // 49,152
constexpr int HALFKP_BUCKETS = 64;
constexpr int MAX_HIDDEN = MAX_NNUE_HIDDEN;
constexpr int DENSE1 = 32;
constexpr int DENSE2 = 32;
constexpr int POLICY_SIZE = 4352;

// The exporter pre-multiplies out_w/out_b by cp_scale=400, so the network's
// final layer already emits side-to-move centipawns.  Do not multiply again.

struct NetInfo {
    int version = 0;
    int arch = 0;
    int inputs = 0;
    int hidden = 0;
    int buckets = 0;
    int quant = 0;
    int policySize = 0;
    float ftScale = 0.0f;
    int ftQuant = 16;  // 16 = exact v1 int16 FT, 8 = in-memory int8 cache
    int fc1Quant = 32; // 32 = float FC1, 8 = quantized int8 FC1
};

// Load / unload a v1 .nnue file.  On failure returns false and fills `error`;
// the previously loaded network (if any) is left untouched on validation
// failure, so callers can decide whether to keep or drop it.  The UCI layer
// explicitly unloads on failure to guarantee the classical fallback.
bool loadFile(const std::string& path, std::string& error);
void unloadFile();
bool fileLoaded();
NetInfo info();
const char* simdName();

// In-memory FT quantization.  "auto" uses int8 for hidden >= 192 (the
// memory/throughput win) and exact int16 below that; "int8"/"int16" force a
// mode.  The on-disk v1 format is unchanged.  Takes effect on the next load.
void setFtQuantMode(const std::string& mode);
const char* ftQuantMode();

// Generation changes whenever the loaded network changes.  Positions use it
// to detect stale accumulators cheaply.
uint32_t generation();

// Eval backend: "cpu" or "none".  Backend none makes evaluate() ignore NNUE
// entirely, even when UseNNUE is true.
void setBackend(const std::string& name);
const char* backendName();
bool backendIsCpu();

// The compiled-in 768->64 residual net is a compatibility fallback used only
// when no EvalFile has ever been configured.  Touching EvalFile (success or
// failure) permanently opts into the v1 path, so a failed load gives exactly
// the classical fallback required by the contract.
bool legacyEnabled();
void disableLegacy();

// Optional experiment knobs.  Defaults give direct network eval (100%).
void setNetWeight(int w);
int getNetWeight();
void setNetClip(int c);
int getNetClip();

// Cheap material correction for the direct v1 path.  Diagnostic only: it is
// an experiment to see whether a simple material term recovers fixed-time
// play; the intended final design is an explicit residual target.
// Returns side-to-move material in centipawns: P=100,N=320,B=330,R=500,Q=900.
void setNetMaterial(int w);
int getNetMaterial();
int materialCp(const Position& pos);

// ---- HalfKP feature function (docs/NNUE_PILOT.md 3.1) --------------------
//
// Own king square is encoded in the bucket and excluded from the piece list.
int halfkpIndex(Color perspective, Piece pc, int sq, int kingSq);
int halfkpFeatures(const Position& pos, Color perspective, int* out, int max);

// ---- Accumulator ---------------------------------------------------------
//
// Positions keep two int32 accumulators.  refreshAccumulator() rebuilds both
// from scratch; accumulatorAfterMake()/AfterUnmake() apply incremental deltas.
// All three are no-ops when no network is loaded (accumulators stay invalid).
void refreshAccumulator(const Position& pos);
bool accumulatorValid(const Position& pos);
bool accumulatorAfterMake(Position& pos, Move m, Undo& u, Color us);
void accumulatorAfterUnmake(Position& pos, Move m, const Undo& u, Color us);
void invalidateAccumulator(const Position& pos);

// ---- Inference -----------------------------------------------------------
//
// Returns a score in centipawns from the side-to-move point of view, using the
// accumulator for pos.side.  Refreshes the accumulator lazily if needed.
int evaluateNetwork(const Position& pos);

} // namespace NNUE
} // namespace chess
