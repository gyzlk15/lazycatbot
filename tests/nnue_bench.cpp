// Microbenchmark for the NNUE eval hot path.
//
// Reports:
//   eval       - evaluateNetwork() on a fixed position (accumulator valid)
//   movepair   - makeMove()+unmakeMove() only (with and without a net)
//   full       - makeMove()+evaluateNetwork()+unmakeMove() (with and without)
//
// Usage: nnue_bench <model.nnue> [iterations]
#include "engine.hpp"
#include "nnue.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace chess;

namespace {
using Clock = std::chrono::steady_clock;

int64_t nsSince(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: nnue_bench <model.nnue> [iterations]\n";
        return 2;
    }
    const int iters = argc >= 3 ? std::atoi(argv[2]) : 200000;
    initAll();

    std::string err;
    const bool loaded = NNUE::loadFile(argv[1], err);
    if (!loaded) {
        std::cerr << "load failed: " << err << '\n';
        return 1;
    }
    const NNUE::NetInfo ni = NNUE::info();
    NNUE::setEnabled(true);
    NNUE::setNetWeight(100);
    NNUE::setNetClip(0);

    Position p;
    p.setFen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    if (!NNUE::accumulatorValid(p)) NNUE::refreshAccumulator(p);

    // Warm up.
    volatile int64_t sink = 0;
    for (int i = 0; i < 1000; ++i) sink += NNUE::evaluateNetwork(p);

    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) sink += NNUE::evaluateNetwork(p);
    auto t1 = Clock::now();
    const double evalNs = double(nsSince(t0, t1)) / iters;

    // Precompute pseudo-random legal moves from the same position so the
    // movepair loop exercises make/unmake without generating moves.
    MoveList list;
    p.generateMoves(list);
    std::vector<Move> moves;
    Position probe;
    probe.setFen(p.fen());
    for (int i = 0; i < list.count; ++i) {
        if (probe.makeMove(list.moves[i])) {
            moves.push_back(list.moves[i]);
            probe.unmakeMove(list.moves[i]);
        }
    }
    if (moves.empty()) {
        std::cerr << "no legal moves\n";
        return 1;
    }

    // Accumulator + board make/unmake with the net loaded.
    Position m;
    m.setFen(p.fen());
    t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        const Move mv = moves[size_t(i) % moves.size()];
        m.makeMove(mv);
        m.unmakeMove(mv);
    }
    t1 = Clock::now();
    const double moveNsNet = double(nsSince(t0, t1)) / iters;

    // Full make + eval + unmake with the net loaded.
    t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        const Move mv = moves[size_t(i) % moves.size()];
        m.makeMove(mv);
        sink += NNUE::evaluateNetwork(m);
        m.unmakeMove(mv);
    }
    t1 = Clock::now();
    const double fullNsNet = double(nsSince(t0, t1)) / iters;

    // Same loops without a net (board work + classical eval only).
    NNUE::unloadFile();
    NNUE::setEnabled(false);
    Position q;
    q.setFen(p.fen());
    for (int i = 0; i < 1000; ++i) sink += evaluate(q);
    t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        const Move mv = moves[size_t(i) % moves.size()];
        q.makeMove(mv);
        q.unmakeMove(mv);
    }
    t1 = Clock::now();
    const double moveNsOff = double(nsSince(t0, t1)) / iters;

    for (int i = 0; i < 1000; ++i) sink += evaluate(q);
    t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        const Move mv = moves[size_t(i) % moves.size()];
        q.makeMove(mv);
        sink += evaluate(q);
        q.unmakeMove(mv);
    }
    t1 = Clock::now();
    const double fullNsOff = double(nsSince(t0, t1)) / iters;

    std::cout << "hidden " << ni.hidden << " iters " << iters
              << " eval_ns " << evalNs
              << " movepair_net_ns " << moveNsNet
              << " movepair_off_ns " << moveNsOff
              << " movepair_overhead_ns " << (moveNsNet - moveNsOff)
              << " full_net_ns " << fullNsNet
              << " full_off_ns " << fullNsOff
              << " full_overhead_ns " << (fullNsNet - fullNsOff)
              << " eval_MNPS " << (1000.0 / evalNs)
              << " full_MNPS " << (1000.0 / fullNsNet)
              << " sink " << sink << '\n';
    return 0;
}
