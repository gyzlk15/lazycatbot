// Unit tests for the HalfKP v1 loader, feature function, accumulator and
// inference.  Run from the repository root: build/nnue_tests.exe
#include "engine.hpp"
#include "nnue.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace chess;

namespace {

int gFailures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++gFailures;
        std::cout << "FAIL: " << what << '\n';
    }
}

bool accumulatorsEqual(const Position& a, const Position& b) {
    return std::memcmp(a.nnueAccumulator, b.nnueAccumulator,
                       sizeof(a.nnueAccumulator)) == 0;
}

void checkAccumulator(const Position& pos, const std::string& tag) {
    Position copy = pos;
    NNUE::refreshAccumulator(copy);
    if (!accumulatorsEqual(pos, copy)) {
        ++gFailures;
        std::cout << "FAIL: accumulator mismatch after " << tag << " fen "
                  << pos.fen() << '\n';
    }
}

bool readFile(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    out.resize(size_t(n));
    in.read(reinterpret_cast<char*>(out.data()), n);
    return bool(in);
}

bool writeFile(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(data.data()),
              std::streamsize(data.size()));
    return bool(out);
}

void testFeatures() {
    Position p;
    check(p.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
          "startpos FEN");
    int feats[64];

    const int nw = NNUE::halfkpFeatures(p, WHITE, feats, 64);
    check(nw == 31, "startpos white feature count 31");
    check(std::find(feats, feats + nw, 3080) != feats + nw,
          "white a2 pawn index 3080");
    check(std::find(feats, feats + nw, 3137) != feats + nw,
          "white b1 knight index 3137");
    check(std::find(feats, feats + nw, 3504) != feats + nw,
          "black a7 pawn index 3504");
    check(std::find(feats, feats + nw, 3836) != feats + nw,
          "black e8 king index 3836");
    // Own king (e1) must be represented by the bucket, not a feature.
    const int ownKingIdx =
        (4 * 12 + 0 * 6 + int(KING)) * 64 + 4;  // 3396
    check(ownKingIdx == 3396, "own-king index arithmetic");
    check(std::find(feats, feats + nw, ownKingIdx) == feats + nw,
          "white own king excluded");

    const int nb = NNUE::halfkpFeatures(p, BLACK, feats, 64);
    check(nb == 31, "startpos black feature count 31");
    check(std::find(feats, feats + nb, 3080) != feats + nb,
          "black a7 pawn oriented index 3080");
    check(std::find(feats, feats + nb, 3137) != feats + nb,
          "black b8 knight oriented index 3137");
    check(std::find(feats, feats + nb, 3504) != feats + nb,
          "white a2 pawn oriented index 3504");
    check(std::find(feats, feats + nb, 3836) != feats + nb,
          "white e1 king oriented index 3836");
    const int blackOwnKingIdx =
        (4 * 12 + 0 * 6 + int(KING)) * 64 + (60 ^ 56);  // 3396? bucket 4, osq 4
    check(std::find(feats, feats + nb, blackOwnKingIdx) == feats + nb,
          "black own king excluded");

    check(NNUE::halfkpIndex(WHITE, W_PAWN, 8, 4) == 3080,
          "halfkpIndex white pawn");
    check(NNUE::halfkpIndex(BLACK, B_PAWN, 48, 60) == 3080,
          "halfkpIndex black pawn");

    Position q;
    check(q.setFen("6k1/8/8/8/8/8/8/4K3 b - - 0 1"), "black king g8 FEN");
    check(NNUE::halfkpIndex(BLACK, W_KING, 4, 62) == 5372,
          "black perspective white king at e1 -> 5372");
    const int blackOwnG8 = NNUE::halfkpIndex(BLACK, B_KING, 62, 62);
    check(blackOwnG8 == (6 * 12 + 5) * 64 + (62 ^ 56),
          "black g8 own king bucket arithmetic");
}

void testLoaderRejectsBadFiles() {
    std::vector<uint8_t> good;
    check(readFile("tests/fixtures/tiny.nnue", good), "read tiny fixture");
    if (good.empty()) return;

    std::string err;
    check(NNUE::loadFile("tests/fixtures/tiny.nnue", err),
          "load tiny fixture: " + err);

    auto bad = good;
    bad[0] = 'X';
    check(writeFile("temp/nnue_bad_magic.nnue", bad), "write bad magic");
    check(!NNUE::loadFile("temp/nnue_bad_magic.nnue", err),
          "reject bad magic");
    check(!err.empty(), "bad magic has an error message");

    bad = good;
    bad[bad.size() - 5] ^= 0x5A;
    check(writeFile("temp/nnue_bad_checksum.nnue", bad), "write bad checksum");
    check(!NNUE::loadFile("temp/nnue_bad_checksum.nnue", err),
          "reject bad checksum");

    bad = good;
    bad.resize(bad.size() - 1);
    check(writeFile("temp/nnue_truncated.nnue", bad), "write truncated");
    check(!NNUE::loadFile("temp/nnue_truncated.nnue", err),
          "reject truncated file");

    // A failed load must not break the currently loaded network in the
    // loader; the UCI layer explicitly unloads if it wants classical fallback.
    check(NNUE::fileLoaded(), "network still loaded after failed attempts");
    check(NNUE::loadFile("tests/fixtures/tiny.nnue", err), "reload tiny");
}

void testSpecialMoves() {
    std::string err;
    check(NNUE::loadFile("tests/fixtures/tiny.nnue", err),
          "load tiny for special-move tests: " + err);

    struct Case {
        const char* fen;
        const char* uci;
    };
    const Case cases[] = {
        {"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"},
        {"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"},
        {"r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8"},
        {"4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"},
        {"4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"},
        {"1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q"},
        {"4k3/8/8/8/8/8/8/4K3 w - - 0 1", "e1e2"},
    };
    for (const Case& c : cases) {
        Position p;
        check(p.setFen(c.fen), std::string("special FEN ") + c.fen);
        checkAccumulator(p, "special start");
        const Move m = p.parseMove(c.uci);
        check(!m.isNone(), std::string("parse special move ") + c.uci);
        if (m.isNone()) continue;
        check(p.makeMove(m), std::string("make special move ") + c.uci);
        checkAccumulator(p, std::string("special make ") + c.uci);
        p.unmakeMove(m);
        checkAccumulator(p, std::string("special unmake ") + c.uci);
    }
}

void testAccumulatorEquivalence() {
    std::string err;
    check(NNUE::loadFile("tests/fixtures/tiny.nnue", err),
          "load tiny for accumulator tests: " + err);
    const NNUE::NetInfo ni = NNUE::info();
    check(ni.hidden == 24, "tiny hidden == 24");

    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "8/8/8/8/8/8/6k1/4K3 w - - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    };

    std::mt19937 rng(1234567);
    for (const char* fen : fens) {
        Position p;
        if (!p.setFen(fen)) {
            check(false, std::string("test FEN rejected: ") + fen);
            continue;
        }
        checkAccumulator(p, "setFen");

        for (int trial = 0; trial < 12; ++trial) {
            check(p.setFen(fen), std::string("reset FEN ") + fen);
            checkAccumulator(p, "reset");

            std::vector<Move> played;
            MoveList list;
            for (int ply = 0; ply < 50; ++ply) {
                p.generateMoves(list);
                if (list.count == 0) break;
                std::vector<int> order(size_t(list.count));
                for (int i = 0; i < list.count; ++i) order[size_t(i)] = i;
                std::shuffle(order.begin(), order.end(), rng);
                bool moved = false;
                for (int i : order) {
                    const Move m = list.moves[i];
                    if (p.makeMove(m)) {
                        played.push_back(m);
                        checkAccumulator(p, "makeMove");
                        moved = true;
                        break;
                    }
                }
                if (!moved) break;
            }

            for (auto it = played.rbegin(); it != played.rend(); ++it) {
                p.unmakeMove(*it);
                checkAccumulator(p, "unmakeMove");
            }
        }
    }
}

void testQuantizedAccumulator() {
    std::string err;
    NNUE::setFtQuantMode("int8");
    check(NNUE::loadFile("tests/fixtures/tiny32.nnue", err),
          "load hidden-32 int8 fixture: " + err);
    const NNUE::NetInfo ni = NNUE::info();
    check(ni.ftQuant == 8, "int8 FT path selected");
    check(ni.fc1Quant == 8, "int8 FC1 path selected");

    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    };
    std::mt19937 rng(98765);
    for (const char* fen : fens) {
        Position p;
        check(p.setFen(fen), std::string("int8 FEN ") + fen);
        checkAccumulator(p, "int8 setFen");
        for (int trial = 0; trial < 6; ++trial) {
            check(p.setFen(fen), "int8 reset");
            std::vector<Move> played;
            MoveList list;
            for (int ply = 0; ply < 40; ++ply) {
                p.generateMoves(list);
                if (!list.count) break;
                std::vector<int> order(size_t(list.count));
                for (int i = 0; i < list.count; ++i) order[size_t(i)] = i;
                std::shuffle(order.begin(), order.end(), rng);
                bool moved = false;
                for (int i : order) {
                    const Move m = list.moves[i];
                    if (p.makeMove(m)) {
                        played.push_back(m);
                        checkAccumulator(p, "int8 makeMove");
                        moved = true;
                        break;
                    }
                }
                if (!moved) break;
            }
            for (auto it = played.rbegin(); it != played.rend(); ++it) {
                p.unmakeMove(*it);
                checkAccumulator(p, "int8 unmakeMove");
            }
        }
        // The int8 path must be deterministic across repeated evals.
        const int a = NNUE::evaluateNetwork(p);
        const int b = NNUE::evaluateNetwork(p);
        check(a == b, "int8 eval deterministic");
    }
    NNUE::setFtQuantMode("auto");
}

void testInferenceAndOptions() {
    std::string err;
    check(NNUE::loadFile("tests/fixtures/tiny_odd.nnue", err),
          "load odd-hidden fixture: " + err);
    check(NNUE::info().hidden == 13, "odd fixture hidden == 13");

    check(NNUE::loadFile("tests/fixtures/tiny.nnue", err),
          "load tiny for inference: " + err);
    check(NNUE::fileLoaded(), "network loaded");
    check(std::string(NNUE::backendName()) == "cpu", "default backend cpu");

    Position p;
    check(p.setFen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
          "inference FEN");
    NNUE::setEnabled(true);
    NNUE::setBackend("cpu");
    NNUE::setNetWeight(100);
    NNUE::setNetClip(0);

    const int net = NNUE::evaluateNetwork(p);
    check(std::abs(net) < 30000, "net output is finite/bounded");
    check(evaluate(p) == net, "direct eval returns the network output");

    NNUE::setBackend("none");
    check(evaluate(p) == classicalEvaluate(p),
          "backend none falls back to classical");
    NNUE::setBackend("cpu");

    NNUE::setNetWeight(0);
    check(evaluate(p) == classicalEvaluate(p),
          "weight 0 falls back to classical");
    NNUE::setNetWeight(100);

    NNUE::setNetClip(10);
    check(std::abs(evaluate(p)) <= 10, "net clip is applied");
    NNUE::setNetClip(0);

    // Every reference FEN must produce a sane value; the exact comparison
    // against Python lives in tests/run_nnue.py.
    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 b - - 0 1",
    };
    for (const char* fen : fens) {
        Position q;
        check(q.setFen(fen), std::string("inference FEN ") + fen);
        const int v = NNUE::evaluateNetwork(q);
        check(std::abs(v) < 30000, std::string("bounded net for ") + fen);
    }

    NNUE::unloadFile();
    check(!NNUE::fileLoaded(), "unload clears the network");
    NNUE::setEnabled(false);
    check(evaluate(p) == classicalEvaluate(p),
          "classical fallback when NNUE is disabled/unloaded");
}

}  // namespace

int main() {
    initAll();
    testFeatures();
    testLoaderRejectsBadFiles();
    testSpecialMoves();
    testAccumulatorEquivalence();
    testQuantizedAccumulator();
    testInferenceAndOptions();

    if (gFailures == 0) {
        std::cout << "nnue_tests: all checks passed\n";
        return 0;
    }
    std::cout << "nnue_tests: " << gFailures << " failure(s)\n";
    return 1;
}
