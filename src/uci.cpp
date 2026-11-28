#include "engine.hpp"
#include "nnue.hpp"

#include <atomic>
#include <iostream>
#include <sstream>
#include <thread>
#include <type_traits>

namespace chess {

class UciEngine {
public:
    UciEngine() : search_(pos_, tt_) {
        tt_.resize(64);
        pos_.setStartpos();
    }

    ~UciEngine() {
        stopAndWait();
        Syzygy::shutdown();
    }

    void loop() {
        std::string line;
        while (std::getline(std::cin, line)) {
            std::istringstream is(line);
            std::string cmd;
            if (!(is >> cmd)) continue;

            if (cmd == "uci") {
                uci();
            } else if (cmd == "isready") {
                Output::line("readyok");
            } else if (cmd == "help") {
                Output::raw(
                    "info string commands: uci isready ucinewgame position go "
                    "stop quit setoption d eval perft bench help\n"
                    "info string Threads 1..16 (default 1): Lazy SMP workers "
                    "sharing one lockless TT; bestmove/info come from the "
                    "main worker only\n"
                    "info string SyzygyPath: optional directory with .rtbw/.rtbz "
                    "files; SyzygyProbeLimit 0..7, SyzygyProbeDepth 1..100\n");
            } else if (cmd == "ucinewgame") {
                stopAndWait();
                pos_.setStartpos();
                tt_.clear();
                search_.clearState();
                search_.setSilent(false);
                for (auto& h : helpers_) h->clearState();
            } else if (cmd == "position") {
                stopAndWait();
                setPosition(is);
            } else if (cmd == "go") {
                go(is);
            } else if (cmd == "stop") {
                stopAndWait();
            } else if (cmd == "quit") {
                stopAndWait();
                break;
            } else if (cmd == "setoption") {
                setOption(is);
            } else if (cmd == "d") {
                stopAndWait();
                Output::raw(pos_.ascii());
            } else if (cmd == "eval") {
                stopAndWait();
                const NNUE::NetInfo ni = NNUE::info();
                int neural = 0;
                if (NNUE::isEnabled() && NNUE::backendIsCpu()) {
                    if (NNUE::fileLoaded())
                        neural = NNUE::evaluateNetwork(pos_);
                    else if (NNUE::legacyEnabled())
                        neural = NNUE::correction(pos_);
                }
                const bool direct = NNUE::fileLoaded()
                                        ? NNUE::getNetWeight() >= 100
                                        : NNUE::direct;
                const int materialCp =
                    NNUE::fileLoaded() ? NNUE::materialCp(pos_) : 0;
                const int netMaterial = materialCp * NNUE::getNetMaterial() / 100;
                std::ostringstream os;
                os << "info string classical(stm) " << classicalEvaluate(pos_)
                   << " neural " << neural << " total " << evaluate(pos_)
                   << " nnue_available "
                   << ((NNUE::isAvailable() || NNUE::fileLoaded()) ? 1 : 0)
                   << " enabled " << (NNUE::isEnabled() ? 1 : 0) << " weight "
                   << NNUE::getWeight() << " clip " << NNUE::clip
                   << " direct " << (direct ? 1 : 0) << " backend "
                   << NNUE::backendName() << " version " << ni.version
                   << " hidden " << ni.hidden << " ft_quant " << ni.ftQuant
                   << " fc1_quant " << ni.fc1Quant
                   << " net_weight "
                   << NNUE::getNetWeight() << " net_clip " << NNUE::getNetClip()
                   << " net_material " << netMaterial << " material_cp "
                   << materialCp
                   << " simd " << NNUE::simdName();
                Output::line(os.str());
            } else if (cmd == "perft") {
                stopAndWait();
                int depth = 1;
                is >> depth;
                depth = std::clamp(depth, 0, 15);
                ensureHistoryRoom(depth);
                auto start = std::chrono::steady_clock::now();
                uint64_t nodes = perft(pos_, depth);
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - start)
                              .count();
                std::ostringstream os;
                os << "perft " << nodes << " nodes " << ms << " ms";
                if (ms > 0)
                    os << " (" << nodes * 1000 / uint64_t(ms) << " nps)";
                Output::line(os.str());
            } else if (cmd == "bench") {
                int depth = 10;
                is >> depth;
                bench(depth);
            } else if (cmd == "ponderhit") {
                // Pondering is not advertised; safely ignore.
            }
        }
    }

private:
    void ensureHelpers(int threads) {
        const size_t want = size_t(std::max(1, threads) - 1);
        while (helpers_.size() < want)
            helpers_.push_back(std::make_unique<Search>(tt_));
        while (helpers_.size() > want) helpers_.pop_back();
    }

    // Keep enough undo/key history for the next search or perft.  Once the
    // fixed MAX_HIST buffer is full, makeMove() refuses every move, which
    // makes a live position look terminal.  Re-anchor the game history at the
    // current FEN when the remaining room is too small.  This only runs for
    // extremely long games (or huge perft depths).
    void ensureHistoryRoom(int extra) {
        if (pos_.histPly <= MAX_HIST - extra - 2) return;
        const std::string current = pos_.fen();
        if (!pos_.setFen(current)) pos_.setStartpos();
    }

    // Fill the per-go shared state and give every worker its own root copy.
    void prepareShared(const SearchLimits& limits) {
        shared_ = std::make_unique<SharedSearchState>();
        shared_->nodeLimit.store(limits.nodes, std::memory_order_relaxed);
        search_.setShared(shared_.get());
        search_.setThreadId(0);
        search_.setMainThread(true);
        search_.resetStop();
        for (size_t i = 0; i < helpers_.size(); ++i) {
            Search& h = *helpers_[i];
            h.loadPosition(pos_);
            h.setShared(shared_.get());
            h.setThreadId(int(i) + 1);
            h.setMainThread(false);
            h.setSilent(true);
            h.resetStop();
        }
    }

    void launchWorkers(const SearchLimits& limits) {
        tt_.newSearch();  // one generation bump for the whole group
        helperThreads_.clear();
        helperThreads_.reserve(helpers_.size());
        for (auto& h : helpers_) {
            Search* worker = h.get();
            helperThreads_.emplace_back(
                [worker, limits]() { worker->run(limits); });
        }
        worker_ = std::thread([this, limits]() {
            search_.run(limits);
            if (shared_)
                shared_->stop.store(true, std::memory_order_relaxed);
        });
    }

    void joinWorkers() {
        if (worker_.joinable()) worker_.join();
        for (auto& t : helperThreads_)
            if (t.joinable()) t.join();
        helperThreads_.clear();
    }

    void releaseShared() {
        search_.setShared(nullptr);
        for (auto& h : helpers_) h->setShared(nullptr);
        shared_.reset();
    }

    Position pos_;
    TranspositionTable tt_;
    Search search_;
    std::thread worker_;
    std::vector<std::unique_ptr<Search>> helpers_;
    std::vector<std::thread> helperThreads_;
    std::unique_ptr<SharedSearchState> shared_;
    int moveOverhead_ = 30;
    int threads_ = 1;

    static int safeInt(const std::string& s, int fallback) noexcept {
        try {
            return std::stoi(s);
        } catch (...) {
            return fallback;
        }
    }

    static bool parseI64(const std::string& s, int64_t& out) noexcept {
        try {
            size_t pos = 0;
            long long v = std::stoll(s, &pos);
            if (pos != s.size()) return false;
            out = int64_t(v);
            return true;
        } catch (...) {
            return false;
        }
    }

    // Read one numeric UCI parameter. Malformed values are ignored instead of
    // leaving the stream failed (which used to swallow later tokens).
    template <typename T>
    static bool readNumber(std::istringstream& is, T& out) {
        std::string token;
        if (!(is >> token)) return false;
        if constexpr (std::is_unsigned_v<T>) {
            // stoull accepts a leading '-' and wraps it to a huge unsigned
            // value.  A negative `go nodes` must be malformed, not a request
            // for an effectively infinite node budget.
            if (!token.empty() && token.front() == '-') return false;
            try {
                size_t pos = 0;
                unsigned long long v = std::stoull(token, &pos);
                if (pos != token.size()) return false;
                out = T(v);
                return true;
            } catch (...) {
                return false;
            }
        } else {
            int64_t v = 0;
            if (!parseI64(token, v)) return false;
            if (v < int64_t(std::numeric_limits<T>::min())) return false;
            if (v > int64_t(std::numeric_limits<T>::max())) return false;
            out = T(v);
            return true;
        }
    }

    void uci() {
        std::ostringstream os;
        os << "id name ChessBot 2.0 NNUE\n";
        os << "id author OpenCode AI\n";
        os << "option name Hash type spin default 64 min 1 max 4096\n";
        os << "option name Threads type spin default 1 min 1 max 16\n";
        os << "option name Move Overhead type spin default 30 min 0 max 5000\n";
        os << "option name EvalFile type string default <empty>\n";
        os << "option name EvalBackend type string default cpu\n";
        os << "option name EvalQuant type string default auto\n";
        os << "option name UseNNUE type check default "
           << (NNUE::enabled ? "true" : "false") << '\n';
        os << "option name NNUEWeight type spin default " << NNUE::weight
           << " min 0 max 200\n";
        os << "option name NNUEClip type spin default " << NNUE::clip
           << " min 0 max 1500\n";
        os << "option name NNUEMaterial type spin default "
           << NNUE::getNetMaterial() << " min 0 max 100\n";
        os << "option name SyzygyPath type string default <empty>\n";
        os << "option name SyzygyProbeDepth type spin default 1 min 1 max 100\n";
        os << "option name Syzygy50MoveRule type check default true\n";
        os << "option name SyzygyProbeLimit type spin default 7 min 0 max 7\n";
        os << "option name Clear Hash type button\n";
        os << "uciok\n";
        Output::raw(os.str());
    }

    void setOption(std::istringstream& is) {
        std::string token, name, value;
        is >> token; // "name"
        while (is >> token && token != "value") {
            if (!name.empty()) name += ' ';
            name += token;
        }
        while (is >> token) {
            if (!value.empty()) value += ' ';
            value += token;
        }

        // Every option that can be read by the search thread stops and joins
        // the worker first. NNUE::enabled/weight are therefore never written
        // while evaluate() is reading them (no atomics required).
        if (name == "Hash") {
            stopAndWait();
            int mb = std::max(1, safeInt(value, 64));
            tt_.resize(std::min(mb, 4096));
        } else if (name == "EvalFile") {
            stopAndWait();
            // Any explicit EvalFile configuration opts into the v1 path.
            NNUE::disableLegacy();
            const std::string path = (value == "<empty>") ? "" : value;
            std::ostringstream os;
            if (path.empty()) {
                NNUE::unloadFile();
                os << "info string EvalFile: cleared; classical evaluation";
            } else {
                std::string err;
                if (NNUE::loadFile(path, err)) {
                    const NNUE::NetInfo ni = NNUE::info();
                    os << "info string EvalFile: loaded \"" << path
                       << "\" version " << ni.version << " hidden "
                       << ni.hidden << " buckets " << ni.buckets
                       << " policy " << ni.policySize << " simd "
                       << NNUE::simdName();
                } else {
                    // Guarantee the classical fallback rather than keeping a
                    // half-initialized or stale network.
                    NNUE::unloadFile();
                    os << "info string EvalFile: failed to load \"" << path
                       << "\": " << err << "; using classical evaluation";
                }
            }
            Output::line(os.str());
        } else if (name == "EvalBackend") {
            stopAndWait();
            if (value == "cpu" || value == "none") {
                NNUE::setBackend(value);
                Output::line("info string EvalBackend: " + value);
            } else {
                Output::line("info string EvalBackend: ignored \"" + value +
                             "\" (expected cpu or none); keeping " +
                             NNUE::backendName());
            }
        } else if (name == "EvalQuant") {
            stopAndWait();
            if (value == "auto" || value == "int8" || value == "int16") {
                NNUE::setFtQuantMode(value);
                Output::line("info string EvalQuant: " + value +
                             " (reload EvalFile to apply)");
            } else {
                Output::line("info string EvalQuant: ignored \"" + value +
                             "\" (expected auto, int8 or int16); keeping " +
                             NNUE::ftQuantMode());
            }
        } else if (name == "Threads") {
            stopAndWait();
            threads_ = std::clamp(safeInt(value, 1), 1, 16);
            ensureHelpers(threads_);
        } else if (name == "Move Overhead") {
            moveOverhead_ = std::max(0, safeInt(value, 30));
        } else if (name == "NNUEWeight") {
            stopAndWait();
            NNUE::setWeight(safeInt(value, 25));
        } else if (name == "NNUEClip") {
            stopAndWait();
            NNUE::clip = std::clamp(safeInt(value, 80), 0, 1500);
            NNUE::setNetClip(NNUE::clip);
        } else if (name == "NNUEMaterial") {
            stopAndWait();
            NNUE::setNetMaterial(safeInt(value, 0));
        } else if (name == "UseNNUE") {
            stopAndWait();
            NNUE::setEnabled(value == "true" || value == "1");
        } else if (name == "SyzygyPath") {
            stopAndWait();
            const std::string path = (value == "<empty>") ? "" : value;
            const bool ok = Syzygy::init(path);
            std::ostringstream os;
            os << "info string Syzygy: ";
            if (path.empty()) {
                os << "disabled";
            } else if (!ok) {
                os << "failed to initialize path \"" << path << "\"";
            } else if (Syzygy::tablebaseMaxPieces() == 0) {
                os << "no tablebase files found in \"" << path << "\"";
            } else {
                os << "loaded " << Syzygy::tablebaseMaxPieces()
                   << "-piece tablebases from \"" << path << "\"";
            }
            Output::line(os.str());
        } else if (name == "SyzygyProbeDepth") {
            stopAndWait();
            Syzygy::setProbeDepth(safeInt(value, 1));
        } else if (name == "Syzygy50MoveRule") {
            stopAndWait();
            Syzygy::setRule50(value == "true" || value == "1");
        } else if (name == "SyzygyProbeLimit") {
            stopAndWait();
            Syzygy::setProbeLimit(safeInt(value, 7));
        } else if (name == "Clear Hash") {
            stopAndWait();
            tt_.clear();
            search_.clearState();
            for (auto& h : helpers_) h->clearState();
        }
    }

    void setPosition(std::istringstream& is) {
        std::string token;
        is >> token;
        if (token == "startpos") {
            pos_.setStartpos();
            is >> token;
        } else if (token == "fen") {
            std::string fen;
            while (is >> token && token != "moves") {
                if (!fen.empty()) fen += ' ';
                fen += token;
            }
            if (!pos_.setFen(fen)) {
                pos_.setStartpos();
                Output::line("info string invalid FEN, using startpos");
                return;
            }
        } else {
            return;
        }

        if (token == "moves") {
            while (is >> token) {
                ensureHistoryRoom(1);
                Move m = pos_.parseMove(token);
                if (m.isNone()) break;
                if (!pos_.makeMove(m)) break;
            }
        }
    }

    void go(std::istringstream& is) {
        stopAndWait();

        if (pos_.kingSq[WHITE] < 0 || pos_.kingSq[BLACK] < 0)
            pos_.setStartpos();

        SearchLimits limits;
        limits.moveOverhead = moveOverhead_;
        limits.depth = MAX_PLY;
        bool hasClock = false;
        bool infinite = false;
        bool hasExplicitLimit = false;
        bool otherLimit = false;
        bool nodesSet = false;
        bool movetimeSet = false;

        std::string token;
        while (is >> token) {
            if (token == "wtime") {
                otherLimit = true;
                if (readNumber(is, limits.wtime) && limits.wtime >= 0) {
                    hasClock = true;
                } else {
                    // Keep malformed/negative clocks finite.  Previously a
                    // bad wtime/btime left hasClock false (or made anyClock
                    // false) and turned the command into a depth-126 search.
                    limits.wtime = 0;
                    hasClock = true;
                }
            } else if (token == "btime") {
                otherLimit = true;
                if (readNumber(is, limits.btime) && limits.btime >= 0) {
                    hasClock = true;
                } else {
                    limits.btime = 0;
                    hasClock = true;
                }
            } else if (token == "winc") {
                otherLimit = true;
                readNumber(is, limits.winc);
            } else if (token == "binc") {
                otherLimit = true;
                readNumber(is, limits.binc);
            } else if (token == "movestogo") {
                otherLimit = true;
                readNumber(is, limits.movestogo);
            } else if (token == "depth") {
                otherLimit = true;
                // Malformed limits get a small finite fallback instead of
                // silently turning into an unintended infinite search.
                if (!readNumber(is, limits.depth)) limits.depth = 8;
                hasExplicitLimit = true;
            } else if (token == "movetime") {
                otherLimit = true;
                if (!readNumber(is, limits.movetime)) limits.movetime = 50;
                movetimeSet = true;
                hasExplicitLimit = true;
            } else if (token == "nodes") {
                otherLimit = true;
                if (!readNumber(is, limits.nodes)) limits.nodes = 100000;
                nodesSet = true;
                hasExplicitLimit = true;
            } else if (token == "mate") {
                otherLimit = true;
                int mate = 0;
                if (!readNumber(is, mate)) mate = 4;
                int64_t d = int64_t(std::max(1, mate)) * 2;
                limits.depth = int(std::min<int64_t>(d, MAX_PLY));
                hasExplicitLimit = true;
            } else if (token == "infinite") {
                infinite = true;
                hasExplicitLimit = true;
            } else if (token == "ponder") {
                // Not advertised; treated as a normal search.
            }
        }

        limits.infinite = infinite;
        if (!hasExplicitLimit && !hasClock) limits.infinite = true;
        // Only a pure infinite search (or `go` with no limits) is gated on
        // `stop`.  Explicit depth/movetime/nodes/mate/clock parameters retain
        // their normal finite behavior.
        limits.waitForStop = limits.infinite && !otherLimit;
        if (limits.depth < 1) limits.depth = 1;
        if (limits.depth > MAX_PLY) limits.depth = MAX_PLY;
        if (movetimeSet && limits.movetime < 0) limits.movetime = 0;
        if (limits.movetime < -1) limits.movetime = 0;
        if (nodesSet && limits.nodes == 0) {
            // "go nodes 0" is degenerate; use a minimal node budget.
            limits.nodes = 1;
        }

        ensureHistoryRoom(MAX_PLY);
        ensureHelpers(threads_);
        prepareShared(limits);
        search_.setSilent(false);
        launchWorkers(limits);
    }

    void stopAndWait() {
        if (worker_.joinable() || shared_) search_.stop();
        joinWorkers();
        releaseShared();
    }

    void bench(int depth) {
        stopAndWait();
        static const char* fens[] = {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "r4rk1/1pp1qppp/p1np1n2/2b1p3/2B1P3/P1NP1N2/1PP1QPPP/R1B1K2R w KQ - 1 10"
        };

        uint64_t total = 0;
        auto start = std::chrono::steady_clock::now();
        ensureHelpers(std::clamp(threads_, 1, 16));
        for (const char* fen : fens) {
            if (!pos_.setFen(fen)) continue;
            SearchLimits limits;
            limits.depth = std::min(depth, MAX_PLY - 2);
            limits.infinite = true;
            prepareShared(limits);
            search_.setSilent(true);
            launchWorkers(limits);
            joinWorkers();
            total += shared_ ? shared_->nodes.load(std::memory_order_relaxed)
                             : search_.nodes();
            releaseShared();
        }
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - start)
                      .count();
        search_.setSilent(false);
        pos_.setStartpos();
        std::ostringstream os;
        os << "bench nodes " << total << " time " << ms << " ms";
        if (ms > 0) os << " nps " << total * 1000 / uint64_t(ms);
        Output::line(os.str());
    }
};

} // namespace chess

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    chess::initAll();
    chess::UciEngine engine;
    engine.loop();
    return 0;
}
