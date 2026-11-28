// Batch classical-evaluation labeler.
//
// Input lines:
//   FEN|sf_cp
// (a line may contain more fields; only the first two are used)
//
// Outputs:
//   --out-classical : FEN|classical_cp|sf_cp
//   --out-blend     : FEN|blend_cp   (with --blend-w, default 0.75)
//
// No search, no NNUE inference, no network access.  Invalid FEN/SF records are
// counted and skipped.  Progress and a small manifest are written for auditing.
#include "engine.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

using namespace chess;

namespace {

struct Args {
    std::string input;
    std::string outClassical;
    std::string outBlend;
    std::string manifest;
    double blendW = 0.75;
    long long maxLines = 0;
    long long progressEvery = 250000;
};

bool parseInt64(const std::string& s, long long& out) {
    if (s.empty()) return false;
    const char* begin = s.data();
    const char* end = s.data() + s.size();
    auto result = std::from_chars(begin, end, out);
    return result.ec == std::errc() && result.ptr == end;
}

bool parseArgs(int argc, char** argv, Args& a) {
    if (argc < 2) return false;
    a.input = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&](std::string& dst) {
            if (i + 1 >= argc) return false;
            dst = argv[++i];
            return true;
        };
        if (k == "--out-classical") {
            if (!next(a.outClassical)) return false;
        } else if (k == "--out-blend") {
            if (!next(a.outBlend)) return false;
        } else if (k == "--manifest") {
            if (!next(a.manifest)) return false;
        } else if (k == "--blend-w") {
            std::string v;
            if (!next(v)) return false;
            char* endp = nullptr;
            a.blendW = std::strtod(v.c_str(), &endp);
            if (endp == v.c_str() || *endp != '\0' || a.blendW < 0.0 ||
                a.blendW > 1.0)
                return false;
        } else if (k == "--max") {
            std::string v;
            if (!next(v) || !parseInt64(v, a.maxLines) || a.maxLines < 0)
                return false;
        } else if (k == "--progress") {
            std::string v;
            if (!next(v) || !parseInt64(v, a.progressEvery) ||
                a.progressEvery < 0)
                return false;
        } else {
            std::cerr << "unknown option: " << k << '\n';
            return false;
        }
    }
    return !a.input.empty();
}

std::string lineForClassical(const std::string& fen, int classical, int sf) {
    std::string out;
    out.reserve(fen.size() + 32);
    out += fen;
    out += '|';
    out += std::to_string(classical);
    out += '|';
    out += std::to_string(sf);
    out += '\n';
    return out;
}

std::string lineForBlend(const std::string& fen, int blend) {
    std::string out;
    out.reserve(fen.size() + 16);
    out += fen;
    out += '|';
    out += std::to_string(blend);
    out += '\n';
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) {
        std::cerr
            << "usage: label_classical <input.txt> [--out-classical PATH]\n"
               "       [--out-blend PATH] [--manifest PATH] [--blend-w 0.75]\n"
               "       [--max N] [--progress N]\n";
        return 2;
    }
    if (args.outClassical.empty() && args.outBlend.empty()) {
        std::cerr << "error: give --out-classical and/or --out-blend\n";
        return 2;
    }

    initAll();

    std::ifstream in(args.input, std::ios::binary);
    if (!in) {
        std::cerr << "error: cannot open input " << args.input << '\n';
        return 2;
    }
    std::ofstream outClassical;
    std::ofstream outBlend;
    if (!args.outClassical.empty()) {
        outClassical.open(args.outClassical, std::ios::binary);
        if (!outClassical) {
            std::cerr << "error: cannot create " << args.outClassical << '\n';
            return 2;
        }
    }
    if (!args.outBlend.empty()) {
        outBlend.open(args.outBlend, std::ios::binary);
        if (!outBlend) {
            std::cerr << "error: cannot create " << args.outBlend << '\n';
            return 2;
        }
    }

    long long lines = 0;
    long long validFen = 0;
    long long writtenClassical = 0;
    long long writtenBlend = 0;
    long long invalidFen = 0;
    long long invalidSf = 0;
    long long missingBar = 0;

    std::string line;
    line.reserve(512);
    Position pos;

    while (std::getline(in, line)) {
        if (args.maxLines > 0 && lines >= args.maxLines) break;
        ++lines;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        const size_t bar = line.find('|');
        if (bar == std::string::npos) {
            ++missingBar;
            continue;
        }
        const std::string fen = line.substr(0, bar);
        std::string sfText = line.substr(bar + 1);
        const size_t bar2 = sfText.find('|');
        if (bar2 != std::string::npos) sfText.resize(bar2);
        long long sf = 0;
        if (!parseInt64(sfText, sf)) {
            ++invalidSf;
            continue;
        }
        const int sfCp =
            int(std::clamp<long long>(sf, -30000, 30000));

        if (!pos.setFen(fen)) {
            ++invalidFen;
            continue;
        }
        ++validFen;
        const int classical = classicalEvaluate(pos);

        if (outClassical) {
            outClassical << lineForClassical(fen, classical, sfCp);
            ++writtenClassical;
        }
        if (outBlend) {
            const long long rounded = std::llround(
                args.blendW * double(classical) +
                (1.0 - args.blendW) * double(sfCp));
            outBlend << lineForBlend(
                fen, int(std::clamp<long long>(rounded, -30000, 30000)));
            ++writtenBlend;
        }

        if (args.progressEvery > 0 && lines % args.progressEvery == 0) {
            std::cerr << "labelled " << lines << " lines, " << validFen
                      << " valid\n";
        }
    }

    if (outClassical && !outClassical.flush()) {
        std::cerr << "error: write failure on " << args.outClassical << '\n';
        return 3;
    }
    if (outBlend && !outBlend.flush()) {
        std::cerr << "error: write failure on " << args.outBlend << '\n';
        return 3;
    }
    outClassical.close();
    outBlend.close();

    std::cerr << "done: lines=" << lines << " valid=" << validFen
              << " classical=" << writtenClassical << " blend=" << writtenBlend
              << " invalid_fen=" << invalidFen << " invalid_sf=" << invalidSf
              << " missing_bar=" << missingBar << '\n';

    if (!args.manifest.empty()) {
        std::ofstream m(args.manifest, std::ios::binary);
        if (!m) {
            std::cerr << "error: cannot create manifest " << args.manifest
                      << '\n';
            return 3;
        }
        m << "{\n"
          << "  \"format\": \"classical-sf-v1\",\n"
          << "  \"input\": \"" << args.input << "\",\n"
          << "  \"classical_output\": \"" << args.outClassical << "\",\n"
          << "  \"blend_output\": \"" << args.outBlend << "\",\n"
          << "  \"blend_weight\": " << args.blendW << ",\n"
          << "  \"sf_weight\": " << (1.0 - args.blendW) << ",\n"
          << "  \"lines\": " << lines << ",\n"
          << "  \"valid_fen\": " << validFen << ",\n"
          << "  \"written_classical\": " << writtenClassical << ",\n"
          << "  \"written_blend\": " << writtenBlend << ",\n"
          << "  \"invalid_fen\": " << invalidFen << ",\n"
          << "  \"invalid_sf\": " << invalidSf << ",\n"
          << "  \"missing_bar\": " << missingBar << "\n"
          << "}\n";
    }
    return 0;
}
