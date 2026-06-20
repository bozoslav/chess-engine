#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>

#include "nnue_training_data_formats.h"

namespace {

void printUsage(const char* program) {
  std::cerr
      << "Usage:\n"
      << "  " << program
      << " plain-to-binpack <input.plain> <output.binpack> [--append]"
         " [--no-validate] [--skip-invalid] [--shard-size N]\n"
      << "  " << program
      << " binpack-to-plain <input.binpack> <output.plain> [--append]"
         " [--no-validate] [--skip-invalid]\n";
}

bool isOption(std::string_view value, std::string_view option) {
  return value == option;
}

std::size_t parseSizeOption(const char* value, std::string_view optionName) {
  std::size_t parsed = 0;
  try {
    parsed = static_cast<std::size_t>(std::stoull(value));
  } catch (const std::exception&) {
    throw std::runtime_error("invalid value for " + std::string(optionName));
  }
  if (parsed == 0) {
    throw std::runtime_error(std::string(optionName) + " must be positive");
  }
  return parsed;
}

std::string stripBinpackSuffix(const std::string& output) {
  constexpr std::string_view kSuffix = ".binpack";
  if (output.size() >= kSuffix.size() &&
      std::string_view(output).substr(output.size() - kSuffix.size()) ==
          kSuffix) {
    return output.substr(0, output.size() - kSuffix.size());
  }
  return output;
}

std::string shardPath(const std::string& outputPrefix, std::size_t shardIndex) {
  std::ostringstream stream;
  stream << outputPrefix << '-' << std::setw(3) << std::setfill('0')
         << shardIndex << ".binpack";
  return stream.str();
}

bool survivesStockfishPackedEntryRoundTrip(
    const binpack::TrainingDataEntry& entry) {
  const binpack::PackedTrainingDataEntry packed = binpack::packEntry(entry);
  const binpack::TrainingDataEntry unpacked = binpack::unpackEntry(packed);

  chess::Position expectedPosition = entry.pos;
  expectedPosition.setPly(entry.ply);

  return unpacked.pos == expectedPosition && unpacked.move == entry.move &&
         unpacked.score == entry.score && unpacked.ply == entry.ply &&
         unpacked.result == entry.result && unpacked.isValid();
}

bool convertPlainToBinpackStrict(const std::string& input,
                                 const std::string& output,
                                 std::ios_base::openmode openMode,
                                 bool validate,
                                 bool skipInvalid,
                                 std::size_t shardSize) {
  constexpr std::size_t kReportEveryNPositions = 1'000'000;
  constexpr std::size_t kInvalidLogLimit = 20;

  if (shardSize > 0 && (openMode & std::ios_base::app) != 0) {
    std::cerr << "--append is not supported with --shard-size\n";
    return false;
  }

  const std::string outputPrefix = stripBinpackSuffix(output);
  if (shardSize == 0) {
    std::cout << "Converting " << input << " to " << output << '\n';
  } else {
    std::cout << "Converting " << input << " to shards " << outputPrefix
              << "-NNN.binpack with " << shardSize
              << " valid positions per shard.\n";
  }

  std::ifstream inputFile(input);
  if (!inputFile) {
    std::cerr << "failed to open input: " << input << '\n';
    return false;
  }

  const std::filesystem::path outputDirectory =
      shardSize == 0 ? std::filesystem::path(output).parent_path()
                     : std::filesystem::path(outputPrefix).parent_path();
  if (!outputDirectory.empty()) {
    std::filesystem::create_directories(outputDirectory);
  }

  std::unique_ptr<binpack::CompressedTrainingDataEntryWriter> writer;
  std::size_t shardIndex = 0;
  std::size_t positionsInShard = 0;
  const auto openNextShard = [&]() {
    const std::string path =
        shardSize == 0 ? output : shardPath(outputPrefix, shardIndex);
    std::cout << "Writing " << path << '\n';
    writer = std::make_unique<binpack::CompressedTrainingDataEntryWriter>(
        path, openMode);
    positionsInShard = 0;
    ++shardIndex;
  };

  openNextShard();

  binpack::TrainingDataEntry entry{};
  std::string key;
  std::string value;
  std::string move;
  std::size_t numProcessedPositions = 0;
  std::size_t numSkippedPositions = 0;

  while (inputFile >> key) {
    if (key == "e") {
      entry.move = chess::uci::uciToMove(entry.pos, move);
      if (validate && !entry.isValid()) {
        if (!skipInvalid) return false;
        if (numSkippedPositions < kInvalidLogLimit) {
          std::cerr << "Illegal move "
                    << chess::uci::moveToUci(entry.pos, entry.move)
                    << " for position " << entry.pos.fen() << '\n';
        } else if (numSkippedPositions == kInvalidLogLimit) {
          std::cerr << "Further invalid positions suppressed.\n";
        }
        ++numSkippedPositions;
        entry = binpack::TrainingDataEntry{};
        move.clear();
        continue;
      }
      if (validate && !survivesStockfishPackedEntryRoundTrip(entry)) {
        if (!skipInvalid) return false;
        if (numSkippedPositions < kInvalidLogLimit) {
          std::cerr << "Packed-entry round-trip mismatch for move "
                    << chess::uci::moveToUci(entry.pos, entry.move)
                    << " in position " << entry.pos.fen() << '\n';
        } else if (numSkippedPositions == kInvalidLogLimit) {
          std::cerr << "Further invalid positions suppressed.\n";
        }
        ++numSkippedPositions;
        entry = binpack::TrainingDataEntry{};
        move.clear();
        continue;
      }

      if (shardSize > 0 && positionsInShard >= shardSize) {
        writer.reset();
        openNextShard();
      }

      writer->addTrainingDataEntry(entry);
      entry = binpack::TrainingDataEntry{};
      move.clear();

      ++numProcessedPositions;
      ++positionsInShard;
      if (numProcessedPositions % kReportEveryNPositions == 0) {
        std::cout << "Processed " << numProcessedPositions << " positions.\n";
      }
      continue;
    }

    inputFile >> std::ws;
    std::getline(inputFile, value, '\n');

    if (key == "fen") {
      entry.pos = chess::Position::fromFen(value);
    } else if (key == "move") {
      move = value;
    } else if (key == "score") {
      entry.score = static_cast<std::int16_t>(std::stoi(value));
    } else if (key == "ply") {
      entry.ply = static_cast<std::uint16_t>(std::stoi(value));
    } else if (key == "result") {
      entry.result = static_cast<std::int16_t>(std::stoi(value));
    } else {
      std::cerr << "unknown plain key: " << key << '\n';
      return false;
    }
  }

  writer.reset();

  std::cout << "Finished. Converted " << numProcessedPositions
            << " positions";
  if (numSkippedPositions > 0) {
    std::cout << " and skipped " << numSkippedPositions << " invalid positions";
  }
  if (shardSize > 0) {
    std::cout << " across " << shardIndex << " shard files";
  }
  std::cout << ".\n";
  return true;
}

bool convertBinpackToPlainStrict(const std::string& input,
                                 const std::string& output,
                                 std::ios_base::openmode openMode,
                                 bool validate,
                                 bool skipInvalid) {
  constexpr std::size_t kBufferSize = binpack::MiB;
  constexpr std::size_t kInvalidLogLimit = 20;

  std::cout << "Converting " << input << " to " << output << '\n';

  std::ofstream outputFile(output, openMode);
  if (!outputFile) {
    std::cerr << "failed to open output: " << output << '\n';
    return false;
  }

  binpack::CompressedTrainingDataEntryReader reader(input, std::ios_base::in);
  std::size_t numProcessedPositions = 0;
  std::size_t numSkippedPositions = 0;
  std::string buffer;
  buffer.reserve(kBufferSize * 2);

  while (reader.hasNext()) {
    const binpack::TrainingDataEntry entry = reader.next();
    if (validate && !entry.isValid()) {
      if (!skipInvalid) return false;
      if (numSkippedPositions < kInvalidLogLimit) {
        std::cerr << "Illegal move "
                  << chess::uci::moveToUci(entry.pos, entry.move)
                  << " for position " << entry.pos.fen() << '\n';
      } else if (numSkippedPositions == kInvalidLogLimit) {
        std::cerr << "Further invalid positions suppressed.\n";
      }
      ++numSkippedPositions;
      continue;
    }

    binpack::emitPlainEntry(buffer, entry);
    ++numProcessedPositions;

    if (buffer.size() > kBufferSize) {
      outputFile << buffer;
      buffer.clear();
      std::cout << "Processed " << numProcessedPositions << " positions.\n";
    }
  }

  if (!buffer.empty()) {
    outputFile << buffer;
  }

  std::cout << "Finished. Converted " << numProcessedPositions
            << " positions";
  if (numSkippedPositions > 0) {
    std::cout << " and skipped " << numSkippedPositions << " invalid positions";
  }
  std::cout << ".\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    printUsage(argv[0]);
    return 2;
  }

  const std::string_view mode = argv[1];
  const std::string input = argv[2];
  const std::string output = argv[3];
  bool append = false;
  bool validate = true;
  bool skipInvalid = false;
  std::size_t shardSize = 0;

  for (int arg = 4; arg < argc; ++arg) {
    const std::string_view option = argv[arg];
    if (isOption(option, "--append")) {
      append = true;
    } else if (isOption(option, "--no-validate")) {
      validate = false;
    } else if (isOption(option, "--skip-invalid")) {
      skipInvalid = true;
    } else if (isOption(option, "--shard-size")) {
      if (arg + 1 >= argc) {
        std::cerr << "missing value for --shard-size\n";
        printUsage(argv[0]);
        return 2;
      }
      shardSize = parseSizeOption(argv[++arg], option);
    } else {
      std::cerr << "unknown option: " << option << '\n';
      printUsage(argv[0]);
      return 2;
    }
  }

  const std::ios_base::openmode openMode =
      append ? std::ios_base::app : std::ios_base::trunc;

  try {
    if (mode == "plain-to-binpack") {
      return convertPlainToBinpackStrict(input, output, openMode, validate,
                                         skipInvalid, shardSize)
                 ? 0
                 : 1;
    }
    if (mode == "binpack-to-plain") {
      if (shardSize != 0) {
        std::cerr << "--shard-size is only valid for plain-to-binpack\n";
        return 2;
      }
      return convertBinpackToPlainStrict(input, output, openMode, validate,
                                         skipInvalid)
                 ? 0
                 : 1;
    }

    std::cerr << "unknown mode: " << mode << '\n';
    printUsage(argv[0]);
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "stockfish_binpack_tool failed: " << error.what() << '\n';
    return 1;
  }
}
