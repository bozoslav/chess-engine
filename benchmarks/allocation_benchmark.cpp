#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <new>

#include "board.h"
#include "nnue.h"
#include "search.h"

namespace {

std::atomic<bool> g_countAllocations{false};
std::atomic<std::uint64_t> g_allocationCount{0};
std::atomic<std::uint64_t> g_allocatedBytes{0};

void* allocate(std::size_t size, std::size_t alignment) {
  if (g_countAllocations.load(std::memory_order_relaxed)) {
    g_allocationCount.fetch_add(1, std::memory_order_relaxed);
    g_allocatedBytes.fetch_add(size, std::memory_order_relaxed);
  }

  void* memory = nullptr;
  if (alignment <= alignof(std::max_align_t)) {
    memory = std::malloc(size);
  } else if (posix_memalign(&memory, alignment, size) != 0) {
    memory = nullptr;
  }
  if (memory == nullptr) throw std::bad_alloc();
  return memory;
}

struct AllocationResult {
  SearchResult search;
  std::uint64_t count;
  std::uint64_t bytes;
};

AllocationResult measuredSearch(Board board, int threads) {
  SearchLimits warmup;
  warmup.depth = 2;
  warmup.threads = threads;
  (void)searchBestMove(board, warmup);

  clearSearchState();
  SearchLimits limits;
  limits.depth = 6;
  limits.threads = threads;
  g_allocationCount.store(0, std::memory_order_relaxed);
  g_allocatedBytes.store(0, std::memory_order_relaxed);
  g_countAllocations.store(true, std::memory_order_release);
  const SearchResult result = searchBestMove(board, limits);
  g_countAllocations.store(false, std::memory_order_release);
  return {result, g_allocationCount.load(std::memory_order_relaxed),
          g_allocatedBytes.load(std::memory_order_relaxed)};
}

}  // namespace

void* operator new(std::size_t size) {
  return allocate(size, alignof(std::max_align_t));
}

void* operator new[](std::size_t size) {
  return allocate(size, alignof(std::max_align_t));
}

void* operator new(std::size_t size, std::align_val_t alignment) {
  return allocate(size, static_cast<std::size_t>(alignment));
}

void* operator new[](std::size_t size, std::align_val_t alignment) {
  return allocate(size, static_cast<std::size_t>(alignment));
}

void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept {
  std::free(memory);
}
void operator delete(void* memory, std::align_val_t) noexcept {
  std::free(memory);
}
void operator delete[](void* memory, std::align_val_t) noexcept {
  std::free(memory);
}
void operator delete(void* memory, std::size_t, std::align_val_t) noexcept {
  std::free(memory);
}
void operator delete[](void* memory, std::size_t, std::align_val_t) noexcept {
  std::free(memory);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: chess_engine_allocation_benchmark <network.nnue>\n";
    return 2;
  }
  if (!nnue::loadNetwork(argv[1])) {
    std::cerr << "failed to load network: " << nnue::lastError() << '\n';
    return 1;
  }

  Board board;
  const AllocationResult single = measuredSearch(board, 1);
  const AllocationResult four = measuredSearch(board, 4);
  std::cout << "threads,allocations,bytes,nodes,bestmove\n";
  std::cout << "1," << single.count << ',' << single.bytes << ','
            << single.search.nodes << ',' << single.search.bestMove.toUci()
            << '\n';
  std::cout << "4," << four.count << ',' << four.bytes << ','
            << four.search.nodes << ',' << four.search.bestMove.toUci() << '\n';
  return single.count == 0 && four.count == 0 ? 0 : 1;
}
