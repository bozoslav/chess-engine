# External Tools

This directory is for local third-party tools used by the engine workflow.

Tracked files here should stay small. Source checkouts, build trees, and local
binaries are ignored by the root `.gitignore`.

Current local tool layout:

- `external/cutechess-src/`: official Cute Chess source checkout.
- `external/cutechess-build/`: local Cute Chess CMake build tree.
- `external/bin/`: local tool binaries or symlinks used by scripts.
