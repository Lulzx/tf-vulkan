#!/usr/bin/env python3
"""Embed a .spv file as a C++ uint32 array header (offline SPIR-V, spec §7)."""
import sys
from pathlib import Path

src, dst, symbol = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
data = src.read_bytes()
assert len(data) % 4 == 0, "SPIR-V must be word aligned"
words = [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]
lines = [", ".join(f"0x{w:08x}u" for w in words[i:i + 6]) for i in range(0, len(words), 6)]
dst.write_text(
    "#pragma once\n#include <cstdint>\n"
    f"inline constexpr uint32_t {symbol}[] = {{\n" + ",\n".join(lines) + "\n};\n"
    f"inline constexpr uint32_t {symbol}_words = {len(words)}u;\n")
