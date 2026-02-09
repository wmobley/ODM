#!/usr/bin/env python3
import sys
from pathlib import Path


def main() -> int:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/PotreeConverter")
    stuff = base / "PotreeConverter" / "include" / "stuff.h"
    if stuff.exists():
        text = stuff.read_text()
        if "#include <filesystem>" not in text:
            text = text.replace("#include <string>\n", "#include <string>\n#include <filesystem>\n", 1)
        if "namespace fs = std::filesystem;" not in text:
            text = text.replace("using std::string;\n", "using std::string;\nnamespace fs = std::filesystem;\n", 1)
        stuff.write_text(text)

    bin_reader = base / "PotreeConverter" / "src" / "BINPointReader.cpp"
    if bin_reader.exists():
        text = bin_reader.read_text()
        if "#include <cstdint>" not in text:
            text = text.replace('#include "BINPointReader.h"\n', '#include "BINPointReader.h"\n#include <cstdint>\n', 1)
        if "#include <cstring>" not in text:
            if '#include "BINPointReader.h"\n#include <cstdint>\n' in text:
                text = text.replace('#include "BINPointReader.h"\n#include <cstdint>\n', '#include "BINPointReader.h"\n#include <cstdint>\n#include <cstring>\n', 1)
            elif '#include "stuff.h"\n' in text:
                text = text.replace('#include "stuff.h"\n', '#include "stuff.h"\n#include <cstring>\n', 1)
            else:
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    if line.startswith("#include"):
                        lines.insert(i + 1, "#include <cstring>")
                        text = "\n".join(lines) + "\n"
                        break
        if "#include <filesystem>" not in text:
            text = text.replace('#include "BINPointReader.h"\n#include <cstdint>\n#include <cstring>\n', '#include "BINPointReader.h"\n#include <cstdint>\n#include <cstring>\n#include <filesystem>\n', 1)
        if "namespace fs = std::filesystem;" not in text:
            text = text.replace("#include <filesystem>\n", "#include <filesystem>\nnamespace fs = std::filesystem;\n", 1)
        bin_reader.write_text(text)

    las_writer = base / "PotreeConverter" / "include" / "LASPointWriter.hpp"
    if las_writer.exists():
        text = las_writer.read_text()
        if '#include "laszip_api.h"' in text:
            text = text.replace('#include "laszip_api.h"\n', '#include <laszip/laszip_api.h>\n', 1)
        las_writer.write_text(text)

    las_reader = base / "PotreeConverter" / "src" / "LASPointReader.cpp"
    if las_reader.exists():
        text = las_reader.read_text()
        if '#include "laszip_api.h"' in text:
            text = text.replace('#include "laszip_api.h"\n', '#include <laszip/laszip_api.h>\n', 1)
        las_reader.write_text(text)

    las_reader_hdr = base / "PotreeConverter" / "include" / "LASPointReader.h"
    if las_reader_hdr.exists():
        text = las_reader_hdr.read_text()
        if '#include "laszip_api.h"' in text:
            text = text.replace('#include "laszip_api.h"\n', '#include <laszip/laszip_api.h>\n', 1)
        las_reader_hdr.write_text(text)

    point_h = base / "PotreeConverter" / "include" / "Point.h"
    if point_h.exists():
        text = point_h.read_text()
        if "#include <cstdint>" not in text:
            text = text.replace('#include "Vector3.h"\n', '#include "Vector3.h"\n#include <cstdint>\n', 1)
        point_h.write_text(text)

    aabb_h = base / "PotreeConverter" / "include" / "AABB.h"
    if aabb_h.exists():
        text = aabb_h.read_text()
        if "#include <limits>" not in text:
            text = text.replace('#include "Vector3.h"\n', '#include "Vector3.h"\n#include <limits>\n', 1)
        aabb_h.write_text(text)

    point_reader_h = base / "PotreeConverter" / "include" / "PointReader.h"
    if point_reader_h.exists():
        text = point_reader_h.read_text()
        if "#include <filesystem>" not in text:
            text = text.replace('#include "PointAttributes.h"\n', '#include "PointAttributes.h"\n#include <filesystem>\n', 1)
        point_reader_h.write_text(text)

    potree_writer = base / "PotreeConverter" / "src" / "PotreeWriter.cpp"
    if potree_writer.exists():
        text = potree_writer.read_text()
        needle = "writer->write(reader->getPoint());"
        if needle in text:
            replacement = "auto point = reader->getPoint();\n\t\t\t\t\t\t\t\t\t\twriter->write(point);"
            text = text.replace(needle, replacement, 1)
        potree_writer.write_text(text)

    cmakelists = base / "CMakeLists.txt"
    if cmakelists.exists():
        text = cmakelists.read_text()
        lines = text.splitlines()
        add_exec_idx = None
        filtered = []
        for i, line in enumerate(lines):
            if line.strip().startswith("add_executable") and "PotreeConverter" in line:
                add_exec_idx = len(filtered)
            if line.strip() == "target_link_libraries(PotreeConverter laszip)":
                continue
            filtered.append(line)
        if add_exec_idx is not None:
            filtered.insert(add_exec_idx + 1, "target_link_libraries(PotreeConverter laszip)")
        else:
            filtered.append("target_link_libraries(PotreeConverter laszip)")
        if "CMP0079" not in text:
            filtered.insert(1, "cmake_policy(SET CMP0079 NEW)")
        cmakelists.write_text("\n".join(filtered) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
