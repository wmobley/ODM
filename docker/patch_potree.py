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
            text = text.replace('#include "BINPointReader.h"\n#include <cstdint>\n', '#include "BINPointReader.h"\n#include <cstdint>\n#include <cstring>\n', 1)
        if "#include <filesystem>" not in text:
            text = text.replace('#include "BINPointReader.h"\n#include <cstdint>\n#include <cstring>\n', '#include "BINPointReader.h"\n#include <cstdint>\n#include <cstring>\n#include <filesystem>\n', 1)
        if "namespace fs = std::filesystem;" not in text:
            text = text.replace("#include <filesystem>\n", "#include <filesystem>\nnamespace fs = std::filesystem;\n", 1)
        bin_reader.write_text(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
