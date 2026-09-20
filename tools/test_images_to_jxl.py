#!/usr/bin/env python3

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("images_to_jxl.py")
SPEC = importlib.util.spec_from_file_location("images_to_jxl", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ImagesToJxlTests(unittest.TestCase):
    def test_flattened_names_are_stable_and_collision_safe(self):
        first = MODULE.flattened_name(Path("one/photo.png"))
        second = MODULE.flattened_name(Path("two/photo.png"))
        self.assertEqual(first, MODULE.flattened_name(Path("one/photo.png")))
        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith(".jxl"))
        self.assertNotIn("/", first)

    def test_discovery_is_recursive_sorted_and_excludes_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            destination = source / "converted"
            (source / "nested").mkdir()
            destination.mkdir()
            (source / "nested" / "b.HEIC").touch()
            (source / "a.png").touch()
            (source / "notes.txt").touch()
            (destination / "old.png").touch()
            discovered = MODULE.discover_images(source, destination)
            self.assertEqual(
                [path.relative_to(source).as_posix() for path in discovered],
                ["a.png", "nested/b.HEIC"],
            )

    def test_png_is_sent_directly_to_cjxl_losslessly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            output = root / "output.jxl"
            source.write_bytes(b"png")
            cjxl = root / "cjxl"
            cjxl.write_text(
                "#!/bin/sh\n"
                "test \"$3\" = -d || exit 2\n"
                "test \"$4\" = 0 || exit 3\n"
                "cp \"$1\" \"$2\"\n"
            )
            cjxl.chmod(0o755)
            MODULE.encode_lossless(source, output, str(cjxl), None)
            self.assertEqual(output.read_bytes(), b"png")

    def test_heic_uses_16_bit_intermediate_then_lossless_cjxl(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.heic"
            output = root / "output.jxl"
            source.write_bytes(b"heic")
            log = root / "magick.log"
            magick = root / "magick"
            magick.write_text(
                "#!/bin/sh\n"
                f"printf '%s\\n' \"$@\" > '{log}'\n"
                "last=\"\"\n"
                "for arg in \"$@\"; do last=\"$arg\"; done\n"
                "last=${last#png:}\n"
                "printf decoded > \"$last\"\n"
            )
            magick.chmod(0o755)
            cjxl = root / "cjxl"
            cjxl.write_text("#!/bin/sh\ncp \"$1\" \"$2\"\n")
            cjxl.chmod(0o755)
            MODULE.encode_lossless(source, output, str(cjxl), str(magick))
            self.assertEqual(output.read_bytes(), b"decoded")
            arguments = log.read_text().splitlines()
            self.assertIn("-auto-orient", arguments)
            self.assertIn("-depth", arguments)
            self.assertIn("16", arguments)


if __name__ == "__main__":
    unittest.main()
