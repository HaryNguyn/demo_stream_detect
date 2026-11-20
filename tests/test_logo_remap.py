"""CLI regression tests for non-destructive logo label migration."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/remap_logo_labels.py"
CLASSES = ROOT / "configs/classes_logo.example.yaml"


class LogoRemapTests(unittest.TestCase):
    def run_script(self, source, output, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--input", str(source),
             "--output", str(output), "--new-classes", str(CLASSES), *args],
            capture_output=True, text=True,
        )

    def test_name_mapping_empty_labels_and_dry_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "input", root / "output"
            (source / "train").mkdir(parents=True)
            original = "21 0.5 0.5 0.2 0.2\n0 0.2 0.2 0.1 0.1\n"
            (source / "train/car.txt").write_text(original, encoding="utf-8")
            (source / "train/negative.txt").touch()
            report = root / "report"
            dry = self.run_script(source, output, "--dry-run", "--report", str(report))
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(report.with_suffix(".json").exists())
            self.assertFalse(report.with_suffix(".csv").exists())
            result = self.run_script(source, output, "--report", str(report))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((output / "train/car.txt").read_text(),
                             "1 0.5 0.5 0.2 0.2\n15 0.2 0.2 0.1 0.1\n")
            self.assertEqual((output / "train/negative.txt").read_text(), "")
            self.assertEqual((source / "train/car.txt").read_text(), original)

    def test_invalid_labels_write_nothing(self):
        for line in ("0 nan 0.5 0.2 0.2", "0 0.5 0.5 0 0.2",
                     "0 0.5 0.5 0.2 0.2 0.9", "24 0.5 0.5 0.2 0.2"):
            with self.subTest(line=line), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "input"
                source.mkdir()
                (source / "bad.txt").write_text(line)
                result = self.run_script(source, root / "output")
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((root / "output").exists())

    def test_report_cannot_modify_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input"
            source.mkdir()
            (source / "car.txt").write_text("21 0.5 0.5 0.2 0.2\n")
            result = self.run_script(source, root / "output", "--report", str(source / "report"))
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "output").exists())
