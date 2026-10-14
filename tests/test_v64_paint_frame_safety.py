import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PaintFrameSafetyTests(unittest.TestCase):
    def test_no_name_is_used_before_a_local_import_in_the_same_function(self):
        # A local "import math" inside paint code made math a local name for the
        # whole method, so the window-drag sparkle animation raised mid-frame.
        problems = []
        for path in ROOT.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                local_imports = {}
                for node in ast.walk(fn):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        for alias in node.names:
                            name = (alias.asname or alias.name).split(".")[0]
                            local_imports.setdefault(name, node.lineno)
                for node in ast.walk(fn):
                    if (isinstance(node, ast.Name) and node.id in local_imports
                            and node.lineno < local_imports[node.id]):
                        problems.append(f"{path.name}:{node.lineno} {fn.name}() uses {node.id}")
        self.assertEqual(problems, [])

    def test_paint_events_always_release_the_painter(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("def _paint_frame("), 2)
        self.assertIn("if painter is not None and painter.isActive():", source)
        self.assertNotIn("WINDOW DRAG ACTIVE", source)


if __name__ == "__main__":
    unittest.main()
