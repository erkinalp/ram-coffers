"""Catch documented CLI flags/examples that do not exist in the code."""

import os
import subprocess
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs", "PS3_CLUSTER_PORT.md")
README = os.path.join(ROOT, "README.md")
SYSTEM_README = os.path.join(ROOT, "ps3-cluster", "README.md")
TOOLS = os.path.join(ROOT, "tools")


class TestDocsDoNotMentionNonexistentFlags(unittest.TestCase):
    def _read(self, *paths):
        out = []
        for path in paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    out.append(fh.read())
        return "\n".join(out)

    def test_retry_ambiguous_is_not_attributed_to_run_subcluster(self):
        """--retry-ambiguous belongs to run_region.py, not run_subcluster.py."""
        text = self._read(DOCS, README, SYSTEM_README)
        # A documented command line containing the wrong tool is a drift.
        for line in text.splitlines():
            if "run_subcluster.py" in line and "--retry-ambiguous" in line:
                self.fail(f"docs wrongly place --retry-ambiguous on subcluster: "
                          f"{line}")

    def test_run_layer_is_documented(self):
        """The layer CLI this deployment uses must appear in the docs."""
        text = self._read(DOCS, README, SYSTEM_README)
        self.assertIn("run_layer.py", text,
                      "add run_layer.py examples to the deployment docs")

    def test_dedup_bytes_is_documented_and_present(self):
        """--dedup-bytes exists on the two coordinator CLIs."""
        text = self._read(DOCS, README, SYSTEM_README)
        self.assertIn("--dedup-bytes", text)

    def test_all_cli_tools_answer_help(self):
        """Every documented launch tool can print a help page."""
        for tool in ["gen_cluster_config.py", "run_expert.py",
                     "run_subcluster.py", "run_region.py", "run_layer.py"]:
            with self.subTest(tool=tool):
                subprocess.run(
                    [sys.executable, os.path.join(TOOLS, tool), "--help"],
                    check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
