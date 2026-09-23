# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from jayce_cli import parse_args, required_extra
from unittest.mock import patch
import jayce


class CommandRoutingTests(unittest.TestCase):
    def test_four_commands_choose_the_right_dependencies(self):
        for arguments, extra in (([], "parent"), (["--no-parent"], None), (["train"], "parent")):
            self.assertEqual(required_extra(parse_args(arguments)), extra)
        for path in ("data/fineweb-edu", "missing.parquet", "data with spaces/shard.parquet"):
            args = parse_args(["train", path])
            self.assertEqual(required_extra(args), "corpus")
            self.assertEqual(args.data, path)

    def test_capacity_and_literal_paths_are_parsed_without_confusion(self):
        args = parse_args(["train", "--capacity", "10000"])
        self.assertEqual(args.capacity, 10000)
        self.assertIsNone(args.data)
        for arguments in (["train", "data", "--capacity=10000"], ["--capacity", "10000", "train", "data"]):
            args = parse_args(arguments)
            self.assertEqual((args.data, args.capacity), ("data", 10000))
        args = parse_args(["train", "--", "--no-parent"])
        self.assertEqual(args.data, "--no-parent")
        self.assertFalse(args.no_parent)

    def test_main_dispatches_all_four_modes(self):
        with patch("jayce.ChatSession") as session, patch("jayce.chat") as chat:
            jayce.main([])
            session.assert_called_once_with(parent=True)
            chat.assert_called_once_with(session.return_value)
        with patch("jayce.ChatSession") as session, patch("jayce.chat"):
            jayce.main(["--no-parent"])
            session.assert_called_once_with(parent=False)
        with patch("jayce_parent_train.run_training") as train:
            jayce.main(["train", "--capacity", "32768"])
            train.assert_called_once_with(32768)
        with patch("jayce_train.run_training") as train:
            jayce.main(["train", "my data.parquet", "--capacity", "32768"])
            train.assert_called_once_with("my data.parquet", 32768)

    def test_removed_controls_and_invalid_capacity_fail_before_any_launch(self):
        for arguments in (["--list-lessons"], ["--prompt=--no-parent"], ["parent"],
                          ["train", "--lessons", "200"], ["train", "--topic", "weather"],
                          ["train", "--data", "folder"], ["train", "--capacity", "0"],
                          ["train", "--capacity", "-1"], ["train", "--no-parent"], ["--capacity", "10"]):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()) as output, self.assertRaises(SystemExit) as error:
                parse_args(arguments)
            self.assertEqual(error.exception.code, 2)
            self.assertIn("./jayce --help", output.getvalue())

    def test_help_only_exposes_the_four_commands_and_capacity(self):
        for arguments in (["--help"], ["train", "--help"], ["--no-parent", "--help"]):
            with redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as error:
                parse_args(arguments)
            self.assertEqual(error.exception.code, 0)
            self.assertIn("./jayce train DATA", output.getvalue())
            self.assertIn("--capacity N", output.getvalue())
            for removed in ("--topic", "--lessons", "--run-dir", "--model", "--prompt"):
                self.assertNotIn(removed, output.getvalue())

    def test_shell_launcher_preserves_data_paths_and_only_installs_selected_extras(self):
        with tempfile.TemporaryDirectory() as folder:
            uv = Path(folder) / "uv"
            # Run the real dispatcher, intercepting only its final uv invocation.
            uv.write_text(f"#!{sys.executable}\n" + """
import json, os, sys
arguments = sys.argv[1:]
index = arguments.index('python')
if arguments[index + 1].endswith('jayce_cli.py'):
    os.execv(sys.executable, [sys.executable, *arguments[index + 1:]])
print(json.dumps(arguments))
""")
            uv.chmod(0o755)
            root = Path(__file__).resolve().parents[1]
            for arguments in ([], ["--no-parent"], ["train", "--capacity", "10000"],
                              ["train", "my data/shard.parquet", "--capacity", "10000"]):
                with self.subTest(arguments=arguments):
                    extra = required_extra(parse_args(arguments))
                    result = subprocess.run([str(root / "jayce"), *arguments], cwd=root, capture_output=True,
                                            text=True, timeout=30, env={**os.environ, "PATH": folder + os.pathsep + os.environ["PATH"]})
                    self.assertEqual(result.returncode, 0, result.stderr)
                    actual = json.loads(result.stdout)
                    self.assertEqual([actual[i + 1] for i, value in enumerate(actual) if value == "--extra"], [extra] if extra else [])
                    self.assertEqual(actual[actual.index(str(root / "jayce.py")) + 1:], arguments)


if __name__ == "__main__":
    unittest.main()
