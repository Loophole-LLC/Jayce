# Copyright (C) 2026 Loophole, LLC.
# SPDX-License-Identifier: AGPL-3.0-only
# See LICENSE.md in the repository root for terms and warranty information.

"""One parser for the four commands; install only the dependencies they need."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


class JayceParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, f"Jayce: {message}\nRun ./jayce --help to see the four commands.\n")


def parse_args(arguments=None):
    parser = JayceParser(
        prog="jayce", allow_abbrev=False,
        usage="jayce [--no-parent]\n       jayce train [DATA] [--capacity N]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "  ./jayce                 Chat with the parent; learn and save automatically.\n"
            "  ./jayce --no-parent     Chat using Jayce's memory alone.\n"
            "  ./jayce train           Learn random lessons from the parent.\n"
            "  ./jayce train DATA      Learn from a Parquet file or directory without a parent."
        ),
        epilog="All modes share memory. Training saves automatically; Ctrl+C pauses it.",
    )
    parser.add_argument("command", nargs="?", choices=["train"], help=argparse.SUPPRESS)
    parser.add_argument("data", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--no-parent", action="store_true", help="Chat without loading the parent")
    parser.add_argument("--capacity", type=int, metavar="N", help="Training memory limit in prototype slots")
    args = parser.parse_intermixed_args(arguments)
    if args.capacity is not None and args.capacity < 1:
        parser.error("--capacity must be a positive number")
    if args.command != "train":
        if args.capacity is not None:
            parser.error("Set capacity when training: ./jayce train --capacity N")
        return args
    if args.no_parent and args.data is None:
        parser.error("Training without a parent needs data: ./jayce train DATA")
    return args


def required_extra(args):
    if args.command == "train":
        return "corpus" if args.data is not None else "parent"
    return None if args.no_parent else "parent"


def main():
    uv, *arguments = sys.argv[1:]
    extra = required_extra(parse_args(arguments))
    root = Path(__file__).resolve().parent
    command = [uv, "run", "--locked", "--inexact"]
    if extra:
        command += ["--extra", extra]
    command += ["--project", str(root), "python", "-u", str(root / "jayce.py"), *arguments]
    os.execv(uv, command)


if __name__ == "__main__":
    main()
