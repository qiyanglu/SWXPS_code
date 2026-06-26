"""Thin command-line wrapper for YAML project workflows."""

from __future__ import annotations

import argparse

from .project import init_project, run_project, validate_project


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swanx")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("project_dir")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("project_yaml")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("project_yaml")
    args = parser.parse_args(argv)

    if args.command == "init":
        project_dir = init_project(args.project_dir)
        print(f"Created SWANX project at: {project_dir}")
        return 0
    if args.command == "validate":
        validate_project(args.project_yaml)
        print(f"Validated {args.project_yaml}")
        return 0
    if args.command == "run":
        output = run_project(args.project_yaml)
        print(f"Wrote {output}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
