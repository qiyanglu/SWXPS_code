"""Command-line entry points for SWANX."""

from __future__ import annotations

import argparse
from pathlib import Path

from .project import init_project, inspect_project, run_project, validate_project


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swanx")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="create an editable YAML project folder")
    init_parser.add_argument("project_dir")
    init_parser.add_argument("--copy-example-data", action="store_true")
    init_parser.add_argument("--data-root")
    init_parser.add_argument(
        "--template",
        choices=("minimal", "multilayer", "fit-demo"),
        default="minimal",
        help="starter ProjectSpec template to generate",
    )

    inspect_parser = subparsers.add_parser("inspect", help="summarize a YAML project without running it")
    inspect_parser.add_argument("project_yaml")

    validate_parser = subparsers.add_parser("validate", help="validate a YAML project")
    validate_parser.add_argument("project_yaml")

    run_parser = subparsers.add_parser("run", help="run a YAML project")
    run_parser.add_argument("project_yaml")

    args = parser.parse_args(argv)
    if args.command == "init":
        project_dir = init_project(
            args.project_dir,
            copy_example_data=args.copy_example_data,
            data_root=args.data_root,
            template=args.template,
        )
        print(f"Created SWANX project: {Path(project_dir)}")
        return 0
    if args.command == "inspect":
        print(inspect_project(args.project_yaml), end="")
        return 0
    if args.command == "validate":
        validate_project(args.project_yaml)
        print(f"Project is valid: {args.project_yaml}")
        return 0
    if args.command == "run":
        output = run_project(args.project_yaml, progress=True)
        print(f"SWANX results written to: {output}")
        return 0
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
