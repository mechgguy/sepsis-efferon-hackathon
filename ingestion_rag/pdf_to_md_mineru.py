#!/usr/bin/env python3
"""
Convert PDF files to Markdown using a local MinerU installation (CPU-optimized).

Features:
- Single PDF or directory input
- Recursive directory scanning
- CPU-safe MinerU execution
- Copies extracted images next to markdown
- Optional artifact retention
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BACKENDS = (
    "auto",
    "pipeline",
    "vlm-http-client",
    "hybrid-http-client",
    "vlm-auto-engine",
    "hybrid-auto-engine",
)

METHODS = (
    "auto",
    "txt",
    "ocr",
)


def default_mineru_bin() -> Path | None:
    candidates: list[Path] = []

    env_bin = os.environ.get("MINERU_BIN")
    if env_bin:
        candidates.append(Path(env_bin).expanduser())

    which_bin = shutil.which("mineru")
    if which_bin:
        candidates.append(Path(which_bin))

    candidates.append(Path.home() / ".venv" / "bin" / "mineru")

    seen: set[Path] = set()

    for candidate in candidates:
        candidate = candidate.expanduser()

        if candidate in seen:
            continue

        seen.add(candidate)

        if candidate.exists():
            return candidate

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDF files to Markdown using MinerU (CPU-optimized).",
    )

    parser.add_argument(
        "input_path",
        type=Path,
        help="PDF file or directory containing PDF files.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Directory for final markdown files.",
    )

    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Directory for MinerU artifacts.",
    )

    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        default=True,
        help="Keep MinerU artifact folders.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories for PDFs.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing markdown files.",
    )

    parser.add_argument(
        "--mineru-bin",
        type=Path,
        default=default_mineru_bin(),
        help="Path to MinerU executable.",
    )

    parser.add_argument(
        "-b",
        "--backend",
        choices=BACKENDS,
        default="pipeline",
        help="MinerU backend.",
    )

    parser.add_argument(
        "-m",
        "--method",
        choices=METHODS,
        default="auto",
        help="Parsing method.",
    )

    parser.add_argument(
        "-l",
        "--lang",
        default="latin",
        help="OCR language hint.",
    )

    parser.add_argument(
        "--formula",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable formula parsing.",
    )

    parser.add_argument(
        "--table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable table parsing.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help='Execution device ("cpu", "cuda", etc).',
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU-safe pipeline backend.",
    )

    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=None,
        help="Start page (0-based).",
    )

    parser.add_argument(
        "-e",
        "--end",
        type=int,
        default=None,
        help="End page (0-based).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing.",
    )

    parser.add_argument(
        "--mineru-extra-arg",
        action="append",
        default=[],
        help="Additional MinerU arguments.",
    )

    return parser.parse_args()


def discover_pdfs(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            return []
        return [input_path]

    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(input_path.glob(pattern))


def resolve_output_dir(
    input_path: Path,
    output_dir: Path | None,
) -> Path:
    if output_dir:
        return output_dir.resolve()

    if input_path.is_file():
        return input_path.parent.resolve()

    return input_path.resolve()


def build_final_md_path(
    input_root: Path,
    pdf_path: Path,
    output_dir: Path,
) -> Path:
    try:
        relative = pdf_path.relative_to(input_root)
        return output_dir / relative.with_suffix(".md")
    except ValueError:
        return output_dir / f"{pdf_path.stem}.md"


def parse_dir_name(backend: str, method: str) -> str:
    return f"{backend}_{method}"


def build_artifacts_root(
    input_root: Path,
    pdf_path: Path,
    keep_artifacts: bool,
    artifacts_dir: Path | None,
    output_dir: Path,
):
    if artifacts_dir:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        return artifacts_dir.resolve(), None

    if keep_artifacts:
        root = output_dir / ".mineru_artifacts"
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve(), None

    temp_dir = tempfile.TemporaryDirectory()
    return Path(temp_dir.name), temp_dir


def resolve_backend(
    args: argparse.Namespace,
    gpu_memory_mb: int | None,
) -> tuple[str, str]:
    if args.force_cpu or args.backend == "pipeline":
        return "pipeline", "Forced CPU-safe pipeline backend."

    return args.backend, f"Using backend: {args.backend}"


def resolve_device(
    args: argparse.Namespace,
    backend: str,
    gpu_memory_mb: int | None,
) -> str | None:
    if args.force_cpu or args.device == "cpu":
        return "cpu"

    return args.device


def build_command(
    args: argparse.Namespace,
    pdf_path: Path,
    artifacts_root: Path,
    backend: str,
    effective_device: str | None,
) -> list[str]:
    cmd = [
        str(args.mineru_bin),
        str(pdf_path),
        "-o",
        str(artifacts_root),
        "-b",
        backend,
        "-m",
        args.method,
    ]

    if effective_device:
        cmd.extend(["--device", effective_device])

    if args.lang:
        cmd.extend(["-l", args.lang])

    if args.formula:
        cmd.append("--formula")

    if args.table:
        cmd.append("--table")

    if args.start is not None:
        cmd.extend(["-s", str(args.start)])

    if args.end is not None:
        cmd.extend(["-e", str(args.end)])

    cmd.extend(args.mineru_extra_arg)

    return cmd


def locate_markdown(
    artifacts_root: Path,
    pdf_stem: str,
    backend: str,
    method: str,
) -> Path:
    parse_dir = parse_dir_name(backend, method)

    md_path = (
        artifacts_root
        / pdf_stem
        / parse_dir
        / f"{pdf_stem}.md"
    )

    if not md_path.exists():
        raise FileNotFoundError(f"Markdown not found: {md_path}")

    return md_path


def rewrite_image_links(
    markdown_text: str,
    final_md_path: Path,
    images_dir: Path,
) -> str:
    if not images_dir.exists():
        return markdown_text

    images_name = f"{final_md_path.stem}_images"

    return markdown_text.replace(
        "images/",
        f"{images_name}/",
    )


def copy_images_to_final(
    artifacts_root: Path,
    pdf_stem: str,
    final_parent: Path,
    backend: str,
    method: str,
):
    parse_dir = parse_dir_name(backend, method)

    images_src = (
        artifacts_root
        / pdf_stem
        / parse_dir
        / "images"
    )

    if not images_src.exists():
        print(f"No images found in {images_src}", file=sys.stderr)
        return

    images_dest = final_parent / f"{pdf_stem}_images"

    shutil.copytree(
        images_src,
        images_dest,
        dirs_exist_ok=True,
    )

    image_count = (
        len(list(images_src.iterdir()))
        if images_src.is_dir()
        else 0
    )

    print(
        f"Copied {image_count} images to {images_dest}",
        file=sys.stdout,
    )


def run_one_pdf(
    pdf_path: Path,
    input_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
    backend: str,
    effective_device: str | None,
) -> tuple[bool, str]:
    final_md_path = build_final_md_path(
        input_root,
        pdf_path,
        output_dir,
    )

    if final_md_path.exists() and not args.overwrite:
        return True, f"Skipped existing: {final_md_path}"

    artifacts_root, temp_dir = build_artifacts_root(
        input_root,
        pdf_path,
        args.keep_artifacts,
        args.artifacts_dir,
        output_dir,
    )

    env = os.environ.copy()

    runtime_cache_dir = artifacts_root / ".runtime"

    env["MPLCONFIGDIR"] = str(runtime_cache_dir / "matplotlib")
    env["YOLO_CONFIG_DIR"] = str(runtime_cache_dir / "ultralytics")
    env["MINERU_DEVICE_MODE"] = effective_device or "cpu"

    cmd = build_command(
        args,
        pdf_path,
        artifacts_root,
        backend,
        effective_device,
    )

    try:
        if args.dry_run:
            return True, " ".join(cmd)

        subprocess.run(
            cmd,
            check=True,
            env=env,
        )

        markdown_path = locate_markdown(
            artifacts_root,
            pdf_path.stem,
            backend,
            args.method,
        )

        markdown_text = markdown_path.read_text(
            encoding="utf-8"
        )

        if args.keep_artifacts:
            images_dir = markdown_path.parent / "images"

            markdown_text = rewrite_image_links(
                markdown_text,
                final_md_path,
                images_dir,
            )

            copy_images_to_final(
                artifacts_root,
                pdf_path.stem,
                final_md_path.parent,
                backend,
                args.method,
            )

        final_md_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        final_md_path.write_text(
            markdown_text,
            encoding="utf-8",
        )

        return True, f"Created {final_md_path}"

    except subprocess.CalledProcessError as exc:
        return False, f"MinerU failed for {pdf_path}: {exc}"

    except Exception as exc:
        return False, f"Failed {pdf_path}: {exc}"

    finally:
        if temp_dir and not args.keep_artifacts:
            temp_dir.cleanup()


def main() -> int:
    args = parse_args()

    if args.force_cpu:
        args.backend = "pipeline"
        args.device = "cpu"

    if args.mineru_bin is None or not args.mineru_bin.exists():
        print(
            "MinerU executable not found. "
            "Set MINERU_BIN or install MinerU.",
            file=sys.stderr,
        )
        return 2

    input_path = args.input_path.resolve()

    if not input_path.exists():
        print(
            f"Input path does not exist: {input_path}",
            file=sys.stderr,
        )
        return 2

    pdfs = discover_pdfs(
        input_path,
        args.recursive,
    )

    if not pdfs:
        print(
            "No PDF files found.",
            file=sys.stderr,
        )
        return 1

    if input_path.is_dir():
        input_root = input_path
    else:
        input_root = input_path.parent

    output_dir = resolve_output_dir(
        input_path,
        args.output_dir,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    backend, backend_msg = resolve_backend(
        args,
        None,
    )

    effective_device = resolve_device(
        args,
        backend,
        None,
    )

    print(backend_msg, file=sys.stderr)
    print(
        "Using CPU-optimized MinerU pipeline.",
        file=sys.stderr,
    )

    failures = 0

    for pdf_path in pdfs:
        ok, message = run_one_pdf(
            pdf_path,
            input_root,
            output_dir,
            args,
            backend,
            effective_device,
        )

        print(message)

        if not ok:
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())