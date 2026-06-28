#!/usr/bin/env python3
"""
PicPicComparison CLI Entry Point.

Compare two images using Linear Algebra and Algorithmic techniques.

Usage:
    python main.py image_a.png image_b.png
    python main.py image_a.png image_b.png --output-dir results/
    python main.py image_a.png image_b.png --size 512 512 --no-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from src.comparator import ImageComparator
from src.loader import ImageLoadError, DimensionMismatchError


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. Defaults to sys.argv[1:].

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        prog="picpiccomparison",
        description=(
            "Compare two images using Linear Algebra (Frobenius, L1, Cosine, SVD) "
            "and Algorithmic (MSE, PSNR, Histogram) metrics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py data/image_A.png data/image_B.png\n"
            "  python main.py img1.jpg img2.jpg --output-dir results/ --size 512 512\n"
            "  python main.py img1.png img2.png --no-plots --json-only\n"
        ),
    )

    parser.add_argument(
        "image_a",
        type=str,
        help="Path to the first image file.",
    )
    parser.add_argument(
        "image_b",
        type=str,
        help="Path to the second image file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output",
        help="Directory to save results and visualizations (default: output/).",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Resize both images to WIDTH HEIGHT before comparison.",
    )
    parser.add_argument(
        "--svd-top-k",
        type=int,
        default=50,
        help="Number of top singular values for SVD comparison (default: 50).",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=256,
        help="Number of histogram bins (default: 256).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating visualization plots.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only output JSON to stdout, no file I/O.",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save the comparison report as JSON to the output directory.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress the summary output.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    args = parse_args(argv)

    # Validate paths
    path_a = Path(args.image_a)
    path_b = Path(args.image_b)

    if not path_a.exists():
        print(f"Error: Image A not found: {path_a}", file=sys.stderr)
        return 1
    if not path_b.exists():
        print(f"Error: Image B not found: {path_b}", file=sys.stderr)
        return 1

    # Build target size
    target_size = None
    if args.size is not None:
        target_size = (args.size[0], args.size[1])

    try:
        comparator = ImageComparator(
            target_size=target_size,
            svd_top_k=args.svd_top_k,
            histogram_bins=args.hist_bins,
        )

        report = comparator.compare(path_a, path_b)

        # JSON-only mode: print to stdout and exit
        if args.json_only:
            print(report.to_json())
            return 0

        # Print summary unless quiet
        if not args.quiet:
            print(report.summary())

        # Output directory
        output_dir = Path(args.output_dir)

        # Save report
        if args.save_report:
            report_path = output_dir / "report.json"
            saved = comparator.save_report(report, report_path)
            print(f"\n  Report saved to: {saved}")

        # Generate visualizations
        if not args.no_plots:
            print(f"\n  Generating visualizations in {output_dir}/...")
            saved_files = comparator.save_visualizations(report, output_dir)
            for f in saved_files:
                print(f"    -> {f}")
            print(f"  Done! {len(saved_files)} plots saved.")

        return 0

    except ImageLoadError as e:
        print(f"Image Load Error: {e}", file=sys.stderr)
        return 1
    except DimensionMismatchError as e:
        print(f"Dimension Mismatch: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
