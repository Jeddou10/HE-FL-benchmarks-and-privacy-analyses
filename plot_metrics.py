import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=str, nargs="+", required=True,
        help="One or more paths to metrics.csv files"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output image file (PNG). If not given, show interactively."
    )
    parser.add_argument(
        "--metric", type=str, default="test_acc",
        help="Column to plot (default: test_acc)"
    )
    args = parser.parse_args()

    plt.figure(figsize=(7,5))

    for csv_path in args.csv:
        df = pd.read_csv(csv_path)

        if args.metric not in df.columns:
            raise ValueError(
                f"Metric '{args.metric}' not found in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )

        label = Path(csv_path).parent.name  # use run directory as label
        plt.plot(df["round"], df[args.metric], marker="o", linestyle="-", label=label)

    plt.xlabel("Round")
    plt.ylabel(args.metric.replace("_", " ").title())
    plt.title(f"{args.metric.replace('_',' ').title()} over Rounds")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    if args.out:
        plt.savefig(args.out, bbox_inches="tight", dpi=150)
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
