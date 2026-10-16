import argparse
import pandas as pd
import pprint
import os
import glob


def _resolve_metrics_parquet(file_path: str) -> str:
    """
    Accept either:
      - aggregator_metric/*weighted_average_metrics*.parquet (preferred)
      - runner_report.parquet (will auto-resolve to aggregator_metric parquet)
    """
    if file_path.endswith("runner_report.parquet"):
        run_dir = os.path.dirname(file_path)
        aggregator_dir = os.path.join(run_dir, "aggregator_metric")
        candidates = sorted(
            glob.glob(os.path.join(aggregator_dir, "*weighted_average_metrics*.parquet")),
            key=os.path.getmtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No aggregator_metric weighted_average_metrics parquet found under '{aggregator_dir}'."
            )
        return candidates[0]

    return file_path

def print_final_score_from_parquet(file_path: str):
    """
    Read the specified Parquet file, locate the row where 'scenario' == 'final_score',
    and pretty-print its metrics.
    :param file_path: Path to the Parquet file.
    """
    # Check file extension
    if not file_path.endswith(".parquet"):
        print(f"Error: file '{file_path}' is not a .parquet file.")
        return

    try:
        resolved_path = _resolve_metrics_parquet(file_path)
        if resolved_path != file_path:
            print(f"Input is runner_report; resolved metrics parquet:\n  {resolved_path}")
        print(f"Processing file: {resolved_path}")
        
        # Read Parquet into DataFrame
        df = pd.read_parquet(resolved_path)
        
        # Filter the row where scenario == 'final_score'
        if "scenario" not in df.columns:
            cols_preview = ", ".join(list(df.columns)[:30])
            raise KeyError(
                f"column 'scenario' missing. Available columns (first 30): {cols_preview}"
            )

        final_score_df = df[df["scenario"] == "final_score"]

        if final_score_df.empty:
            print(f"Error: no 'final_score' scenario found in '{resolved_path}'.")
            return
            
        # Convert first matching record to dict
        final_score_dict = final_score_df.to_dict(orient="records")[0]
        
        # Pretty print dictionary
        print("\n" + "="*25 + " FINAL SCORE " + "="*25)
        pprint.PrettyPrinter(indent=4).pprint(final_score_dict)
        print("="*63 + "\n")

    except FileNotFoundError:
        print(f"Error: file not found '{file_path}'")
    except KeyError:
        print(f"Error: column 'scenario' missing or invalid in file '{file_path}'.")
    except IndexError:
        print(f"Error: 'scenario' column exists but no 'final_score' row in '{file_path}'.")
    except Exception as e:
        # Handle any other unexpected errors
        print(f"Error while processing '{file_path}': {e}")
def main():
    """
    Parse command-line arguments and invoke the processing function.
    """
    parser = argparse.ArgumentParser(
        description="Read and print 'final_score' metrics from a single Parquet file."
    )
    parser.add_argument(
        "file_path",
        help="Path to the Parquet file (prefer: .../aggregator_metric/...weighted_average_metrics_*.parquet).",
    )

    args = parser.parse_args()
    print_final_score_from_parquet(args.file_path)

if __name__ == "__main__":
    # Script entrypoint
    main()
