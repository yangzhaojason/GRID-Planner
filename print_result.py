"""Print only complete, finite, unambiguous nuPlan results."""
import argparse
import pprint

from flow_planner.onpolicy.sim_report import parse_final_score_from_runner_report


def print_final_score_from_parquet(file_path: str):
    result = parse_final_score_from_runner_report(file_path)
    print("========================= FINAL SCORE =========================")
    pprint.PrettyPrinter(indent=4).pprint(result.components)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_path", help="Runner report or associated weighted-average metrics parquet.")
    args = parser.parse_args()
    try:
        print_final_score_from_parquet(args.file_path)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Invalid simulation result: {exc}\n")


if __name__ == "__main__":
    main()
