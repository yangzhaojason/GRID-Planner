import os
import argparse
import json
from pathlib import Path

def get_filter_parameters(num_scenarios_per_type=None, limit_total_scenarios=None, shuffle=True, scenario_tokens=None, log_names=None):

    scenario_types = None

    scenario_tokens                      # List of scenario tokens to include
    log_names = log_names                # Filter scenarios by log names
    map_names = None                     # Filter scenarios by map names

    num_scenarios_per_type               # Number of scenarios per type
    limit_total_scenarios                # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type
    timestamp_threshold_s = None         # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
    ego_displacement_minimum_m = None    # Whether to remove scenarios where the ego moves less than a certain amount

    expand_scenarios = True              # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals = False          # Whether to remove scenarios where the mission goal is invalid
    shuffle                              # Whether to shuffle the scenarios

    ego_start_speed_threshold = None     # Limit to scenarios where the ego reaches a certain speed from below
    ego_stop_speed_threshold = None      # Limit to scenarios where the ego reaches a certain speed from above
    speed_noise_tolerance = None         # Value at or below which a speed change between two timepoints should be ignored as noise.

    return scenario_types, scenario_tokens, log_names, map_names, num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, ego_displacement_minimum_m, \
           expand_scenarios, remove_invalid_goals, shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance

def parse_bool(value):
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def write_data_list(save_path, output_json):
    """Write a stable index of processed files, relative to the dataset root."""
    npz_files = sorted(p.name for p in Path(save_path).glob("*.npz") if p.is_file())
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(npz_files, indent=2) + "\n", encoding="utf-8")
    return len(npz_files)


def build_parser():
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--data_path', required=True, type=str, help='path to raw data')
    parser.add_argument('--map_path', required=True, type=str, help='path to map data')

    parser.add_argument('--save_path', default='./cache', type=str, help='path to save processed data')
    parser.add_argument('--scenarios_per_type', type=int, default=None, help='number of scenarios per type')
    parser.add_argument('--total_scenarios', type=int, default=10, help='limit total number of scenarios')
    parser.add_argument('--shuffle_scenarios', type=parse_bool, default=True, help='shuffle scenarios')



    parser.add_argument('--log_names', default=str(Path(__file__).resolve().with_name('nuplan_train.json')),
                        help='JSON list of training log names')
    parser.add_argument('--output_json', default=None,
                        help='dataset index output (default: SAVE_PATH/flow_planner_training.json)')
    return parser


def main():
    args = build_parser().parse_args()
    args.data_path = str(Path(args.data_path).expanduser().resolve())
    args.map_path = str(Path(args.map_path).expanduser().resolve())
    args.save_path = str(Path(args.save_path).expanduser().resolve())
    output_json = args.output_json or str(Path(args.save_path) / 'flow_planner_training.json')

    # Keep --help and manifest utilities usable before installing the training stack.
    from flow_planner.data.data_process.data_processor import DataProcessor

    from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder

    # create save folder
    os.makedirs(args.save_path, exist_ok=True)

    sensor_root = None
    db_files = None

    # Only preprocess the training data
    with open(Path(args.log_names).expanduser(), "r", encoding="utf-8") as file:
        log_names = json.load(file)
    print("loading nuplan dataset... (This might take few minutes)")
    map_version = "nuplan-maps-v1.0"    
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version)
    scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type, args.total_scenarios, args.shuffle_scenarios, log_names=log_names))

    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")

    # process data
    del worker, builder, scenario_filter
    processor = DataProcessor(args.save_path)
    processor.work(scenarios)

    count = write_data_list(args.save_path, output_json)
    print(f"Saved {count} .npz file names to {output_json}")


if __name__ == "__main__":
    main()
