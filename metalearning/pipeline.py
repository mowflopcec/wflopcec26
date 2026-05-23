import subprocess
import argparse
import sys
from pathlib import Path
import json
import importlib.util


def _load_build_indices():
	comp_path = Path(__file__).parent / 'components' / 'build_indices_dict.py'
	spec = importlib.util.spec_from_file_location('build_indices_dict', comp_path)
	module = importlib.util.module_from_spec(spec)
	assert spec and spec.loader
	spec.loader.exec_module(module)
	return module.build_indices_for_indicator

build_indices_for_indicator = _load_build_indices()

merit_table_builder_path = 'components/merit_table_builder.py'
regression_metrics_path = 'components/regression_metrics.py'
figure_builder_metrics_path = 'components/figure_builder.py'

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--label", "-l", default="epsilon", choices=["epsilon", "hipervolume", "igd"])
	parser.add_argument("--python", default=sys.executable)
	args = parser.parse_args()

	label = args.label

	print(f"===================== Building indices dict ({label}) =====================")
	try:
		indices = build_indices_for_indicator(label, tie_strategy='all')
		json_dir = Path('result') / label / 'indices'
		json_dir.mkdir(parents=True, exist_ok=True)
		json_path = json_dir / f"{label}_dict.json"
		with open(json_path, 'w', encoding='utf-8') as f:
			json.dump(indices, f, ensure_ascii=False, indent=2)
		print(f"Indices JSON saved at: {json_path}")
	except Exception as e:
		print(f"[WARNING] Could not build indices dict for '{label}': {e}")
	print("=========================================================================")

	print("\n\n")

	print(f"===================== Running merit_table_builder.py ({label}) =====================")
	subprocess.run([args.python, merit_table_builder_path, label], check=True)
	print("=========================================================================")

	print("\n\n")

	print(f"===================== Running regression_metrics.py ({label}) =====================")
	subprocess.run([args.python, regression_metrics_path, label], check=True)
	print("=========================================================================")
	print("\n")

	print(f"===================== Running figure_builder.py ({label}) =====================")
	subprocess.run([args.python, figure_builder_metrics_path, label], check=True)
	print("=========================================================================")
	print("\n")
	print("Pipeline finished.")

if __name__ == "__main__":
	main()