import os
import sys
import pandas as pd

if len(sys.argv) != 2:
    print("Usage: python build_pareto_adaptive.py <indicator>")
    sys.exit(1)

indicator_arg = sys.argv[1].lower()
if indicator_arg == 'hv':
    indicator_arg = 'hipervolume'

valid_indicators = {"epsilon", "hipervolume", "igd"}
if indicator_arg not in valid_indicators:
    print(f"Invalid indicator: {indicator_arg}. Use one of: {', '.join(valid_indicators)}")
    sys.exit(1)

base_path = 'meta_features/dataset/'
analysis_path = 'Analysis/'
output_path = f'meta_dataset/{indicator_arg}/pareto_based/adaptive_walk/'

os.makedirs(output_path, exist_ok=True)

r_values = ["0.05", "0.1", "0.25", "0.5", "1.0"]

combined_data = {f"r{r}": [] for r in r_values}

indicator_config = {
    'epsilon': {
        'subdir': 'epsilon_additive',
        'files': {
            'MOEAD': 'esp_ad_moead.out',
            'NSGAII': 'esp_ad_nsga2.out',
            'COMOLSD': 'esp_ad_comolsd.out'
        }
    },
    'hipervolume': {
        'subdir': 'hypervolume',
        'files': {
            'MOEAD': 'HV_moead.out',
            'NSGAII': 'HV_nsga2.out',
            'COMOLSD': 'HV_comolsd.out'
        }
    },
    'igd': {
        'subdir': 'igd',
        'files': {
            'MOEAD': 'IGD_moead.out',
            'NSGAII': 'IGD_nsga2.out',
            'COMOLSD': 'IGD_comolsd.out'
        }
    }
}

cfg = indicator_config[indicator_arg]

for instance in os.listdir(base_path):
    pareto_adaptive_walk_path = os.path.join(base_path, instance, 'pareto_based', 'adaptative_walk')
    analysis_instance_path = os.path.join(analysis_path, instance)

    if not os.path.isdir(pareto_adaptive_walk_path) or not os.path.isdir(analysis_instance_path):
        continue

    metric_dir = os.path.join(analysis_instance_path, cfg['subdir'])
    files_map = {alg: os.path.join(metric_dir, fname) for alg, fname in cfg['files'].items()}

    def compute_mean(filepath):
        if not os.path.exists(filepath):
            print(f"[WARN] File not found: {filepath}")
            return None
        numbers = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line = line.replace(',', '.')
                tokens = [t for chunk in line.split() for t in chunk.split(';')]
                flat_tokens = []
                for t in tokens:
                    if ',' in t and t.count('.') == 0:
                        t = t.replace(',', '.')
                    if ',' in t:
                        flat_tokens.extend(t.split(','))
                    else:
                        flat_tokens.append(t)
                for tok in flat_tokens:
                    try:
                        numbers.append(float(tok))
                    except ValueError:
                        continue
        if not numbers:
            print(f"[WARN] No valid numeric values in: {filepath}")
            return None
        return sum(numbers) / len(numbers)

    means = {alg: compute_mean(path) for alg, path in files_map.items()}

    for r in r_values:
        dir_name = f"r{r}"
        dir_path = os.path.join(pareto_adaptive_walk_path, dir_name)

        if not os.path.isdir(dir_path):
            continue

        csv_file = next((f for f in os.listdir(dir_path) if f.startswith('ofe_') and f.endswith('.csv')), None)

        if csv_file:
            csv_path = os.path.join(dir_path, csv_file)
            df = pd.read_csv(csv_path)

            df['NSGAII'] = means['NSGAII']
            df['MOEAD'] = means['MOEAD']
            df['COMOLSD'] = means['COMOLSD']

            combined_data[dir_name].append(df)

for config, dataframes in combined_data.items():
    if dataframes:
        final_df = pd.concat(dataframes, ignore_index=True)
        final_path = os.path.join(output_path, f"{config}.csv")
        final_df.to_csv(final_path, index=False)
        print(f"Generated archive ({indicator_arg}): {final_path}")

print("Process completed")
