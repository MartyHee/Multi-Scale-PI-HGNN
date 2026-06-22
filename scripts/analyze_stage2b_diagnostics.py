"""
Stage 2-B Diagnostic Evaluation Script.

Reads artifacts from all 5 completed Stage 2-A/2-B baselines (MLP, GCN, GAT, RGCN, HGT)
and produces fine-grained diagnostic outputs to guide Stage 3 (Ours-base) design.

Usage:
    python scripts/analyze_stage2b_diagnostics.py \\
        --artifact-dir remote_artifacts \\
        --output-dir outputs/diagnostics/stage2b \\
        [--outputs-dir outputs/baselines]

No new training is performed. No model code is modified.
"""

import argparse, csv, io, json, os, re, sys, textwrap, tarfile, traceback
from datetime import datetime
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISP_COMPONENTS = ['Dx', 'Dy', 'Dz', 'Rx', 'Ry', 'Rz']
FORCE_COMPONENTS = [
    'Fx_I', 'Fy_I', 'Fz_I', 'Mx_I', 'My_I', 'Mz_I',
    'Fx_J', 'Fy_J', 'Fz_J', 'Mx_J', 'My_J', 'Mz_J',
]
FORCE_COMPONENTS_SHORT = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

DISP_R2_KNOWN = {
    'MLP': 0.8554, 'GCN': 0.8476, 'GAT': 0.8421,
    'RGCN': 0.9366, 'HGT': 0.9769,
}
FORCE_R2_KNOWN = {
    'MLP': 0.9824, 'GCN': 0.9696, 'GAT': 0.9632,
    'RGCN': 0.9878, 'HGT': 0.9891,
}
RELMAE_KNOWN = {
    'MLP': 0.0884, 'GCN': 0.1227, 'GAT': 0.1361,
    'RGCN': 0.0724, 'HGT': 0.0683,
}

REPORTED_METRICS = {}
for m in ['MLP', 'GCN', 'GAT', 'RGCN', 'HGT']:
    REPORTED_METRICS[m] = {
        'Disp R2': DISP_R2_KNOWN[m],
        'Force R2': FORCE_R2_KNOWN[m],
        'RelMAE': RELMAE_KNOWN[m],
    }

MODEL_ORDER = ['MLP', 'GCN', 'GAT', 'RGCN', 'HGT']
MODEL_LABELS = {
    'MLP': 'MLP', 'GCN': 'GCN', 'GAT': 'GAT',
    'RGCN': 'RGCN', 'HGT': 'HGT',
}
MODEL_COLORS = {
    'MLP': '#7f7f7f', 'GCN': '#1f77b4', 'GAT': '#ff7f0e',
    'RGCN': '#2ca02c', 'HGT': '#d62728',
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def fmt_r2(v):
    """Format R2 value."""
    return f'{v:.4f}'


def fmt_mae(v):
    """Format MAE value."""
    if abs(v) < 1:
        return f'{v:.6f}'
    return f'{v:.2f}'


def fmt_rel(v):
    """Format relative metric."""
    return f'{v:.4f}'


def fmt_time(sec):
    """Format seconds to human-readable."""
    if sec < 60:
        return f'{sec:.1f}s'
    elif sec < 3600:
        return f'{sec / 60:.1f}min'
    else:
        return f'{sec / 3600:.1f}h'


def safe_get(d, *keys, default=None):
    """Safely navigate nested dict."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, {})
    return d if d != {} else default


# ---------------------------------------------------------------------------
# Artifact Reader
# ---------------------------------------------------------------------------


class ArtifactReader:
    """Read artifact files from either a tar.gz or an extracted directory."""

    def __init__(self, path):
        self.path = path
        self._is_tar = path.endswith('.tar.gz') or path.endswith('.tgz')
        self._tar = None
        self._names = None

    def __enter__(self):
        if self._is_tar:
            self._tar = tarfile.open(self.path, 'r:gz')
            self._names = self._tar.getnames()
        else:
            self._names = []
            for root, dirs, files in os.walk(self.path):
                for f in files:
                    self._names.append(os.path.join(root, f))
        return self

    def __exit__(self, *args):
        if self._tar:
            self._tar.close()

    def _find(self, suffix):
        for n in self._names:
            if n.endswith(suffix):
                return n
        return None

    def read_text(self, suffix):
        """Read a file by suffix, return str or None."""
        name = self._find(suffix)
        if name is None:
            return None
        if self._is_tar:
            f = self._tar.extractfile(name)
            return f.read().decode('utf-8', errors='replace')
        else:
            with open(name, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()

    def read_json(self, suffix):
        text = self.read_text(suffix)
        if text is None:
            return None
        return json.loads(text)

    def read_csv_rows(self, suffix):
        text = self.read_text(suffix)
        if text is None:
            return None
        reader = csv.DictReader(io.StringIO(text))
        return list(reader)

    def has_file(self, suffix):
        return self._find(suffix) is not None

    def list_files(self):
        return self._names[:]


# ---------------------------------------------------------------------------
# 1. Artifact Consistency Check
# ---------------------------------------------------------------------------


class ArtifactCheck:
    """Per-model artifact consistency verification."""

    def __init__(self, model_name, artifact_path):
        self.model = model_name
        self.artifact_path = artifact_path
        self.results = {}
        self.passed = True
        self.metrics = None
        self.train_log = None
        self.model_summary_data = None
        self.config_text = None
        self.server_log_text = None

    def run(self):
        with ArtifactReader(self.artifact_path) as reader:
            self._check_files_exist(reader)
            self._check_train_log(reader)
            self._check_metrics_summary(reader)
            self._check_model_summary(reader)
            self._check_config(reader)
            self._check_server_log(reader)
            self._check_consistency()
        return self.results

    def _record(self, key, value, passed=True):
        self.results[key] = {'value': value, 'passed': passed}
        if not passed:
            self.passed = False

    def _check_files_exist(self, reader):
        required = [
            'metrics_summary.json', 'model_summary.json',
            'train_log.csv', 'config_resolved.yaml', 'server_output.log',
            'best_model.pt',
        ]
        for f in required:
            exists = reader.has_file(f)
            self._record(f'has_{f}', exists, exists)
            if not exists:
                print(f'    [WARN] Missing: {f}')

    def _check_train_log(self, reader):
        rows = reader.read_csv_rows('train_log.csv')
        if rows is None:
            self._record('train_log_exists', False, False)
            return
        self.train_log = rows
        n = len(rows)
        epochs = [int(r['epoch']) for r in rows]
        continuous = (epochs == list(range(1, n + 1)))

        nan_count = inf_count = 0
        for row in rows:
            for k, v in row.items():
                if k == 'epoch':
                    continue
                try:
                    fv = float(v)
                    if np.isnan(fv):
                        nan_count += 1
                    elif np.isinf(fv):
                        inf_count += 1
                except (ValueError, TypeError):
                    pass

        # best epoch from val_loss
        best_val_loss = float('inf')
        best_epoch_from_val = None
        for row in rows:
            try:
                vl = float(row['val_loss'])
                if vl < best_val_loss:
                    best_val_loss = vl
                    best_epoch_from_val = int(row['epoch'])
            except (KeyError, ValueError):
                pass

        first_vl = float(rows[0]['val_loss']) if rows else 0
        last_vl = float(rows[-1]['val_loss']) if rows else 0
        improving = last_vl < first_vl

        self._record('train_log_epochs', n, n >= 80)
        self._record('train_log_continuous', continuous, continuous)
        self._record('train_log_nan', nan_count, nan_count == 0)
        self._record('train_log_inf', inf_count, inf_count == 0)
        self._record('train_log_improving', improving, improving)
        self._record('best_epoch_from_log', best_epoch_from_val)
        self._record('first_val_loss', round(first_vl, 6))
        self._record('last_val_loss', round(last_vl, 6))

    def _check_metrics_summary(self, reader):
        mt = reader.read_json('metrics_summary.json')
        if mt is None:
            self._record('metrics_summary_exists', False, False)
            return
        self.metrics = mt

        num_params = mt.get('num_params', -1)
        best_epoch = safe_get(mt, 'training', 'best_epoch')
        total_time = safe_get(mt, 'training', 'total_time_seconds', default=0)
        early_stopped = safe_get(mt, 'training', 'early_stopped', default=None)

        dataset_sizes = mt.get('dataset_sizes', {})
        train_size = dataset_sizes.get('train', -1)
        val_size = dataset_sizes.get('val', -1)
        test_size = dataset_sizes.get('test', -1)

        test = mt.get('test', {})
        disp = test.get('disp', {})
        force = test.get('force', {})

        disp_r2 = disp.get('macro_avg_r2', None)
        force_r2 = force.get('macro_avg_r2', None)
        relmae = test.get('combined_rel_mae', None)
        disp_mae = disp.get('macro_avg_mae', None)
        force_mae = force.get('macro_avg_mae', None)

        disp_per_r2 = disp.get('per_component_r2', [])
        force_per_r2 = force.get('per_component_r2', [])
        disp_per_mae = disp.get('per_component_mae', [])
        force_per_mae = force.get('per_component_mae', [])

        self._record('num_params', num_params)
        self._record('metrics_best_epoch', best_epoch)
        self._record('total_time_seconds', total_time)
        self._record('early_stopped', early_stopped)
        self._record('train_size', train_size)
        self._record('test_disp_r2', disp_r2)
        self._record('test_force_r2', force_r2)
        self._record('test_combined_rel_mae', relmae)
        self._record('test_disp_mae', disp_mae)
        self._record('test_force_mae', force_mae)
        self._record('disp_per_component_r2', disp_per_r2)
        self._record('force_per_component_r2', force_per_r2)
        self._record('disp_per_component_mae', disp_per_mae)
        self._record('force_per_component_mae', force_per_mae)

        # vs reported
        r = REPORTED_METRICS.get(self.model, {})
        if disp_r2 is not None and 'Disp R2' in r:
            disp_ok = abs(disp_r2 - r['Disp R2']) / max(abs(r['Disp R2']), 1e-10) < 0.005
            self._record('vs_reported_disp_r2', f'{disp_r2:.4f} vs {r["Disp R2"]:.4f}', disp_ok)
        if force_r2 is not None and 'Force R2' in r:
            force_ok = abs(force_r2 - r['Force R2']) / max(abs(r['Force R2']), 1e-10) < 0.005
            self._record('vs_reported_force_r2', f'{force_r2:.4f} vs {r["Force R2"]:.4f}', force_ok)
        if relmae is not None and 'RelMAE' in r:
            relmae_ok = abs(relmae - r['RelMAE']) / max(abs(r['RelMAE']), 1e-10) < 0.01
            self._record('vs_reported_relmae', f'{relmae:.6f} vs {r["RelMAE"]:.6f}', relmae_ok)

    def _check_model_summary(self, reader):
        ms = reader.read_json('model_summary.json')
        if ms is None:
            self._record('model_summary_exists', False, False)
            return
        self.model_summary_data = ms
        self._record('model_name', ms.get('model_name', ''))
        self._record('model_class', ms.get('model_class', ''))

    def _check_config(self, reader):
        cfg = reader.read_text('config_resolved.yaml')
        if cfg is None:
            self._record('config_exists', False, False)
            return
        self.config_text = cfg
        self._record('config_exists', True)

        # Extract key fields
        for key in ['split_mode', 'epochs', 'batch_size', 'seed']:
            m = re.search(rf'^{key}:\s*(.*)', cfg, re.MULTILINE)
            if m:
                self._record(f'cfg_{key}', m.group(1).strip())

    def _check_server_log(self, reader):
        log = reader.read_text('server_output.log')
        if log is None:
            self._record('server_log_exists', False, False)
            return
        self.server_log_text = log
        log_size_kb = len(log) // 1024
        has_summary = 'Training Complete' in log
        has_test_metrics = 'Test Disp R2' in log
        has_traceback = 'Traceback' in log
        has_oom = 'out of memory' in log.lower() or 'OOM' in log
        completed = has_summary and has_test_metrics and not has_traceback and not has_oom
        self._record('server_log_size_kb', log_size_kb)
        self._record('server_has_summary', has_summary, has_summary)
        self._record('server_has_test_metrics', has_test_metrics, has_test_metrics)
        self._record('server_has_traceback', has_traceback, not has_traceback)
        self._record('server_has_oom', has_oom, not has_oom)
        self._record('server_completed', completed, completed)

    def _check_consistency(self):
        """Cross-check best_epoch between metrics_summary and train_log."""
        mt_be = self.results.get('metrics_best_epoch', {}).get('value')
        lg_be = self.results.get('best_epoch_from_log', {}).get('value')
        if mt_be is not None and lg_be is not None:
            match = (mt_be == lg_be)
            self._record('best_epoch_consistency',
                         f'metrics={mt_be}, log={lg_be}', match)


# ---------------------------------------------------------------------------
# 2. Aggregate Metrics Comparison
# ---------------------------------------------------------------------------


def build_aggregate_table(all_checks):
    """Build aggregate metrics table from all model checks."""
    rows = []
    for m in MODEL_ORDER:
        c = all_checks.get(m)
        if c is None:
            continue
        r = c.results
        rows.append({
            'Model': m,
            'Params': r.get('num_params', {}).get('value', 'N/A'),
            'Best Epoch': r.get('metrics_best_epoch', {}).get('value', 'N/A'),
            'Train Time (s)': r.get('total_time_seconds', {}).get('value', 'N/A'),
            'Train Time': fmt_time(r.get('total_time_seconds', {}).get('value') or 0),
            'Disp R2': r.get('test_disp_r2', {}).get('value'),
            'Force R2': r.get('test_force_r2', {}).get('value'),
            'Disp MAE': r.get('test_disp_mae', {}).get('value'),
            'Force MAE': r.get('test_force_mae', {}).get('value'),
            'RelMAE': r.get('test_combined_rel_mae', {}).get('value'),
        })
    return rows


def write_aggregate_csv(rows, path):
    fieldnames = ['Model', 'Params', 'Best Epoch', 'Train Time (s)', 'Train Time',
                  'Disp R2', 'Force R2', 'Disp MAE', 'Force MAE', 'RelMAE']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f'    Written: {path}')


def write_aggregate_md(rows, path):
    """Write markdown table."""
    lines = [
        '| Model | Params | Best Epoch | Train Time | Disp R2 | Force R2 | Disp MAE | Force MAE | RelMAE |',
        '|-------|-------:|-----------:|-----------:|--------:|---------:|---------:|----------:|-------:|',
    ]
    for row in rows:
        dr2 = fmt_r2(row['Disp R2']) if row['Disp R2'] is not None else 'N/A'
        fr2 = fmt_r2(row['Force R2']) if row['Force R2'] is not None else 'N/A'
        dmae = fmt_mae(row['Disp MAE']) if row['Disp MAE'] is not None else 'N/A'
        fmae = fmt_mae(row['Force MAE']) if row['Force MAE'] is not None else 'N/A'
        rmae = fmt_rel(row['RelMAE']) if row['RelMAE'] is not None else 'N/A'
        params = f'{row["Params"]:,}' if isinstance(row['Params'], (int, float)) else str(row['Params'])
        lines.append(f'| {row["Model"]} | {params} | {row["Best Epoch"]} | {row["Train Time"]} | {dr2} | {fr2} | {dmae} | {fmae} | {rmae} |')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'    Written: {path}')


# ---------------------------------------------------------------------------
# 3. Per-Component Metrics
# ---------------------------------------------------------------------------


def build_per_component_table(all_checks):
    """Build per-component R2 table."""
    disp_rows = []
    force_rows = []

    # Displacement per component
    for m in MODEL_ORDER:
        c = all_checks.get(m)
        if c is None:
            continue
        r = c.results
        per_r2 = r.get('disp_per_component_r2', {}).get('value', [None]*6)
        per_mae = r.get('disp_per_component_mae', {}).get('value', [None]*6)
        row = {'Model': m}
        for i, comp in enumerate(DISP_COMPONENTS):
            row[f'{comp}_R2'] = per_r2[i] if i < len(per_r2) else None
            row[f'{comp}_MAE'] = per_mae[i] if i < len(per_mae) else None
        row['Macro_Disp_R2'] = r.get('test_disp_r2', {}).get('value')
        disp_rows.append(row)

    # Force per component (12-dim I/J)
    for m in MODEL_ORDER:
        c = all_checks.get(m)
        if c is None:
            continue
        r = c.results
        per_r2 = r.get('force_per_component_r2', {}).get('value', [None]*12)
        per_mae = r.get('force_per_component_mae', {}).get('value', [None]*12)
        row = {'Model': m}
        for i, comp in enumerate(FORCE_COMPONENTS):
            row[f'{comp}_R2'] = per_r2[i] if i < len(per_r2) else None
            row[f'{comp}_MAE'] = per_mae[i] if i < len(per_mae) else None
        row['Macro_Force_R2'] = r.get('test_force_r2', {}).get('value')
        force_rows.append(row)

    return disp_rows, force_rows


def write_per_component_csv(disp_rows, force_rows, base_path):
    # Disp components
    disp_fields = ['Model'] + [f'{c}_R2' for c in DISP_COMPONENTS] + ['Macro_Disp_R2']
    disp_mae_fields = ['Model'] + [f'{c}_MAE' for c in DISP_COMPONENTS]
    with open(base_path.replace('.csv', '_disp_r2.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=disp_fields)
        w.writeheader()
        for row in disp_rows:
            w.writerow({k: row[k] for k in disp_fields})
    with open(base_path.replace('.csv', '_disp_mae.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=disp_mae_fields)
        w.writeheader()
        for row in disp_rows:
            w.writerow({k: row[k] for k in disp_mae_fields})

    # Force components
    force_fields = ['Model'] + [f'{c}_R2' for c in FORCE_COMPONENTS] + ['Macro_Force_R2']
    force_mae_fields = ['Model'] + [f'{c}_MAE' for c in FORCE_COMPONENTS]
    with open(base_path.replace('.csv', '_force_r2.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=force_fields)
        w.writeheader()
        for row in force_rows:
            w.writerow({k: row[k] for k in force_fields})
    with open(base_path.replace('.csv', '_force_mae.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=force_mae_fields)
        w.writeheader()
        for row in force_rows:
            w.writerow({k: row[k] for k in force_mae_fields})
    print(f'    Written: {base_path.replace(".csv", "_disp_r2.csv")} etc.')


def write_per_component_md(disp_rows, force_rows, base_path):
    """Write per-component markdown tables."""
    # Disp R2 table
    lines = ['## Per-Component Displacement R2\n']
    header = '| Component | ' + ' | '.join(MODEL_ORDER) + ' | HGT Δ vs RGCN |'
    sep = '|-----------|' + '|'.join(':---:' for _ in MODEL_ORDER) + '|:-------------:|'
    lines.append(header)
    lines.append(sep)
    for i, comp in enumerate(DISP_COMPONENTS):
        vals = []
        for m in MODEL_ORDER:
            row = next((r for r in disp_rows if r['Model'] == m), None)
            v = row.get(f'{comp}_R2', None) if row else None
            vals.append(fmt_r2(v) if v is not None else 'N/A')
        # HGT Δ vs RGCN
        hgt_row = next((r for r in disp_rows if r['Model'] == 'HGT'), None)
        rgcn_row = next((r for r in disp_rows if r['Model'] == 'RGCN'), None)
        hgt_v = hgt_row.get(f'{comp}_R2', None) if hgt_row else None
        rgcn_v = rgcn_row.get(f'{comp}_R2', None) if rgcn_row else None
        delta = f'{hgt_v - rgcn_v:+.4f}' if (hgt_v is not None and rgcn_v is not None) else 'N/A'
        lines.append(f'| **{comp}** | ' + ' | '.join(vals) + f' | {delta} |')
    # Macro avg
    mac_vals = []
    for m in MODEL_ORDER:
        row = next((r for r in disp_rows if r['Model'] == m), None)
        v = row.get('Macro_Disp_R2', None) if row else None
        mac_vals.append(fmt_r2(v) if v is not None else 'N/A')
    hgt_mac = next((r for r in disp_rows if r['Model'] == 'HGT'), {}).get('Macro_Disp_R2')
    rgcn_mac = next((r for r in disp_rows if r['Model'] == 'RGCN'), {}).get('Macro_Disp_R2')
    delta_mac = f'{hgt_mac - rgcn_mac:+.4f}' if (hgt_mac is not None and rgcn_mac is not None) else 'N/A'
    lines.append(f'| **Macro avg** | ' + ' | '.join(mac_vals) + f' | {delta_mac} |')

    # Force R2 table
    lines.append('\n## Per-Component Force R2\n')
    # Use short names for display
    force_headers = FORCE_COMPONENTS
    header = '| Component | ' + ' | '.join(MODEL_ORDER) + ' | HGT Δ vs RGCN |'
    sep = '|-----------|' + '|'.join(':---:' for _ in MODEL_ORDER) + '|:-------------:|'
    lines.append(header)
    lines.append(sep)
    for i, comp in enumerate(force_headers):
        vals = []
        for m in MODEL_ORDER:
            row = next((r for r in force_rows if r['Model'] == m), None)
            v = row.get(f'{comp}_R2', None) if row else None
            vals.append(fmt_r2(v) if v is not None else 'N/A')
        hgt_v = next((r for r in force_rows if r['Model'] == 'HGT'), {}).get(f'{comp}_R2')
        rgcn_v = next((r for r in force_rows if r['Model'] == 'RGCN'), {}).get(f'{comp}_R2')
        delta = f'{hgt_v - rgcn_v:+.4f}' if (hgt_v is not None and rgcn_v is not None) else 'N/A'
        lines.append(f'| {comp} | ' + ' | '.join(vals) + f' | {delta} |')
    mac_vals = []
    for m in MODEL_ORDER:
        row = next((r for r in force_rows if r['Model'] == m), None)
        v = row.get('Macro_Force_R2', None) if row else None
        mac_vals.append(fmt_r2(v) if v is not None else 'N/A')
    hgt_mac = next((r for r in force_rows if r['Model'] == 'HGT'), {}).get('Macro_Force_R2')
    rgcn_mac = next((r for r in force_rows if r['Model'] == 'RGCN'), {}).get('Macro_Force_R2')
    delta_mac = f'{hgt_mac - rgcn_mac:+.4f}' if (hgt_mac is not None and rgcn_mac is not None) else 'N/A'
    lines.append(f'| **Macro avg** | ' + ' | '.join(mac_vals) + f' | {delta_mac} |')

    with open(base_path.replace('.csv', '.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'    Written: {base_path.replace(".csv", ".md")}')


# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------


def _model_short(m):
    return MODEL_LABELS.get(m, m)


def fig_model_ranking(aggregate_rows, output_dir):
    """Bar chart: Disp R2, Force R2, RelMAE for all 5 models."""
    models = [r['Model'] for r in aggregate_rows]
    disp_r2 = [r['Disp R2'] if r['Disp R2'] is not None else 0 for r in aggregate_rows]
    force_r2 = [r['Force R2'] if r['Force R2'] is not None else 0 for r in aggregate_rows]
    relmae = [r['RelMAE'] if r['RelMAE'] is not None else 0 for r in aggregate_rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = [MODEL_COLORS[m] for m in models]

    axes[0].bar(models, disp_r2, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_title('Test Disp R2', fontsize=13, fontweight='bold')
    axes[0].set_ylim(0.7, 1.0)
    axes[0].axhline(y=0.85, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    for i, v in enumerate(disp_r2):
        axes[0].text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=9)

    axes[1].bar(models, force_r2, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_title('Test Force R2', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0.92, 1.0)
    for i, v in enumerate(force_r2):
        axes[1].text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=9)

    axes[2].bar(models, relmae, color=colors, edgecolor='black', linewidth=0.5)
    axes[2].set_title('Combined RelMAE (lower is better)', fontsize=13, fontweight='bold')
    for i, v in enumerate(relmae):
        axes[2].text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=9)

    fig.suptitle('Stage 2-B: Model Ranking Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'stage2b_model_ranking.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Figure saved: {path}')


def fig_dy_r2_comparison(all_checks, output_dir):
    """Bar chart: Dy R2 across all models."""
    models = []
    dy_vals = []
    for m in MODEL_ORDER:
        c = all_checks.get(m)
        if c is None:
            continue
        r = c.results
        per_r2 = r.get('disp_per_component_r2', {}).get('value', [None]*6)
        dy = per_r2[1] if len(per_r2) > 1 else None
        if dy is not None:
            models.append(m)
            dy_vals.append(dy)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [MODEL_COLORS[m] for m in models]
    bars = ax.bar(models, dy_vals, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
    ax.set_title('Dy R2 Trajectory: Stage 2-A → Stage 2-B', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('R2', fontsize=12)
    ax.axhline(y=0.9, color='green', linestyle='--', linewidth=1, alpha=0.6, label='R2=0.9 threshold')

    for i, v in enumerate(dy_vals):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')

    # Annotations
    ax.annotate('Homogeneous\nmethods fail', xy=(1, 0.18), xytext=(0.5, 0.08),
                fontsize=9, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax.annotate('Typed Conv\nbreaks bottleneck', xy=(3, 0.67), xytext=(2.5, 0.45),
                fontsize=9, color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=1))
    ax.annotate('Typed Attn\nnearly solves', xy=(4, 0.91), xytext=(3.5, 0.78),
                fontsize=9, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1))

    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, 'stage2b_dy_r2_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Figure saved: {path}')


def fig_per_component_heatmap(all_checks, output_dir):
    """Heatmap: models × components, values = R2."""
    # Only include models that have data
    available_models = [m for m in MODEL_ORDER if m in all_checks]
    if len(available_models) < 2:
        print('    [SKIP] Not enough models for heatmap')
        return

    # Disp components
    comps = DISP_COMPONENTS
    data = []
    for m in available_models:
        c = all_checks.get(m)
        if c is None:
            continue
        r = c.results
        per_r2 = r.get('disp_per_component_r2', {}).get('value', [None]*6)
        data.append([v if v is not None else 0 for v in per_r2[:6]])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    arr = np.array(data)
    im = ax.imshow(arr, cmap='RdYlGn', vmin=0, vmax=1.0, aspect='auto')

    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels(comps, fontsize=11)
    ax.set_yticks(range(len(available_models)))
    ax.set_yticklabels(available_models, fontsize=11)
    ax.set_title('Displacement Per-Component R2', fontsize=13, fontweight='bold')

    for i in range(len(available_models)):
        for j in range(len(comps)):
            val = arr[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(output_dir, 'stage2b_per_component_heatmap_disp.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Figure saved: {path}')

    # Force components (use I/J average for compactness)
    force_short_labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    force_data = []
    for m in available_models:
        c = all_checks.get(m)
        if c is None:
            continue
        r = c.results
        per_r2 = r.get('force_per_component_r2', {}).get('value', [None]*12)
        # Average I and J
        avg = []
        for k in range(6):
            vi = per_r2[k] if k < len(per_r2) and per_r2[k] is not None else 0
            vj = per_r2[k+6] if (k+6) < len(per_r2) and per_r2[k+6] is not None else 0
            avg.append((vi + vj) / 2)
        force_data.append(avg)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    arr_f = np.array(force_data)
    im = ax.imshow(arr_f, cmap='RdYlGn', vmin=0.9, vmax=1.0, aspect='auto')

    ax.set_xticks(range(len(force_short_labels)))
    ax.set_xticklabels(force_short_labels, fontsize=11)
    ax.set_yticks(range(len(available_models)))
    ax.set_yticklabels(available_models, fontsize=11)
    ax.set_title('Force Per-Component R2 (I/J averaged)', fontsize=13, fontweight='bold')

    for i in range(len(available_models)):
        for j in range(len(force_short_labels)):
            val = arr_f[i, j]
            ax.text(j, i, f'{val:.4f}', ha='center', va='center', fontsize=9, color='black')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(output_dir, 'stage2b_per_component_heatmap_force.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Figure saved: {path}')


def fig_remaining_error_hgt(all_checks, output_dir):
    """HGT remaining error (1 - R2) per component."""
    c = all_checks.get('HGT')
    if c is None:
        return
    r = c.results
    disp_r2 = r.get('disp_per_component_r2', {}).get('value', [None]*6)
    force_r2 = r.get('force_per_component_r2', {}).get('value', [None]*12)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Disp remaining error
    comps = DISP_COMPONENTS
    remaining = [max(1 - v, 0) if v is not None else 0 for v in disp_r2[:6]]
    colors_d = ['#d62728' if v > 0.05 else '#2ca02c' for v in remaining]
    axes[0].bar(comps, remaining, color=colors_d, edgecolor='black', linewidth=0.5)
    axes[0].set_title('HGT: Displacement Remaining Error (1 - R2)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('1 - R2', fontsize=11)
    for i, v in enumerate(remaining):
        axes[0].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9, fontweight='bold' if v > 0.05 else 'normal')
    axes[0].axhline(y=0.05, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='5% threshold')
    axes[0].legend(fontsize=9)

    # Force remaining error
    force_short = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    remaining_f = []
    for k in range(6):
        vi = force_r2[k] if k < len(force_r2) and force_r2[k] is not None else 0
        vj = force_r2[k+6] if (k+6) < len(force_r2) and force_r2[k+6] is not None else 0
        remaining_f.append(max(1 - (vi + vj) / 2, 0))
    colors_f = ['#d62728' if v > 0.02 else '#2ca02c' for v in remaining_f]
    axes[1].bar(force_short, remaining_f, color=colors_f, edgecolor='black', linewidth=0.5)
    axes[1].set_title('HGT: Force Remaining Error (1 - Avg I/J R2)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('1 - R2', fontsize=11)
    for i, v in enumerate(remaining_f):
        axes[1].text(i, v + 0.0005, f'{v:.4f}', ha='center', fontsize=9)
    axes[1].axhline(y=0.02, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='2% threshold')

    fig.suptitle('HGT Remaining Error: Where is there still room for improvement?',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'stage2b_remaining_error_hgt.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Figure saved: {path}')


def fig_training_curves(all_checks, output_dir):
    """Overlay training loss curves for typed models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    typed_models = ['RGCN', 'HGT']
    styles = {'RGCN': '-', 'HGT': '--'}
    colors_curves = {'RGCN': '#2ca02c', 'HGT': '#d62728'}

    for m in typed_models:
        c = all_checks.get(m)
        if c is None or c.train_log is None:
            continue
        rows = c.train_log
        epochs = [int(r['epoch']) for r in rows]
        val_loss = [float(r['val_loss']) for r in rows]
        val_disp_r2 = [float(r['val_disp_r2']) for r in rows]
        axes[0].plot(epochs, val_loss, styles[m], color=colors_curves[m],
                     label=f'{m} val_loss', linewidth=1.5)
        axes[1].plot(epochs, val_disp_r2, styles[m], color=colors_curves[m],
                     label=f'{m} val_disp_r2', linewidth=1.5)

    axes[0].set_title('Validation Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Validation Disp R2', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R2')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Training Curves: Typed Models (RGCN vs HGT)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'stage2b_training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Figure saved: {path}')


# ---------------------------------------------------------------------------
# 5. Diagnostic Report (Missing Predictions / Region Labels)
# ---------------------------------------------------------------------------


def check_predictions_availability(all_checks, artifact_dir, outputs_dir):
    """Check if full test predictions are available."""
    findings = {
        'has_predictions': False,
        'prediction_files': [],
        'notes': [],
    }

    # Check extracted artifact directories
    artifact_dir_path = os.path.abspath(artifact_dir)
    if os.path.isdir(artifact_dir_path):
        for root, dirs, files in os.walk(artifact_dir_path):
            for f in files:
                if 'predict' in f.lower() and (f.endswith('.csv') or f.endswith('.pt') or f.endswith('.pth')):
                    findings['prediction_files'].append(os.path.join(root, f))

    # Check outputs dirs
    if outputs_dir and os.path.isdir(outputs_dir):
        for root, dirs, files in os.walk(outputs_dir):
            for f in files:
                if 'predict' in f.lower() and (f.endswith('.csv') or f.endswith('.pt') or f.endswith('.pth')):
                    findings['prediction_files'].append(os.path.join(root, f))

    # Check in tar.gz too
    if os.path.isdir(artifact_dir_path):
        for fname in os.listdir(artifact_dir_path):
            if fname.endswith('.tar.gz'):
                try:
                    with tarfile.open(os.path.join(artifact_dir_path, fname), 'r:gz') as tar:
                        for m in tar.getnames():
                            if 'predict' in m.lower() and m.endswith('.csv'):
                                findings['prediction_files'].append(f'{fname}:{m}')
                except Exception:
                    pass

    if findings['prediction_files']:
        findings['has_predictions'] = True

    return findings


def write_missing_predictions_doc(path, prediction_check):
    """Document why tail-error / region metrics can't be computed yet."""
    lines = [
        '# Diagnostic: Missing Full Test Predictions\n',
    ]
    if prediction_check['has_predictions']:
        lines.append('## Status: Predictions Available\n')
        lines.append(f'Found {len(prediction_check["prediction_files"])} prediction-related files:\n')
        for pf in prediction_check['prediction_files'][:20]:
            lines.append(f'- {pf}')
        lines.append('\nTail error and region metrics can be computed.')
    else:
        lines.append('## Status: Predictions NOT Available\n')
        lines.append(
            'Current artifacts do not contain `test_predictions.csv` or any full prediction output. '
            'The standard artifact template does not export full test predictions.\n'
        )
        lines.append('### Impact\n')
        lines.append('The following diagnostics are currently BLOCKED:\n')
        lines.append('1. **Tail error metrics**: P50/P90/P95/P99 absolute/relative error')
        lines.append('2. **High-response subset error**: Top 10% by true value magnitude')
        lines.append('3. **Low-response subset error**: Bottom 10% by true value magnitude')
        lines.append('4. **Region-wise metrics**: Support, midspan, connection regions')
        lines.append('5. **Physical consistency**: Support BC residual, structural_link smoothness\n')
        lines.append('### Required Action\n')
        lines.append(
            'To enable these diagnostics, a new `eval-only` mode should be added to `train_baseline.py` '
            'that loads a trained `best_model.pt` and runs full test set inference, saving:\n'
        )
        lines.append('- `test_predictions.csv` with columns: sample_id, component, true_value, pred_value')
        lines.append('- `test_predictions_disp.csv`')
        lines.append('- `test_predictions_force.csv`\n')
        lines.append('### Recommendation\n')
        lines.append(
            'Add eval-only export as a prerequisite task before entering Stage 3, '
            'so that tail error and region analysis can be completed.\n'
        )
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'    Written: {path}')


def write_region_metric_requirements(path):
    """Document region label construction plan (no dataset changes)."""
    lines = [
        '# Diagnostic: Region Metric Requirements\n',
        '## Purpose',
        '',
        'Region-wise metrics evaluate whether models perform uniformly across physically distinct '
        'parts of the steel truss girder, or whether certain regions (supports, midspan, connections) '
        'concentrate prediction errors.',
        '',
        '## Available Node Coordinates',
        '',
        '`hetero_graph_dataset_v2` stores mesh_node features (15-dim) which include:',
        '- Nodal coordinates (x, y, z) as part of mesh_node.x',
        '',
        '## Proposed Region Construction (Dataset-Agnostic)',
        '',
        'Regions should be computed on-the-fly in the diagnostic script from node coordinates and '
        'the existing edge structure. This does NOT require modifying dataset schema:\n',
        '',
        '### 1. Support Region',
        '- Nodes that are `general_supports` (boundary conditions)',
        '- Identified via: nodes where BC constraints (first 6 dims of mesh_node.x) are active',
        '- Approximately: nodes near the girder ends (min/max x-coordinate)',
        '',
        '### 2. Midspan Region',
        '- Nodes in the central 1/3 of the girder span along the x-axis',
        '- Identified via: x-coordinate percentile (33%–67%)',
        '',
        '### 3. Beam-Plate Connection Region',
        '- Nodes that belong to both beam and plate elements',
        '- Identified via: mesh_node nodes that have both `belongs_to_beam` and `belongs_to_plate` edges',
        '',
        '### 4. High-Response Region',
        '- Nodes where true displacement magnitude is in the top 10% of the test set',
        '- Requires full test predictions to be exported first',
        '',
        '## Fields Required',
        '',
        '| Field | Source | Status |',
        '|-------|--------|--------|',
        '| Node coordinates | mesh_node.x[:, 0:3] | ✅ Available',
        '| Support flag | mesh_node.x[:, 3:9] BC constraints | ✅ Available',
        '| Edge incidence | edge_index_dict | ✅ Available',
        '| True displacement | From test set | ✅ Available (in DataLoaders)',
        '| Predicted displacement | From model inference | ❌ Not exported',
        '| Region assignment | Computed from above fields | 🟡 Not yet implemented',
        '',
        '## Implementation Path',
        '',
        '1. Add eval-only prediction export (prerequisite)',
        '2. Implement `RegionAssigner` class that assigns region labels per sample per node',
        '3. Compute region-wise metrics using true vs predicted values',
        '4. Generate region-wise tables and figures',
        '',
        '## Notes',
        '',
        '- Region labels are sample-specific because beam positions vary slightly across designs',
        '- Region metrics should be computed per-sample then aggregated, not pooled across samples',
        '- The truss girder topology is consistent across all 70 samples, so region definitions '
        'can be based on the canonical topology',
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'    Written: {path}')


# ---------------------------------------------------------------------------
# 6. Physical Diagnostic Plan
# ---------------------------------------------------------------------------


def write_physical_diagnostic_plan(path):
    """Document lightweight physical consistency diagnostics plan."""
    lines = [
        '# Diagnostic: Physical Consistency Plan\n',
        '## Purpose',
        '',
        'Evaluate whether model predictions respect basic physical constraints of the steel truss girder. '
        'These are NOT rigorous FEM consistency checks — they are lightweight, interpretable diagnostics '
        'that can reveal physically unrealistic predictions.\n',
        '',
        '## Proposed Diagnostics (Need Full Predictions)',
        '',
        '### 1. Support Boundary Condition Residual',
        '- At support nodes (where BC constraints are active), displacement should be near-zero in '
        'constrained DOFs.',
        '- **Metric**: Mean absolute displacement at support-constrained DOFs.',
        '- **Expected**: Should be very small; large values indicate physically unrealistic predictions.',
        '',
        '### 2. Structural Link Smoothness',
        '- `structural_link` edges represent rigid connections between mesh nodes.',
        '- Two mesh nodes connected by a rigid link should have identical or nearly identical displacements.',
        '- **Metric**: Mean absolute displacement difference across structural_link endpoints.',
        '- **Expected**: Near-zero for RIGID links.',
        '',
        '### 3. Beam Endpoint Displacement Consistency',
        '- A beam element with endpoints I and J should have displacements consistent with the mesh_node '
        'features at those endpoints.',
        '- **Metric**: Correlation between beam endpoint displacement predictions and the associated '
        'mesh_node predictions at the same physical locations.',
        '',
        '### 4. Approximate Beam Force Balance (Qualitative)',
        '- For a beam element in static equilibrium, I/J end forces should satisfy basic relationships '
        '(sum of forces ≈ 0 for unloaded beams).',
        '- **Metric**: Residual of I/J force sum for beams with no distributed load.',
        '- **Note**: This is approximate because the model predicts total forces, not the equilibrium residual.',
        '',
        '## Prerequisites',
        '',
        '| Prerequisite | Status | Notes |',
        '|--------------|--------|-------|',
        '| Full test predictions | ❌ Missing | Need eval-only export first |',
        '| Support BC node mask | ✅ Available | From mesh_node.x BC dims |',
        '| structural_link edge list | ✅ Available | From edge_index_dict |',
        '| Beam endpoint node IDs | ✅ Available | From belongs_to_beam edges |',
        '',
        '## Recommendation',
        '',
        'Implement eval-only prediction export first, then add physical diagnostics as a follow-up.',
        'The physical diagnostics script should be a separate module (`scripts/physical_diagnostics.py`) '
        'that reads predictions + dataset and computes the above metrics.',
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'    Written: {path}')


# ---------------------------------------------------------------------------
# 7. Full Prediction Analysis (when predictions are available)
# ---------------------------------------------------------------------------


def load_predictions(pred_dir, model_name='*'):
    """Load NPZ predictions for a model from the prediction export dir.

    Args:
        pred_dir: Root prediction directory (e.g. outputs/predictions/stage2b)
        model_name: Model name glob (e.g. 'hgt', '*')

    Returns:
        dict of model_name -> {'mesh': npz, 'beam': npz, 'summary': dict}
        or None if not found.
    """
    import glob as _glob
    pred_dir = os.path.abspath(pred_dir)
    if not os.path.isdir(pred_dir):
        return None

    results = {}
    # Search for prediction_summary.json files
    pattern = os.path.join(pred_dir, model_name, '*', 'prediction_summary.json')
    for sum_path in sorted(_glob.glob(pattern)):
        try:
            with open(sum_path, 'r') as f:
                summary = json.load(f)
            mname = summary.get('model_name', 'unknown')
            base_dir = os.path.dirname(sum_path)
            mesh_path = os.path.join(base_dir, 'mesh_node_predictions.npz')
            beam_path = os.path.join(base_dir, 'beam_element_predictions.npz')

            if os.path.exists(mesh_path) and os.path.exists(beam_path):
                results[mname] = {
                    'mesh': np.load(mesh_path),
                    'beam': np.load(beam_path),
                    'summary': summary,
                    'dir': base_dir,
                }
                print(f'    Loaded predictions for {mname}: {base_dir}')
        except Exception as e:
            print(f'    [WARN] Failed to load predictions from {sum_path}: {e}')

    return results if results else None


def compute_tail_error_metrics(pred_data, output_dir):
    """Compute P50/P90/P95/P99 absolute and relative error per component.

    Args:
        pred_data: dict from load_predictions() for a single model
        output_dir: output directory for CSV results
    """
    mesh = pred_data['mesh']
    beam = pred_data['beam']

    disp_true = mesh['y_true_disp']
    disp_pred = mesh['y_pred_disp']
    force_true = beam['y_true_force']
    force_pred = beam['y_pred_force']

    def _compute_tail(true, pred, comp_names, label):
        abs_err = np.abs(pred - true)
        rel_err = np.abs(pred - true) / (np.abs(true) + 1e-10)

        percentiles = [50, 90, 95, 99]
        rows = []
        for i, comp in enumerate(comp_names):
            row = {'Component': comp}
            for p in percentiles:
                row[f'AbsErr_P{p}'] = np.percentile(abs_err[:, i], p)
                row[f'RelErr_P{p}'] = np.percentile(rel_err[:, i], p)
            rows.append(row)
        return rows

    disp_tail = _compute_tail(disp_true, disp_pred,
                              ['Dx', 'Dy', 'Dz', 'Rx', 'Ry', 'Rz'], 'Disp')
    force_tail = _compute_tail(force_true, force_pred,
                               ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz',
                                'Fx_J', 'Fy_J', 'Fz_J', 'Mx_J', 'My_J', 'Mz_J'],
                               'Force')

    # Save
    import csv
    for tail_data, suffix in [(disp_tail, 'disp'), (force_tail, 'force')]:
        path = os.path.join(output_dir, f'tail_error_{suffix}.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            if tail_data:
                w = csv.DictWriter(f, fieldnames=tail_data[0].keys())
                w.writeheader()
                w.writerows(tail_data)
        print(f'    Written: {path}')

    return disp_tail, force_tail


def compute_high_low_response(pred_data, output_dir):
    """Compute error on high/low response subsets (top/bottom 10% by true magnitude)."""
    mesh = pred_data['mesh']
    beam = pred_data['beam']

    results = []
    for name, true, pred in [
        ('Disp', mesh['y_true_disp'], mesh['y_pred_disp']),
        ('Force', beam['y_true_force'], beam['y_pred_force']),
    ]:
        magnitude = np.abs(true).mean(axis=1)
        top10 = np.percentile(magnitude, 90)
        bot10 = np.percentile(magnitude, 10)

        high_mask = magnitude >= top10
        low_mask = magnitude <= bot10

        for mask_label, mask in [('HighResponse_top10', high_mask),
                                  ('LowResponse_bot10', low_mask)]:
            if mask.sum() == 0:
                continue
            sub_true = true[mask]
            sub_pred = pred[mask]

            mae = np.abs(sub_pred - sub_true).mean()
            rmse = np.sqrt(((sub_pred - sub_true) ** 2).mean())
            ss_res = ((sub_pred - sub_true) ** 2).sum()
            ss_tot = ((sub_true - sub_true.mean(axis=0, keepdims=True)) ** 2).sum()
            r2 = float(1 - ss_res / (ss_tot + 1e-10))

            results.append({
                'Task': name,
                'Subset': mask_label,
                'Count': mask.sum(),
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
            })

    # Save
    import csv
    path = os.path.join(output_dir, 'high_low_response.csv')
    if results:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f'    Written: {path}')
    return results


def compute_physical_diagnostics_from_predictions(pred_data, dataset, output_dir):
    """Compute lightweight physical diagnostics from predictions.

    NOTE: dataset argument is not yet wired — structural_link analysis needs
    edge_index_dict which is only available when iterating the DataLoader.
    This function provides a placeholder for future implementation.
    """
    mesh = pred_data['mesh']
    support_flags = mesh.get('support_flags', None)
    disp_true = mesh['y_true_disp']
    disp_pred = mesh['y_pred_disp']

    diagnostics = {}

    # Support BC residual: for constrained DOFs, displacement should be near-zero
    if support_flags is not None:
        support_mask = support_flags > 0.5
        if support_mask.any():
            bc_residual = np.abs(disp_pred[support_mask] - disp_true[support_mask])
            diagnostics['support_bc_residual'] = {
                'mean_abs': float(bc_residual.mean()),
                'median_abs': float(np.median(bc_residual)),
                'p95': float(np.percentile(bc_residual, 95)),
                'n_constrained_dof': int(support_mask.sum()),
            }

    # Structural link diagnostics require edge_index — not available from NPZ alone
    # This requires iterating the dataset to find structural_link edges
    diagnostics['structural_link'] = {
        'note': 'Requires edge_index from DataLoader. Not computed from NPZ alone.',
        'requirement': 'Run with --dataset-dir pointing to processed/hetero_graph_dataset_v2',
    }

    # Save
    path = os.path.join(output_dir, 'physical_diagnostics.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(diagnostics, f, indent=2)
    print(f'    Written: {path}')

    return diagnostics


def analyze_predictions(predictions_dir, output_dir):
    """Run all prediction-based analyses.

    Args:
        predictions_dir: Path to root prediction directory.
        output_dir: Output directory for analysis results.
    """
    print(f'\n{"=" * 60}')
    print('Full Prediction Analysis')
    print(f'{"=" * 60}')

    # Load all available predictions
    pred_data = load_predictions(predictions_dir, model_name='*')

    if pred_data is None or len(pred_data) == 0:
        print('\n  [WARN] No prediction files found.')
        print(f'  Searched in: {predictions_dir}')
        print('  Run scripts/export_full_predictions.py first.')
        return False

    print(f'\n  Found predictions for: {list(pred_data.keys())}')

    for model_name, data in pred_data.items():
        model_out_dir = ensure_dir(os.path.join(output_dir, model_name))
        print(f'\n  --- {model_name} ---')

        # Tail error
        try:
            compute_tail_error_metrics(data, model_out_dir)
        except Exception as e:
            print(f'    [WARN] Tail error failed: {e}')

        # High/low response
        try:
            compute_high_low_response(data, model_out_dir)
        except Exception as e:
            print(f'    [WARN] High/low response failed: {e}')

        # Physical diagnostics (limited without dataset)
        try:
            compute_physical_diagnostics_from_predictions(data, None, model_out_dir)
        except Exception as e:
            print(f'    [WARN] Physical diagnostics failed: {e}')

    return True


def discover_artifacts(artifact_dir):
    """Discover model artifacts from directory."""
    artifacts = {}
    artifact_dir = os.path.abspath(artifact_dir)

    # Map model name to keyword - use more specific patterns to avoid
    # 'rgcn' matching as 'gcn' or 'hgt' matching 'gat'
    import re
    # Use negative lookahead/lookbehind for letters only, so _ before/after still matches
    model_keys = [
        ('MLP', re.compile(r'(?<![a-z])mlp(?![a-z])')),
        ('GCN', re.compile(r'(?<![a-z])gcn(?![a-z])')),   # 'gcn' but not 'rgcn'
        ('GAT', re.compile(r'(?<![a-z])gat(?![a-z])')),   # 'gat' but not 'hgat'
        ('RGCN', re.compile(r'(?<![a-z])rgcn(?![a-z])')),
        ('HGT', re.compile(r'(?<![a-z])hgt(?![a-z])')),
    ]

    if not os.path.isdir(artifact_dir):
        print(f'[WARN] Artifact directory not found: {artifact_dir}')
        return artifacts

    entries = sorted(os.listdir(artifact_dir))
    print(f'  Scanning {len(entries)} entries in {artifact_dir}')
    for fname in entries:
        fpath = os.path.join(artifact_dir, fname)

        # Check tar.gz (prefer extracted dir over tar.gz)
        if fname.endswith('.tar.gz'):
            for model, pattern in model_keys:
                if pattern.search(fname.lower()):
                    if model not in artifacts:
                        artifacts[model] = fpath
                        print(f'    Found tar.gz: {fname} -> {model}')
                    break

        # Check extracted directories (prefer over tar.gz)
        if os.path.isdir(fpath):
            if os.path.isfile(os.path.join(fpath, 'metrics_summary.json')):
                for model, pattern in model_keys:
                    if pattern.search(fname.lower()):
                        artifacts[model] = fpath
                        print(f'    Found extracted dir: {fname} -> {model}')
                        break
            else:
                # Check subdirectories one level deep
                try:
                    for subname in os.listdir(fpath):
                        subpath = os.path.join(fpath, subname)
                        if os.path.isdir(subpath) and os.path.isfile(os.path.join(subpath, 'metrics_summary.json')):
                            for model, pattern in model_keys:
                                if pattern.search(subname.lower()) or pattern.search(fname.lower()):
                                    artifacts[model] = subpath
                                    print(f'    Found nested dir: {fname}/{subname} -> {model}')
                                    break
                except (PermissionError, OSError):
                    pass

    return artifacts


def main():
    # Set stdout to utf-8 for Windows compatibility
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(
        description='Stage 2-B Diagnostic Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/analyze_stage2b_diagnostics.py --artifact-dir remote_artifacts --output-dir outputs/diagnostics/stage2b
              python scripts/analyze_stage2b_diagnostics.py --help
        """),
    )
    parser.add_argument('--artifact-dir', type=str, default='remote_artifacts',
                        help='Directory containing model artifacts (tar.gz or extracted)')
    parser.add_argument('--output-dir', type=str, default='outputs/diagnostics/stage2b',
                        help='Output directory for diagnostic results')
    parser.add_argument('--outputs-dir', type=str, default='outputs/baselines',
                        help='Directory containing smoke test outputs (fallback)')
    parser.add_argument('--predictions-dir', type=str, default=None,
                        help='Directory with exported full predictions '
                             '(outputs/predictions/stage2b) for tail-error analysis')
    args = parser.parse_args()

    print('=' * 80)
    print('STAGE 2-B DIAGNOSTIC EVALUATION')
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 80)

    # ---- Resolve paths ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Multi-Scale-PI-HGNN/
    artifact_dir = os.path.join(project_root, args.artifact_dir) \
        if not os.path.isabs(args.artifact_dir) else args.artifact_dir
    output_dir = os.path.join(project_root, args.output_dir) \
        if not os.path.isabs(args.output_dir) else args.output_dir
    outputs_baselines_dir = os.path.join(project_root, args.outputs_dir) \
        if not os.path.isabs(args.outputs_dir) else args.outputs_dir

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_dir_run = os.path.join(output_dir, timestamp)
    figures_dir = ensure_dir(os.path.join(output_dir_run, 'figures'))
    print(f'\nOutput dir: {output_dir_run}')

    # ---- Discover artifacts ----
    print(f'\n{"=" * 60}')
    print('Step 0: Artifact Discovery')
    print(f'{"=" * 60}')
    artifacts = discover_artifacts(artifact_dir)
    if not artifacts:
        print(f'[ERROR] No artifacts found in {artifact_dir}')
        print(f'  Expected files: server_mlp_full_*.tar.gz, server_gcn_full_*.tar.gz, etc.')
        sys.exit(1)

    print(f'  Found artifacts:')
    for model, path in sorted(artifacts.items()):
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        if os.path.isdir(path):
            total = sum(os.path.getsize(os.path.join(dirpath, f))
                        for dirpath, _, filenames in os.walk(path) for f in filenames)
            size = total
        print(f'    {model}: {path} ({size // 1024} KB)')

    # ---- 1. Artifact Consistency Check ----
    print(f'\n{"=" * 60}')
    print('Step 1: Artifact Consistency Check')
    print(f'{"=" * 60}')

    all_checks = OrderedDict()
    for model in MODEL_ORDER:
        if model not in artifacts:
            print(f'\n  [SKIP] {model}: no artifact found')
            continue
        print(f'\n  --- {model} ---')
        checker = ArtifactCheck(model, artifacts[model])
        checker.run()
        all_checks[model] = checker
        print(f'    Artifact check: {"PASS" if checker.passed else "FAIL"}')

    # Save artifact check CSV
    check_csv_path = os.path.join(output_dir_run, 'diagnostic_artifact_check.csv')
    check_fields = ['Model', 'Check', 'Value', 'Passed']
    with open(check_csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(check_fields)
        for model, checker in all_checks.items():
            for key, val in checker.results.items():
                value = val.get('value')
                passed = val.get('passed')
                if isinstance(value, list):
                    value = str(value[:6]) + '...' if len(value) > 6 else str(value)
                w.writerow([model, key, str(value)[:80] if value is not None else 'N/A',
                            'PASS' if passed else 'FAIL'])
    print(f'\n  Saved: {check_csv_path}')

    # ---- 2. Aggregate Metrics ----
    print(f'\n{"=" * 60}')
    print('Step 2: Aggregate Metrics Comparison')
    print(f'{"=" * 60}')
    agg_rows = build_aggregate_table(all_checks)
    if agg_rows:
        write_aggregate_csv(agg_rows, os.path.join(output_dir_run, 'stage2b_main_metrics.csv'))
        write_aggregate_md(agg_rows, os.path.join(output_dir_run, 'stage2b_main_metrics.md'))
        print('\n  Aggregate metrics:')
        print(f'  {"Model":<8s} {"Params":>10s} {"Best Ep":>8s} {"Time":>12s} {"Disp R2":>10s} {"Force R2":>10s} {"RelMAE":>10s}')
        print('  ' + '-' * 68)
        for row in agg_rows:
            dr2 = fmt_r2(row['Disp R2']) if row['Disp R2'] is not None else 'N/A'
            fr2 = fmt_r2(row['Force R2']) if row['Force R2'] is not None else 'N/A'
            rmae = fmt_rel(row['RelMAE']) if row['RelMAE'] is not None else 'N/A'
            params = f'{row["Params"]:,}' if isinstance(row['Params'], (int, float)) else str(row['Params'])
            print(f'  {row["Model"]:<8s} {params:>10s} {str(row["Best Epoch"]):>8s} {row["Train Time"]:>12s} {dr2:>10s} {fr2:>10s} {rmae:>10s}')

    # ---- 3. Per-Component Metrics ----
    print(f'\n{"=" * 60}')
    print('Step 3: Per-Component Metrics')
    print(f'{"=" * 60}')
    disp_rows, force_rows = build_per_component_table(all_checks)
    if disp_rows:
        write_per_component_csv(disp_rows, force_rows,
                                os.path.join(output_dir_run, 'stage2b_per_component_metrics.csv'))
        write_per_component_md(disp_rows, force_rows,
                               os.path.join(output_dir_run, 'stage2b_per_component_metrics.csv'))
        print('\n  Displacement per-component R2:')
        print(f'  {"Model":<8s}', end='')
        for c in DISP_COMPONENTS:
            print(f'{c:>8s}', end='')
        print(f'{"Macro":>8s}')
        print('  ' + '-' * 64)
        for row in disp_rows:
            print(f'  {row["Model"]:<8s}', end='')
            for c in DISP_COMPONENTS:
                v = row.get(f'{c}_R2')
                print(f'{fmt_r2(v) if v is not None else "N/A":>8s}', end='')
            v = row.get('Macro_Disp_R2')
            print(f'{fmt_r2(v) if v is not None else "N/A":>8s}')

        # HGT still weakest dispersion component
        hgt_row = next((r for r in disp_rows if r['Model'] == 'HGT'), None)
        if hgt_row:
            worst_comp = min(DISP_COMPONENTS, key=lambda c: hgt_row.get(f'{c}_R2', 1) or 1)
            worst_val = hgt_row.get(f'{worst_comp}_R2', 0)
            print(f'\n  HGT weakest disp component: {worst_comp} (R2={worst_val:.4f})')
            print(f'  HGT strongest disp component: {max(DISP_COMPONENTS, key=lambda c: hgt_row.get(f"{c}_R2", 0) or 0)}')

    if force_rows:
        print('\n  Force per-component R2 (I/J averaged):')
        print(f'  {"Model":<8s}', end='')
        for c in FORCE_COMPONENTS_SHORT:
            print(f'{c:>8s}', end='')
        print(f'{"Macro":>8s}')
        print('  ' + '-' * 64)
        for row in force_rows:
            print(f'  {row["Model"]:<8s}', end='')
            for i, c in enumerate(FORCE_COMPONENTS_SHORT):
                vi = row.get(f'{FORCE_COMPONENTS[i]}_R2')
                vj = row.get(f'{FORCE_COMPONENTS[i+6]}_R2')
                avg = (vi + vj) / 2 if (vi is not None and vj is not None) else None
                print(f'{fmt_r2(avg) if avg is not None else "N/A":>8s}', end='')
            v = row.get('Macro_Force_R2')
            print(f'{fmt_r2(v) if v is not None else "N/A":>8s}')

    # ---- 4. Tail Error / Predictions Check ----
    print(f'\n{"=" * 60}')
    print('Step 4: Prediction Availability Check')
    print(f'{"=" * 60}')
    prediction_check = check_predictions_availability(all_checks, artifact_dir, outputs_baselines_dir)
    if prediction_check['has_predictions']:
        print(f'  [OK] Found {len(prediction_check["prediction_files"])} prediction files')
    else:
        print('  [WARN] No full test predictions found in any artifact or output directory')
    write_missing_predictions_doc(
        os.path.join(output_dir_run, 'diagnostic_missing_predictions.md'),
        prediction_check,
    )

    # ---- 5. Region Metrics Requirements ----
    print(f'\n{"=" * 60}')
    print('Step 5: Region Metrics Requirements')
    print(f'{"=" * 60}')
    print('  No region labels in current dataset schema.')
    print('  Generating region metric requirements document...')
    write_region_metric_requirements(
        os.path.join(output_dir_run, 'diagnostic_region_metric_requirements.md'),
    )

    # ---- 6. Physical Diagnostic Plan ----
    print(f'\n{"=" * 60}')
    print('Step 6: Physical Consistency Diagnostic Plan')
    print(f'{"=" * 60}')
    print('  Generating physical diagnostic plan document...')
    write_physical_diagnostic_plan(
        os.path.join(output_dir_run, 'diagnostic_physical_diagnostic_plan.md'),
    )

    # ---- 7. Figures ----
    print(f'\n{"=" * 60}')
    print('Step 7: Generating Figures')
    print(f'{"=" * 60}')
    if agg_rows:
        fig_model_ranking(agg_rows, figures_dir)
    if all_checks:
        fig_dy_r2_comparison(all_checks, figures_dir)
        fig_per_component_heatmap(all_checks, figures_dir)
        fig_remaining_error_hgt(all_checks, figures_dir)
        fig_training_curves(all_checks, figures_dir)

    # ---- 8. Prediction-based Analysis (optional) ----
    if args.predictions_dir:
        print(f'\n{"=" * 60}')
        print('Step 8: Full Prediction Analysis (tail error, high/low response)')
        print(f'{"=" * 60}')
        pred_dir = os.path.join(project_root, args.predictions_dir) \
            if not os.path.isabs(args.predictions_dir) else args.predictions_dir
        analyze_predictions(pred_dir, output_dir_run)
    else:
        print(f'\n  [SKIP] Step 8: No --predictions-dir specified.')
        print(f'  To enable tail-error analysis, run scripts/export_full_predictions.py first,')
        print(f'  then re-run this script with --predictions-dir outputs/predictions/stage2b')

    # ---- Summary ----
    print(f'\n{"=" * 80}')
    print('DIAGNOSTIC COMPLETE')
    print(f'{"=" * 80}')
    print(f'\nOutput directory: {output_dir_run}')
    print(f'\nFiles generated:')
    for root, dirs, files in os.walk(output_dir_run):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            rel = os.path.relpath(fpath, output_dir_run)
            print(f'  {rel:60s} {size:>8,} bytes')

    print(f'\n{"=" * 80}')
    print('SUMMARY: Model Read Status')
    print(f'{"=" * 80}')
    print(f'  {"Model":<8s} {"Artifact":>10s} {"Checks":>8s}')
    print(f'  {"-" * 28}')
    for model in MODEL_ORDER:
        if model in all_checks:
            status = 'PASS' if all_checks[model].passed else 'FAIL'
            print(f'  {model:<8s} {"[OK]":>10s} {status:>8s}')
        else:
            print(f'  {model:<8s} {"[MISS]":>10s} {"SKIP":>8s}')
    print()


if __name__ == '__main__':
    main()
