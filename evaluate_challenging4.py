"""
Batch files are now supported

Evaluate DyGFormer/other DyGlib models on challenging case subsets

DyGLib contains memory-based algorithms which can't evaluate on a subset alone

Uses node_id_map from test_trajectory.pkl to map scene_id -> (node_id, ts).
node_id_map format: "{scene_id}_{agent_id}" -> node_id

Usage:
    # Step 1: run main script with --save_predictions
    nohup python -u train_traj_prediction_alg_he_intermodal_all4.py \
        --model_name TGN \
        --data_dir /scratch/yiran/v2x/v2x_cig_alg_selfloop \
        --num_modes 6 \
        --use_intermodal_attn --intermodal_attn_layers 2 \
        --fde_weight 1.0 --cls_weight 0.0 --time_weight none \
        --pred_horizon 50 \
        --batch_size 32 \
        --use_map \
        --gpu 6 \
        --eval_only \
        --resume logs/tgn/inter/trial2/best_model.pt \
        --save_predictions \
        --output_dir logs/tgn/inter/trial2/eval_challenging > logs/tgn/inter/trial2/eval_challenging/eval_challenging.log 2>&1 &
    tail -f logs/tgn/inter/trial2/eval_challenging/eval_challenging.log
    
    # Step 2: run this 
    python evaluate_challenging.py \
        --data_dir /scratch/yiran/v2x/v2x_cig_alg_selfloop \
        --predictions_path logs/.../test_predictions.pt
"""

import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from collections import defaultdict




def build_scene_node_mapping(data_dir):
    """
    Build mapping from scene_id to (node_id, ts) keys.
    
    Uses node_id_map from test_trajectory.pkl.
    node_id_map keys: "{scene_id}_{agent_id}" -> node_id
    current_state keys: (node_id, ts) -> [x, y, vx, vy, heading]
    """
    data_dir = Path(data_dir)
    
    # Support both single file and batched files
    single_path = data_dir / 'test_trajectory.pkl'
    if single_path.exists():
        with open(single_path, 'rb') as f:
            traj_data = pickle.load(f)
    else:
        # Merge batched trajectory files
        batch_files = sorted(data_dir.glob('test_batch*_trajectory.pkl'))
        if not batch_files:
            raise FileNotFoundError(f"No test trajectory files found in {data_dir}")
        print(f"  Found {len(batch_files)} batched trajectory files, merging...")
        merged_node_id_map = {}
        merged_current_state = {}
        merged_future_traj = {}
        merged_history_traj = {}
        for bf in batch_files:
            with open(bf, 'rb') as f:
                batch_data = pickle.load(f)
            if 'node_id_map' in batch_data:
                merged_node_id_map.update(batch_data['node_id_map'])
            if 'current_state' in batch_data:
                merged_current_state.update(batch_data['current_state'])
            if 'future_traj' in batch_data:
                merged_future_traj.update(batch_data['future_traj'])
            if 'history_traj' in batch_data:
                merged_history_traj.update(batch_data['history_traj'])
        traj_data = {
            'node_id_map': merged_node_id_map,
            'current_state': merged_current_state,
            'future_traj': merged_future_traj,
            'history_traj': merged_history_traj,
        }
    
    node_id_map = traj_data['node_id_map']
    current_state = traj_data['current_state']
    
    print(f"  node_id_map entries: {len(node_id_map)}")
    print(f"  current_state entries: {len(current_state)}")
    
    # node_id -> scene_id
    nodeid_to_scene = {}
    # node_id -> agent_id (integer)
    nodeid_to_agent = {}
    scene_to_nodeids = defaultdict(set)
    
    for agent_key, node_id in node_id_map.items():
        parts = agent_key.rsplit('_', 1)
        if len(parts) == 2:
            scene_id = parts[0]
            agent_id = parts[1]
            nodeid_to_scene[node_id] = scene_id
            nodeid_to_agent[node_id] = int(agent_id)
            scene_to_nodeids[scene_id].add(node_id)
    
    # scene_id -> [(node_id, ts)]
    scene_to_keys = defaultdict(list)
    key_to_scene = {}
    
    for (nid, ts) in current_state.keys():
        scene_id = nodeid_to_scene.get(nid)
        if scene_id:
            scene_to_keys[scene_id].append((nid, ts))
            key_to_scene[(nid, ts)] = scene_id
    
    print(f"  Scenes with test data: {len(scene_to_keys)}")
    print(f"  Total (node_id, ts) keys: {len(key_to_scene)}")
    
    return {
        'scene_to_keys': dict(scene_to_keys),
        'key_to_scene': key_to_scene,
        'nodeid_to_scene': nodeid_to_scene,
        'nodeid_to_agent': nodeid_to_agent,
        'traj_data': traj_data,  
    }


def load_predictions(predictions_path):
    """Load saved test predictions from main script."""
    data = torch.load(predictions_path, map_location='cpu')
    
    print(f"\nLoaded predictions from {predictions_path}")
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")
        elif isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__}")
    
    return data


def analyze_by_case_type(predictions, mapping, cases_dir):
    """Compute per-case-type metrics following main script's V2X-Graph protocol."""
    cases_dir = Path(cases_dir)
    
    node_ids = predictions['node_ids']
    timestamps = predictions['timestamps']
    preds = predictions['preds']
    gts = predictions['gts']
    states = predictions['states']
    is_target = predictions['is_target']
    v2x_ade = predictions['v2x_ade']
    v2x_fde = predictions['v2x_fde']
    
    N = len(preds)
    
    # scene_id -> [sample_idx]
    scene_to_indices = defaultdict(list)
    for i in range(N):
        nid = int(node_ids[i])
        scene_id = mapping['nodeid_to_scene'].get(nid)
        if scene_id:
            scene_to_indices[scene_id].append(i)
    
    case_types = ['A_approaching', 'B_departing', 'C_overtaking', 
                  'D_crossing', 'E_lane_change', 'F_fast']
    
    results = {}
    
    # Overall metrics
    target_mask = is_target.bool()
    target_count = target_mask.sum().item()
    
    overall_ade = v2x_ade.mean().item()
    overall_fde = v2x_fde.mean().item()
    overall_mr = (v2x_fde > 2).float().mean().item()
    
    target_ade = v2x_ade[target_mask].mean().item() if target_count > 0 else 0
    target_fde = v2x_fde[target_mask].mean().item() if target_count > 0 else 0
    target_mr = (v2x_fde[target_mask] > 2).float().mean().item() if target_count > 0 else 0
    

    print("CHALLENGING CASES EVALUATION (V2X-Graph Protocol)")

    
    print(f"\n[Overall Test Set]")
    print(f"  All agents (N={N}):        ADE={overall_ade:.3f}m  FDE={overall_fde:.3f}m  MR@2m={overall_mr*100:.1f}%")
    print(f"  Target agents (N={target_count}): ADE={target_ade:.3f}m  FDE={target_fde:.3f}m  MR@2m={target_mr*100:.1f}%")
    
    results['overall'] = {
        'all': {'ade': overall_ade, 'fde': overall_fde, 'mr_2m': overall_mr, 'n': N},
        'target': {'ade': target_ade, 'fde': target_fde, 'mr_2m': target_mr, 'n': target_count},
    }
    
    print(f"\n[Per Case Type - All Agents in Challenging Scenes]")
    print(f"  {'Case':<22} {'Scenes':>7} {'N':>7} {'ADE':>8} {'FDE':>8} {'MR@2m':>8}")
    
    for case_type in case_types:
        details_path = cases_dir / f'{case_type}_details.json'
        if not details_path.exists():
            continue
        
        with open(details_path) as f:
            entries = json.load(f)
        
        val_scene_ids = {e['scene_id'] for e in entries if e['split'] == 'val'}
        
        case_indices = []
        matched_scenes = set()
        for scene_id in val_scene_ids:
            indices = scene_to_indices.get(scene_id, [])
            if indices:
                case_indices.extend(indices)
                matched_scenes.add(scene_id)
        
        if not case_indices:
            print(f"  {case_type:<22} {'(no matches)':>7}")
            results[case_type] = None
            continue
        
        idx = torch.tensor(case_indices)
        case_ade = v2x_ade[idx]
        case_fde = v2x_fde[idx]
        case_target = is_target[idx]
        case_states = states[idx]
        
        ade_mean = case_ade.mean().item()
        fde_mean = case_fde.mean().item()
        mr = (case_fde > 2).float().mean().item()
        
        print(f"  {case_type:<22} {len(matched_scenes):>7} {len(case_indices):>7} "
              f"{ade_mean:>8.3f} {fde_mean:>8.3f} {mr*100:>7.1f}%")
        
        target_idx_mask = case_target.bool()
        target_n = target_idx_mask.sum().item()
        
        case_result = {
            'all': {'ade': ade_mean, 'fde': fde_mean, 'mr_2m': mr, 'n': len(case_indices)},
            'matched_scenes': len(matched_scenes),
            'total_val_scenes': len(val_scene_ids),
        }
        
        if target_n > 0:
            t_ade = case_ade[target_idx_mask].mean().item()
            t_fde = case_fde[target_idx_mask].mean().item()
            t_mr = (case_fde[target_idx_mask] > 2).float().mean().item()
            case_result['target'] = {'ade': t_ade, 'fde': t_fde, 'mr_2m': t_mr, 'n': target_n}
        
        # Speed stratification
        speeds = torch.norm(case_states[:, 2:4], dim=-1)
        for speed_name, mask in [('slow', speeds < 5), ('med', (speeds >= 5) & (speeds < 15)), ('fast', speeds >= 15)]:
            if mask.sum() > 0:
                case_result[f'speed_{speed_name}'] = {
                    'ade': case_ade[mask].mean().item(),
                    'fde': case_fde[mask].mean().item(),
                    'mr_2m': (case_fde[mask] > 2).float().mean().item(),
                    'n': mask.sum().item(),
                }
        
        results[case_type] = case_result
    
    print(f"\n[Per Case Type - Target Agents Only] (comparable with V2X-Graph)")
    print(f"  {'Case':<22} {'N':>7} {'ADE':>8} {'FDE':>8} {'MR@2m':>8}")
    print("  " + "-" * 54)
    
    for case_type in case_types:
        r = results.get(case_type)
        if r and 'target' in r:
            t = r['target']
            print(f"  {case_type:<22} {t['n']:>7} {t['ade']:>8.3f} {t['fde']:>8.3f} {t['mr_2m']*100:>7.1f}%")
    
    return results




CASE_COLORS = {
    'A_approaching': '#e74c3c', 'B_departing': '#3498db',
    'C_overtaking': '#2ecc71', 'D_crossing': '#f39c12',
    'E_lane_change': '#9b59b6', 'F_fast': '#e67e22',
}


def get_rating(ade, fde):
    if ade < 0.5 and fde < 1.0:
        return "EXCELLENT"
    elif ade < 1.0 and fde < 2.0:
        return "GOOD"
    elif ade < 2.0 and fde < 4.0:
        return "ACCEPTABLE"
    return "POOR"


def load_scene_csv(scene_id, split='val'):
    """Load raw cooperative trajectory CSV."""
    v2x_root = Path('/scratch/maiqi/autodriving/v2xseq/V2X-Seq-TFD/cooperative-vehicle-infrastructure')
    csv_path = v2x_root / 'cooperative-trajectories' / split / f'{scene_id}.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df = df.rename(columns={'v_x': 'vx', 'v_y': 'vy', 'theta': 'heading'})
    return df


def plot_scene_with_prediction(ax, scene_id, agent_id, abs_curr, rel_pred, rel_gt, 
                                ade, fde, is_tgt, case_agent_ids=None, hist_steps=50):
    """
    Plot one prediction example with full scene context.
    
    Colors:
        Current agent history: #2ca02c (green) solid
        GT future:             #3498db (blue) dashed  
        Pred future:           #e74c3c (red) dashed
        Case agent (pair):     #e67e22 (orange) solid hist / dashed future
        Background agents:     gray
    
    case_agent_ids: set of agent ids involved in the case (excluding current agent)
    
    Coordinate transform (get_batch is translation-only):
        absolute = relative + [curr_x, curr_y]
    """
    df = load_scene_csv(scene_id, split='val')
    if df is None:
        ax.text(0.5, 0.5, f'Scene {scene_id}\nCSV not found',
                ha='center', va='center', transform=ax.transAxes)
        return
    
    timestamps = sorted(df['timestamp'].unique())
    hist_ts = set(timestamps[:hist_steps])
    fut_ts = set(timestamps[hist_steps:])
    
    curr_x, curr_y = abs_curr[0], abs_curr[1]
    
    # > convert relative pred/gt to absolute (translation only, no rotation in get_batch)
    abs_pred = rel_pred.copy()
    abs_pred[:, 0] += curr_x
    abs_pred[:, 1] += curr_y
    
    abs_gt = rel_gt.copy()
    abs_gt[:, 0] += curr_x
    abs_gt[:, 1] += curr_y
    
    if case_agent_ids is None:
        case_agent_ids = set()
    
    # > other agents: case agents (orange) vs background (gray)
    for aid in df['id'].unique():
        if aid == agent_id:
            continue
        adf = df[df['id'] == aid].sort_values('timestamp')
        hist = adf[adf['timestamp'].isin(hist_ts)]
        fut = adf[adf['timestamp'].isin(fut_ts)]
        
        is_case = (aid in case_agent_ids)
        color = '#e67e22' if is_case else 'gray'
        alpha = 1.0 if is_case else 0.3
        lw = 2.5 if is_case else 0.8
        s = 40 if is_case else 10
        zord = 5 if is_case else 3
        
        if len(hist) > 0:
            ax.plot(hist['x'].values, hist['y'].values, '-', 
                    color=color, alpha=alpha, lw=lw)
            ax.scatter(hist['x'].iloc[0], hist['y'].iloc[0], 
                      color=color, marker='o', s=s, alpha=alpha, zorder=zord)
        if len(fut) > 0:
            ax.plot(fut['x'].values, fut['y'].values, '--', 
                    color=color, alpha=alpha * 0.7, lw=lw)
            ax.scatter(fut['x'].iloc[-1], fut['y'].iloc[-1],
                      color=color, marker='x', s=s, alpha=alpha * 0.7, zorder=zord)
    
    adf = df[df['id'] == agent_id].sort_values('timestamp')
    hist = adf[adf['timestamp'].isin(hist_ts)]
    
    # > history (green solid)
    if len(hist) > 0:
        ax.plot(hist['x'].values, hist['y'].values, '-', 
                color='#2ca02c', lw=2.5, alpha=0.9, zorder=5)
        ax.scatter(hist['x'].iloc[0], hist['y'].iloc[0], 
                  color='#2ca02c', marker='o', s=50, zorder=6, 
                  edgecolors='white', linewidths=0.5)
    
    # > observation point (current position = black diamond)
    ax.scatter(curr_x, curr_y, color='black', marker='D', s=70, 
              zorder=10, edgecolors='white', linewidths=0.5)
    
    # > gt future (blue dashed)
    ax.plot(abs_gt[:, 0], abs_gt[:, 1], '--', 
            color='#3498db', lw=2.5, alpha=0.8, zorder=5)
    ax.scatter(abs_gt[-1, 0], abs_gt[-1, 1], 
              color='#3498db', marker='x', s=60, linewidths=2, zorder=6)
    
    # > pred future (red dashed)
    ax.plot(abs_pred[:, 0], abs_pred[:, 1], '--', 
            color='#e74c3c', lw=2.5, alpha=0.8, zorder=5)
    ax.scatter(abs_pred[-1, 0], abs_pred[-1, 1], 
              color='#e74c3c', marker='x', s=60, linewidths=2, zorder=6)
    
    # > fde error line at endpoints
    ax.plot([abs_gt[-1, 0], abs_pred[-1, 0]], [abs_gt[-1, 1], abs_pred[-1, 1]],
            ':', color='red', lw=1.2, alpha=0.4)
    
    # Title
    rating = get_rating(ade, fde)
    tgt_str = "★TARGET" if is_tgt else "other"
    ax.set_title(f"Scene {scene_id} ({tgt_str})\n"
                 f"ADE={ade:.2f}m  FDE={fde:.2f}m  [{rating}]",
                 fontsize=9, fontweight='bold')
    
    ax.set_xlabel('X (m)', fontsize=8)
    ax.set_ylabel('Y (m)', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=7)


def visualize_case_predictions(predictions, mapping, cases_dir, output_dir, data_dir, n_per_case=6):
    """Visualize prediction examples for each case type with full scene context."""
    cases_dir = Path(cases_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # > absolute coordinates from pkl
    traj_data = mapping['traj_data']
    abs_current_state = traj_data['current_state']  # (nid, ts) -> [x,y,vx,vy,heading]
    
    node_ids = predictions['node_ids']
    timestamps = predictions['timestamps']
    preds = predictions['preds']
    gts = predictions['gts']
    is_target = predictions['is_target']
    v2x_ade = predictions['v2x_ade']
    v2x_fde = predictions['v2x_fde']
    
    N = len(preds)
    
    # > scene_id -> [sample_idx]
    scene_to_indices = defaultdict(list)
    for i in range(N):
        scene_id = mapping['nodeid_to_scene'].get(int(node_ids[i]))
        if scene_id:
            scene_to_indices[scene_id].append(i)
    
    case_types = ['A_approaching', 'B_departing', 'C_overtaking',
                  'D_crossing', 'E_lane_change', 'F_fast']
    
    legend_handles = [
        mlines.Line2D([], [], color='#2ca02c', lw=2.5, label='Agent history'),
        mlines.Line2D([], [], color='#3498db', lw=2.5, ls='--', label='GT future'),
        mlines.Line2D([], [], color='#e74c3c', lw=2.5, ls='--', label='Pred future'),
        mlines.Line2D([], [], color='#e67e22', lw=2.5, label='Case agent'),
        mlines.Line2D([], [], color='gray', lw=1.0, alpha=0.5, label='Background'),
        mlines.Line2D([], [], marker='D', color='w', markerfacecolor='black', 
                      markersize=8, label='Observation'),
    ]
    
    for case_type in case_types:
        details_path = cases_dir / f'{case_type}_details.json'
        if not details_path.exists():
            continue
        
        with open(details_path) as f:
            entries = json.load(f)
        val_entries = [e for e in entries if e['split'] == 'val']
        val_scene_ids = [e['scene_id'] for e in val_entries]
        
        # > build scene_id -> details for extracting case agent ids
        scene_details = {e['scene_id']: e.get('details', {}) for e in val_entries}
        
        # Prefer target agent samples
        target_samples = []
        other_samples = []
        for scene_id in val_scene_ids:
            for idx in scene_to_indices.get(scene_id, []):
                if is_target[idx]:
                    target_samples.append((scene_id, idx))
                else:
                    other_samples.append((scene_id, idx))
        
        all_samples = target_samples if target_samples else other_samples
        if not all_samples:
            print(f"  {case_type}: no visualization samples")
            continue
        
        np.random.seed(42)
        n = min(n_per_case, len(all_samples))
        chosen = [all_samples[i] for i in 
                  np.random.choice(len(all_samples), n, replace=False)]
        
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
        if n == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes).flatten()
        
        for i, (scene_id, idx) in enumerate(chosen):
            ax = axes[i]
            nid = int(node_ids[idx])
            ts = float(timestamps[idx])
            
            agent_id = mapping['nodeid_to_agent'].get(nid)
            
            abs_curr = abs_current_state.get((nid, ts))
            if abs_curr is None:
                ax.text(0.5, 0.5, f'Scene {scene_id}\nno abs state',
                        ha='center', va='center', transform=ax.transAxes)
                continue
            
            rel_pred = preds[idx].numpy().copy()
            rel_gt = gts[idx].numpy().copy()
            ade = v2x_ade[idx].item()
            fde = v2x_fde[idx].item()
            is_tgt = bool(is_target[idx].item())
            
            # > extract case agent ids from details (the "other" agent in the case)
            details = scene_details.get(scene_id, {})
            case_agent_ids = set()
            pair = details.get('pair')
            if pair:
                # A/B/C/D: pair = [id1, id2], highlight the one that isn't current
                case_agent_ids = {int(a) for a in pair if a is not None} - {agent_id}
            agent_single = details.get('agent')
            if agent_single is not None and agent_single != agent_id:
                case_agent_ids.add(int(agent_single))
            agents_list = details.get('agents', [])
            for a in agents_list:
                aid = a[0] if isinstance(a, (list, tuple)) else a
                if aid != agent_id:
                    case_agent_ids.add(int(aid))
            
            plot_scene_with_prediction(
                ax, scene_id, agent_id, abs_curr,
                rel_pred, rel_gt, ade, fde, is_tgt,
                case_agent_ids=case_agent_ids
            )
        
        for i in range(n, len(axes)):
            axes[i].set_visible(False)
        
        case_title = case_type.replace('_', ' ').title()
        case_color = CASE_COLORS.get(case_type, 'black')
        fig.suptitle(f'Predictions: {case_title} ({n} test examples)', 
                     fontsize=14, fontweight='bold', color=case_color, y=1.02)
        
        fig.legend(handles=legend_handles, loc='lower center', ncol=6, fontsize=10,
                   bbox_to_anchor=(0.5, -0.02))
        
        plt.tight_layout()
        save_path = output_dir / f'{case_type}_predictions.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  {case_type}: {n} examples -> {save_path}")



def create_comparison_table(results, output_path):
    """Create a comparison table image."""
    case_types = ['A_approaching', 'B_departing', 'C_overtaking',
                  'D_crossing', 'E_lane_change', 'F_fast']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    headers = ['Case Type', 'N (all)', 'ADE', 'FDE', 'MR@2m',
               'N (target)', 'ADE*', 'FDE*', 'MR@2m*']
    
    rows = []
    o = results['overall']
    rows.append([
        'Overall (full test)',
        str(o['all']['n']),
        f"{o['all']['ade']:.3f}",
        f"{o['all']['fde']:.3f}",
        f"{o['all']['mr_2m']*100:.1f}%",
        str(o['target']['n']),
        f"{o['target']['ade']:.3f}",
        f"{o['target']['fde']:.3f}",
        f"{o['target']['mr_2m']*100:.1f}%",
    ])
    
    for case_type in case_types:
        r = results.get(case_type)
        if r is None:
            rows.append([case_type] + ['-'] * 8)
            continue
        
        a = r['all']
        t = r.get('target', {})
        rows.append([
            case_type,
            str(a['n']),
            f"{a['ade']:.3f}",
            f"{a['fde']:.3f}",
            f"{a['mr_2m']*100:.1f}%",
            str(t.get('n', '-')),
            f"{t['ade']:.3f}" if 'ade' in t else '-',
            f"{t['fde']:.3f}" if 'fde' in t else '-',
            f"{t['mr_2m']*100:.1f}%" if 'mr_2m' in t else '-',
        ])
    
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    
    for j in range(len(headers)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    for j in range(len(headers)):
        table[1, j].set_facecolor('#ecf0f1')
        table[1, j].set_text_props(fontweight='bold')
    
    for i in range(2, len(rows) + 1):
        for j in [4, 8]:
            text = table[i, j].get_text().get_text()
            if text != '-':
                val = float(text.replace('%', ''))
                if val > 40:
                    table[i, j].set_facecolor('#fadbd8')
                elif val > 25:
                    table[i, j].set_facecolor('#fdebd0')
                else:
                    table[i, j].set_facecolor('#d5f5e3')
    
    ax.set_title("DyGFormer on Challenging Cases (V2X-Graph Protocol)\n"
                 "(*Target Agent Only, comparable to V2X-Graph)",
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Comparison table -> {output_path}")




def main():
    parser = argparse.ArgumentParser('Evaluate DyGFormer on Challenging Cases')
    parser.add_argument('--data_dir', type=str, 
                       default='/scratch/yiran/v2x/v2x_cig_alg_selfloop')
    parser.add_argument('--predictions_path', type=str, required=True)
    parser.add_argument('--cases_dir', type=str,
                       default='/home/yiran/challenging_cases')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--n_vis', type=int, default=6)
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = str(Path(args.cases_dir) / 'evaluation')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Building scene-node mapping from test trajectory data")
    mapping = build_scene_node_mapping(args.data_dir)
    
    predictions = load_predictions(args.predictions_path)
    
    results = analyze_by_case_type(predictions, mapping, args.cases_dir)
    
    print("\nGenerating prediction visualizations")
    visualize_case_predictions(
        predictions, mapping, args.cases_dir, output_dir, 
        args.data_dir, args.n_vis
    )
    
    create_comparison_table(results, output_dir / 'comparison_table.png')
    
    json_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            json_results[k] = {}
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    json_results[k][k2] = {
                        k3: (float(v3) if isinstance(v3, (torch.Tensor, np.floating, float)) else int(v3) if isinstance(v3, (np.integer, int)) else v3)
                        for k3, v3 in v2.items()
                    }
                else:
                    json_results[k][k2] = float(v2) if isinstance(v2, (torch.Tensor, np.floating, float)) else v2
        else:
            json_results[k] = v
    
    results_path = output_dir / 'challenging_results.json'
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"All outputs in: {output_dir}")


if __name__ == '__main__':
    main()