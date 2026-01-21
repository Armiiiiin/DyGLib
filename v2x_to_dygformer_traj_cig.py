"""
V2X-Seq-TFD to DyGFormer format converter for traj prediction
input: V2X-Seq-TFD cooperative-trajectories (fused vehicle + infra)

V2X-Graph CIG 

1. directed edges src -> dst and dst -> src are 2 seperate adges
2. local coord of TARGET node
3. relative pos = src_pos - dst_pos
4. relative heading = src_heading - dst_heading

V2X train 80% -> my train
V2X train 20% -> my val
V2X val 100% -> my test
for car with speed < 0.5 m/s, future_traj (x, y) = (x0, y0), that way the relative coord can be (0, 0) and the noise can be reduced
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle


class V2XToDyGFormerTrajectory:
    
    def __init__(self, v2x_root: str, output_dir: str = './processed_data/v2x_traj'):
        self.v2x_root = Path(v2x_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.coop_dir = self.v2x_root / 'cooperative-vehicle-infrastructure'
        
        self.pred_horizon = 50
        self.hist_horizon = 50
        
        self.max_lanes = 20
        self.lane_search_radius = 50.0
        self.points_per_lane = 20
        
        self.load_lane_map()
    
    def load_lane_map(self):
        map_dir = self.v2x_root / 'map_files'
        
        bbox_path = map_dir / 'yizhuang_PEK_halluc_bbox_table.npy'
        idx_map_path = map_dir / 'yizhuang_PEK_tableidx_to_laneid_map.json'
        vector_map_path = map_dir / 'yizhuang_PEK_vector_map.json'
        
        if not all(p.exists() for p in [bbox_path, idx_map_path, vector_map_path]):
            print("Warning: Lane map files not found")
            self.has_lane_map = False
            return
        
        print("Loading lane map")
        self.lane_bbox = np.load(bbox_path)
        with open(idx_map_path, 'r') as f:
            self.idx_to_laneid = json.load(f)
        with open(vector_map_path, 'r') as f:
            self.vector_map = json.load(f)
        
        self.has_lane_map = True
        print(f"  Loaded {len(self.lane_bbox)} lanes")
    
    def get_nearby_lanes(self, x: float, y: float) -> dict:
        if not self.has_lane_map:
            return None
        
        in_x = (self.lane_bbox[:, 0] - self.lane_search_radius <= x) & \
               (x <= self.lane_bbox[:, 2] + self.lane_search_radius)
        in_y = (self.lane_bbox[:, 1] - self.lane_search_radius <= y) & \
               (y <= self.lane_bbox[:, 3] + self.lane_search_radius)
        
        candidate_indices = np.where(in_x & in_y)[0]
        
        if len(candidate_indices) == 0:
            return None
        
        centerlines = []
        lane_types = []
        distances = []
        
        for idx in candidate_indices:
            lane_id = self.idx_to_laneid.get(str(idx))
            if lane_id is None or lane_id not in self.vector_map:
                continue
            
            lane_info = self.vector_map[lane_id]
            centerline = np.array(lane_info['centerline'])
            
            if len(centerline) == 0:
                continue
            
            dists = np.sqrt((centerline[:, 0] - x)**2 + (centerline[:, 1] - y)**2)
            min_dist = dists.min()
            
            if min_dist > self.lane_search_radius:
                continue
            
            if len(centerline) >= self.points_per_lane:
                indices = np.linspace(0, len(centerline)-1, self.points_per_lane, dtype=int)
                sampled = centerline[indices]
            else:
                sampled = np.zeros((self.points_per_lane, 2))
                sampled[:len(centerline)] = centerline
                sampled[len(centerline):] = centerline[-1]
            
            centerlines.append(sampled)
            lane_types.append(lane_info.get('lane_type', 'UNKNOWN'))
            distances.append(min_dist)
        
        if len(centerlines) == 0:
            return None
        
        sorted_idx = np.argsort(distances)[:self.max_lanes]
        centerlines = np.array([centerlines[i] for i in sorted_idx])
        lane_types = [lane_types[i] for i in sorted_idx]
        
        return {
            'centerlines': centerlines.astype(np.float32),
            'lane_types': lane_types
        }
    
    def load_trajectory_file(self, csv_file: Path) -> pd.DataFrame:
        """
        ['city', 'timestamp', 'id', 'type', 'sub_type', 'tag', 'x', 'y', 'z', 
         'length', 'width', 'height', 'theta', 'v_x', 'v_y', 'intersect_id']
        """
        df = pd.read_csv(csv_file)
        
        rename_map = {
            'theta': 'heading',
            'v_x': 'vx',
            'v_y': 'vy'
        }
        df = df.rename(columns=rename_map)
        
        return df
    
    def calculate_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['id', 'timestamp']).copy()
        
        if 'vx' in df.columns and 'vy' in df.columns:
            if not (df['vx'].isna().any() or df['vy'].isna().any()):
                return df
        
        for agent_id in df['id'].unique():
            mask = df['id'] == agent_id
            agent_data = df[mask].copy()
            
            if len(agent_data) > 1:
                dx = agent_data['x'].diff()
                dy = agent_data['y'].diff()
                dt = agent_data['timestamp'].diff()
                
                vx = (dx / dt).fillna(0).replace([np.inf, -np.inf], 0).clip(-30, 30)
                vy = (dy / dt).fillna(0).replace([np.inf, -np.inf], 0).clip(-30, 30)
                
                df.loc[mask, 'vx'] = vx.values
                df.loc[mask, 'vy'] = vy.values
        
        return df
    
    def get_future_trajectory(self, df: pd.DataFrame, agent_id: int, 
                              current_ts: float, horizon: int) -> np.ndarray:
        agent_data = df[df['id'] == agent_id].sort_values('timestamp')
        future_data = agent_data[agent_data['timestamp'] > current_ts].head(horizon)
        
        if len(future_data) < horizon:
            return None
        
        trajectory = future_data[['x', 'y', 'vx', 'vy', 'heading']].values
        
        # > check traj quality to avoid weird gt
        x, y = trajectory[:, 0], trajectory[:, 1]
        step_dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    
        # > leap
        if step_dist.max() > 5.0:
            return None
    
        # > stuck
        if (step_dist < 0.001).sum() > 5:
            return None
    
        return trajectory
    
    def get_history_trajectory(self, df: pd.DataFrame, agent_id: int,
                           current_ts: float, horizon: int) -> np.ndarray:

        agent_data = df[df['id'] == agent_id].sort_values('timestamp')
        hist_data = agent_data[agent_data['timestamp'] <= current_ts].tail(horizon)
    
        if len(hist_data) < horizon:
            return None
    
        trajectory = hist_data[['x', 'y', 'vx', 'vy']].values  # [50, 4]
    
        x, y = trajectory[:, 0], trajectory[:, 1]
        step_dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    
        if step_dist.max() > 5.0:
            return None
    
        return trajectory.astype(np.float32)
    
    def check_isolated(self, agents: List[dict], agent_idx: int, 
                       threshold: float = 20.0) -> bool:
        agent = agents[agent_idx]
        for j, other in enumerate(agents):
            if j != agent_idx:
                dist = np.sqrt((agent['x'] - other['x'])**2 + 
                              (agent['y'] - other['y'])**2)
                if dist < threshold:
                    return False 
        return True
    
    def process_scenes(self, scene_files: List[Path], split_name: str,
                       coop_traj_dir: Path,
                       spatial_threshold: float = 20.0, skip_isolated: bool = True):
        
        print(f"Processing {split_name} split ({len(scene_files)} scenes)")
        print(f"Prediction horizon: {self.pred_horizon} frames ({self.pred_horizon * 0.1:.1f}s)")
        print(f"Skip isolated nodes: {skip_isolated}")
        
        all_edges = []
        all_edge_feats = []
        node_id_map = {}
        current_node_id = 0
        edge_idx = 0
        
        node_future_traj = {}
        node_current_state = {}
        node_history_traj = {}
        node_lanes = {}
        node_is_target = {}
        
        total_agents = 0
        skipped_isolated = 0
        skipped_incomplete = 0
        target_agent_count = 0
        
        for scene_idx, scene_file in enumerate(scene_files):
            scene_id = scene_file.stem
            
            coop_file = coop_traj_dir / f'{scene_id}.csv'
            
            if not coop_file.exists():
                continue
            
            all_df = self.load_trajectory_file(coop_file)
            all_df = self.calculate_velocity(all_df)
            
            timestamps = sorted(all_df['timestamp'].unique())
            if len(timestamps) >= self.hist_horizon + self.pred_horizon:
                valid_timestamps = [timestamps[self.hist_horizon - 1]]
            else:
                valid_timestamps = []
            
            if len(valid_timestamps) == 0:
                continue
            
            base_ts = timestamps[0]
            ts_map = {orig: (orig - base_ts) for orig in timestamps}
            
            for _, row in all_df.iterrows():
                agent_key = f"{scene_id}_{row['id']}"
                if agent_key not in node_id_map:
                    current_node_id += 1
                    node_id_map[agent_key] = current_node_id
            
            for ts in valid_timestamps:
                frame_df = all_df[all_df['timestamp'] == ts]
                normalized_ts = ts_map[ts]
                
                agents = []
                for _, row in frame_df.iterrows():
                    agent_key = f"{scene_id}_{row['id']}"
                    node_id = node_id_map[agent_key]
                    
                    future_traj = self.get_future_trajectory(all_df, row['id'], ts, self.pred_horizon)
                    
                    if future_traj is None:
                        skipped_incomplete += 1
                        continue
                    
                    history_traj = self.get_history_trajectory(all_df, row['id'], ts, self.hist_horizon)

                    if history_traj is None:
                        skipped_incomplete += 1
                        continue
                    
                    nearby_lanes = None
                    if self.has_lane_map:
                        nearby_lanes = self.get_nearby_lanes(row['x'], row['y'])
                    is_target = (row.get('tag', '') == 'TARGET_AGENT')
                    
                    agents.append({
                        'node_id': node_id,
                        'x': row['x'],
                        'y': row['y'],
                        'vx': row.get('vx', 0),
                        'vy': row.get('vy', 0),
                        'heading': row.get('heading', 0),
                        'type': row.get('type', 'UNKNOWN'),
                        'is_target': is_target,
                        'future_traj': future_traj,
                        'history_traj': history_traj,
                        'nearby_lanes': nearby_lanes
                    })
                
                total_agents += len(agents)
                
                valid_agents = []
                for i, agent in enumerate(agents):
                    if skip_isolated and self.check_isolated(agents, i, spatial_threshold):
                        skipped_isolated += 1
                        continue
                    valid_agents.append(agent)
                    
                    key = (agent['node_id'], normalized_ts)
                    
                    # > for car with speed < 0.5 m/s, future_traj (x, y) = (x0, y0)
                    speed = np.sqrt(agent['vx']**2 + agent['vy']**2)
                    if speed < 0.5 and not agent['is_target']:
                        agent['future_traj'][:, 0] = agent['x']
                        agent['future_traj'][:, 1] = agent['y']
                        
                    
                        
                    node_future_traj[key] = agent['future_traj']
                    node_current_state[key] = np.array([
                        agent['x'], agent['y'], 
                        agent['vx'], agent['vy'], 
                        agent['heading']
                    ], dtype=np.float32)
                    node_history_traj[key] = agent['history_traj']
                    node_is_target[key] = agent['is_target']
                    
                    if agent['is_target']:
                        target_agent_count += 1
                    
                    if agent['nearby_lanes'] is not None:
                        node_lanes[key] = agent['nearby_lanes']
                
                agents = valid_agents
                
                # > V2X-Graph CIG
                for src_idx, src_agent in enumerate(agents):
                    for dst_idx, dst_agent in enumerate(agents):
                        if src_idx == dst_idx:
                            continue
                        
                        dist = np.sqrt((src_agent['x'] - dst_agent['x'])**2 + 
                                      (src_agent['y'] - dst_agent['y'])**2)
                        
                        if dist < spatial_threshold:
                            edge_idx += 1
                            
                            # > coord system: dst heading
                            heading_dst = dst_agent['heading']
                            cos_h = np.cos(heading_dst)
                            sin_h = np.sin(heading_dst)
                            
                            # > relative pos: src - dst
                            dx = src_agent['x'] - dst_agent['x']
                            dy = src_agent['y'] - dst_agent['y']
                            
                            # > dst agent coord
                            rel_x = cos_h * dx + sin_h * dy
                            rel_y = -sin_h * dx + cos_h * dy
                            
                            # > angle in dst agent coord
                            angle_in_dst_frame = np.arctan2(rel_y, rel_x)
                            
                            # > src_v converted into dst node coord
                            src_vx_local = cos_h * src_agent['vx'] + sin_h * src_agent['vy']
                            src_vy_local = -sin_h * src_agent['vx'] + cos_h * src_agent['vy']
                            
                            # > dst_v in dst coord system
                            dst_vx_local = cos_h * dst_agent['vx'] + sin_h * dst_agent['vy']
                            dst_vy_local = -sin_h * dst_agent['vx'] + cos_h * dst_agent['vy']
                            
                            # > relative v in dst coord system
                            rel_vx = src_vx_local - dst_vx_local
                            rel_vy = src_vy_local - dst_vy_local
                            
                            # > relative heading = src_heading - dst_heading
                            heading_diff = src_agent['heading'] - dst_agent['heading']
                            heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
                            
                            all_edges.append({
                                'u': src_agent['node_id'],  
                                'i': dst_agent['node_id'],  
                                'ts': normalized_ts,
                                'label': 1,
                                'idx': edge_idx
                            })
                            
                            edge_feat = [
                                dist,
                                angle_in_dst_frame, # > src angle in dst coord sys
                                src_vx_local, src_vy_local,     
                                dst_vx_local, dst_vy_local, 
                                rel_vx, rel_vy,
                                np.cos(heading_diff), # > relative heading cos
                                np.sin(heading_diff), # > relative heading sin
                                dist / spatial_threshold, # > normalized dist
                                np.sqrt(rel_vx**2 + rel_vy**2), # > scalar
                                1.0 if np.cos(heading_diff) > 0.5 else 0.0,  # > sae heading flag
                                1.0 if src_agent['type'] == dst_agent['type'] else 0.0 
                            ]
                            all_edge_feats.append(edge_feat)
            
            if (scene_idx + 1) % 100 == 0:
                print(f"  Processed {scene_idx + 1}/{len(scene_files)} scenes")
        
        print(f"\n{split_name} Summary:")
        print(f"  Total nodes: {len(node_id_map)}")
        print(f"  Total edges: {len(all_edges)}")
        print(f"  Traj samples: {len(node_future_traj)}")
        print(f"  TARGET_AGENT samples: {target_agent_count}")
        print(f"  Lane samples: {len(node_lanes)}")
        print(f"  Skipped (isolated): {skipped_isolated}")
        print(f"  Skipped (incomplete): {skipped_incomplete}")
        
        slow_target = sum(1 for (nid, ts), is_target in node_is_target.items() 
                          if is_target and np.linalg.norm(node_current_state[(nid, ts)][2:4]) < 0.5)
        print(f"  Slow TARGET_AGENTs (speed < 0.5): {slow_target}/{target_agent_count}")
        
        return {
            'edges': all_edges,
            'edge_feats': all_edge_feats,
            'num_nodes': len(node_id_map),
            'node_future_traj': node_future_traj,
            'node_current_state': node_current_state,
            'node_history_traj': node_history_traj,
            'node_is_target': node_is_target,
            'node_lanes': node_lanes,
            'node_id_map': node_id_map
        }
    
    def save_split(self, data: dict, split_name: str, node_feat_dim: int = 172, skip_lanes: bool = False):
        print(f"\nSaving {split_name} split.")
        
        edges = data['edges']
        edge_feats = data['edge_feats']
        num_nodes = data['num_nodes']
        
        df = pd.DataFrame(edges)
        csv_path = self.output_dir / f'{split_name}.csv'
        df.to_csv(csv_path, index=False)
        
        edge_feat_array = np.array(edge_feats, dtype=np.float32)
        empty_row = np.zeros((1, edge_feat_array.shape[1]), dtype=np.float32)
        edge_feat_with_zero = np.vstack([empty_row, edge_feat_array])
        np.save(self.output_dir / f'{split_name}.npy', edge_feat_with_zero)
        
        node_feats = np.zeros((num_nodes + 1, node_feat_dim), dtype=np.float32)
        np.save(self.output_dir / f'{split_name}_node.npy', node_feats)
        
        traj_data = {
            'future_traj': data['node_future_traj'],
            'current_state': data['node_current_state'],
            'history_traj': data['node_history_traj'],
            'is_target': data['node_is_target'],
            'pred_horizon': self.pred_horizon,
            'hist_horizon': self.hist_horizon,
            'node_id_map': data['node_id_map']
        }
        with open(self.output_dir / f'{split_name}_trajectory.pkl', 'wb') as f:
            pickle.dump(traj_data, f)
        
        if not skip_lanes:
            lane_data = {
                'lanes': data['node_lanes'],
                'max_lanes': self.max_lanes,
                'points_per_lane': self.points_per_lane,
                'search_radius': self.lane_search_radius
            }
            with open(self.output_dir / f'{split_name}_lanes.pkl', 'wb') as f:
                pickle.dump(lane_data, f)
        
            print(f"  Saved: {split_name}.csv, .npy, _node.npy, _trajectory.pkl, _lanes.pkl")
        else:
            print(f"  Saved: {split_name}.csv, .npy, _node.npy, _trajectory.pkl (skipped lanes)")
    
if __name__ == "__main__":
    
    V2X_ROOT = "/scratch/maiqi/autodriving/v2xseq/V2X-Seq-TFD"
    
    OUTPUT_DIR = "/scratch/yiran/v2x/v2x_cig"
    
    converter = V2XToDyGFormerTrajectory(v2x_root=V2X_ROOT, output_dir=OUTPUT_DIR)
    converter.has_lane_map = True
    
    np.random.seed(89)
    
    # > use cooperative-trajectories directly
    coop_train_dir = converter.coop_dir / 'cooperative-trajectories' / 'train'
    coop_test_dir = converter.coop_dir / 'cooperative-trajectories' / 'val'
    
    all_train_files = sorted(list(coop_train_dir.glob('*.csv')))
    all_test_files = sorted(list(coop_test_dir.glob('*.csv')))
    
    np.random.shuffle(all_train_files)
    
    val_ratio = 0.2
    split_idx = int(len(all_train_files) * (1 - val_ratio))
    my_train_files = all_train_files[:split_idx]
    my_val_files = all_train_files[split_idx:]
    
    print(f"V2X cooperative train: {len(all_train_files)} scenes")
    print(f"My train: {len(my_train_files)} scenes")
    print(f"My val: {len(my_val_files)} scenes")
    print(f"My test: {len(all_test_files)} scenes")
    
    BATCH_SIZE = 2000
    
    def process_batched(files, split_name, coop_dir):
        print(f"Processing {split_name}: {len(files)} scenes (batched)")
        
        all_results = []
        for batch_start in range(0, len(files), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(files))
            batch_files = files[batch_start:batch_end]
            
            print(f"\nBatch {batch_start//BATCH_SIZE + 1}: scenes {batch_start}-{batch_end}")
            
            batch_data = converter.process_scenes(
                scene_files=batch_files,
                split_name=f'{split_name}_batch{batch_start//BATCH_SIZE}',
                coop_traj_dir=coop_dir,
                spatial_threshold=20.0,
                skip_isolated=False
            )
            all_results.append(batch_data)
            print(f"Batch done. Edges: {len(batch_data['edges'])}")
        
        print(f"\nMerging {split_name} batches")
        merged = {
            'edges': [], 'edge_feats': [], 'num_nodes': 0,
            'node_future_traj': {}, 'node_current_state': {},
            'node_history_traj': {}, 'node_is_target': {},
            'node_lanes': {}, 'node_id_map': {}
        }
        
        node_offset = 0
        edge_offset = 0
        
        for batch_data in all_results:
            old_to_new = {}
            for old_key, old_id in batch_data['node_id_map'].items():
                new_id = old_id + node_offset
                old_to_new[old_id] = new_id
                merged['node_id_map'][old_key] = new_id
            
            for edge in batch_data['edges']:
                merged['edges'].append({
                    'u': old_to_new[edge['u']],
                    'i': old_to_new[edge['i']],
                    'ts': edge['ts'],
                    'label': edge['label'],
                    'idx': edge['idx'] + edge_offset
                })
            
            merged['edge_feats'].extend(batch_data['edge_feats'])
            
            for (old_nid, ts), traj in batch_data['node_future_traj'].items():
                merged['node_future_traj'][(old_to_new[old_nid], ts)] = traj
                
            for (old_nid, ts), hist in batch_data['node_history_traj'].items():
                merged['node_history_traj'][(old_to_new[old_nid], ts)] = hist
            
            for (old_nid, ts), state in batch_data['node_current_state'].items():
                merged['node_current_state'][(old_to_new[old_nid], ts)] = state
            
            for (old_nid, ts), is_target in batch_data['node_is_target'].items():
                merged['node_is_target'][(old_to_new[old_nid], ts)] = is_target
                
            for (old_nid, ts), lanes in batch_data['node_lanes'].items():
                merged['node_lanes'][(old_to_new[old_nid], ts)] = lanes
            
            node_offset += batch_data['num_nodes']
            edge_offset += len(batch_data['edges'])
        
        merged['num_nodes'] = node_offset
        
        target_count = sum(1 for v in merged['node_is_target'].values() if v)
        print(f"{split_name} merged: {merged['num_nodes']} nodes, {len(merged['edges'])} edges, {len(merged['node_future_traj'])} trajectories, {target_count} TARGET_AGENTs")
        return merged
    
    train_data = process_batched(my_train_files, 'train', coop_train_dir)
    converter.save_split(train_data, 'train', skip_lanes=False)
    
    val_data = process_batched(my_val_files, 'val', coop_train_dir)
    converter.save_split(val_data, 'val', skip_lanes=False)
    
    test_data = process_batched(all_test_files, 'test', coop_test_dir)
    converter.save_split(test_data, 'test', skip_lanes=False)
    
    split_info = {
        'train_scenes': [f.stem for f in my_train_files],
        'val_scenes': [f.stem for f in my_val_files],
        'test_scenes': [f.stem for f in all_test_files],
        'train_size': len(my_train_files),
        'val_size': len(my_val_files),
        'test_size': len(all_test_files),
        'data_source': 'cooperative-trajectories',
        'graph_style': 'v2xg-cig-directed-dst-centric',
        'edge_features': [
            'distance',
            'angle_in_dst_frame',
            'src_vx_local', 'src_vy_local',
            'dst_vx_local', 'dst_vy_local',
            'rel_vx', 'rel_vy',
            'rel_heading_cos', 'rel_heading_sin',
            'normalized_distance',
            'rel_speed',
            'same_direction_flag',
            'same_type_flag'
        ]
    }
    with open(converter.output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print("\nDone")