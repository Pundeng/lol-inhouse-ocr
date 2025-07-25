import os
import pandas as pd

def parse_lists(player_names, kda_cs_gold, objectives, vision, vision_gold, damages):
    # summary data
    summary_data = {
        "player_name": [],
        "kills": [],
        "deaths": [],
        "assists": [],
        "cs": [],
        "gold": []
    }
    player_names.pop(5)
    for name, line in zip(player_names, kda_cs_gold):
        summary_data["player_name"].append(name)
        try:
            line = line.split(" ")
            k = int(line[0])
            d = int(line[2])
            a = int(line[4])
            cs = int(line[5])
            gold = int(line[6].replace(",", ""))
        except Exception as e:
            print(f"[ParseError] summary line: {line}, err: {e}")
            k, d, a, cs, gold = 0, 0, 0, 0, 0
        summary_data["kills"].append(k)
        summary_data["deaths"].append(d)
        summary_data["assists"].append(a)
        summary_data["cs"].append(cs)
        summary_data["gold"].append(gold)
    
    # vision data
    vision_data = {
        "vision_score": [],
        "wards_placed": [],
        "wards_destroyed": [],
        "control_wards_purchased": []
    }
    
    for key, line in zip(vision_data.keys(), vision[1:5]):
        skip = 3 if key == "control_wards_purchased" else 2
        try:
            vision_data[key] = list(map(int, line.split()[skip:]))
        except ValueError:
            print(f"[ParseError] vision line: {line}")
            vision_data[key] = [0] * 10

    # damage data
    damage_data = {
        "damage_dealt": [],
        "tower_damage": [],
        "healing": [],
        "damage_taken": [],
    }

    def parse_dmg_line_four(line):
        return [int(x.replace(",", "").replace(".", "")) for x in line.split()[4:]]
    
    def parse_dmg_line_two(line):
        return [int(x.replace(",", "").replace(".", "")) for x in line.split()[2:]]

    damage_data["damage_dealt"] = parse_dmg_line_four(damages[0])
    damage_data["tower_damage"] = parse_dmg_line_four(damages[9])
    damage_data["healing"] = parse_dmg_line_two(damages[-2])
    damage_data["damage_taken"] = parse_dmg_line_two(damages[-1])
    
    # objectives data
    team_stats = []
    for i, line in enumerate(objectives):
        if line.startswith("BANS"):
            continue
        try:
            parts = list(map(int, line.split()))
            
            if i + 2 < len(objectives):
                team1_score = int(parts[0])
                team2_score = int(objectives[i + 2].split()[0])
                victory = 1 if team1_score > team2_score else 0
            else:
                victory = 1
                
            stats = {
                "team": i // 2 + 1,
                "turrets_destroyed": parts[0],
                "barons": parts[2],
                "dragons": parts[3],
                "heralds": parts[4],
                "voidgrubs": parts[5],
                "victory": victory
            }
            team_stats.append(stats)
        except Exception as e:
            print(f"[ParseError] team objective line: {line}, err: {e}")
    
    reordered_summary = reorder_by_vision_gold(summary_data, vision_gold)

    return reordered_summary, damage_data, vision_data, team_stats
    
    
def reorder_by_vision_gold(summary_data, vision_gold_line):
    if isinstance(vision_gold_line, list):
        vision_gold_line = vision_gold_line[0]

    gold_strs = vision_gold_line.replace("Gold Earned", "").strip().split()
    vision_gold = [int(g.replace(",", "")) for g in gold_strs]

    summary_gold = summary_data["gold"]

    summary_used = set()
    reorder_map = []

    for vg in vision_gold:
        min_diff = float('inf')
        best_idx = -1
        for i, sg in enumerate(summary_gold):
            if i in summary_used:
                continue
            diff = abs(vg - sg)
            if diff < min_diff:
                min_diff = diff
                best_idx = i
        reorder_map.append(best_idx)
        summary_used.add(best_idx)

    def reorder_dict(d):
        return {k: [d[k][i] for i in reorder_map] for k in d}

    reordered_summary = reorder_dict(summary_data)

    return reordered_summary
    
    
def create_player_dataframe(summary_data, damage_data, vision_data, match_code, match_date):
    num_players = 10
    player_rows = []

    for i in range(num_players):
        row = {
            "match_code": match_code,
            "match_date": match_date,
            "player_name": summary_data.get("player_name", [""]*num_players)[i],
            "position": summary_data.get("positions", [""]*num_players)[i],
            "champion": summary_data.get("champions", [""]*num_players)[i],
            "kills": summary_data.get("kills", [0]*num_players)[i],
            "deaths": summary_data.get("deaths", [0]*num_players)[i],
            "assists": summary_data.get("assists", [0]*num_players)[i],
            "cs": summary_data.get("cs", [0]*num_players)[i],
            "gold": summary_data.get("gold", [0]*num_players)[i],
            "damage_dealt": damage_data.get("damage_dealt", [0]*num_players)[i],
            "tower_damage": damage_data.get("tower_damage", [0]*num_players)[i],
            "healing": damage_data.get("healing", [0]*num_players)[i],
            "damage_taken": damage_data.get("damage_taken", [0]*num_players)[i],
            "vision_score": vision_data.get("vision_score", [0]*num_players)[i],
            "wards_placed": vision_data.get("wards_placed", [0]*num_players)[i],
            "wards_destroyed": vision_data.get("wards_destroyed", [0]*num_players)[i],
            "control_wards_purchased": vision_data.get("control_wards_purchased", [0]*num_players)[i]
        }
        player_rows.append(row)
    
    return pd.DataFrame(player_rows)


def save_to_csv(df, match_code, save_dir="data_csv"):
    os.makedirs(save_dir, exist_ok=True)
    csv_filename = f"{match_code}.csv"
    csv_path = os.path.join(save_dir, csv_filename)

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path
