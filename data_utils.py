import os
import pandas as pd
import re
import difflib

nickname_map = {
    "Pundeng": "정윤재",
    "Happier": "허균",
    "광야의 정글러": "이승연",
    "ChefChoi": "최진석",
    "BbeunNa": "이수진",
    "man from nowhere": "장우주",
    "혜 윰": "문상휘",
    "fleur de peau": "문상휘",
    "진짜힘들어": "심시온",
    "네전공은나": "최광호",
    "lty369": "임태우",
    "shosho ebi": "이곤섭",
    "이렐킬각연구원": "김주헌",
    "상 체 파 괴 자": "김주헌",
    "Sony A7R V": "한종인",
    "SoLa": "김태형",
    "ClimbingIsFun": "이정웅",
    "crocaw": "최명진",
    "krim123456789": "임태윤",
    "dono jelly": "황승훈",
    "mushroom farmer": "김민준",
}

def parse_lists(player_names, kda_cs_gold, objectives, vision, vision_gold, damages, victory_time):
    """
    Parses raw OCR lines into structured dictionaries for summary, damage, vision data, and team stats.
    """
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
    player_names_mapped = map_player_names(player_names, nickname_map)

    for name, line in zip(player_names_mapped, kda_cs_gold):
        summary_data["player_name"].append(name)
        try:
            nums = [int(n.replace(",", "").replace(".", "")) for n in re.findall(r"\d[\d,.]*", line)]
            k, d, a, cs, gold = nums[:5] if len(nums) >= 5 else (0, 0, 0, 0, 0)
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
        try:
            print(line)
            matches = re.findall(r'\d[\d,]*', line)
            cleaned = [int(m.replace(',', '')) for m in matches]
            return cleaned[:10] + [0] * (10 - len(cleaned))
        except Exception as e:
            print(f"[ParseError] parse_dmg_line_four: {line}, err: {e}")
            return [0] * 10    
    
    def parse_dmg_line_two(line):
        try:
            print(line)
            matches = re.findall(r'\d[\d,]*', line)
            cleaned = [int(m.replace(',', '')) for m in matches]
            return cleaned[:10] + [0] * (10 - len(cleaned))
        except Exception as e:
            print(f"[ParseError] parse_dmg_line_two: {line}, err: {e}")
            return [0] * 10
        
    damage_data["damage_dealt"] = parse_dmg_line_four(damages[0])
    damage_data["tower_damage"] = parse_dmg_line_four(damages[9])
    damage_data["healing"] = parse_dmg_line_two(damages[-2])
    damage_data["damage_taken"] = parse_dmg_line_two(damages[-1])
    
    # objectives data
    team_stats = []
    team = 0
    for i, line in enumerate(objectives):
        if not re.fullmatch(r"[0-9\s]+", line.strip()):
            continue
        try:
            parts = list(map(int, line.strip().split()))
                
            stats = {
                "team": team,
                "turrets_destroyed": parts[0],
                "barons": parts[2],
                "dragons": parts[3],
                "heralds": parts[4],
                "voidgrubs": parts[5]
            }
            team_stats.append(stats)
            team += 1
        except Exception as e:
            print(f"[ParseError] team objective line: {line}, err: {e}")
    
    reordered_summary = reorder_by_vision_gold(summary_data, vision_gold)
    team1_victory, game_duration = parse_outcome_and_duration(victory_time)

    return reordered_summary, damage_data, vision_data, team_stats, team1_victory, game_duration
 
def map_player_names(player_names, nickname_map, cutoff=0.5):
    """
    Map player names to real names using string similarity matching.
    """
    mapped_names = []
    nickname_keys = list(nickname_map.keys())

    for name in player_names:
        match = difflib.get_close_matches(name, nickname_keys, n=1, cutoff=cutoff)
        if match:
            mapped_names.append(nickname_map[match[0]])
        else:
            mapped_names.append(name)  # fallback to original
    return mapped_names
    
def parse_outcome_and_duration(victory_time):
    """
    Extracts the match outcome (victory/defeat) and duration from the summary image lines.
    """
    try:
        team1_victory = 1 if victory_time[0].upper() == "VICTORY" else 0
        time_match = re.search(r"(\d{1,2})[:.,;](\d{2})", victory_time[1])
        
        if not time_match:
            raise ValueError("No valid time format found")
        
        minutes = int(time_match.group(1))
        seconds = int(time_match.group(2))
        game_duration = minutes + (1 if seconds >= 30 else 0)

        return team1_victory, game_duration
    
    except Exception as e:
        print(f"[ParseError] outcome/duration: {victory_time}, err: {e}")
        return 0, 0

def reorder_by_vision_gold(summary_data, vision_gold_line):
    """
    Reorders summary and damage data based on matching gold values from vision data to align player order.
    """
    if isinstance(vision_gold_line, list):
        vision_gold_line = vision_gold_line[1]

    cleaned = re.sub(r"[A-Za-z]", "", vision_gold_line).strip()
    gold_strs = cleaned.split()
    vision_gold = [int(g.replace(",", "").replace(".","")) for g in gold_strs]

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
    
def create_player_dataframe(summary_data, damage_data, vision_data, match_code, match_date, team1_victory, game_duration):
    """
    Combines parsed player data into a structured DataFrame ready for SQL export or CSV storage.
    """
    num_players = 10
    player_rows = []

    def get_list_or_default(d, key, default_value):
        lst = d.get(key, [])
        if len(lst) < num_players:
            lst += [default_value] * (num_players - len(lst))
        return lst[:num_players]

    player_names = get_list_or_default(summary_data, "player_name", "")
    positions = get_list_or_default(summary_data, "positions", "")
    champions = get_list_or_default(summary_data, "champions", "")
    kills = get_list_or_default(summary_data, "kills", 0)
    deaths = get_list_or_default(summary_data, "deaths", 0)
    assists = get_list_or_default(summary_data, "assists", 0)
    cs = get_list_or_default(summary_data, "cs", 0)
    gold = get_list_or_default(summary_data, "gold", 0)
    damage_dealt = get_list_or_default(damage_data, "damage_dealt", 0)
    tower_damage = get_list_or_default(damage_data, "tower_damage", 0)
    healing = get_list_or_default(damage_data, "healing", 0)
    damage_taken = get_list_or_default(damage_data, "damage_taken", 0)
    vision_score = get_list_or_default(vision_data, "vision_score", 0)
    wards_placed = get_list_or_default(vision_data, "wards_placed", 0)
    wards_destroyed = get_list_or_default(vision_data, "wards_destroyed", 0)
    control_wards_purchased = get_list_or_default(vision_data, "control_wards_purchased", 0)

    for i in range(num_players):
        team = 1 if i < 5 else 2
        victory = team1_victory if team == 1 else 1 - team1_victory

        row = {
            "match_code": match_code,
            "match_date": match_date,
            "team": team,
            "player_name": player_names[i],
            "position": positions[i],
            "champion": champions[i],
            "kills": kills[i],
            "deaths": deaths[i],
            "assists": assists[i],
            "pentakill": 0,
            "first_blood": 0,
            "cs": cs[i],
            "gold": gold[i],
            "damage_dealt": damage_dealt[i],
            "tower_damage": tower_damage[i],
            "healing": healing[i],
            "damage_taken": damage_taken[i],
            "vision_score": vision_score[i],
            "wards_placed": wards_placed[i],
            "wards_destroyed": wards_destroyed[i],
            "control_wards_purchased": control_wards_purchased[i],
            "victory": victory,
            "game_duration": game_duration,
        }
        player_rows.append(row)

    return pd.DataFrame(player_rows)

def save_to_csv(df, match_code, save_dir="data_csv"):
    """
    Saves the player DataFrame into a UTF-8-BOM encoded CSV file with proper naming based on match code.
    """
    ordered_columns = [
        "match_date", "match_code", "team", "player_name", "position", "champion",
        "kills", "deaths", "assists", "cs", "gold", "first_blood", "pentakill",
        "damage_dealt", "tower_damage", "healing", "damage_taken",
        "vision_score", "wards_placed", "wards_destroyed", "victory", "game_duration"
    ]
    
    os.makedirs(save_dir, exist_ok=True)
    csv_filename = f"{match_code}.csv"
    csv_path = os.path.join(save_dir, csv_filename)
    print("saved to", csv_path)

    df.to_csv(csv_path, index=False, encoding="utf-8-sig", columns=ordered_columns)
    return csv_path
