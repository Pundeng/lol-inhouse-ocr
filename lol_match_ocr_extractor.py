# LOL Match OCR Extractor
# Summary, Vision, Damage images → match_data.csv + match_team_stats.csv (auto-aligned by gold values)

import cv2
import numpy as np
import easyocr
import pandas as pd
from datetime import datetime
from scipy.optimize import linear_sum_assignment

# --- CONFIG: match code ---
match_code = "20250101-1"
match_date = datetime.strptime(match_code.split("-")[0], "%Y%m%d").strftime("%Y-%m-%d")

# --- File paths ---
summary_path = f"{match_code}_summary.PNG"
damage_path = f"{match_code}_damage.PNG"
vision_path = f"{match_code}_vision.PNG"

# --- Load images ---
summary_img = cv2.imread(summary_path)
damage_img = cv2.imread(damage_path)
vision_img = cv2.imread(vision_path)

reader = easyocr.Reader(['en', 'ko'], gpu=False)

# --- Bounding boxes (replace with actual data if needed) ---
player_name_boxes = [(143,270,300,299), (142,307,304,335), (144,343,293,367), (143,376,296,406), (144,411,293,439), (145,489,294,514), (142,525,295,556), (143,559,291,588), (144,595,291,620), (144,626,275,661)]
kda_boxes = [(526,273,634,301), (525,307,628,335), (529,343,630,370), (524,379,627,406), (525,414,625,441), (522,490,628,517), (525,525,630,552), (520,559,633,589), (523,590,632,622), (525,625,627,655)]
cs_boxes = [(659,275,699,301), (659,307,699,335), (658,343,703,371), (658,376,700,406), (660,412,699,439), (658,490,700,518), (658,523,698,552), (657,557,699,587), (662,596,695,619), (660,628,699,654)]
gold_boxes = [(723,275,785,300), (722,306,787,336), (724,343,773,369), (725,377,784,406), (723,413,781,440), (722,489,783,515), (724,526,782,550), (724,560,781,583), (723,594,782,621), (724,629,786,656)]
vision_gold_boxes = [(292,454,355,481), (367,453,433,482), (439,455,503,482), (514,456,573,483), (585,456,649,483), (664,455,724,479), (735,454,801,482), (807,455,874,482), (882,455,949,481), (960,456,1022,479)]

# --- Damage boxes ---
damage_boxes = {
    "damage_dealt": [(295,270,353,295), (371,271,424,295), (442,272,500,297), (516,271,575,297), (587,270,646,296),
                      (665,271,719,292), (737,269,796,292), (817,272,872,297), (883,268,945,296), (962,271,1014,295)],
    "tower_damage": [(296,504,346,530), (373,503,425,529), (445,503,500,532), (518,503,574,532), (583,501,649,529),
                      (660,502,725,530), (734,501,800,529), (810,504,869,529), (887,504,942,529), (957,502,1017,529)],
    "healing": [(296,599,347,619), (371,599,422,622), (441,599,500,623), (517,598,574,626), (585,598,651,625),
                (665,599,724,623), (732,598,795,628), (810,598,868,623), (884,598,942,623), (959,598,1020,625)],
    "damage_taken": [(296,623,350,652), (368,624,427,652), (440,623,497,650), (517,625,576,650), (586,623,649,652),
                     (663,625,724,648), (734,624,797,651), (812,623,871,651), (882,623,944,652), (958,625,1020,649)]
}

# --- Vision boxes ---
vision_boxes = {
    "vision_score": [(302,308,341,331), (376,310,417,333), (450,308,488,331), (522,311,570,334), (588,333,648,358),
                      (674,306,715,334), (742,310,790,335), (812,307,860,334), (889,307,938,337), (969,310,1008,337)],
    "wards_placed": [(305,333,337,360), (375,334,418,359), (452,337,488,359), (525,334,560,360), (599,337,635,360),
                     (673,337,716,358), (744,333,787,359), (821,334,860,361), (891,333,938,360), (965,333,1006,362)],
    "wards_destroyed": [(303,358,341,382), (376,359,422,387), (448,359,494,385), (518,358,566,387), (598,360,638,386),
                        (673,362,715,387), (746,360,789,387), (824,360,863,386), (896,359,933,387), (972,361,1010,386)]
}

# --- Objective boxes ---
team_stat_boxes = {
    1: (854, 379, 1031, 404),
    2: (857, 597, 1031, 623)
}

# --- OCR extraction helper ---
def extract_text(img, box):
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]

    # === image adjusting ===
    crop = cv2.resize(crop, None, fx=2, fy=2)  # zoom
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)  # black and white
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # binarization

    result = reader.readtext(crop, detail=0)
    if not result:
        return ""
    
    text = result[0]
    cleaned = (
        text.replace(",", "")
            .replace(" ", "")
            .replace(".", "")
            .replace("O", "0")
            .replace("o", "0")
            .replace("I", "1")
            .replace("l", "1")
            .replace("S", "5")
            .replace("B", "8")
    )
    return cleaned

# --- Summary Extraction ---
summary_data = []
summary_gold = []
for i in range(10):
    name = extract_text(summary_img, player_name_boxes[i])
    kda = extract_text(summary_img, kda_boxes[i])
    cs = extract_text(summary_img, cs_boxes[i])
    gold = extract_text(summary_img, gold_boxes[i])
    try:
        kills, deaths, assists = map(int, kda.split("/"))
    except:
        kills, deaths, assists = (0, 0, 0)
    summary_gold.append(int(gold) if gold.isdigit() else 0)
    summary_data.append({"player_name": name, "kills": kills, "deaths": deaths, "assists": assists, "cs": int(cs) if cs.isdigit() else 0, "gold": int(gold) if gold.isdigit() else 0})

# --- Vision gold for matching ---
vision_gold = [int(extract_text(vision_img, box)) for box in vision_gold_boxes]

# --- Gold matching ---
def match_by_gold(summary_golds, other_golds):
    cost = np.abs(np.array(summary_golds)[:, None] - np.array(other_golds)[None, :])
    _, idx = linear_sum_assignment(cost)
    return idx

vision_idx_map = match_by_gold(summary_gold, vision_gold)
print("Reordering index:", vision_idx_map)

# --- Damage field extraction & reordering ---
damage_data = {}
for field, boxes in damage_boxes.items():
    values = []
    for box in boxes:
        val = extract_text(damage_img, box)
        try:
            values.append(int(val))
        except:
            values.append(0)
    reordered = [values[i] for i in vision_idx_map]  # reorder to match summary
    damage_data[field] = reordered


# --- Vision field extraction & reordering ---
vision_data = {}
for field, boxes in vision_boxes.items():
    values = []
    for box in boxes:
        val = extract_text(vision_img, box)
        try:
            values.append(int(val))
        except:
            values.append(0)
    reordered = [values[i] for i in vision_idx_map]  # reorder to match summary
    vision_data[field] = reordered


# --- Team stats extraction ---
team_stats = []
for team in [1, 2]:
    raw = extract_text(summary_img, team_stat_boxes[team])
    numbers = [int(s) for s in raw if s.isdigit()]
    if len(numbers) < 6:
        numbers += [0] * (6 - len(numbers))
    turrets, _, barons, dragons, heralds, voidgrubs = numbers[:6]  # skip inhibitor (2nd number)
    team_stats.append({
        "match_code": match_code,
        "team": team,
        "turrets_destroyed": turrets,
        "barons": barons,
        "dragons": dragons,
        "heralds": heralds,
        "voidgrubs": voidgrubs,
        "victory": 1 if (team == 2) else 0
    })

# --- Combine all into match_data ---
positions = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
champions = ['챔피언', '챔피언', '챔피언', '챔피언', '챔피언', '챔피언', '챔피언', '챔피언', '챔피언', '챔피언']
match_data = []
for i in range(10):
    match_data.append({
        "match_date": match_date,
        "match_code": match_code,
        "team": 1 if i < 5 else 2,
        "player_name": summary_data[i]["player_name"],
        "position": positions[i],
        "champion": champions[i],
        "kills": summary_data[i]["kills"],
        "deaths": summary_data[i]["deaths"],
        "assists": summary_data[i]["assists"],
        "cs": summary_data[i]["cs"],
        "gold": summary_data[i]["gold"],
        "first_blood": 0,
        "pentakill": 0,
        "damage_dealt": damage_data["damage_dealt"][i],
        "tower_damage": damage_data["tower_damage"][i],
        "healing": damage_data["healing"][i],
        "damage_received": damage_data["damage_taken"][i],
        "vision_score": vision_data["vision_score"][i],
        "wards_placed": vision_data["wards_placed"][i],
        "wards_destroyed": vision_data["wards_destroyed"][i],
        "victory": 1 if (i >= 5) else 0,
        "game_duration": 0
    })

# --- Save to CSV ---
pd.DataFrame(match_data).to_csv(f"{match_code}_match_data.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(team_stats).to_csv(f"{match_code}_match_team_stats.csv", index=False, encoding="utf-8-sig")

print("CSV convertion complete!")
