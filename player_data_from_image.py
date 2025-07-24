import cv2 as cv
import numpy as np
import easyocr
import pandas as pd

# NOTE: For whatever reason, if numpy is unavailable after installing easyocr run this command
# pip install "numpy<2"

match_code = "20250519-3"
match_date = f"{match_code.split('-')[0][:4]}-{match_code.split('-')[0][4:6]}-{match_code.split('-')[0][6:]}"

def match_template(source_path, template_path):
    img = cv.imread(source_path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"

    template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
    assert template is not None, "file could not be read, check with os.path.exists()"

    w, h = template.shape[::-1]

    # Apply template Matching
    res = cv.matchTemplate(img, template, getattr(cv, "TM_CCOEFF_NORMED"))
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    x1, y1 = max_loc
    x2, y2 = x1 + w, y1 + h

    # cv.imshow("Matched Image", img[y1:y2, x1:x2])
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return img[y1:y2, x1:x2]


def process_image(source, dilate=True):
    img = source

    # Strong blur to model the background
    blurred = cv.GaussianBlur(img, (51, 51), 0)

    # Subtract blurred background from original
    img = cv.subtract(img, blurred)

    _, img = cv.threshold(img, 20, 255, cv.THRESH_TOZERO)

    # Optional: normalize contrast
    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)

    img = cv.resize(img, None, fx=3.0, fy=3.0, interpolation=cv.INTER_CUBIC)

    # Apply sharpening kernel before thresholding
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv.filter2D(img, -1, kernel)

    if dilate:
        kernel = np.ones((2, 2), np.uint8)
        img = cv.dilate(img, kernel, iterations=1)

    # cv.imshow("New Image", img)
    # cv.waitKey()

    return img


def extract_text(img):
    reader = easyocr.Reader(["en", "ko"])  # English + Korean
    # results = reader.readtext(img, detail=1)
    results = reader.readtext(img, detail=1, contrast_ths=0.05, adjust_contrast=0.7)

    texts, y, i = [], None, -1

    # Group texts in similar y positions together
    for bbox, text, confidence in results:
        if y is None or bbox[0][1] > y + 50:
            # Start a new row
            texts.append(text)
            y = bbox[0][1]
            i += 1
        else:
            # join the current text to the row
            texts[i] += f" {text}"

        # print(f"Detected: '{text}' (confidence: {confidence:.2f}) at {bbox}")

    # [print(text) for text in texts]

    return texts

def create_player_dataframe(summary_data, damage_data, vision_data, match_code, match_date):
    num_players = 10
    player_rows = []

    for i in range(num_players):
        row = {
            "match_code": match_code,
            "match_date": match_date,
            "player_name": summary_data.get("player_names", [""]*num_players)[i],
            "position": summary_data.get("positions", [""]*num_players)[i],
            "champion": summary_data.get("champions", [""]*num_players)[i],
            "kills": summary_data.get("kills", [0]*num_players)[i],
            "deaths": summary_data.get("deaths", [0]*num_players)[i],
            "assists": summary_data.get("assists", [0]*num_players)[i],
            "cs": summary_data.get("cs", [0]*num_players)[i],
            "gold": summary_data.get("gold", [0]*num_players)[i],
            "total_damage": damage_data.get("total_damage", [0]*num_players)[i],
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
    for key, line in zip(vision_data.keys(), vision[1:4]):
        vision_data[key] = list(map(int, line.split()[2:]))

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
            stats = {
                "team": i // 2 + 1,
                "turrets_destroyed": parts[0],
                "barons": parts[2],
                "dragons": parts[3],
                "heralds": parts[4],
                "voidgrubs": parts[5],
                "victory": 1 if parts[0] > (objectives[i + 2].split()[0] if i + 2 < len(objectives) else -1) else 0
            }
            team_stats.append(stats)
        except Exception as e:
            print(f"[ParseError] team objective line: {line}, err: {e}")

    print(summary_data)
    print(vision_data)
    print(damage_data)
    print(team_stats)

    return summary_data, damage_data, vision_data, team_stats

    # print(objectives)


def main():
    # source = f"../lol_inhouse_images/{match_code}_summary.PNG"
    # template = "image_template/summary_players.PNG"

    # img = match_template(source, template)
    # img = process_image(img, False)  # image dilation breaks player name a bit too much
    # player_names = extract_text(img)

    # template = "image_template/summary_player_details.PNG"

    # img = match_template(source, template)
    # img = process_image(img)
    # kda_cs_gold = extract_text(img)

    # template = "image_template/summary_objectives.PNG"

    # img = match_template(source, template)
    # img = process_image(img)
    # objectives = extract_text(img)

    # source = f"../lol_inhouse_images/{match_code}_vision.PNG"
    # template = "image_template/vision_vision.PNG"

    # img = match_template(source, template)
    # img = process_image(img)
    # vision = extract_text(img)

    # template = "image_template/vision_gold.PNG"

    # img = match_template(source, template)
    # img = process_image(img)
    # vision_gold = extract_text(img)

    # source = f"../lol_inhouse_images/{match_code}_damage.PNG"
    # template = "image_template/damage_damage.PNG"

    # img = match_template(source, template)
    # img = process_image(img)
    # damages = extract_text(img)

    player_names = ['Pundeng', '이월길각 연구원', '네전공은나', 'Ity369', 'shosho ebi', '14 1 35 1 18 8', '헤음', '진자힘들어', 'Sony AZR V', 'Chefchoi', 'Happier']
    kda_cs_gold = ['9 / 1 / 13 193 12,553', '9 / 2 / 3 207 13,450', '4 / 1 / 5 151 8,763', '11 / 8 / 5 135 10,946', '2 / 2 / 13 31 8,476', '3 / 7 / 8 27 7,537', '4 / 7 / 3 155 9,838', '2 / 7 / 6 129 8,067', '4 / 9 / 1 121 8,883', '1 / 5 / 0 167 7,824']
    objectives = ['BANS + OBJECTIVES', '8 1 0 3 1 0', 'BANS + OBJECTIVES', '2 0 0 0 0 3']
    vision = ['VISION', 'Vision Score 60 12 12 13 16 44 18 12 14 13', 'Wards Placed 28 8 6 8 2 12 4 7 5 8', 'Wards Destroyed 6 1 2 2 3 5 1 0 3 2', 'Control Wards Purchased 7 0 0 0 0 3 3 0 3 4']
    damages = ['Total Damage to Champions 20,758 21,518 12,677 12,204 20,397 9,171 16,422 11,665 14,002 4,552', 'Physical Damage to Champions 945 18,659 1,953 9,506 19,877 1,799 13,240 10,343 344 1,582', 'Magic Damage to Champions 19,812 1,627 10,724 2,697 0 6,361 1,839 434 13,658 2,970', 'True Damage to Champions 0 1,232 0 0 520 1.009 1,343 832 0 0', 'Total Damage Dealt 47,835 105,068 119,769 140 097 249,320 25,719 197,034 79,030 71,011 101,346', 'Physical Damage Dealt 4,498 94,206 43,532 125,658 176,844 5,989 109,656 70,260 7,887 26,338', 'Magic Damage Dealt 41,279 3,397 58,962 13,906 0 13,727 20.434 775 62,488 65,337', 'True Damage Dealt 2,058 7,465 17,275 532 72,475 6,002 66,942 7995 636 9,670', 'Largest Critical Strike 0 386 6 0 294 0 7 0 0 179', 'Total Damage To Turrets 727 4,182 3,417 10,506 6,151 102 121 3,505 2,364 3,646', 'Total Damage To Objectives 9,830 12,976 5,593 11,993 36,533 447 8,658 3,755 2,364 4,298', 'DAMAGE TAKEN AND HEALED', 'Damage Healed 2,781 2,496 1,188 7,832 17,493 1,878 9,843 3,287 3,174 2,154', 'Damage Taken 13,116 17,366 8,397 16,570 27,939 17,740 26,852 18,140 27,496 18,896']
    vision_gold = ['Gold Earned 8,476 10,946 8,763 13,450 12,553 7537 9,838 8,067 8,883 7824']

    summary_data, vision_data, damage_data, team_stats = parse_lists(player_names, kda_cs_gold, objectives, vision, vision_gold, damages)

    # output_path = f"{match_code}_match.csv"
    # df = create_player_dataframe(summary_data, damage_data, vision_data, match_code, match_date)
    # df.to_csv(output_path, index=False, encoding="utf-8-sig")
    # print(f"CSV saved to: {output_path}")




if __name__ == "__main__":
    main()
