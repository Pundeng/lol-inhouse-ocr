
import pytesseract
from PIL import Image
import csv
import os
import re
from datetime import datetime


# Tesseract 경로 설정
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 이미지 경로
image_paths = {
    "summary": "20250519-5_summary.png",
    "damage": "20250519-5_damage.png",
    "vision": "20250519-5_vision.png"
}

# OCR config
OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789"

# 좌표 (예시: 플레이어 10명 기준, 좌표는 직접 수정)
COORDS = {
    "summary": {
        "kills":      [(531, 276, 556, 296),
                        (525, 308, 555, 333),
                        (524, 344, 559, 366),
                        (527, 376, 556, 400),
                        (530, 414, 556, 436),
                        (525, 491, 555, 514),
                        (528, 523, 552, 549),
                        (527, 561, 550, 581),
                        (529, 593, 553, 617),
                        (528, 629, 553, 655)],
        "deaths":     [(564, 274, 589, 296),
                        (562, 310, 591, 332),
                        (561, 345, 591, 367),
                        (561, 381, 594, 401),
                        (562, 416, 593, 437),
                        (562, 492, 592, 513),
                        (563, 523, 594, 548),
                        (559, 564, 593, 581),
                        (562, 595, 591, 616),
                        (563, 632, 590, 653)],
        "assists":    [(597, 277, 626, 298),
                        (600, 310, 627, 333),
                        (598, 345, 624, 368),
                        (600, 379, 628, 403),
                        (601, 414, 626, 436),
                        (599, 492, 624, 511),
                        (601, 527, 628, 550),
                        (602, 561, 626, 583),
                        (598, 595, 626, 617),
                        (598, 631, 629, 654)],
        "cs":         [(659, 275, 699, 301),
                        (659, 307, 699, 335),
                        (658, 343, 703, 371),
                        (658, 376, 700, 406),
                        (660, 412, 699, 439),
                        (658, 490, 700, 518),
                        (658, 523, 698, 552),
                        (657, 557, 699, 587),
                        (662, 596, 695, 619),
                        (660, 628, 699, 654)],
        "gold":       [(723, 275, 785, 300),
                        (722, 306, 787, 336),
                        (724, 343, 773, 369),
                        (725, 377, 784, 406),
                        (723, 413, 781, 440),
                        (722, 489, 783, 515),
                        (724, 526, 782, 550),
                        (724, 560, 781, 583),
                        (723, 594, 782, 621),
                        (724, 629, 786, 656)] ,
    },
    "damage": {
        "damage_dealt": [(297, 272, 347, 292),
                        (373, 272, 422, 292),
                        (447, 272, 492, 290),
                        (521, 274, 570, 292),
                        (594, 272, 646, 293),
                        (666, 272, 718, 294),
                        (740, 272, 794, 292),
                        (815, 273, 865, 292),
                        (888, 273, 937, 294),
                        (960, 272, 1013, 292)],
        "damage_received":     [(300, 627, 346, 647),
                        (371, 628, 423, 649),
                        (446, 628, 494, 649),
                        (520, 627, 567, 649),
                        (591, 627, 644, 647),
                        (666, 626, 719, 648),
                        (739, 626, 792, 647),
                        (813, 627, 867, 648),
                        (888, 625, 942, 649),
                        (963, 626, 1014, 647)],
    },
    "vision": {
        "vision_score": [(306, 309, 336, 330),
                        (381, 312, 411, 331),
                        (458, 311, 481, 330),
                        (530, 312, 563, 332),
                        (605, 311, 634, 332),
                        (679, 310, 708, 333),
                        (750, 309, 782, 331),
                        (829, 311, 860, 332),
                        (898, 311, 933, 332),
                        (972, 312, 1001, 330)],
        "wards_placed": [(309, 337, 333, 356),
                        (382, 335, 409, 358),
                        (454, 337, 487, 358),
                        (531, 335, 558, 358),
                        (604, 337, 633, 356),
                        (678, 336, 710, 358),
                        (756, 335, 781, 356),
                        (823, 335, 857, 358),
                        (897, 336, 932, 359),
                        (973, 337, 1002, 357)]
    }
}

# 파일명에서 match_code, match_date 추출
basename = os.path.basename(image_paths["summary"])
match = re.search(r"(?P<date>\d{8})-(?P<code_num>\d+)", basename)

match_code = f"{match.group('date')}-{match.group('code_num')}" if match else "UNKNOWN"
match_date = match.group("date") if match else "00000000"
formatted_date = datetime.strptime(match_date, "%Y%m%d").strftime("%B %d, %Y")


# 결과 리스트 초기화
results = []
for i in range(10):
    results.append({
        "team": "Team1" if i < 5 else "Team2",
        "match_code": match_code,
        "match_date": formatted_date,
        "player_name": "",
        "position": "Top",
        "champion": "None",
        "objectives_score": 0,
        "victory": True
    })

# 이미지별 OCR 수행
for category, fields in COORDS.items():
    try:
        image = Image.open(image_paths[category])
    except FileNotFoundError:
        print(f"[오류] 이미지 파일이 존재하지 않습니다: {image_paths[category]}")
        continue

    for field, boxes in fields.items():
        for i, box in enumerate(boxes):
            crop = image.crop(box)
            text = pytesseract.image_to_string(crop, config=OCR_CONFIG, lang="eng").strip()
            cleaned = text.replace("\n", "").replace(" ", "")
            results[i][field] = cleaned if cleaned else "None"
            
# 콘솔 출력
for idx, r in enumerate(results):
    print(f"Player {idx+1}: {r}")
    
# CSV 저장
csv_filename = "playerwise_ocr_result.csv"
with open(csv_filename, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"[완료] 결과가 {csv_filename}에 저장되었습니다.")
