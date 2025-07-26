import cv2 as cv
import numpy as np
import easyocr
import pandas as pd
# import os
# import datetime
# import traceback

from data_utils import parse_lists, create_player_dataframe, save_to_csv

# NOTE: For whatever reason, if numpy is unavailable after installing easyocr run this command
# pip install "numpy<2"

def match_template(source_path, template_path):
    """
    Finds and crops the region in the source image that matches the given template using OpenCV.
    """
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
    """
    Applies preprocessing steps like grayscale, thresholding, and optional dilation to enhance OCR accuracy.
    """
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


def extract_all_images(images_with_templates):
    """
    Extracts and preprocesses a set of images given as (source, template) path tuples for OCR input.
    """
    reader = easyocr.Reader(["en", "ko"], gpu=True)
    
    total = len(images_with_templates)
    results = []
    
    for idx, (source_path, template_path) in enumerate(images_with_templates):
        progress_bar(template_path, idx + 1, total)

        img = match_template(source_path, template_path)
        img = process_image(img)
        text = extract_text_with_reader(img, reader)
        results.append(text)
        
    return results


def extract_text_with_reader(img, reader):
    """
    Uses EasyOCR to extract grouped text lines from a processed image with an existing OCR reader.
    """
    results = reader.readtext(img, detail=1, contrast_ths=0.05, adjust_contrast=0.7)

    texts, y, i = [], None, -1
    for bbox, text, confidence in results:
        if y is None or bbox[0][1] > y + 50:
            texts.append(text)
            y = bbox[0][1]
            i += 1
        else:
            texts[i] += f" {text}"
    return texts


def progress_bar(task_name, step, total_steps, bar_width=20):
    """
    Prints a dynamic CLI progress bar to visualize which image is currently being processed.
    """
    percent = int((step / total_steps) * 100) - 1
    filled = int(bar_width * step / total_steps)
    bar = "o" * filled + "-" * (bar_width - filled)
    print(f"{bar} {percent}%")
    print(f'Scanning "{task_name}"\n')
    

LOG_PATH = "process_log.txt"
FAIL_PATH = "failed_matches.txt"

def run_all_from_folder(image_folder="../lol_inhouse_images"):
    """
    Processes all match images in the specified folder, extracts data, and saves results to CSVs.
    """
    log("Scanning folder for match codes...")
    match_codes = set()

    for filename in os.listdir(image_folder):
        if filename.endswith("_summary.PNG"):
            match_code = filename.replace("_summary.PNG", "")
            match_codes.add(match_code)
        elif filename.endswith("_summary.png"):
            match_code = filename.replace("_summary.png", "")
            match_codes.add(match_code)

    match_codes = sorted(match_codes)
    log(f"Found {len(match_codes)} matches to process")

    for match_code in match_codes:
        try:
            log(f"Processing {match_code}...")
            main(match_code)  # 너의 main(match_code) 함수가 여기 있어야 함
            log(f"Finished {match_code}")
        except Exception as e:
            log(f"Error on {match_code}: {e}")
            log_failed(match_code, traceback.format_exc())


def log(message):
    """
    Writes a timestamped log message to both the console and a persistent 'run_log.txt' file.
    """
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    full_msg = f"{timestamp} {message}"
    print(full_msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")


def log_failed(match_code, err_msg=""):
    """
    Records the match code and optional error message to 'failed_matches.txt' if processing fails.
    """
    with open(FAIL_PATH, "a", encoding="utf-8") as f:
        f.write(f"{match_code} - {err_msg}\n")
   
        
def main(match_code):
    """
    Orchestrates the full flow of image extraction, parsing, and saving for a single match code.
    """
    
    match_date = f"{match_code.split('-')[0][:4]}-{match_code.split('-')[0][4:6]}-{match_code.split('-')[0][6:]}"
    images_to_ocr = [
        (f"../lol_inhouse_images/{match_code}_summary.png", "image_template/summary_players.PNG"),
        (f"../lol_inhouse_images/{match_code}_summary.png", "image_template/summary_player_details.PNG"),
        (f"../lol_inhouse_images/{match_code}_summary.png", "image_template/summary_objectives.PNG"),
        (f"../lol_inhouse_images/{match_code}_summary.png", "image_template/summary_time.PNG"),
        (f"../lol_inhouse_images/{match_code}_vision.png", "image_template/vision_vision.PNG"),
        (f"../lol_inhouse_images/{match_code}_vision.png", "image_template/vision_gold.PNG"),
        (f"../lol_inhouse_images/{match_code}_damage.png", "image_template/damage_damage.PNG"),
    ]

    data = extract_all_images(images_to_ocr)
    
    player_names = data[0]
    kda_cs_gold = data[1]
    objectives = data[2]
    victory_time = data[3]
    vision = data[4]
    vision_gold = data[5]
    damages = data[6]

    print(player_names)
    print(kda_cs_gold)
    print(objectives)
    print(victory_time)
    print(vision)
    print(damages)
    print(vision_gold)

    # for i in range(len(damages)):
    #     print(i, " ", damages[i])

    summary_data, damage_data, vision_data, team_data, team1_victory, game_duration = parse_lists(
                                                                                    player_names, kda_cs_gold, 
                                                                                    objectives, vision, 
                                                                                    vision_gold, damages, 
                                                                                    victory_time)

    # print("------------------------------------")
    # print(summary_data)
    # print(damage_data)
    # print(vision_data)
    # print(team_data)

    # vision_data["control_wards_purchased"] = [1, 6, 0, 4, 3, 1, 0, 3, 0, 0]

    df = create_player_dataframe(summary_data, damage_data, vision_data, match_code, match_date, team1_victory, game_duration)

    save_to_csv(df, match_code)
    
    # images_to_ocr = [
    #     (f"../lol_inhouse_images/{match_code}_summary.png", "image_template/summary_time.PNG"),
    # ]

    # data = extract_all_images(images_to_ocr)
    # print(data)


if __name__ == "__main__":
    run_all_from_folder()
    # main("20250725-3")
