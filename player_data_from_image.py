import cv2 as cv
import numpy as np
import easyocr

# NOTE: For whatever reason, if numpy is unavailable after installing easyocr run this command
# pip install "numpy<2"


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

    cv.imshow("New Image", img)
    cv.waitKey()

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

    [print(text) for text in texts]

    return texts


def main():
    source = "../lol_inhouse_images/20250519-3_summary.PNG"
    template = "image_template/summary_players.PNG"

    img = match_template(source, template)
    img = process_image(img, False)  # image dilation breaks player name a bit too much
    extract_text(img)

    template = "image_template/summary_player_details.PNG"

    img = match_template(source, template)
    img = process_image(img)
    extract_text(img)

    template = "image_template/summary_objectives.PNG"

    img = match_template(source, template)
    img = process_image(img)
    extract_text(img)

    source = "../lol_inhouse_images/20250519-3_vision.PNG"
    template = "image_template/vision_vision.PNG"

    img = match_template(source, template)
    img = process_image(img)
    extract_text(img)

    source = "../lol_inhouse_images/20250519-3_damage.PNG"
    template = "image_template/damage_damage.PNG"

    img = match_template(source, template)
    img = process_image(img)
    extract_text(img)


if __name__ == "__main__":
    main()
