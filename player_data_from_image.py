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
    return (x1, y1, x2, y2)


def process_image(source, rect=None):
    if rect is not None:
        img = cv.imread(source)
        assert img is not None, "file could not be read, check with os.path.exists()"

        x1, y1, x2, y2 = rect
        img = img[y1:y2, x1:x2]
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        img = source

    alpha, beta = 1.5, 0.0

    img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

    img = cv.resize(img, None, fx=3.0, fy=3.0, interpolation=cv.INTER_CUBIC)

    cv.imshow("New Image", img)
    cv.waitKey()

    return img


def extract_text(img):
    reader = easyocr.Reader(["en", "ko"])  # English + Korean
    results = reader.readtext(img, detail=1)

    for bbox, text, confidence in results:
        print(f"Detected: '{text}' (confidence: {confidence:.2f}) at {bbox}")


def main():
    source = "test_summary.PNG"
    template = "image_template/summary_player_details.PNG"

    img = match_template(source, template)
    # img = process_image(source, img)
    img = process_image(img)

    extract_text(img)


if __name__ == "__main__":
    # test_ocr_from_image()
    # test_template_matching()

    main()
