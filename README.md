# League of Legends In-House OCR

A Python-based OCR pipeline that extracts player and match statistics from League of Legends post-game screenshots and converts them into structured CSV data.

## Overview

Manually recording statistics from in-house League of Legends matches was slow and repetitive. This project automates that process by locating relevant regions of post-game screenshots, preprocessing the images for OCR, extracting text, parsing noisy OCR output, and exporting structured player and team statistics.

The project was originally built to process real match screenshots from our in-house games.

## How It Works

```text
Post-game screenshots
        ↓
OpenCV template matching
        ↓
Image preprocessing
        ↓
EasyOCR
        ↓
Parsing and validation
        ↓
Pandas DataFrames
        ↓
CSV output
```

The image-processing pipeline uses template matching to locate relevant UI regions instead of relying entirely on fixed coordinates. Images are then processed using techniques including background subtraction, contrast normalization, resizing, sharpening, and dilation before being passed to EasyOCR.

OCR output is parsed and normalized into structured statistics such as:

* Player names
* Kills / deaths / assists
* CS and gold
* Damage statistics
* Vision statistics
* Team objectives
* Match result
* Game duration

## Tech Stack

* Python
* OpenCV
* EasyOCR
* NumPy
* pandas

## Project Structure

```text
player_data_from_image.py   Main OCR and image-processing pipeline
data_utils.py               Parsing, normalization, and CSV generation
champ_ocr_test.py           Experimental champion-image recognition
image_template/             Templates used to locate UI regions
data_csv/                   Example generated player statistics
```

## Running the Project

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Place the required League of Legends screenshots in the configured image directory and run:

```bash
python player_data_from_image.py
```

The application processes available matches and writes structured results to CSV files.

> The project was developed around a specific set of League of Legends post-game screenshot layouts, so additional configuration may be required for different resolutions or UI versions.

## My Contribution

This project was developed collaboratively with one other developer.

I implemented the majority of the project, including the core OCR workflow, image-processing pipeline, data parsing, and integration of extracted statistics into structured output. My collaborator contributed to portions of the implementation and helped with development and testing.

I understand and can explain the complete pipeline, including the collaboratively written portions.

## Limitations

* The current template-matching approach is designed around specific League of Legends post-game UI layouts.
* OCR accuracy can vary with screenshot resolution and image quality.
* Some parsing rules assume a standard 5v5 match structure.
* Champion recognition remains experimental.

## Future Improvements

* Add automated tests for OCR parsing and preprocessing.
* Replace hard-coded paths with command-line arguments or configuration files.
* Improve resolution-independent template detection.
* Separate image processing, OCR, parsing, and application orchestration into dedicated modules.
* Add stronger validation and confidence handling for extracted statistics.

## Privacy

Player identifiers and personal information have been anonymized in the public version of this repository.
