DDRAGON_URL = "https://ddragon.leagueoflegends.com/cdn/"


def identify_champion(source, template):
    source_img = source
    template_img = cv.imdecode(template, 0)

    # resize to match source and template champion portrait size
    # source_img = cv.resize(source, None, fx=2.0, fy=2.0, interpolation=cv.INTER_CUBIC)
    template_img = cv.resize(
        template_img, None, fx=0.3, fy=0.3, interpolation=cv.INTER_CUBIC
    )

    w, h = template_img.shape[::-1]
    offset = 8
    template_img = template_img[
        offset : h - offset, offset : w - offset
    ]  # crop around the template image

    cv.imshow("source", source_img)
    cv.waitKey()

    cv.imshow("template", template_img)
    cv.waitKey()

    # Apply template Matching
    res = cv.matchTemplate(source_img, template_img, getattr(cv, "TM_CCOEFF_NORMED"))
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    x1, y1 = max_loc
    x2, y2 = x1 + w, y1 + h

    cv.imshow("Matched Image", source_img[y1:y2, x1:x2])
    cv.waitKey(0)

    x1, y1 = min_loc
    x2, y2 = x1 + w, y1 + h

    cv.imshow("Matched Image", source_img[y1:y2, x1:x2])
    cv.waitKey(0)
    cv.destroyAllWindows()

    return


def find_champions(summary, vision):
    champions_in_play_template1 = "image_template/summary_champions.PNG"
    champions_in_play_template2 = "image_template/vision_champions.PNG"
    champions_banned_template = "image_template/summary_objectives.PNG"

    # get part of the image that matches the template
    champs_in_play_img1 = match_template(summary, champions_in_play_template1)
    champs_in_play_img2 = match_template(vision, champions_in_play_template2)
    champs_banned_img = match_template(summary, champions_banned_template)

    # get latest version of ddragon api
    versions = requests.get(
        "https://ddragon.leagueoflegends.com/api/versions.json"
    ).json()
    latest = versions[0]

    champ_data = requests.get(f"{DDRAGON_URL}{latest}/data/en_US/champion.json").json()

    # save champion name and its associated image string
    champion_img_urls = {
        champ.lower(): data["image"]["full"]
        for champ, data in champ_data["data"].items()
    }

    # get image of champion using the image string
    req = urllib.request.urlopen(
        f"{DDRAGON_URL}{latest}/img/champion/{champion_img_urls["zeri"]}"
    )
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)

    test = identify_champion(champs_in_play_img1, arr)

    return
