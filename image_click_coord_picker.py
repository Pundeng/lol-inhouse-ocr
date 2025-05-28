
import cv2

# 이미지 경로 (여기에 사용할 파일명 넣어주세요)
IMAGE_PATH = "test_vision.png"

coords = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coords.append((x, y))
        if len(coords) % 2 == 0:
            print(f"{coords[-2]} → {coords[-1]}\n")

# 이미지 열기
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"이미지 파일을 찾을 수 없습니다: {IMAGE_PATH}")
else:
    print("이미지에서 좌상단과 우하단 순서로 두 번 클릭하세요 (좌표 출력됨)\n")
    cv2.imshow("Click to Get Bounding Box", img)
    cv2.setMouseCallback("Click to Get Bounding Box", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
