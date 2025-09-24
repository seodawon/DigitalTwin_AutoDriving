import cv2
import numpy as np

# 영상 경로
video_path = '/home/rokey/turtlebot3_ws/src/output.mp4'

# 초기 추정 HSV 범위
yellow_lower_init = np.array([15, 80, 80])
yellow_upper_init = np.array([35, 255, 255])
white_lower_init = np.array([0, 0, 200])
white_upper_init = np.array([180, 60, 255])

# 누적 최소/최대값 초기화
yellow_min_all = np.array([180, 255, 255])
yellow_max_all = np.array([0, 0, 0])
white_min_all = np.array([180, 255, 255])
white_max_all = np.array([0, 0, 0])

# === 보정 함수 ===
def auto_exposure(frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def white_balance_simple(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg_a = np.average(lab[:, :, 1])
    avg_b = np.average(lab[:, :, 2])
    lab[:, :, 1] -= ((avg_a - 128) * (lab[:, :, 0] / 255.0) * 1.1)
    lab[:, :, 2] -= ((avg_b - 128) * (lab[:, :, 0] / 255.0) * 1.1)
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# === 영상 처리 시작 ===
cap = cv2.VideoCapture(0)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # 리사이즈
    scale_width = 640
    scale_ratio = scale_width / frame.shape[1]
    new_height = int(frame.shape[0] * scale_ratio)
    frame = cv2.resize(frame, (scale_width, new_height))

    # 노출 보정 + 화이트 밸런스
    frame = auto_exposure(frame)
    frame = white_balance_simple(frame)

    # HSV 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # === 노란색 추출 ===
    yellow_mask = cv2.inRange(hsv, yellow_lower_init, yellow_upper_init)
    yellow_pixels = hsv[yellow_mask > 0]
    if yellow_pixels.size > 0:
        yellow_min = np.min(yellow_pixels, axis=0)
        yellow_max = np.max(yellow_pixels, axis=0)
        yellow_min_all = np.minimum(yellow_min_all, yellow_min)
        yellow_max_all = np.maximum(yellow_max_all, yellow_max)

    # === 흰색 추출 ===
    white_mask = cv2.inRange(hsv, white_lower_init, white_upper_init)
    white_pixels = hsv[white_mask > 0]
    if white_pixels.size > 0:
        white_min = np.min(white_pixels, axis=0)
        white_max = np.max(white_pixels, axis=0)
        white_min_all = np.minimum(white_min_all, white_min)
        white_max_all = np.maximum(white_max_all, white_max)

    # 결과 출력
    print(f"Frame {frame_count}:")
    print(f"Yellow min HSV: {yellow_min_all}, max HSV: {yellow_max_all}")
    print(f"White min HSV: {white_min_all}, max HSV: {white_max_all}")

    # 마스크 시각화
    yellow_result = cv2.bitwise_and(frame, frame, mask=yellow_mask)
    white_result = cv2.bitwise_and(frame, frame, mask=white_mask)
    combined = np.hstack((yellow_result, white_result))
    cv2.imshow('Yellow(Left) and White(Right) Mask', combined)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 종료
        break

cap.release()
cv2.destroyAllWindows()

# === 최종 결과 ===
print(f"\n총 프레임 수: {frame_count}")
print("====== 노란색 HSV 범위 ======")
print("최소 HSV:", yellow_min_all)
print("최대 HSV:", yellow_max_all)
print("\n====== 흰색 HSV 범위 ======")
print("최소 HSV:", white_min_all)
print("최대 HSV:", white_max_all)
