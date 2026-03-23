# logic.py
import cv2
import numpy as np
from google import genai
from google.genai import types

def get_gemini_traced_image(api_key, image_bytes, prompt, model_id, gap_fill_kernel=0):
    """Gemini APIを呼び出して、劣化箇所に指定色を描画した画像データを取得する"""
    # ⚠️ (ここは以前と変更なし)
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_id,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                ],
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                raw_bytes = part.inline_data.data
                composite_bytes = _composite_red_on_original(image_bytes, raw_bytes, gap_fill_kernel)
                return composite_bytes, raw_bytes
        return None, None
    except Exception as e:
        raise Exception(f"APIエラー: {e}")

def _extract_color_mask(bgr_img, target_rgb=(255, 0, 0), saturation_threshold=150):
    """HSV色空間で指定色のピクセルを検出する。色相はtarget_rgbから自動計算。"""
    # target_rgb → BGRに変換してOpenCVでHSV化
    target_bgr = np.uint8([[[target_rgb[2], target_rgb[1], target_rgb[0]]]])
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0][0]
    h = int(target_hsv[0])  # OpenCVの色相: 0〜180

    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    tol = 10
    h_low  = (h - tol) % 180
    h_high = (h + tol) % 180

    if h_low <= h_high:
        mask = cv2.inRange(hsv, np.array([h_low,  saturation_threshold, 80]),
                                np.array([h_high, 255, 255]))
    else:  # 色相が0/180をまたぐ場合（赤など）
        mask1 = cv2.inRange(hsv, np.array([h_low, saturation_threshold, 80]),
                                 np.array([180,   255, 255]))
        mask2 = cv2.inRange(hsv, np.array([0,     saturation_threshold, 80]),
                                 np.array([h_high,255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
    return mask

def _composite_red_on_original(original_bytes, traced_bytes, gap_fill_kernel=0, saturation_threshold=150, target_rgb=(255, 0, 0)):
    orig = cv2.imdecode(np.frombuffer(original_bytes, np.uint8), cv2.IMREAD_COLOR)
    traced = cv2.imdecode(np.frombuffer(traced_bytes, np.uint8), cv2.IMREAD_COLOR)
    if traced.shape[:2] != orig.shape[:2]:
        traced = cv2.resize(traced, (orig.shape[1], orig.shape[0]))

    color_mask = _extract_color_mask(traced, target_rgb, saturation_threshold)

    # ギャップ埋め・線つなぎ: Closing（膨張→収縮）で途切れを補完
    if gap_fill_kernel > 1:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_fill_kernel, gap_fill_kernel))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_close)

    result = orig.copy()
    # 検出色をそのままキャンバスに描画（BGR順）
    result[color_mask > 0] = [target_rgb[2], target_rgb[1], target_rgb[0]]
    _, enc = cv2.imencode(".png", result)
    return enc.tobytes()


def reprocess_from_raw(image_bytes, raw_bytes, gap_fill_kernel=0, saturation_threshold=150, target_rgb=(255, 0, 0)):
    """raw画像からパラメータを変えて再処理する（Gemini再呼び出し不要）"""
    return _composite_red_on_original(image_bytes, raw_bytes, gap_fill_kernel, saturation_threshold, target_rgb)


def process_yolo_segmentation(traced_bytes, original_width, original_height, min_area_px=10, exclusion_rects=None, class_id=0, saturation_threshold=150, target_rgb=(255, 0, 0)):
    """指定色の描画画像からYOLO用テキストと可視化画像を生成する (除外領域処理付き)"""

    # バイト列をOpenCV形式に変換
    nparr = np.frombuffer(traced_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 色領域の抽出（HSV）
    mask = _extract_color_mask(img, target_rgb, saturation_threshold)

    # ⚠️ 【新機能】 除外領域（矩形リスト）の処理
    # exclusion_rects: [{'left': x, 'top': y, 'width': w, 'height': h}, ...]
    if exclusion_rects:
        for rect in exclusion_rects:
            # 座標を取得
            x = int(rect['left'])
            y = int(rect['top'])
            w = int(rect['width'])
            h = int(rect['height'])
            
            # マスク画像の指定された矩形領域を黒（0：検出なし）で塗りつぶす
            # 確実に削除するため、塗りつぶし (thickness=-1)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)
            
            # 可視化画像の方にも、除外領域をグレーの矩形で描画（確認用）
            cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 2)

    # 以降は以前と同じ処理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_lines = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area_px:
            continue
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        coords = []
        for pt in approx:
            nx = max(0.0, min(1.0, pt[0][0] / original_width))
            ny = max(0.0, min(1.0, pt[0][1] / original_height))
            coords.append(f"{nx:.6f} {ny:.6f}")
        yolo_lines.append(f"{class_id} {' '.join(coords)}")

    return "\n".join(yolo_lines), img