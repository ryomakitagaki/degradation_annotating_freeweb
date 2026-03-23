# app.py
import streamlit as st
import io
import zipfile
import base64
import cv2
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from pathlib import Path
# streamlit-drawable-canvas が使う内部API互換パッチ
# 旧: image_to_url(image, width:int, clamp, channels, fmt, key)
# 新: image_to_url(image, layout_config:LayoutConfig, clamp, channels, fmt, key)
# このあたり画像読み込みー出力を堅牢化するために、streamlitのバージョン差異を吸収する互換関数を定義している。
import streamlit.elements.image as _st_img_mod
if not hasattr(_st_img_mod, "image_to_url"):
    try:
        from streamlit.elements.lib.image_utils import image_to_url as _new_image_to_url
        from streamlit.elements.lib.layout_utils import LayoutConfig as _LayoutConfig

        def _compat_image_to_url(image, width_or_config, *args, **kwargs):
            if isinstance(width_or_config, int):
                width_or_config = _LayoutConfig(width=width_or_config)
            return _new_image_to_url(image, width_or_config, *args, **kwargs)

        _st_img_mod.image_to_url = _compat_image_to_url
    except ImportError:
        pass

from streamlit_drawable_canvas import st_canvas

import logic

# --- ヘルパー関数 (エラー回避のため外側に定義.) ---
def get_exclusion_mask(image_data, target_w, target_h):
    """canvas の image_data から除外マスクを target サイズで作成する"""
    if image_data is None:
        return None
    # 4チャンネル目(アルファチャンネル)が描画された部分
    alpha = image_data[:, :, 3]
    if alpha.max() == 0:
        return None
    # OpenCVでリサイズ。描画領域を255(白)にする
    mask = cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return mask

# --- UI設定 ---
st.set_page_config(page_title="Degradation Analysis & Annotation Tool", layout="wide")
st.title("🏗️ Degradation Analysis & Annotation Tool")

# --- セッション状態の初期化 ---
if 'file_index' not in st.session_state:
    st.session_state.file_index = 0
if 'results_dict' not in st.session_state:
    st.session_state.results_dict = {}
if 'file_names' not in st.session_state:
    st.session_state.file_names = []
if 'file_bytes_dict' not in st.session_state:
    st.session_state.file_bytes_dict = {}
if 'zoom_orig' not in st.session_state:
    st.session_state.zoom_orig = False

# --- 基本プロンプト定義 ---
V1="""
写真にうつる建築物の表面を解析し，ひび割れを特定してください。
直線的なタイルやブロックの目地，建材の稜線，塗料の剥がれ部，異種材料の境界部分はひび割れではありません。
建材表面の幾何学的な模様や陰影はひび割れではありません。
特定したひび割れの上に、RGB(255, 0, 0)の不透明な線を，描画した画像を生成してください。
"""
V2="""
写真に写る建築物の表面を解析し，欠損部や剥離部をすべて特定し、
その範囲にRGB(255, 0, 0)の不透明な描画した画像を返してください。
"""
V3="""
写真に写る建築物の表面を解析し，エフロレッセンス（白華現象，efflorescence）が見られる領域をすべて特定し、
その範囲にRGB(255, 0, 0)の不透明な色で塗りつぶした画像を返してください。
"""
PROMPT_MAP = {
    "Cracks": V1,
    "Chipped/Delaminated": V2,
    "Efflorescence/Other": V3
}
CLASS_MAP = {
    "Cracks":              {"id": 0, "suffix": "_cracks"},
    "Chipped/Delaminated": {"id": 1, "suffix": "_chipped"},
    "Efflorescence/Other": {"id": 2, "suffix": "_efflorescence"},
}

# --- 1. ファイル読み込み（サイドバーより先に実行してfile_namesを確定させる）---
uploaded_files = st.file_uploader("Load images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
current_names = [f.name for f in uploaded_files] if uploaded_files else []

if current_names != st.session_state.file_names:
    st.session_state.file_names = current_names
    st.session_state.file_bytes_dict = {f.name: f.read() for f in uploaded_files} if uploaded_files else {}
    st.session_state.file_index = 0
    st.session_state.results_dict = {}

# --- 2. サイドバー（file_namesが確定した後に描画）---
with st.sidebar:
    st.header("🔑 Setting and model Loading")
    api_key = st.text_input("Gemini API Key", type="password")
    model_id = st.selectbox("Model", ["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"])
    prompt_type = st.radio("Degradation Type", ["Cracks", "Chipped/Delaminated", "Efflorescence/Other"])
    st.divider()
    st.header("📊 Current status")
    if st.session_state.file_names:
        total = len(st.session_state.file_names)
        current = st.session_state.file_index + 1
        st.write(f"Progress: {current} / {total}")
        st.progress(current / total)

        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️ Previous", disabled=(st.session_state.file_index == 0)):
                st.session_state.file_index -= 1
                st.rerun()
        with col_next:
            if st.button("Next ➡️", disabled=(st.session_state.file_index == total - 1)):
                st.session_state.file_index += 1
                st.rerun()

# --- 3. 個別処理エリア ---
if st.session_state.file_names:
    filename = st.session_state.file_names[st.session_state.file_index]
    st.subheader(f"📂 current image: {filename}")

    image_bytes = st.session_state.file_bytes_dict[filename]
    nparr = np.frombuffer(image_bytes, np.uint8)
    _cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_np = cv2.cvtColor(_cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_np)
    w, h = pil_img.size
    _img_buf = io.BytesIO()
    pil_img.save(_img_buf, format="PNG")
    _img_b64 = base64.b64encode(_img_buf.getvalue()).decode()

    # キャンバス幅を先に計算して全画像と揃える
    CANVAS_PAD = 10
    display_w = 600 if w > 600 else w
    display_h = int(h * display_w / w)
    canvas_total_w = display_w + 2 * CANVAS_PAD

    def crop_canvas_padding(image_data):
        if image_data is None:
            return None
        p = CANVAS_PAD
        ch, cw = image_data.shape[:2]
        return image_data[p:ch - p, p:cw - p]

    # --- 上段: Original Image | Refinement Prompt ---
    col_top_l, col_top_r = st.columns([1, 1])

    with col_top_l:
        with st.expander("📷 Original Image", expanded=True):
            st.markdown(
                f'<img src="data:image/png;base64,{_img_b64}" width="{canvas_total_w}">',
                unsafe_allow_html=True,
            )
            if st.button("🔍 Zoom", key=f"zoom_btn_{filename}"):
                st.session_state.zoom_orig = not st.session_state.zoom_orig

    if st.session_state.zoom_orig:
        st.markdown(
            f'<img src="data:image/png;base64,{_img_b64}" style="width:100%">',
            unsafe_allow_html=True,
        )
        st.caption(filename)
        if st.button("✕ Close zoom", key=f"zoom_close_{filename}"):
            st.session_state.zoom_orig = False
            st.rerun()

    with col_top_r:
        st.markdown("#### 🤖 Refinement Prompt")
        refine_key = f"refine_text_{filename}"
        user_refinement = st.text_area(
            "Add instructions for better detection:",
            placeholder="Ex: 'Ignore the vertical tile joints on the right.'",
            key=refine_key
        )
        picked_hex = st.color_picker("Annotation color", "#FF0000", key=f"color_{filename}")
        pr = int(picked_hex[1:3], 16)
        pg = int(picked_hex[3:5], 16)
        pb = int(picked_hex[5:7], 16)
        target_rgb = (pr, pg, pb)

        if st.button("🚀 Analyze / Refine with AI", use_container_width=True):
            if not api_key:
                st.error("Please enter API key")
            else:
                with st.spinner("Analyzing..."):
                    final_prompt = PROMPT_MAP[prompt_type].replace(
                        "RGB(255, 0, 0)", f"RGB({pr}, {pg}, {pb})"
                    )
                    if user_refinement:
                        final_prompt += f"\n\n**Additional instructions:**\n{user_refinement}"
                    traced_data, raw_data = logic.get_gemini_traced_image(api_key, image_bytes, final_prompt, model_id)
                    if filename not in st.session_state.results_dict:
                        st.session_state.results_dict[filename] = {}
                    st.session_state.results_dict[filename]["traced_data"] = traced_data
                    st.session_state.results_dict[filename]["raw_data"] = raw_data
                    st.session_state.results_dict[filename]["target_rgb"] = target_rgb
                    st.success("Analysis completed!")

    # --- 下段: AI raw output | Post-processing (canvas) ---
    if filename in st.session_state.results_dict and st.session_state.results_dict[filename].get("traced_data"):
        res = st.session_state.results_dict[filename]

        st.divider()

        col_bot_l, col_bot_r = st.columns([1, 1])

        with col_bot_l:
            st.markdown("#### Post-processing")
            # 右カラムのコントロール高さ分スペーサー（余白）を入れて下端を揃える
            st.markdown('<div style="height: 40px"></div>', unsafe_allow_html=True)
            with st.expander("🔍 AI raw output", expanded=True):
                if res.get("raw_data"):
                    _raw_b64 = base64.b64encode(res["raw_data"]).decode()
                    st.markdown(
                        f'<img src="data:image/png;base64,{_raw_b64}" width="{canvas_total_w}">',
                        unsafe_allow_html=True,
                    )

        with col_bot_r:
            ctrl1, ctrl2, ctrl3 = st.columns(3)
            with ctrl1:
                min_area = st.number_input(
                    "Min polygon area (px)",
                    value=0, min_value=0, key=f"min_area_{filename}"
                )
            with ctrl2:
                gap_fill_kernel = st.slider(
                    "Gap fill kernel (0=off)", 0, 100, 0,
                    key=f"gap_fill_{filename}",
                    help="途切れた線をつなぐ・穴を埋める。値を大きくするほど強く補完。"
                )
            with ctrl3:
                sat_thresh = st.slider(
                    "Saturation threshold", 0, 255, 150,
                    key=f"sat_{filename}",
                    help="高いほど純粋な指定色のみ検出。低くするとピンク・オレンジも含まれる。変更するとキャンバスに即反映。"
                )

        saved_target_rgb = res.get("target_rgb", (255, 0, 0))
        if res.get("raw_data"):
            traced_bytes_live = logic.reprocess_from_raw(image_bytes, res["raw_data"], int(gap_fill_kernel), int(sat_thresh), saved_target_rgb)
        else:
            traced_bytes_live = res["traced_data"]
        _traced_cv = cv2.imdecode(np.frombuffer(traced_bytes_live, np.uint8), cv2.IMREAD_COLOR)
        traced_pil = Image.fromarray(cv2.cvtColor(_traced_cv, cv2.COLOR_BGR2RGB))

        with col_bot_r:
            st.write("Mark erroneous detection areas (Polygons)")
            pad_orig = max(1, round(CANVAS_PAD * w / display_w))
            padded_traced = ImageOps.expand(traced_pil, border=pad_orig, fill=(160, 160, 160))

            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.5)",
                stroke_width=2,
                background_image=padded_traced,
                update_streamlit=True,
                height=display_h + 2 * CANVAS_PAD,
                width=display_w + 2 * CANVAS_PAD,
                drawing_mode="polygon",
                key=f"canvas_{filename}",
            )

        # --- Manual Exclusion Preview（下段全幅）---
        if canvas_result.image_data is not None:
            cropped_data = crop_canvas_padding(canvas_result.image_data)
            mask = get_exclusion_mask(cropped_data, w, h)
            if mask is not None:
                traced_np = np.array(traced_pil.convert("RGB"))
                orig_np = img_np.copy()
                preview_np = np.where(mask[:, :, np.newaxis] > 0, orig_np, traced_np)
                _, _prev_buf = cv2.imencode(".png", cv2.cvtColor(preview_np.astype(np.uint8), cv2.COLOR_RGB2BGR))
                _prev_b64 = base64.b64encode(_prev_buf.tobytes()).decode()
                st.markdown(
                    f'<img src="data:image/png;base64,{_prev_b64}" style="width:100%">',
                    unsafe_allow_html=True,
                )
                st.caption("Manual Exclusion Preview")

        # 保存済みクラスの表示
        saved_classes = list(st.session_state.results_dict.get(filename, {}).get("class_annotations", {}).keys())
        if saved_classes:
            labels_str = ", ".join(CLASS_MAP[c]["suffix"].lstrip("_") for c in saved_classes)
            st.info(f"Saved classes for this image: **{labels_str}**")

        if st.button("✅ Confirm and save", use_container_width=True):
                traced_bytes_to_use = traced_bytes_live

                if canvas_result.image_data is not None:
                    nparr = np.frombuffer(traced_bytes_to_use, np.uint8)
                    traced_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    th, tw = traced_cv.shape[:2]
                    cropped_data = crop_canvas_padding(canvas_result.image_data)
                    mask = get_exclusion_mask(cropped_data, tw, th)

                    if mask is not None:
                        orig_cv = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                        orig_resized = cv2.resize(orig_cv, (tw, th))
                        traced_cv[mask > 0] = orig_resized[mask > 0]

                    _, enc = cv2.imencode(".png", traced_cv)
                    traced_bytes_to_use = enc.tobytes()

                class_id = CLASS_MAP[prompt_type]["id"]
                yolo_txt, vis_img = logic.process_yolo_segmentation(
                    traced_bytes_to_use, w, h, int(min_area), [], class_id, int(sat_thresh), saved_target_rgb
                )
                res = st.session_state.results_dict[filename]
                if "class_annotations" not in res:
                    res["class_annotations"] = {}
                if "class_vis_imgs" not in res:
                    res["class_vis_imgs"] = {}
                res["class_annotations"][prompt_type] = yolo_txt
                res["class_vis_imgs"][prompt_type] = vis_img
                res["completed"] = True
                saved_names = ", ".join(CLASS_MAP[c]["suffix"].lstrip("_") for c in res["class_annotations"])
                st.success(f"Saved [{prompt_type}] for {filename}  |  All saved: {saved_names}")

# --- 4. 一括書き出し ---
if st.session_state.results_dict:
    st.divider()
    completed_count = sum(1 for r in st.session_state.results_dict.values() if r.get("completed"))
    if st.button(f"📁 Make a ZIP ({completed_count} of data set)", use_container_width=True):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for fname, data in st.session_state.results_dict.items():
                if data.get("completed"):
                    stem = Path(fname).stem
                    ext = Path(fname).suffix
                    # 全クラスのアノテーションを結合してラベルファイルに保存
                    all_annotations = "\n".join(
                        txt for txt in data.get("class_annotations", {}).values() if txt.strip()
                    )
                    # 元画像を images/ に保存（YOLO dataset 形式）
                    zf.writestr(f"images/{fname}", st.session_state.file_bytes_dict[fname])
                    # ラベルを labels/ に保存
                    zf.writestr(f"labels/{stem}.txt", all_annotations)
                    # クラスごとの可視化画像を visualized/ にサフィックス付きで保存
                    for pt, vis_img in data.get("class_vis_imgs", {}).items():
                        suffix = CLASS_MAP.get(pt, CLASS_MAP["Cracks"])["suffix"]
                        vis_fname = f"{stem}{suffix}{ext}"
                        _, img_enc = cv2.imencode(".jpg", vis_img)
                        zf.writestr(f"visualized/{vis_fname}", img_enc.tobytes())
            # data.yaml を追加
            yaml_content = (
                "nc: 3\n"
                "names:\n"
                "  - cracks\n"
                "  - chipped_delaminated\n"
                "  - efflorescence\n"
                "train: images/\n"
                "val: images/\n"
            )
            zf.writestr("data.yaml", yaml_content)
        st.download_button("🔥 ZIP Download", zip_buffer.getvalue(), "dataset.zip", "application/zip", use_container_width=True)
