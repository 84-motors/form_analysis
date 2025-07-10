import cv2
import mediapipe as mp
import streamlit as st
import tempfile
import os
import numpy as np
import time

# Streamlit UI
st.title("姿勢推定アプリ")

# 再生速度の選択
play_speed = st.selectbox(
    "再生速度を選択してください",
    options=["0.25倍", "0.5倍", "1倍", "1.5倍", "2倍"],
    index=2
)
speed_map = {
    "0.25倍": 0.25,
    "0.5倍": 0.5,
    "1倍": 1.0,
    "1.5倍": 1.5,
    "2倍": 2.0
}
speed_factor = speed_map[play_speed]

# 動画アップロード
uploaded_file = st.file_uploader("動画を選択", type=["mp4", "mov", "avi"])

# MediaPipe初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

if uploaded_file is not None:
    # 一時保存
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_file.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 出力動画設定（横に2画面なので幅は2倍）
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(tempfile.gettempdir(), "output_combined.mp4")
    out = cv2.VideoWriter(output_path, codec, fps, (width * 2, height))

    # ストリーム表示枠
    stframe = st.empty()

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_image)

            # 元フレームに描画
            original_with_landmarks = frame.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    original_with_landmarks,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            # 黒背景に描画
            black_image = np.zeros_like(frame)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    black_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            # 横連結
            combined_frame = np.concatenate((original_with_landmarks, black_image), axis=1)

            # 表示
            stframe.image(combined_frame, channels="BGR")

            # 出力
            out.write(combined_frame)

            # 再生速度に応じた待機時間（秒）
            delay = (1.0 / fps) / speed_factor
            time.sleep(delay)

        cap.release()
        out.release()
        os.remove(temp_file.name)

    # 処理完了・再生とダウンロード
    st.success("処理が完了しました！")
    st.video(output_path)
    st.download_button(
        label="動画をダウンロード",
        data=open(output_path, "rb").read(),
        file_name="pose_estimation_comparison.mp4",
        mime="video/mp4"
    )
