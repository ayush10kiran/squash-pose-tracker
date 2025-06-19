import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
import tempfile
import numpy as np

st.set_page_config(layout="centered")
st.title("Squash Pose Tracker")

st.markdown(
    """
    <div style='padding: 10px; border-radius: 100px;'>
        <b>📷 Video Guidelines:</b><br>
        For the best results, record the player from a <b>straight front-facing view</b> standing just outside the court,<br>
        ideally from the <b>corner behind the player</b> where they hit the ball.<br>
        Make sure the full body is visible and unobstructed.
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_video = st.file_uploader("🎥 Upload your squash video", type=["mp4", "mov", "avi"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(tfile.name)
    frame_data = []
    frame_id = 0
    form_scores = []
    impact_frames = []

    st.subheader("🔍 Analyzing video...")
    stframe = st.empty()
    prev_left_wrist = None
    velocity_threshold = 0.03

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_row = {'frame': frame_id}
            for i, lm in enumerate(landmarks):
                frame_row[f'x_{i}'] = lm.x
                frame_row[f'y_{i}'] = lm.y
                frame_row[f'z_{i}'] = lm.z
            frame_data.append(frame_row)

            try:
                def score_metric(metric, ranges):
                    for score, (low, high) in ranges.items():
                        if low <= metric <= high:
                            return score
                    return min(ranges.keys()) + 0.25

                left_ankle_x = landmarks[27].x
                right_ankle_x = landmarks[28].x
                foot_dist = abs(left_ankle_x - right_ankle_x)

                left_shoulder_x = landmarks[11].x
                left_hip_x = landmarks[23].x
                torso_align = abs(left_shoulder_x - left_hip_x)

                left_knee_y = landmarks[25].y
                left_ankle_y = landmarks[27].y
                knee_bend = left_ankle_y - left_knee_y

                foot_ranges = {
                    2.5: (0.18, 0.37),
                    1.875: (0.14, 0.4),
                    1.25: (0.1, 0.45),
                    0.625: (0.05, 0.5),
                }

                torso_ranges = {
                    2.5: (0, 0.035),
                    1.875: (0.035, 0.065),
                    1.25: (0.065, 0.1),
                    0.625: (0.1, 0.14),
                }

                knee_ranges = {
                    2.5: (0.04, 1),
                    1.875: (0.02, 0.04),
                    1.25: (0.005, 0.02),
                    0.625: (0, 0.005),
                }

                foot_score = score_metric(foot_dist, foot_ranges)
                torso_score = score_metric(torso_align, torso_ranges)
                knee_score = score_metric(knee_bend, knee_ranges)

                total_form_score = round(
                    ((foot_score / 2.5) * 4) +
                    ((torso_score / 2.5) * 3) +
                    ((knee_score / 2.5) * 3), 2
                )

                left_elbow = landmarks[13]
                left_wrist = landmarks[15]
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                left_hip = landmarks[23]
                right_hip = landmarks[24]

                arm_extension = np.sqrt((left_wrist.x - left_elbow.x) ** 2 + (left_wrist.y - left_elbow.y) ** 2)

                shoulder_vec = np.array([right_shoulder.x - left_shoulder.x, right_shoulder.y - left_shoulder.y])
                hip_vec = np.array([right_hip.x - left_hip.x, right_hip.y - left_hip.y])

                cos_theta = np.dot(shoulder_vec, hip_vec) / (np.linalg.norm(shoulder_vec) * np.linalg.norm(hip_vec) + 1e-6)
                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                hip_rotation = angle

                if prev_left_wrist is not None:
                    wrist_velocity = np.sqrt(
                        (left_wrist.x - prev_left_wrist.x) ** 2 +
                        (left_wrist.y - prev_left_wrist.y) ** 2
                    )
                else:
                    wrist_velocity = 0
                prev_left_wrist = left_wrist

                if arm_extension > 0.1 or hip_rotation > 0.15 or wrist_velocity > velocity_threshold:
                    impact_frames.append(frame_id)
                    form_scores.append(total_form_score)
                else:
                    form_scores.append(None)

            except:
                form_scores.append(None)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        preview = cv2.resize(image, (480, 360))
        stframe.image(preview, channels='BGR')
        frame_id += 1

    cap.release()

    df = pd.DataFrame(frame_data)
    score_df = pd.DataFrame({'frame': range(len(form_scores)), 'form_score': form_scores})
    df = df.merge(score_df, on='frame', how='left')

    st.success("✅ Pose keypoints extracted and impact frames scored!")

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Keypoints + Score CSV",
        data=csv,
        file_name='pose_keypoints_scored.csv',
        mime='text/csv',
    )

    impact_scores = [s for s in form_scores if s is not None]
    if impact_scores:
        avg_score = round(np.mean(impact_scores), 2)
        st.metric("📊 Average Form Score (Impact Only)", f"{avg_score}/10")
