import cv2


def read_video(video_path):
    """Read a video file and return its frames as a list of numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_view_frames, output_video_path, fps=24):
    """Save a list of frames (numpy arrays) as a video file."""
    if len(output_view_frames) == 0:
        raise ValueError("No frames to save.")
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs as needed
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_view_frames[0].shape[1], output_view_frames[0].shape[0]))
    
    for frame in output_view_frames:
        video.write(frame)
    
    video.release()