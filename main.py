from utils import save_video, read_video
from trackers import Tracker

def main():
    # Read Video
    video_frames = read_video('input_videos/D35bd9041_1.mp4')


    # Initialize Tracker
    tracker = Tracker('models/best.pt') # Path to your YOLO model

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_video_path='stubs/track_stub.pkl')

    # Draw output
    # Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)



    # Save Video
    save_video(output_video_frames, 'output_videos/output.avi', fps=24)


if __name__ == "__main__":
    main()
