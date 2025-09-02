from utils import save_video, read_video

def main():
    # Read Video
    video_frames = read_video('input_videos/D35bd9041_1.mp4')


    # Save Video
    save_video(video_frames, 'output_videos/output.avi', fps=24)

    
if __name__ == "__main__":
    main()
