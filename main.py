from utils import save_video, read_video
from trackers import Tracker
from team_assigner import TeamAssigner

import cv2

def main():
    # Read Video
    video_frames = read_video('input_videos/D35bd9041_1.mp4')

   
    # Initialize Tracker
    tracker = Tracker('models/best.pt') # Path to your YOLO model

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_video_path='stubs/track_stub.pkl')

    # Initialize Player Team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

  
    # Assign team to each player with loop
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
           team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'], 
                                                 player_id)
           
            # Save Player to team introducing team key in track
           tracks['players'][frame_num][player_id]['team'] = team

           # Get the color from the cluster
           tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    
    # Draw output
    # Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)



    # Save Video
    save_video(output_video_frames, 'output_videos/output.avi', fps=24)


if __name__ == "__main__":
    main()
