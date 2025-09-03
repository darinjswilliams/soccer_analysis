from utils import save_video, read_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigners import PlayerBallAssigner
from team_ball_control import TeamBallControl
import numpy as np

import cv2

def main():
    # Read Video
    video_frames = read_video('input_videos/A1606b0e6_0.mp4') # Path to your input video

   
    # Initialize Tracker
    tracker = Tracker('models/best.pt') # Path to your YOLO model

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_video_path='stubs/track_stub.pkl')

   
   # Interpolate Ball Positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
   
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


    # Assign Ball Aquisition to Player
    player_assigner = PlayerBallAssigner()

    # Track Team Ball Control
    team_ball_control = TeamBallControl() # Initialize with 'None' for the first frame

    
    for frame_num, player_track in enumerate(tracks['players']):

        # Get ball bbox for the frame and assign to player
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)


        # If a player is assigned, mark has_ball True with new key in track as has_all
        team_control_of_ball = np.array(team_ball_control.get_team_ball_control(tracks, frame_num, assigned_player))


    # Draw output
    # Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_control_of_ball)



    # Save Video
    save_video(output_video_frames, 'output_videos/output.avi', fps=24)


if __name__ == "__main__":
    main()
