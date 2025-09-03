from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import pandas as pd

import sys

#we need to get the folder for utils
sys.path.append('../')
from utils import get_center_bbox, get_bbox_width



class Tracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)  # Load a pretrained model
        self.tracker = sv.ByteTrack()


    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()

        #handle edge cases if it is missing in the start or end
        df_ball_positions = df_ball_positions.bfill()

        # Convert back to list of dictionaries
        ball_positions = [{1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()] 

        return ball_positions  


    
    def detect_frames(self, frames):

        # Batch size to avoid memeory issues
        batch_size=20
        detections = []

        # Incrementation is done by batch size 20, 40 etc..
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
   
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_video_path=None):


        if read_from_stub and stub_video_path is not None and os.path.exists(stub_video_path):
            with open(stub_video_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # Update tracks with new detections
        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "ball": [],
            "referees": []
        }

        # overwrite goal keeper with player
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
    

            # Convert the detection to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)
        

            
            # Convert Goal to Player object
            for object_id, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_id] = cls_names_inv['player']


            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)


            tracks['players'].append({})
            tracks['ball'].append({})
            tracks['referees'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()  # Bounding box coordinates
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {"bbox": bbox}
                
                if class_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {"bbox": bbox}

             # Loop over detection without track id
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == cls_names_inv['ball']:
                     tracks['ball'][frame_num][1] = {"bbox": bbox}

        # Save to stub file
        if stub_video_path is not None:
            with open(stub_video_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks


    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        # center of circle is bouding box

        x_center, _ = get_center_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame,
                    center=(x_center, y2),
                    axes=(int(width), int(0.35 * width)),
                    angle=0,
                    startAngle=-45,
                    endAngle=235,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_4
                    )
        
        #draw rectangle for id
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height//2) + 15  #15 pixels below the ellipse center adding random buffer
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame, 
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)), 
                          color,
                          cv2.FILLED)


             # Write the track id number
            x1_text = x1_rect+12

            if track_id > 99:
                  x1_text -= 10

            cv2.putText(frame,
                          f'{track_id}',
                          (int(x1_text), int(y1_rect)+15),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5,
                          (255, 255, 255),
                          2)             

        
        return frame

    def draw_trangle(self, frame, bbox, color):
        y = int(bbox[1])
        # center of circle is bouding box
        x_center, _ = get_center_bbox(bbox)

        triangle_points = np.array([ [x_center, y],
                                     [x_center - 10, y - 20],
                                     [x_center + 10, y - 20]
                                  ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2) #black border

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw simi -transparent rectangle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        alpha = 0.4  # Transparency factor

        # add to orginal frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Calcualte the percentage of ball control
        team_ball_control_till_frame = team_ball_control[:frame_num+1]

        # Get number of times each team has ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        # Calculate percentage for statistics
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        # Write the text to frame
        cv2.putText(frame, f'Team 1 Ball Control: {team_1*100: .2f}%', (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f'Team 2 Ball Control: {team_2*100: .2f}%', (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame


    def draw_annotations(self, video_frames, tracks, team_ball_control):

        output_video_frame= []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # Draw circles using the tracks dictionary
            player_dict = tracks.get('players', [])[frame_num]
            ball_dict = tracks.get('ball', [])[frame_num]
            referee_dict = tracks.get('referees', [])[frame_num]


            # Draw players
            for track_id, player in player_dict.items():
                color = player.get('team_color', (0, 0, 255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)


                # If player has ball, draw a triangle above the player with
                if player.get('has_ball', False):
                    
                    frame = self.draw_trangle(frame, player['bbox'], (255, 255, 255))


            # Draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (255, 255, 0))

            
            # Draw ball
            for _, ball in ball_dict.items():
                frame = self.draw_trangle(frame, ball['bbox'], (0, 255, 0))


            # Draw team ball control text
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)


            # append frame to output video
            output_video_frame.append(frame)

        return output_video_frame