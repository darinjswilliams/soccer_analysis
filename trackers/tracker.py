from ultralytics import YOLO
import supervision as sv
import pickle
import os



class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # Load a pretrained model
        self.tracker = sv.ByteTrack()


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

        


 