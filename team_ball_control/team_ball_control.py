class TeamBallControl:
    def __init__(self):
        self.team_ball_control = []

    def get_team_ball_control(self, tracks, frame_num, assigned_player):
        if frame_num < 0 or frame_num >= len(tracks['players']):
            raise IndexError("frame_num is out of bounds")
        if assigned_player != -1:
            # if assigned_player < 0 or assigned_player >= len(tracks['players'][frame_num]):
                # raise IndexError("assigned_player is out of bounds")
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            self.team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if self.team_ball_control:
                self.team_ball_control.append(self.team_ball_control[-1])

        return self.team_ball_control