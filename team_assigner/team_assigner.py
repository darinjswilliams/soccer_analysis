from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

   
    def get_clustering_model(self, image):
        # Reshape the image to 2d Array
        image_2d = image.reshape((-1, 3))

        # Convert to KMeans Cluster with 2 clusters
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1).fit(image_2d)
       
       # Return the kmeans
        return kmeans
   
   
    def get_player_color(self, frame, bbox):

        #crop player from frame using bbox and convet integer
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # top half of the player
        top_half_image = image[0:int(image.shape[0]/2), :]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster lables
        labels = kmeans.labels_

        # Reshape labels back to image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]

        #Non player cluster is the color that appears most in the corners
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)

        # Player cluster
        player_cluster = 1 - non_player_cluster

        # Get the player color
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color
   
   
    def assign_team_color(self, frame, player_detections):
       
        player_colors = []

        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

         # divde player colors into two teams using kmeans to cluster
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10).fit(player_colors) 


        self.kmeans = kmeans

        # Assign team colors
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self, frame, bbox, player_id):

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, bbox) 

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] 
        team_id += 1

        #lets check to see if the goal keeper has the ball if he has ball assign id 2
        if player_id == 91:
            team_id = 2 if 2 not in self.team_colors else 1
        
        #lets save before returning

        self.player_team_dict[player_id] = team_id

        return team_id
