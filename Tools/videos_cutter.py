from moviepy.editor import VideoFileClip

# Load the video file
video = VideoFileClip("L:/ENVIROBASS_NOE/TESTS/APRENDIZAJE/S_26_29_MAYO/S_26_29_D1/S_26_29_D1_B1/GX016343.MP4")

# Cut the video from the start (0 seconds) to 4 minutes and 30 seconds (270 seconds)
cortado = video.subclip(0, 60)

# Save the cut video to the specified output file
cortado.write_videofile("D:/Chris/Rocio/Lubinas/Laberinto/cortado_laberinto3.mp4")
