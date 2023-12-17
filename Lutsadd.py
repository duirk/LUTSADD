import torch
import moviepy.editor as mp

epoch = 50

video_input = "1.mp4"
audio_input = "2.mp4"
output_video = "resul.mp4"

# Cargar el archivo .pt en PyTorch
checkpoint_path = f"./imagenes/checkpoint_epoch_{epoch}.pt"
checkpoint = torch.load(checkpoint_path)


# Cargar los videos y audios
video1 = mp.VideoFileClip(video_input)
video2 = mp.VideoFileClip("1.mp4")
audio = mp.AudioFileClip(audio_input)

# Aplicar el efecto de color al video1
video1 = video1.fx(mp.vfx.colorx, 1.5)

# Aplicar el modelo de PyTorch al video2
# ...

# Combinar el video1 y el video2
final_video = video1.set_audio(audio)

# Guardar el resultado en el archivo de salida
final_video.write_videofile(output_video, codec="libx264", audio_codec="aac", fps=final_video.fps)
