from moviepy.editor import *

def main():
    input_video_path = 'fight.mp4'
    output_audio_path = 'fight.wav'
    
    video = VideoFileClip(input_video_path)
    video.audio.write_audiofile(output_audio_path)

    input_video_path = 'KO.mp4'
    output_audio_path = 'KO.wav'
    
    video = VideoFileClip(input_video_path)
    video.audio.write_audiofile(output_audio_path)

if __name__ == '__main__':
    main()
