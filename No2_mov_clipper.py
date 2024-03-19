from moviepy.editor import AudioFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate
import os

def find_most_similar_audio_section(video_audio_path, reference_audio_path):
    sr_v, audio_v = wavfile.read(video_audio_path)
    sr_r, audio_r = wavfile.read(reference_audio_path)

    if audio_v.ndim > 1:
        audio_v = np.mean(audio_v, axis=1)
    if audio_r.ndim > 1:
        audio_r = np.mean(audio_r, axis=1)

    correlation = correlate(audio_v, audio_r, mode='valid')
    max_corr_index = np.argmax(correlation)

    return max_corr_index / sr_v

def process_video_files(folder_path, reference_audio_path):
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4') and "_round" in f]
    
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        video_audio_path = os.path.join(folder_path, 'temp_audio.wav')
        output_video_path = os.path.join(folder_path, 'clip_' + video_file)
        audio_clip = AudioFileClip(video_path)
        audio_clip.write_audiofile(video_audio_path)
        split_time = find_most_similar_audio_section(video_audio_path, reference_audio_path)
        ffmpeg_extract_subclip(video_path, 0, split_time, targetname=output_video_path)
        os.remove(video_audio_path)

def main():
    folder_path = 'work'
    reference_audio_path = 'KO.wav'
    process_video_files(folder_path, reference_audio_path)
    
    print("動画の切り出しが完了しました。")

if __name__ == '__main__':
    main()
