import moviepy.editor as mp
import librosa
import numpy as np
from scipy.signal import correlate
from scipy.signal import find_peaks
import os

from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def process_video(video_path):
    base_video_name = os.path.splitext(os.path.basename(video_path))[0]

    audio_from_video_path = f'temp_audio_{base_video_name}.wav'
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_from_video_path)
    
    audio_A, sr_A = librosa.load(audio_from_video_path)
    audio_B, sr_B = librosa.load('fight.wav')
    
    correlation = correlate(audio_A, audio_B, mode='full')
    correlation = np.abs(correlation)
    
    peaks, _ = find_peaks(correlation, height=np.max(correlation)*0.8)
    
    frame_rate = video.fps
    frames = [(peak / sr_A) * frame_rate for peak in peaks]
    
    times = [frame / frame_rate for frame in frames]
    times.append(video.duration)
    
    for i, (start_time, end_time) in enumerate(zip(times[:-1], times[1:])):
        output_filename = f"work/{base_video_name}_round{i+1:01d}.mp4"
        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_filename)
    
    print(f"ビデオ '{base_video_name}' の切り出しが完了しました。")

def main():
    folder_path = 'mov'
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
    
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        process_video(video_path)

if __name__ == '__main__':
    main()