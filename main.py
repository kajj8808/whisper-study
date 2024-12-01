
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, save_audio
from moviepy.audio.io.AudioFileClip import AudioFileClip
import whisper
import uuid
import os

AUDIO_DIR_PATH = "./sample/audio"
AUDIO_FILE_PATH = f"{AUDIO_DIR_PATH}/1_uma_5_min_(Vocals).wav"

whisper_model = whisper.load_model("large-v2")

vad_model = load_silero_vad()
wav = read_audio(AUDIO_FILE_PATH)
speech_time_stamps = get_speech_timestamps(
    wav, model=vad_model, return_seconds=True, min_speech_duration_ms=50)

# 결과를 저장할 리스트
transcription_results = []


for index, time_stamp in enumerate(speech_time_stamps):
    try:
        # 고유 파일 이름 생성
        unique_filename = str(uuid.uuid4())
        temp_audio_path = f"./segments/{unique_filename}.wav"

        # 오디오 파일을 자르고 임시 파일로 저장
        audio_clip = AudioFileClip(filename=AUDIO_FILE_PATH)
        new_clip = audio_clip.subclip(time_stamp["start"], time_stamp["end"])
        new_clip.write_audiofile(temp_audio_path)

        # Whisper 모델로 자른 부분의 텍스트를 추출
        result = whisper_model.transcribe(
            temp_audio_path, language="Japanese", temperature=0.2
        )

        # 결과를 출력하고 리스트에 저장
        transcription_results.append({
            "index": index,
            "start": time_stamp["start"],
            "end": time_stamp["end"],
            "text": result['text']
        })

        print(
            f"Index {index}/{speech_time_stamps.count()}: Transcription: {result['text']}")

    except Exception as e:
        print(f"오류 발생 (Index {index}): {e}")

    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# VTT 파일로 저장
vtt_file_path = f"{AUDIO_DIR_PATH}/output.vtt"

with open(vtt_file_path, 'w', encoding='utf-8') as vtt_file:
    vtt_file.write("WEBVTT\n\n")

    for result in transcription_results:
        start_time = result["start"]
        end_time = result["end"]
        text = result["text"]

        # 시간을 VTT 포맷 (HH:MM:SS.mmm)으로 변환
        start_time_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{start_time % 60:06.3f}"
        end_time_str = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{end_time % 60:06.3f}"

        # VTT 형식으로 자막을 작성
        vtt_file.write(f"{result['index'] + 1}\n")
        vtt_file.write(f"{start_time_str} --> {end_time_str}\n")
        vtt_file.write(f"{text}\n\n")

print(f"VTT 파일이 '{vtt_file_path}'로 저장되었습니다.")
