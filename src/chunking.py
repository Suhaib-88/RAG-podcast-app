import os
import pandas as pd
from pydub import AudioSegment

def save_uploadedfile(uploadedfile, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, uploadedfile.name), 'wb') as f:
        f.write(uploadedfile.getbuffer())
    return f"Saved File: {uploadedfile.name}"

def split_m4a(mp3_file_folder, mp3_chunk_folder, episode_id, chunk_length_ms, overlap_ms):
    audio = AudioSegment.from_mp3(mp3_file_folder + "/" + episode_id + ".mp3")
    num_chunks = len(audio) // (chunk_length_ms - overlap_ms) + (1 if len(audio) % chunk_length_ms else 0)

    for i in range(num_chunks):
        start_ms = i * chunk_length_ms - (i * overlap_ms)
        end_ms = start_ms + chunk_length_ms
        chunk = audio[start_ms:end_ms]
        export_fp = mp3_chunk_folder + "/" + episode_id + f"_chunk{i + 1}.mp3"
        chunk.export(export_fp, format="mp3")

    return chunk

def dataframe_chunking(mp3_chunk_folder):
    episode_metadata_df = pd.read_csv('episode_metadata.csv')
    chunk_fps = os.listdir(mp3_chunk_folder)
    episode_chunk_df = pd.DataFrame({
        'filepath': [mp3_chunk_folder + '/' + fp for fp in chunk_fps],
        'episode_id': [fp.split('_chunk')[0] for fp in chunk_fps]
    })
    episodes_df = episode_chunk_df.merge(episode_metadata_df, on='episode_id')
    return episodes_df