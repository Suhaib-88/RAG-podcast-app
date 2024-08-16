import time
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document

def audio_to_text(filepath, client, model):
    with open(filepath, "rb") as file:
        transcriptions = client.audio.translations.create(
            file=(filepath, file.read()),
            model=model,
        )
    return transcriptions.text

def documentation(episodes_df, text_splitter, client, model):
    documents = []
    cnt = 0
    for index, row in episodes_df.iterrows():
        cnt += 1
        audio_filepath = row['filepath']
        transcript = audio_to_text(audio_filepath, client, model)
        time.sleep(2)
        chunks = text_splitter.split_text(transcript)
        for chunk in chunks:
            header = f"Date: {row['published_date']}\nEpisode Title: {row['title']}\n\n"
            documents.append(Document(page_content=header + chunk, metadata={"source": "local"}))

        if cnt % (round(len(episodes_df) / 5)) == 0:
            print(round(cnt / len(episodes_df), 2) * 100, '% of transcripts processed...')

        print('# Transcription Chunks: ', len(documents))
    return documents