from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
from tqdm import tqdm


data_path = '/media/capstone/HDD2/source_data/youtube_links'


api_key = "AIzaSyCIwWZGc_QKKkm83YE270eM0uhtK6r3lyw"
youtube = build('youtube', 'v3', developerKey=api_key)


playlist_id = "PL_RnEKSdOwG0s8qm9hNMSLHj7cVnWiBfG"

video_links = []

next_page_token = None
while True:
    request = youtube.playlistItems().list(
        part="snippet",
        playlistId=playlist_id,
        maxResults=500,
        pageToken=next_page_token,
    )

    response = request.execute()

    for item in tqdm(response['items']):
        video_links.append("https://www.youtube.com/watch?v=" + item['snippet']['resourceId']['videoId'])

    next_page_token = response.get('nextPageToken')

    if not next_page_token:
        break

with open(f"{data_path}{os.sep}{playlist_id}.txt", "w") as f:
    for video in video_links:
        f.write(video + "\n")

print(len(video_links))



# channel_id = "UCR9HdGYs477tb37IBcD0PqQ"
#
# video_links = []
#
# next_page_token = None
# while True:
#     request = youtube.search().list(
#         part="id",
#         channelId=channel_id,
#         maxResults=500,
#         order="date",
#         pageToken=next_page_token,
#         type="video"
#     )
#
#     response = request.execute()
#
#     for item in tqdm(response['items']):
#         video_links.append("https://www.youtube.com/watch?v=" + item['id']['videoId'])
#
#     next_page_token = response.get('nextPageToken')
#
#     if not next_page_token:
#         break
#
# with open(f"{data_path}{os.sep}{channel_id}.txt", "w") as f:
#     for video in video_links:
#         f.write(video + "\n")
#
# print(len(video_links))
