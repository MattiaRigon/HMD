from googleapiclient.discovery import build

# Replace with your API key
API_KEY = "YOUR_API_KEY"

# Replace with your video ID
VIDEO_ID = "dQw4w9WgXcQ"

# Build the YouTube API client
youtube = build("youtube", "v3", developerKey=API_KEY)

def fetch_comments(video_id):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100  # Maximum allowed per request
    )
    while request:
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        # Check if there are more pages
        request = youtube.commentThreads().list_next(request, response)
    return comments

# Fetch and print comments
comments = fetch_comments(VIDEO_ID)
for i, comment in enumerate(comments[:10], start=1):  # Displaying only the first 10 comments
    print(f"{i}: {comment}")
