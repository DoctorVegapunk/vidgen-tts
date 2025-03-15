import pysrt
from sentence_transformers import SentenceTransformer, util
import requests
import numpy as np
import re
from collections import defaultdict

# Replace with your Pexels API key
PEXELS_API_KEY = "0SzM0QyylRqJK4Vj2EZ9JI8oLyrAd1n0reU7MadOQ0k2SMQ9T9Shv8wr"
PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"

# Initialize Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_keywords(text):
    """
    Extract meaningful keywords from text using simple but effective rules.
    """
    # Common words to filter out
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
        'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
        'will', 'with', 'yeah', 'um', 'uh', 'like', 'gonna', 'wanna', 'okay', 'oh',
        'hey', 'just', 'but', 'they', 'this', 'what', 'when', 'where', 'who', 'how',
        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'than', 'too', 'very', 'can', 'could', 'may', 'might', 'must', 'need', 'ought',
        'shall', 'should', 'would', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their',
        'mine', 'yours', 'hers', 'ours', 'theirs'
    }
    
    # Convert to lowercase and split into words
    words = text.lower().split()
    
    # Filter out stop words and short words
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Limit to the most meaningful words (up to 3)
    if len(keywords) > 3:
        keywords = keywords[:3]
    
    return ' '.join(keywords)

def clean_query(text):
    """
    Clean and polish the search query to improve search results.
    """
    # Remove special characters and normalize spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove subtitle artifacts and timestamps
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3}', '', text)
    
    # Extract keywords
    keywords = extract_keywords(text)
    
    return keywords.strip() or "nature"  # Return "nature" if no keywords found

def search_video_candidates(query, per_page=15, retries=3):
    """
    Search Pexels API for video candidates with retry logic and fallback options.
    """
    headers = {"Authorization": PEXELS_API_KEY}
    
    for attempt in range(retries):
        try:
            # Try with original query
            if attempt == 0:
                search_query = query
            # First fallback: Try with first word only
            elif attempt == 1:
                search_query = query.split()[0] if query else "nature"
            # Final fallback: Use "nature"
            else:
                search_query = "nature"
            
            params = {
                "query": search_query,
                "per_page": per_page,
                "size": "large",
                "orientation": "landscape"
            }
            
            response = requests.get(PEXELS_VIDEO_SEARCH_URL, headers=headers, params=params)
            
            if response.status_code == 200:
                videos = response.json().get("videos", [])
                if videos:  # If we got any videos, return them
                    return videos
                    
        except Exception as e:
            print(f"Error in attempt {attempt + 1}: {str(e)}")
            continue
    
    # If all attempts fail, return a default nature video search
    params = {"query": "nature", "per_page": per_page}
    response = requests.get(PEXELS_VIDEO_SEARCH_URL, headers=headers, params=params)
    return response.json().get("videos", [])

def get_best_quality_video_url(video_files):
    """
    Get the highest quality video URL from available video files.
    """
    if not video_files:
        return None
        
    # Sort by height (resolution) in descending order
    sorted_files = sorted(video_files, key=lambda x: x.get('height', 0), reverse=True)
    
    # Return the highest quality video URL
    return sorted_files[0].get('link') if sorted_files else None

def main(srt_file):
    subs = pysrt.open(srt_file)
    results = []
    used_videos = set()
    
    for i, sub in enumerate(subs):
        # Clean and extract keywords from current subtitle only
        cleaned_query = clean_query(sub.text)
        
        # Retrieve candidate videos
        candidates = search_video_candidates(cleaned_query)
        
        video_url = None
        best_candidate = None
        
        if candidates:
            # Get first unused video
            for candidate in candidates:
                video_id = candidate.get('id')
                if video_id not in used_videos:
                    best_candidate = candidate
                    used_videos.add(video_id)
                    video_url = get_best_quality_video_url(candidate.get('video_files', []))
                    break
            
            # If all candidates are used, take the first one anyway
            if not video_url and candidates:
                best_candidate = candidates[0]
                video_url = get_best_quality_video_url(best_candidate.get('video_files', []))
        
        results.append({
            "subtitle": sub.text,
            "cleaned_query": cleaned_query,
            "video_url": video_url
        })

    # Print the results
    for res in results:
        print("Subtitle:", res["subtitle"])
        print("Cleaned Query:", res["cleaned_query"])
        print("Matched Video URL:", res["video_url"])
        print("-" * 50)

if __name__ == "__main__":
    main("subtitles.srt")