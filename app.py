# -*- coding: utf-8 -*- 
from flask import Flask, request, send_file
import io
import zipfile
import json
import soundfile as sf
import numpy as np
from datetime import timedelta
import re
import requests
import spacy
from spacy.tokenizer import Tokenizer
from sentence_transformers import SentenceTransformer, util
from pydub import AudioSegment
from pydub.effects import normalize
from kokoro import KPipeline
from flask_cors import CORS

# ------------------ Updated Pexels API & Video Search Functions ------------------
PEXELS_API_KEY = "0SzM0QyylRqJK4Vj2EZ9JI8oLyrAd1n0reU7MadOQ0k2SMQ9T9Shv8wr"
PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"

# Global model for video search embeddings (kept for compatibility)
video_search_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----- New Functions from Code 2 -----
def extract_keywords(text):
    """
    Extract meaningful keywords from text using simple but effective rules.
    """
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
    words = text.lower().split()
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    if len(keywords) > 3:
        keywords = keywords[:3]
    return ' '.join(keywords)

def clean_query(text):
    """
    Clean and polish the search query to improve search results.
    """
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3}', '', text)
    keywords = extract_keywords(text)
    return keywords.strip() or "nature"

def search_video_candidates(query, per_page=15, retries=3):
    """
    Search Pexels API for video candidates with retry logic and fallback options.
    """
    headers = {"Authorization": PEXELS_API_KEY}
    
    for attempt in range(retries):
        try:
            if attempt == 0:
                search_query = query
            elif attempt == 1:
                search_query = query.split()[0] if query else "nature"
            else:
                search_query = "nature"
            
            params = {
                "query": search_query,
                "per_page": per_page,
                "size": "medium",
                "orientation": "landscape"
            }
            
            response = requests.get(PEXELS_VIDEO_SEARCH_URL, headers=headers, params=params)
            
            if response.status_code == 200:
                videos = response.json().get("videos", [])
                if videos:
                    return videos
                    
        except Exception as e:
            print(f"Error in attempt {attempt + 1}: {str(e)}")
            continue
    
    # Final fallback search
    params = {"query": "nature", "per_page": per_page}
    response = requests.get(PEXELS_VIDEO_SEARCH_URL, headers=headers, params=params)
    return response.json().get("videos", [])

def get_best_quality_video_url(video_files):
    """
    Get the highest quality video URL from available video files.
    """
    if not video_files:
        return None
    sorted_files = sorted(video_files, key=lambda x: x.get('height', 0), reverse=True)
    return sorted_files[0].get('link') if sorted_files else None

def get_video_link_for_subtitle(index, subtitles_list, used_videos):
    """
    Updated video link retrieval using Code 2 functionality.
    """
    subtitle_text = subtitles_list[index]
    cleaned_query = clean_query(subtitle_text)
    candidates = search_video_candidates(cleaned_query)
    video_url = None
    if candidates:
        for candidate in candidates:
            video_id = candidate.get('id')
            if video_id not in used_videos:
                used_videos.add(video_id)
                video_url = get_best_quality_video_url(candidate.get('video_files', []))
                break
        if not video_url and candidates:
            candidate = candidates[0]
            video_url = get_best_quality_video_url(candidate.get('video_files', []))
    return video_url

# ------------------ Scene Generation & Subtitle Processing ------------------
nlp = spacy.load("en_core_web_sm")
infixes = [r'''[\(\)\[\]\,\?\!\:\;\…\–\—]''']
infix_re = spacy.util.compile_infix_regex(infixes)
nlp.tokenizer = Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer, rules=nlp.Defaults.tokenizer_exceptions)

class SceneGenerator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.customize_tokenizer()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    def customize_tokenizer(self):
        infixes = [r'''[\(\)\[\]\,\?\!\:\;\…\–\—]''']
        infix_re = spacy.util.compile_infix_regex(infixes)
        self.nlp.tokenizer = Tokenizer(
            self.nlp.vocab,
            infix_finditer=infix_re.finditer,
            rules=self.nlp.Defaults.tokenizer_exceptions
        )
    def clean_text(self, text: str):
        return re.sub(r"(\w)[’'](\w)", r"\1\2", text)
    def extract_scene_elements(self, text: str):
        doc = self.nlp(self.clean_text(text))
        descriptions = []
        verbs = [
            token for token in doc 
            if token.pos_ == "VERB" 
            and not any(c in token.text for c in ["'", "’"])
        ]
        for verb in verbs:
            main_verb = verb.lemma_.lower()
            subject, obj = self.find_subject_object(verb)
            if not (subject or obj):
                continue
            target_noun = self.validate_noun(subject) or self.validate_noun(obj)
            if not target_noun:
                continue
            modifier = self.get_modifier(doc, target_noun)
            action = self.get_progressive_verb(main_verb)
            description = self.format_description(action, target_noun, modifier)
            if description and len(description.split()) >= 2:
                descriptions.append(description)
        return descriptions if descriptions else None
    def validate_noun(self, noun: str):
        return noun if noun and len(noun) > 1 and noun.isalpha() else None
    def find_subject_object(self, verb):
        subj = next(
            (child.text for child in verb.children 
             if child.dep_ == "nsubj" and child.pos_ in ("NOUN", "PROPN") and len(child.text) > 1),
            None
        )
        obj = next(
            (child.text for child in verb.children 
             if child.dep_ in ["dobj", "pobj"] and child.pos_ in ("NOUN", "PROPN") and len(child.text) > 1),
            None
        )
        return subj, obj
    def get_progressive_verb(self, verb: str):
        verb = verb.lower()
        if verb == "be":
            return "being"
        elif verb.endswith("ee"):
            return verb + "ing"
        elif verb.endswith("ie"):
            return verb[:-2] + "ying"
        elif verb.endswith("e") and len(verb) > 1 and verb != "ee":
            return verb[:-1] + "ing"
        else:
            return verb + "ing"
    def get_modifier(self, doc, subject: str):
        for token in doc:
            if token.text == subject and token.dep_ != "poss":
                for child in token.children:
                    if child.dep_ in ["amod", "compound"] and child.pos_ in ("ADJ", "NOUN"):
                        return child.text
        return None
    def format_description(self, action: str, subject: str, modifier: str = None):
        subject = subject.lower()
        if modifier:
            modifier = modifier.lower()
            description = f"{action} {modifier} {subject}"
        else:
            description = f"{action} {subject}"
        words = description.split()
        if 2 <= len(words) <= 4:
            return " ".join(words)
        elif len(words) > 4:
            return " ".join(words[:4])
        else:
            return None
    def describe_scene(self, sentence: str):
        descriptions = self.extract_scene_elements(sentence)
        if descriptions:
            return descriptions[0].title()
        doc = self.nlp(self.clean_text(sentence))
        meaningful_words = [
            token.text for token in doc 
            if token.pos_ in ("NOUN", "VERB", "ADJ") and len(token.text) > 3
        ][:4]
        return " ".join(meaningful_words).title() if meaningful_words else "Scene Transition"

def process_subtitles(input_text):
    input_text = re.sub(r'([,!.:;?—])', r' \1 ', input_text)
    input_text = re.sub(r'\s{2,}', ' ', input_text)
    doc = nlp(input_text)
    processed = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if len(sent_text) < 1:
            continue
        if len(sent) > 14:
            chunks = []
            current_chunk = []
            for token in sent:
                if token.text in [",", ";", "—", ":"] and len(current_chunk) > 4:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                current_chunk.append(token.text)
            if current_chunk:
                chunks.append(" ".join(current_chunk))
        else:
            chunks = [sent_text]
        processed.extend(chunks)
    return [c.strip() for c in processed if c.strip()]

# ------------------ Flask App & Pipeline Integration ------------------
pipeline = KPipeline(lang_code='a')
app = Flask(__name__)
CORS(app) 

@app.route('/generate', methods=['POST'])
def generate_content():
    data = request.get_json()
    text_input = data.get('text', '')
    voice = data.get('voice', 'af_heart')
    speed = data.get('speed', 1.2)
    edit_video_data = data.get('edit_video_data', False)
    
    # Process text into subtitle lines
    processed_lines = process_subtitles(text_input)
    scene_generator = SceneGenerator()
    
    full_audio = []
    subtitles = []
    subtitle_texts = []   # Plain text for each subtitle (for video search)
    durations_list = []   # Duration (in seconds) for each subtitle
    current_time = 0.0
    used_videos = set()   # Track used video IDs
    
    # Generate content using your pipeline
    generator = pipeline(text_input, voice=voice, speed=speed, split_pattern=r'[.!?]+')
    for i, (graphemes, phonemes, audio) in enumerate(generator):
        duration = len(audio) / 24000.0
        end_time = current_time + duration
        
        subtitle_text = graphemes.strip()
        subtitle_texts.append(subtitle_text)
        
        sub_start = str(timedelta(seconds=current_time)).split('.')[0] + ',000'
        sub_end = str(timedelta(seconds=end_time + (duration * 0.05))).split('.')[0] + ',000'
        subtitles.append(f"{i+1}\n{sub_start} --> {sub_end}\n{subtitle_text}\n\n")
        
        full_audio.append(audio)
        durations_list.append(round(duration, 2))
        current_time = end_time
    
    # Process combined audio
    combined_audio = np.concatenate(full_audio)
    raw_audio_buffer = io.BytesIO()
    sf.write(raw_audio_buffer, combined_audio, 24000, format='WAV')
    raw_audio_buffer.seek(0)
    audio_seg = normalize(AudioSegment.from_wav(raw_audio_buffer))
    audio_seg = audio_seg.low_pass_filter(1600).high_pass_filter(100).fade_in(200).fade_out(500)
    final_audio = io.BytesIO()
    audio_seg.export(final_audio, format="wav")
    final_audio.seek(0)
    
    # Generate video_data: include subtitle text when editing manually
    video_data = []
    if edit_video_data:
        for i, duration in enumerate(durations_list):
            query = clean_query(subtitle_texts[i])
            video_data.append({
                "subtitle": subtitle_texts[i],  # Include the original subtitle line
                "search_query": query,
                "duration": duration
            })
    else:
        for i, duration in enumerate(durations_list):
            video_url = get_video_link_for_subtitle(i, subtitle_texts, used_videos)
            video_data.append({
                "video_url": video_url,
                "duration": duration
            })
    
    # Create a zip package with audio, subtitles, and video_data JSON
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('audio.wav', final_audio.getvalue())
        zf.writestr('subtitles.srt', ''.join(subtitles))
        zf.writestr('video_data.json', json.dumps(video_data, indent=2))
    
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', download_name='content_package.zip')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
