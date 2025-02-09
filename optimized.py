# -*- coding: utf-8 -*-
from flask import Flask, request, send_file
import io
import zipfile
import json
import soundfile as sf
import numpy as np
from datetime import timedelta
import re
import gc
import tempfile
from kokoro import KPipeline
import spacy
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Memory optimization: Load heavy models once and reuse
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
pipeline = KPipeline(lang_code='a')

# Configure spaCy tokenizer once
infix_re = spacy.util.compile_infix_regex([r'''[\(\)\[\]\,\?\!\:\;\…\–\—]'''])
nlp.tokenizer = spacy.tokenizer.Tokenizer(
    nlp.vocab,
    infix_finditer=infix_re.finditer,
    rules=nlp.Defaults.tokenizer_exceptions
)

class OptimizedSceneGenerator:
    def __init__(self, nlp, sentence_model):
        self.nlp = nlp
        self.model = sentence_model
        self.clean_text = re.compile(r"(\w)[’'](\w)").sub
        self.verb_cache = {}
    
  
    def customize_tokenizer(self):
        """Customize tokenizer to handle hyphens and apostrophes properly."""
        infixes = [r'''[\(\)\[\]\,\?\!\:\;\…\–\—]''']
        infix_re = spacy.util.compile_infix_regex(infixes)
        self.nlp.tokenizer = Tokenizer(
            self.nlp.vocab,
            infix_finditer=infix_re.finditer,
            rules=self.nlp.Defaults.tokenizer_exceptions
        )

    def clean_text(self, text: str):
        """Remove problematic hyphens and apostrophes pre-processing."""
        return re.sub(r"(\w)[’'](\w)", r"\1\2", text)

    def extract_scene_elements(self, text: str):
        """Extract key visual elements with enhanced noun validation."""
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
        """Ensure nouns are valid words."""
        return noun if noun and len(noun) > 1 and noun.isalpha() else None

    def find_subject_object(self, verb):
        """Find valid subject/object with length checks."""
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
        """Convert verb to its progressive form."""
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
        """Find modifiers for the subject."""
        for token in doc:
            if token.text == subject and token.dep_ != "poss":
                for child in token.children:
                    if child.dep_ in ["amod", "compound"] and child.pos_ in ("ADJ", "NOUN"):
                        return child.text
        return None

    def format_description(self, action: str, subject: str, modifier: str = None):
        """Format the description into 2-4 words."""
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
        """Generate a single scene description for a sentence."""
        descriptions = self.extract_scene_elements(sentence)
        if descriptions:
            return descriptions[0].title()
        
        # Fallback: Use first meaningful words as description
        doc = self.nlp(self.clean_text(sentence))
        meaningful_words = [
            token.text for token in doc 
            if token.pos_ in ("NOUN", "VERB", "ADJ") and len(token.text) > 3
        ][:4]
        return " ".join(meaningful_words).title() if meaningful_words else "Scene Transition"


def process_subtitles(input_text):
    input_text = re.sub(r'([,!.:;?—])', r' \1 ', input_text)[:500]  # Limit input size
    return [sent.text.strip() for sent in nlp(input_text).sents if sent.text.strip()]

@app.route('/generate', methods=['POST'])
def generate_content():
    data = request.get_json()
    text = data.get('text', '')[:2000]  # Limit input size
    voice = data.get('voice', 'af_heart')
    speed = max(0.5, min(float(speed), 2.0))  # Validate speed

    # Initialize components with shared models
    scene_generator = OptimizedSceneGenerator(nlp, sentence_model)
    processed_lines = process_subtitles(text)

    # Use temporary files instead of in-memory buffers
    with tempfile.NamedTemporaryFile() as audio_temp, \
         tempfile.NamedTemporaryFile(mode='w+') as subs_temp, \
         tempfile.NamedTemporaryFile(mode='w+') as video_temp:

        # Process audio in chunks
        with sf.SoundFile(audio_temp.name, mode='w', samplerate=24000, channels=1) as f:
            current_time = 0.0
            subtitles = []
            video_data = []
            
            for i, (graphemes, phonemes, audio) in enumerate(
                pipeline(text, voice=voice, speed=speed, split_pattern=r'[.!?]+')
            ):
                # Process audio chunk
                f.write(audio.astype(np.float32))
                duration = len(audio) / 24000
                end_time = current_time + duration

                # Generate metadata
                description = scene_generator.describe_scene(graphemes.strip())
                subtitles.append(f"{i+1}\n"
                    f"{timedelta(seconds=current_time)} --> {timedelta(seconds=end_time)}\n"
                    f"{graphemes.strip()}\n\n")
                video_data.append({
                    "scene_description": description,
                    "duration": round(duration, 2)
                })
                current_time = end_time

                # Manual garbage collection
                del audio
                if i % 10 == 0:
                    gc.collect()

            # Write metadata files
            subs_temp.write(''.join(subtitles))
            video_temp.write(json.dumps(video_data, indent=2))
            subs_temp.seek(0)
            video_temp.seek(0)

            # Create zip in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zf:
                zf.write(audio_temp.name, 'audio.wav')
                zf.writestr('subtitles.srt', subs_temp.read())
                zf.writestr('video_data.json', video_temp.read())
            
            zip_buffer.seek(0)
            return send_file(
                zip_buffer,
                mimetype='application/zip',
                download_name='content_package.zip'
            )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=False)  # Reduce threading overhead