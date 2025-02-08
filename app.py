# -*- coding: utf-8 -*-
from flask import Flask, request, send_file
import io
import zipfile
import json
import soundfile as sf
import numpy as np
from datetime import timedelta
import re
from kokoro import KPipeline
import spacy
from spacy.tokenizer import Tokenizer
from sentence_transformers import SentenceTransformer
from pydub import AudioSegment
from pydub.effects import normalize

app = Flask(__name__)

# Initialize NLP components
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

# Initialize the pipeline
pipeline = KPipeline(lang_code='a')

# Original text
text = "Imagine waking up one day and realizing that your body is no longer just yours. Every heartbeat, every breath, every meal—it’s now shared with another life. But here's the thing: pregnancy doesn’t just change a person for nine months. It changes them forever. We all know about the baby bump, but what about the rest? Pregnancy alters nearly every system in the body. The heart grows bigger to pump 50% more blood. Ligaments stretch, bones shift, and the brain actually shrinks—but in a good way! Studies show this may enhance emotional intelligence and maternal instincts. And that’s just the beginning. The body becomes a master of adaptation. The immune system recalibrates itself to protect both mother and child, sometimes leading to a temporary weakening that makes women more susceptible to colds and infections. Blood sugar levels rise, the body stores more fat, and the respiratory system expands to meet the increasing oxygen demands. These changes aren’t just temporary inconveniences; they’re profound evolutionary adaptations ensuring the survival of both lives. Hormones surge like a roller-coaster, leading to mood swings, anxiety, and even postpartum depression. But there’s another side: an increased capacity for empathy, sharper problem-solving skills, and a deep emotional bond that rewires the brain forever. The mind and body sync in an extraordinary way, preparing the mother not just to give birth but to nurture and protect. And here’s what many don’t talk about—pregnancy leaves a lasting mark. Some women develop new allergies, others experience changes in their hair or voice. And incredibly, fetal cells stay in the mother’s body for decades, influencing her immune system and possibly even protecting her from diseases like Alzheimer’s. Scientists call this phenomenon micro-chimerism, and it’s one of the most fascinating biological mysteries we’re just beginning to understand. Pregnancy is not just about creating life—it’s about transformation in ways we’re only beginning to understand. So the next time you see a mother, remember: she’s not just carrying a child, she’s carrying a new version of herself. And that, my friends, is one of nature’s most profound miracles."""

# Modified process_subtitles function with better punctuation handling
def process_subtitles(input_text):
    # Add space around punctuation for better TTS processing
    input_text = re.sub(r'([,!.:;?—])', r' \1 ', input_text)
    input_text = re.sub(r'\s{2,}', ' ', input_text)
    
    doc = nlp(input_text)
    processed = []
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if len(sent_text) < 1:
            continue
            
        # Split long sentences at natural pause points
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

processed_lines = process_subtitles(text)

@app.route('/generate', methods=['POST'])
def generate_content():
    data = request.get_json()
    text = data.get('text', '')
    voice = data.get('voice', 'af_heart')
    speed = data.get('speed', 1.2)
    
    # Process text
    processed_lines = process_subtitles(text)
    scene_generator = SceneGenerator()
    
    # Generate content
    generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'[.!?]+')
    
    full_audio, subtitles, video_data = [], [], []
    current_time = 0.0

    for i, (graphemes, phonemes, audio) in enumerate(generator):
        duration = len(audio) / 24000
        end_time = current_time + duration
        
        description = scene_generator.describe_scene(graphemes.strip())
        sub_start = str(timedelta(seconds=current_time)).split('.')[0] + ',000'
        sub_end = str(timedelta(seconds=end_time + (duration * 0.05))).split('.')[0] + ',000'
        
        subtitles.append(f"{i+1}\n{sub_start} --> {sub_end}\n{graphemes.strip()}\n\n")
        full_audio.append(audio)
        video_data.append({"scene_description": description, "duration": round(duration, 2)})
        current_time = end_time

    # Process audio
    combined_audio = np.concatenate(full_audio)
    raw_audio_buffer = io.BytesIO()
    sf.write(raw_audio_buffer, combined_audio, 24000, format='WAV')
    
    raw_audio_buffer.seek(0)
    audio = normalize(AudioSegment.from_wav(raw_audio_buffer))
    audio = audio.low_pass_filter(1600).high_pass_filter(100).fade_in(200).fade_out(500)
    
    final_audio = io.BytesIO()
    audio.export(final_audio, format="wav")
    final_audio.seek(0)

    # Create zip package
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('audio.wav', final_audio.getvalue())
        zf.writestr('subtitles.srt', ''.join(subtitles))
        zf.writestr('video_data.json', json.dumps(video_data, indent=2))
    
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', download_name='content_package.zip')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)