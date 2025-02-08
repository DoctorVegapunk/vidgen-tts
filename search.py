# -*- coding: utf-8 -*-
from kokoro import KPipeline
import soundfile as sf
import numpy as np
from datetime import timedelta
import json
import spacy
from collections import Counter

# Load English language model
nlp = spacy.load("en_core_web_sm")

class ContextualTermGenerator:
    def __init__(self, full_text):
        self.full_doc = nlp(full_text)
        self.theme_words = self._identify_core_themes()
        
    def _identify_core_themes(self):
        """Extract 3 main themes from the entire text"""
        nouns = [token.text.lower() for token in self.full_doc 
                if token.pos_ in ("NOUN", "PROPN") and not token.is_stop]
        return [word for word, count in Counter(nouns).most_common(3)]

    def generate_term(self, sentence):
        """Create context-aware search term for a sentence"""
        doc = nlp(sentence)
        
        # Extract meaningful components
        main_entities = [ent.text for ent in doc.ents]
        verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        nouns = [token.text for token in doc 
                if token.pos_ in ("NOUN", "PROPN") and not token.is_stop]
        
        # Prioritize entities then nouns then verbs
        components = main_entities + nouns + verbs
        
        # Add theme context if completely missing
        if not any(theme in " ".join(components).lower() for theme in self.theme_words):
            components += self.theme_words
            
        # Select 3-4 most significant words
        selected = []
        for word in components:
            if word.lower() not in selected and len(selected) < 4:
                selected.append(word)
        
        return " ".join(selected[:4]).title()

# Initialize pipeline once
pipeline = KPipeline(lang_code='a')

def process_speech(text, voice='af_heart'):
    # Create context analyzer
    term_generator = ContextualTermGenerator(text)
    
    # Generate audio and subtitles
    generator = pipeline(text, voice=voice, speed=1, split_pattern=r'\n+')
    
    # Processing variables
    full_audio = []
    subtitles = []
    video_data = []
    current_time = 0.0

    for i, (graphemes, phonemes, audio) in enumerate(generator):
        duration = len(audio) / 24000
        end_time = current_time + duration
        
        # Generate smart search term
        search_term = term_generator.generate_term(graphemes)
        
        # Store results
        subtitles.append(
            f"{i+1}\n{timedelta(seconds=current_time)} --> {timedelta(seconds=end_time)}\n{graphemes.strip()}\n\n"
        )
        video_data.append({
            "term": search_term,
            "duration": round(duration, 2)
        })
        full_audio.append(audio)
        current_time = end_time
    
    # Save outputs
    sf.write("output.wav", np.concatenate(full_audio), 24000)
    with open("subtitles.srt", "w", encoding='utf-8') as f:
        f.writelines(subtitles)
    
    return video_data

# Example usage with pregnancy text
pregnancy_text = """Imagine waking up one day and realizing that your body is no longer just yours. Every heartbeat, every breath, every meal—it’s now shared with another life. But here’s the thing: pregnancy doesn’t just change a person for nine months. It changes them forever. We all know about the baby bump, but what about the rest? Pregnancy alters nearly every system in the body. The heart grows bigger to pump 50% more blood. Ligaments stretch, bones shift, and the brain actually shrinks—but in a good way! Studies show this may enhance emotional intelligence and maternal instincts. And that’s just the beginning. The body becomes a master of adaptation. The immune system recalibrates itself to protect both mother and child, sometimes leading to a temporary weakening that makes women more susceptible to colds and infections. Blood sugar levels rise, the body stores more fat, and the respiratory system expands to meet the increasing oxygen demands. These changes aren’t just temporary inconveniences; they’re profound evolutionary adaptations ensuring the survival of both lives. Hormones surge like a rollercoaster, leading to mood swings, anxiety, and even postpartum depression. But there’s another side: an increased capacity for empathy, sharper problem-solving skills, and a deep emotional bond that rewires the brain forever. The mind and body sync in an extraordinary way, preparing the mother not just to give birth but to nurture and protect. And here’s what many don’t talk about—pregnancy leaves a lasting mark. Some women develop new allergies, others experience changes in their hair or voice. And incredibly, fetal cells stay in the mother’s body for decades, influencing her immune system and possibly even protecting her from diseases like Alzheimer’s. Scientists call this phenomenon micro-chimerism, and it’s one of the most fascinating biological mysteries we’re just beginning to understand. Pregnancy is not just about creating life—it’s about transformation in ways we’re only beginning to understand. So the next time you see a mother, remember: she’s not just carrying a child, she’s carrying a new version of herself. And that, my friends, is one of nature’s most profound miracles."""  # Your original text
results = process_speech(pregnancy_text)
print(json.dumps(results, indent=2))
