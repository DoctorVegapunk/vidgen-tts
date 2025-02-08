import spacy
from spacy.tokenizer import Tokenizer
from sentence_transformers import SentenceTransformer
import re

class SceneGenerator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.customize_tokenizer()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def customize_tokenizer(self):
        """Customize tokenizer to handle hyphens and apostrophes properly."""
        infixes = [r'''[\(\)\[\]\,\?\!\:\;\…\–\—]''']  # Keep hyphens as single tokens
        infix_re = spacy.util.compile_infix_regex(infixes)
        self.nlp.tokenizer = Tokenizer(
            self.nlp.vocab,
            infix_finditer=infix_re.finditer,
            rules=self.nlp.Defaults.tokenizer_exceptions
        )

    def clean_text(self, text: str):
        """Remove problematic hyphens and apostrophes pre-processing."""
        return re.sub(r"(\w)[’'](\w)", r"\1\2", text)  # Fix contractions like "don’t"

    def extract_scene_elements(self, text: str):
        """Extract key visual elements with enhanced noun validation."""
        doc = self.nlp(self.clean_text(text))
        descriptions = []
        
        # Filter out verbs from contractions (checking both ' and ’)
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
                
            # Validate noun phrases are complete words
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
        """Ensure nouns are valid words (2+ characters, no fragments)."""
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
        """Convert verb to its progressive form with enhanced rules."""
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
        """Find modifiers (amod or compound) for the subject, skipping possessives."""
        for token in doc:
            if token.text == subject and token.dep_ != "poss":
                for child in token.children:
                    if child.dep_ in ["amod", "compound"] and child.pos_ in ("ADJ", "NOUN"):
                        return child.text
        return None

    def format_description(self, action: str, subject: str, modifier: str = None):
        """Format the description into 2-4 words with proper casing."""
        subject = subject.lower()
        if modifier:
            modifier = modifier.lower()
            description = f"{action} {modifier} {subject}"
        else:
            description = f"{action} {subject}"
        
        # Split into words and ensure correct length
        words = description.split()
        if 2 <= len(words) <= 4:
            return " ".join(words)
        elif len(words) > 4:
            return " ".join(words[:4])  # Truncate to 4 words
        else:
            return None

    def process_text(self, text: str):
        """Process entire text and generate scene descriptions."""
        doc = self.nlp(self.clean_text(text))
        descriptions = []
        
        for sent in doc.sents:
            sentence_descriptions = self.extract_scene_elements(sent.text)
            if sentence_descriptions:
                descriptions.extend(sentence_descriptions)
        
        return descriptions

def main():
    text = """
    Imagine waking up one day and realizing that your body is no longer just yours. Every heartbeat, every breath, every meal—it’s now shared with another life. But here’s the thing: pregnancy doesn’t just change a person for nine months. It changes them forever. We all know about the baby bump, but what about the rest? Pregnancy alters nearly every system in the body. The heart grows bigger to pump 50% more blood. Ligaments stretch, bones shift, and the brain actually shrinks—but in a good way! Studies show this may enhance emotional intelligence and maternal instincts. And that’s just the beginning. The body becomes a master of adaptation. The immune system recalibrates itself to protect both mother and child, sometimes leading to a temporary weakening that makes women more susceptible to colds and infections. Blood sugar levels rise, the body stores more fat, and the respiratory system expands to meet the increasing oxygen demands. These changes aren’t just temporary inconveniences; they’re profound evolutionary adaptations ensuring the survival of both lives. Hormones surge like a rollercoaster, leading to mood swings, anxiety, and even postpartum depression. But there’s another side: an increased capacity for empathy, sharper problem-solving skills, and a deep emotional bond that rewires the brain forever. The mind and body sync in an extraordinary way, preparing the mother not just to give birth but to nurture and protect. And here’s what many don’t talk about—pregnancy leaves a lasting mark. Some women develop new allergies, others experience changes in their hair or voice. And incredibly, fetal cells stay in the mother’s body for decades, influencing her immune system and possibly even protecting her from diseases like Alzheimer’s. Scientists call this phenomenon microchimerism, and it’s one of the most fascinating biological mysteries we’re just beginning to understand. Pregnancy is not just about creating life—it’s about transformation in ways we’re only beginning to understand. So the next time you see a mother, remember: she’s not just carrying a child, she’s carrying a new version of herself. And that, my friends, is one of nature’s most profound miracles.
    """
    
    generator = SceneGenerator()
    descriptions = generator.process_text(text)
    
    print("Visual Descriptions:")
    for i, desc in enumerate(descriptions, 1):
        print(f"{i}. {desc.capitalize()}")

if __name__ == "__main__":
    main()