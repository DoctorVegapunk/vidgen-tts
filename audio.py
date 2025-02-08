# -*- coding: utf-8 -*-
from kokoro import KPipeline
import soundfile as sf
import numpy as np

# Initialize pipeline with American English
pipeline = KPipeline(lang_code='a')

text = "Imagine waking up one day and realizing that your body is no longer just yours. Every heartbeat, every breath, every meal—it's now shared with another life. But here's the thing: pregnancy doesn't just change a person for nine months. It changes them forever. We all know about the baby bump, but what about the rest? Pregnancy alters nearly every system in the body. The heart grows bigger to pump 50% more blood. Ligaments stretch, bones shift, and the brain actually shrinks—but in a good way! Studies show this may enhance emotional intelligence and maternal instincts. And that's just the beginning. The body becomes a master of adaptation. The immune system recalibrates itself to protect both mother and child, sometimes leading to a temporary weakening that makes women more susceptible to colds and infections. Blood sugar levels rise, the body stores more fat, and the respiratory system expands to meet the increasing oxygen demands. These changes aren't just temporary inconveniences; they're profound evolutionary adaptations ensuring the survival of both lives. Hormones surge like a roller-coaster, leading to mood swings, anxiety, and even postpartum depression. But there's another side: an increased capacity for empathy, sharper problem-solving skills, and a deep emotional bond that rewires the brain forever. The mind and body sync in an extraordinary way, preparing the mother not just to give birth but to nurture and protect. And here's what many don't talk about—pregnancy leaves a lasting mark. Some women develop new allergies, others experience changes in their hair or voice. And incredibly, fetal cells stay in the mother's body for decades, influencing her immune system and possibly even protecting her from diseases like Alzheimer's. Scientists call this phenomenon micro-chimerism, and it's one of the most fascinating biological mysteries we're just beginning to understand. Pregnancy is not just about creating life—it's about transformation in ways we're only beginning to understand. So the next time you see a mother, remember: she's not just carrying a child, she's carrying a new version of herself. And that, my friends, is one of nature's most profound miracles."

# Generate audio
generator = pipeline(
    text,
    voice='af_heart',
    speed=1.2,
    split_pattern=r'[.!?]+'
)

# Collect all audio segments
audio_segments = []
for _, _, audio in generator:
    audio_segments.append(audio)

# Merge and save all segments
merged_audio = np.concatenate(audio_segments)
sf.write('merged_output.wav', merged_audio, 24000)
print("Merged audio saved as merged_output.wav")