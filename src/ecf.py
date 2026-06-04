"""
EMOTION-CONSISTENCY FORENSICS (ECF) MODULE ★
Unique feature: Cross-checks facial emotion against speech sentiment.

Pipeline:
  1. FER → Detect facial emotion from video frames
  2. Whisper → Transcribe speech from audio
  3. VADER → Analyze sentiment of transcription
  4. Compare emotion vs sentiment → ECF Inconsistency Score
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image

from config.config import (
    WHISPER_MODEL_SIZE, ECF_WEIGHT, VISUAL_WEIGHT,
    EMOTION_SENTIMENT_MAP, DEVICE
)


# ──────────────────────────────────────────────
# 1. FACIAL EMOTION RECOGNITION (FER)
# ──────────────────────────────────────────────

def detect_facial_emotion(face_image):
    """
    Detects the dominant facial emotion from a face image.
    Uses the FER library (pre-trained on AffectNet).
    
    Args:
        face_image: PIL Image of a face
    
    Returns:
        dict: {'dominant_emotion': str, 'emotions': dict}
    """
    from fer import FER
    
    detector = FER(mtcnn=False)  # Use default detector (face already cropped)
    
    # FER expects numpy array
    img_array = np.array(face_image)
    
    try:
        result = detector.detect_emotions(img_array)
        if result:
            emotions = result[0]['emotions']
            dominant = max(emotions, key=emotions.get)
            return {
                'dominant_emotion': dominant,
                'emotions': emotions
            }
    except Exception as e:
        print(f"⚠️ FER error: {e}")
    
    # Fallback: use DeepFace or return neutral
    return _fer_fallback(face_image)


def _fer_fallback(face_image):
    """
    Fallback emotion detection using a simple approach.
    """
    try:
        from deepface import DeepFace
        result = DeepFace.analyze(
            img_path=np.array(face_image),
            actions=['emotion'],
            enforce_detection=False
        )
        if isinstance(result, list):
            result = result[0]
        dominant = result['dominant_emotion']
        emotions = result['emotion']
        return {
            'dominant_emotion': dominant,
            'emotions': emotions
        }
    except:
        pass
    
    # Ultimate fallback
    return {
        'dominant_emotion': 'neutral',
        'emotions': {'neutral': 1.0}
    }


def analyze_emotions_across_frames(face_crops):
    """
    Analyzes emotions across multiple frames and returns
    the most common (dominant) emotion.
    
    Args:
        face_crops: List of PIL Images (face crops from frames)
    
    Returns:
        dict with keys: 'dominant_emotion', 'frame_emotions', 'emotion_counts'
    """
    frame_emotions = []
    emotion_counts = {}
    
    print("  😊 Analyzing facial emotions across frames...")
    
    for i, face in enumerate(face_crops):
        result = detect_facial_emotion(face)
        emotion = result['dominant_emotion']
        frame_emotions.append(emotion)
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # Most frequent emotion
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    
    print(f"     Dominant emotion: {dominant_emotion}")
    print(f"     Distribution: {emotion_counts}")
    
    return {
        'dominant_emotion': dominant_emotion,
        'frame_emotions': frame_emotions,
        'emotion_counts': emotion_counts
    }


# ──────────────────────────────────────────────
# 2. SPEECH-TO-TEXT (Whisper)
# ──────────────────────────────────────────────

_transcribe_model = None

def transcribe_audio(audio_path):
    """
    Transcribes audio file to text using OpenAI Whisper.
    
    Args:
        audio_path: Path to WAV/MP3 audio file
    
    Returns:
        dict: {'text': str, 'language': str}
    """
    global _transcribe_model
    
    print("  🎤 Transcribing audio with Whisper...")
    
    if _transcribe_model is None:
        import whisper
        print(f"     Loading Whisper '{WHISPER_MODEL_SIZE}' model...")
        _transcribe_model = whisper.load_model(WHISPER_MODEL_SIZE)
    
    try:
        result = _transcribe_model.transcribe(str(audio_path))
        text = result['text'].strip()
        language = result.get('language', 'unknown')
        
        print(f"     Language: {language}")
        print(f"     Transcript: {text[:100]}{'...' if len(text)>100 else ''}")
        
        return {
            'text': text,
            'language': language
        }
    except Exception as e:
        print(f"  ❌ Transcription failed: {e}")
        return {'text': '', 'language': 'unknown'}


# ──────────────────────────────────────────────
# 3. SENTIMENT ANALYSIS (VADER)
# ──────────────────────────────────────────────

_analyzer = None

def analyze_sentiment(text):
    """
    Analyzes sentiment of text using VADER.
    
    Args:
        text: Input text string
    
    Returns:
        dict: {'sentiment': str, 'scores': dict, 'compound': float}
    """
    global _analyzer
    
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    
    scores = _analyzer.polarity_scores(text)
    compound = scores['compound']
    
    # Classify sentiment
    if compound >= 0.05:
        sentiment = 'positive'
    elif compound <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return {
        'sentiment': sentiment,
        'scores': scores,
        'compound': compound
    }


# ──────────────────────────────────────────────
# 4. ECF INCONSISTENCY SCORING
# ──────────────────────────────────────────────

def compute_ecf_score(facial_emotion, speech_sentiment):
    """
    Computes the ECF Inconsistency Score (0 to 1).
    
    Logic:
      - If facial emotion sentiment matches speech sentiment → low score (consistent)
      - If they mismatch → high score (inconsistent, likely deepfake)
    
    Args:
        facial_emotion: str (e.g., 'happy', 'sad', 'angry', ...)
        speech_sentiment: str ('positive', 'negative', 'neutral')
    
    Returns:
        float: ECF score between 0 (consistent) and 1 (inconsistent)
    """
    # Map facial emotion to expected sentiment
    expected_sentiment = EMOTION_SENTIMENT_MAP.get(
        facial_emotion.lower(), 'neutral'
    )
    
    if expected_sentiment == 'neutral' or speech_sentiment == 'neutral':
        # Neutral is compatible with anything → low inconsistency
        return 0.2
    
    if expected_sentiment == speech_sentiment:
        # Emotion matches sentiment → consistent → low score
        return 0.1
    else:
        # Mismatch → inconsistent → high score
        # Stronger mismatches get higher scores
        if (expected_sentiment == 'positive' and speech_sentiment == 'negative') or \
           (expected_sentiment == 'negative' and speech_sentiment == 'positive'):
            return 0.9  # Strong mismatch
        else:
            return 0.6  # Moderate mismatch


# ──────────────────────────────────────────────
# 5. COMPLETE ECF PIPELINE
# ──────────────────────────────────────────────

def run_ecf_analysis(face_crops, audio_path):
    """
    COMPLETE ECF ANALYSIS PIPELINE
    
    Args:
        face_crops: List of PIL Images (face crops from video frames)
        audio_path: Path to extracted audio file
    
    Returns:
        dict with all ECF analysis results
    """
    print("\n" + "="*50)
    print("🔍 ECF ANALYSIS (Emotion-Consistency Forensics)")
    print("="*50)
    
    # Step 1: Facial Emotion Recognition
    emotion_result = analyze_emotions_across_frames(face_crops)
    facial_emotion = emotion_result['dominant_emotion']
    text = ''
    # Step 2: Speech-to-Text
    if audio_path and Path(audio_path).exists():
        transcription = transcribe_audio(audio_path)
        text = transcription['text']
        
        # Step 3: Sentiment Analysis
        if text.strip():
            sentiment_result = analyze_sentiment(text)
            speech_sentiment = sentiment_result['sentiment']
        else:
            sentiment_result = {'sentiment': 'neutral', 'scores': {}, 'compound': 0.0}
            speech_sentiment = 'neutral'
            print("  ⚠️  No speech detected, defaulting to neutral sentiment")
    else:
        transcription = {'text': '', 'language': 'N/A'}
        sentiment_result = {'sentiment': 'neutral', 'scores': {}, 'compound': 0.0}
        speech_sentiment = 'neutral'
        print("  ⚠️  No audio available, defaulting to neutral sentiment")
    
    # Step 4: Compute Inconsistency Score
    ecf_score = compute_ecf_score(facial_emotion, speech_sentiment)
    
    # Compile report
    report = {
        'facial_emotion': facial_emotion,
        'emotion_distribution': emotion_result['emotion_counts'],
        'transcription': transcription['text'],
        'transcription_language': transcription.get('language', 'N/A'),
        'speech_sentiment': speech_sentiment,
        'sentiment_scores': sentiment_result.get('scores', {}),
        'ecf_inconsistency_score': ecf_score,
        'verdict': 'INCONSISTENT' if ecf_score > 0.5 else 'CONSISTENT'
    }
    
    # Print report
    print(f"\n📋 ECF REPORT:")
    print(f"  Facial Emotion:   {facial_emotion}")
    print(f"  Speech Sentiment: {speech_sentiment}")
    print(f"  ECF Score:        {ecf_score:.2f} (0=consistent, 1=inconsistent)")
    print(f"  Verdict:          {report['verdict']}")
    
    if text:
        print(f"  Transcript:       {text[:150]}...")
    
    return report