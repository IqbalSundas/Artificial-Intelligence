"""
FUSION MODULE
Combines Visual CNN score + ECF score into a final verdict.
"""

from config.config import ECF_WEIGHT, VISUAL_WEIGHT, LABEL_MAP


def fuse_scores(visual_score, ecf_score, visual_weight=None, ecf_weight=None):
    """
    Fuses the visual CNN fake probability with the ECF inconsistency score
    using a weighted average.
    
    Args:
        visual_score: float [0,1] — probability of being FAKE from CNN
        ecf_score: float [0,1] — ECF inconsistency score
        visual_weight: weight for visual score (default from config)
        ecf_weight: weight for ECF score (default from config)
    
    Returns:
        dict: {
            'final_score': float,
            'verdict': str,
            'confidence': str,
            'visual_contribution': float,
            'ecf_contribution': float
        }
    """
    vw = visual_weight if visual_weight is not None else VISUAL_WEIGHT
    ew = ecf_weight if ecf_weight is not None else ECF_WEIGHT
    
    # Weighted average
    final_score = (vw * visual_score) + (ew * ecf_score)
    # Normalize
    total_weight = vw + ew
    final_score = final_score / total_weight
    
    # Determine verdict
    if final_score >= 0.5:
        verdict = "FAKE"
    else:
        verdict = "REAL"
    
    # Confidence level
    confidence = abs(final_score - 0.5) * 2  # 0 to 1, how far from decision boundary
    if confidence > 0.6:
        confidence_label = "HIGH"
    elif confidence > 0.3:
        confidence_label = "MEDIUM"
    else:
        confidence_label = "LOW"
    
    return {
        'final_score': final_score,
        'verdict': verdict,
        'confidence_label': confidence_label,
        'confidence_pct': final_score * 100 if verdict == "FAKE" else (1 - final_score) * 100,
        'visual_score': visual_score,
        'ecf_score': ecf_score,
        'visual_contribution': (vw * visual_score) / total_weight,
        'ecf_contribution': (ew * ecf_score) / total_weight,
        'weights_used': {'visual': vw, 'ecf': ew}
    }


def generate_final_report(fusion_result, ecf_report, visual_details=None):
    """
    Generates a comprehensive text report.
    
    Args:
        fusion_result: Output from fuse_scores()
        ecf_report: Output from run_ecf_analysis()
        visual_details: Optional dict with per-frame visual predictions
    
    Returns:
        str: Formatted report
    """
    report = []
    report.append("=" * 60)
    report.append("🔒 DEEPFAKE DETECTION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Verdict
    report.append(f"  VERDICT: {fusion_result['verdict']}")
    report.append(f"  Confidence: {fusion_result['confidence_pct']:.1f}% ({fusion_result['confidence_label']})")
    report.append(f"  Combined Score: {fusion_result['final_score']:.3f}")
    report.append("")
    
    # Visual Analysis
    report.append("  📸 VISUAL ANALYSIS (CNN)")
    report.append(f"     Visual Fake Score: {fusion_result['visual_score']:.3f}")
    report.append(f"     Contribution: {fusion_result['visual_contribution']:.3f}")
    report.append("")
    
    # ECF Analysis
    report.append("  🎭 ECF ANALYSIS (Emotion-Consistency)")
    report.append(f"     Facial Emotion: {ecf_report['facial_emotion']}")
    report.append(f"     Speech Sentiment: {ecf_report['speech_sentiment']}")
    report.append(f"     ECF Inconsistency: {ecf_report['ecf_inconsistency_score']:.3f}")
    report.append(f"     ECF Verdict: {ecf_report['verdict']}")
    if ecf_report.get('transcription'):
        report.append(f"     Transcript: {ecf_report['transcription'][:200]}")
    report.append("")
    
    # Weights
    report.append(f"  ⚖️  FUSION WEIGHTS")
    report.append(f"     Visual: {fusion_result['weights_used']['visual']}")
    report.append(f"     ECF: {fusion_result['weights_used']['ecf']}")
    report.append("")
    
    report.append("=" * 60)
    
    text = "\n".join(report)
    print(text)
    
    return text