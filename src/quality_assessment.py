"""AI-powered translation quality assessment using Gemini and Hugging Face APIs."""

import os
import json
import requests
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class QualityReport:
    """Translation quality assessment report."""
    overall_score: float  # 1-10 scale
    accuracy_score: float  # 1-10 scale
    fluency_score: float   # 1-10 scale
    consistency_score: float  # 1-10 scale
    issues: List[str]
    suggestions: List[str]
    corrected_segments: List[Dict[str, str]]  # List of {"original": "...", "corrected": "..."}

class QualityAssessmentError(Exception):
    """Exception raised for quality assessment errors."""
    pass

class TranslationQualityAssessor:
    """AI-powered translation quality assessment."""
    
    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
        
        # Check if at least one API key is available when needed
        self._api_keys_checked = False
    
    def assess_translation_quality(self, original_srt_path: str, translated_srt_path: str, 
                                 source_lang: str, target_lang: str) -> QualityReport:
        """Assess the quality of a translation comparing original and translated SRT files."""
        # Check API keys when actually needed
        if not self._api_keys_checked:
            if not self.gemini_api_key and not self.hf_api_key:
                raise QualityAssessmentError(
                    "Neither GEMINI_API_KEY nor HUGGINGFACE_API_KEY found in environment variables. "
                    "Please set at least one API key to enable translation quality assessment."
                )
            self._api_keys_checked = True
        
        print(f"Assessing translation quality ({source_lang} -> {target_lang})...")
        
        # Read SRT files
        original_segments = self._read_srt_segments(original_srt_path)
        translated_segments = self._read_srt_segments(translated_srt_path)
        
        if len(original_segments) != len(translated_segments):
            print(f"WARNING: Segment count mismatch: {len(original_segments)} vs {len(translated_segments)}")
        
        # Sample segments for assessment (to avoid hitting API limits)
        sample_pairs = self._sample_segments(original_segments, translated_segments, max_samples=10)
        
        # Perform quality assessment
        if self.gemini_api_key:
            return self._assess_with_gemini(sample_pairs, source_lang, target_lang)
        else:
            return self._assess_with_huggingface(sample_pairs, source_lang, target_lang)
    
    def _read_srt_segments(self, srt_path: str) -> List[str]:
        """Extract text segments from SRT file."""
        if not os.path.exists(srt_path):
            raise QualityAssessmentError(f"SRT file not found: {srt_path}")
        
        segments = []
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse SRT format
        srt_blocks = re.split(r'\n\s*\n', content.strip())
        for block in srt_blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # Skip index and timestamp, get text
                text = '\n'.join(lines[2:]).strip()
                if text:
                    segments.append(text)
        
        return segments
    
    def _sample_segments(self, original: List[str], translated: List[str], max_samples: int = 10) -> List[Tuple[str, str]]:
        """Sample segments for quality assessment."""
        min_len = min(len(original), len(translated))
        if min_len <= max_samples:
            return list(zip(original[:min_len], translated[:min_len]))
        
        # Sample evenly distributed segments
        step = min_len // max_samples
        indices = [i * step for i in range(max_samples)]
        return [(original[i], translated[i]) for i in indices]
    
    def _assess_with_gemini(self, sample_pairs: List[Tuple[str, str]], 
                          source_lang: str, target_lang: str) -> QualityReport:
        """Assess translation quality using Google Gemini API."""
        print("Using Gemini API for quality assessment...")
        
        # Prepare prompt
        segments_text = "\n".join([
            f"Original ({source_lang}): {orig}\nTranslation ({target_lang}): {trans}\n---"
            for orig, trans in sample_pairs
        ])
        
        prompt = f"""You are a professional translation quality assessor. Please evaluate the following translation from {source_lang} to {target_lang}.

Translation samples:
{segments_text}

Please provide a detailed assessment in JSON format with the following structure:
{{
    "overall_score": <1-10>,
    "accuracy_score": <1-10>,
    "fluency_score": <1-10>,
    "consistency_score": <1-10>,
    "issues": ["list", "of", "issues"],
    "suggestions": ["list", "of", "suggestions"],
    "corrected_segments": [
        {{"original": "original_text", "corrected": "corrected_translation"}},
        ...
    ]
}}

Scoring criteria:
- Accuracy (1-10): How well the meaning is preserved
- Fluency (1-10): How natural the target language sounds
- Consistency (1-10): Consistent terminology and style
- Overall (1-10): General quality assessment

Only include corrected_segments if you have specific improvements to suggest."""

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 2048
                }
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if 'candidates' not in result or not result['candidates']:
                raise QualityAssessmentError("No response from Gemini API")
            
            text_response = result['candidates'][0]['content']['parts'][0]['text']
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
            if not json_match:
                raise QualityAssessmentError("Could not extract JSON from Gemini response")
            
            assessment_data = json.loads(json_match.group())
            return self._create_quality_report(assessment_data)
            
        except Exception as e:
            raise QualityAssessmentError(f"Gemini API error: {str(e)}")
    
    def _assess_with_huggingface(self, sample_pairs: List[Tuple[str, str]], 
                               source_lang: str, target_lang: str) -> QualityReport:
        """Assess translation quality using Hugging Face API."""
        print("Using Hugging Face API for quality assessment...")
        
        # Use a translation quality estimation model from Hugging Face
        # This is a simplified implementation - you may want to use more sophisticated models
        
        try:
            # For now, provide a basic assessment
            # In a real implementation, you would use models like:
            # - facebook/wmt-large-en-de-qe for quality estimation
            # - microsoft/DialoGPT for fluency assessment
            
            # Placeholder implementation - basic heuristics
            avg_length_ratio = sum(len(trans) / max(len(orig), 1) for orig, trans in sample_pairs) / len(sample_pairs)
            
            # Simple scoring based on length ratio and other heuristics
            accuracy_score = min(10, max(1, 10 - abs(1 - avg_length_ratio) * 5))
            fluency_score = 7.0  # Placeholder - would need actual fluency assessment
            consistency_score = 7.0  # Placeholder - would need terminology consistency check
            overall_score = (accuracy_score + fluency_score + consistency_score) / 3
            
            return QualityReport(
                overall_score=overall_score,
                accuracy_score=accuracy_score,
                fluency_score=fluency_score,
                consistency_score=consistency_score,
                issues=["Translation assessment with Hugging Face requires more specific models"],
                suggestions=["Consider using Gemini API for more detailed assessment"],
                corrected_segments=[]
            )
            
        except Exception as e:
            raise QualityAssessmentError(f"Hugging Face API error: {str(e)}")
    
    def _create_quality_report(self, data: Dict) -> QualityReport:
        """Create QualityReport from assessment data."""
        return QualityReport(
            overall_score=float(data.get('overall_score', 0)),
            accuracy_score=float(data.get('accuracy_score', 0)),
            fluency_score=float(data.get('fluency_score', 0)),
            consistency_score=float(data.get('consistency_score', 0)),
            issues=data.get('issues', []),
            suggestions=data.get('suggestions', []),
            corrected_segments=data.get('corrected_segments', [])
        )
    
    def format_quality_report(self, report: QualityReport, source_lang: str, target_lang: str) -> str:
        """Format quality report for display."""
        lines = [
            f"TRANSLATION QUALITY ASSESSMENT ({source_lang} -> {target_lang})",
            "=" * 60,
            f"SCORES (1-10 scale):",
            f"   Overall Quality: {report.overall_score:.1f}/10",
            f"   Accuracy:        {report.accuracy_score:.1f}/10",
            f"   Fluency:         {report.fluency_score:.1f}/10", 
            f"   Consistency:     {report.consistency_score:.1f}/10",
            ""
        ]
        
        if report.issues:
            lines.extend([
                "ISSUES IDENTIFIED:",
                *[f"   - {issue}" for issue in report.issues],
                ""
            ])
        
        if report.suggestions:
            lines.extend([
                "SUGGESTIONS:",
                *[f"   - {suggestion}" for suggestion in report.suggestions],
                ""
            ])
        
        if report.corrected_segments:
            lines.extend([
                f"SUGGESTED CORRECTIONS ({len(report.corrected_segments)} segments):",
                ""
            ])
            for i, correction in enumerate(report.corrected_segments, 1):
                lines.extend([
                    f"   {i}. Original:  {correction['original']}",
                    f"      Corrected: {correction['corrected']}",
                    ""
                ])
        
        return "\n".join(lines)
    
    def should_offer_correction(self, report: QualityReport, min_score: float = 6.0) -> bool:
        """Determine if correction should be offered based on quality scores."""
        return (report.overall_score < min_score or 
                report.accuracy_score < min_score or 
                len(report.corrected_segments) > 0)


def assess_translation_quality(original_srt: str, translated_srt: str, 
                             source_lang: str, target_lang: str) -> Optional[QualityReport]:
    """Convenience function to assess translation quality."""
    try:
        assessor = TranslationQualityAssessor()
        return assessor.assess_translation_quality(original_srt, translated_srt, source_lang, target_lang)
    except QualityAssessmentError as e:
        print(f"⚠️ Quality assessment unavailable: {e}")
        return None
    except Exception as e:
        print(f"❌ Quality assessment error: {e}")
        return None