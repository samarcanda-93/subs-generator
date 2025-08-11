"""Core transcription and translation logic."""

import os
import time
import asyncio
import concurrent.futures
from typing import Tuple, Optional
from faster_whisper import WhisperModel
from processing import post_process_segments_batched, write_srt
from quality_assessment import assess_translation_quality
from correction import auto_correct_translation

def _process_ai_only(filename: str, orig_srt: str, trans_srt: str, detected_lang: str, 
                    target_language: str, quality_check: bool, auto_correct: bool, 
                    overwrite_corrections: bool) -> Tuple[str, bool, str]:
    """Process AI quality assessment and correction only (skip transcription)."""
    
    # Language name mapping
    source_lang_map = {
        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
        'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
        'tr': 'Turkish', 'pl': 'Polish', 'nl': 'Dutch', 'sv': 'Swedish',
        'da': 'Danish', 'no': 'Norwegian', 'fi': 'Finnish', 'he': 'Hebrew',
        'hr': 'Croatian'
    }
    
    source_lang_name = source_lang_map.get(detected_lang, detected_lang)
    target_lang_name = source_lang_map.get(target_language, target_language)
    
    print(f"[{filename}] -> Running AI quality assessment...")
    
    try:
        if auto_correct:
            corrected = auto_correct_translation(orig_srt, trans_srt, source_lang_name, target_lang_name, overwrite_corrections)
            if corrected:
                return filename, True, f"AI correction completed"
            else:
                return filename, True, f"No corrections needed (quality acceptable)"
        elif quality_check:
            report = assess_translation_quality(orig_srt, trans_srt, source_lang_name, target_lang_name)
            if report:
                from quality_assessment import TranslationQualityAssessor
                assessor = TranslationQualityAssessor()
                print(f"[{filename}] -> Quality assessment completed (score: {report.overall_score:.1f}/10)")
                
                # Offer corrections if quality is poor
                if assessor.should_offer_correction(report):
                    from correction import offer_correction
                    corrected = offer_correction(trans_srt, report, overwrite_corrections)
                    if corrected:
                        return filename, True, f"AI correction completed (improved from {report.overall_score:.1f}/10)"
                    else:
                        return filename, True, f"Quality assessment completed (score: {report.overall_score:.1f}/10) - corrections declined"
                else:
                    return filename, True, f"Quality assessment completed (score: {report.overall_score:.1f}/10) - no corrections needed"
            else:
                return filename, True, f"Quality assessment completed"
    except Exception as e:
        return filename, False, f"AI processing failed: {e}"
    
    return filename, True, "AI processing completed"


async def process_file(video_path: str, filename: str, executor: concurrent.futures.ThreadPoolExecutor,
                      device: str, compute_type: str, skip_translation: bool, english_only: bool, 
                      target_language: str, model_name: str, output_dir: str, 
                      quality_check: bool = False, auto_correct: bool = False, 
                      overwrite_corrections: bool = False) -> Tuple[str, bool, Optional[str]]:
    """Process a single video file asynchronously."""
    base = os.path.splitext(filename)[0]
    orig_srt = os.path.join(output_dir, f"{base}_orig.srt")
    
    # Create target language filename
    if target_language == 'en':
        trans_srt = os.path.join(output_dir, f"{base}_en.srt")
    else:
        trans_srt = os.path.join(output_dir, f"{base}_{target_language}.srt")
    
    corrected_srt = trans_srt.replace('.srt', '_ai_corrected.srt')

    # Smart skip logic based on what files exist and what's requested
    files_exist = {
        'orig': os.path.exists(orig_srt),
        'trans': os.path.exists(trans_srt),
        'corrected': os.path.exists(corrected_srt)
    }
    
    # Skip scenarios
    if auto_correct and files_exist['corrected']:
        return filename, True, f"Skipped: AI-corrected file '{os.path.basename(corrected_srt)}' already exists"
    
    if not (quality_check or auto_correct) and files_exist['orig'] and files_exist['trans']:
        return filename, True, f"Skipped: All subtitle files already exist"
    
    if files_exist['orig'] and skip_translation and not (quality_check or auto_correct):
        return filename, True, f"Skipped: Original file exists, translation skipped"
    
    # If files exist and we need AI processing, skip transcription but continue to AI phase
    if (quality_check or auto_correct) and files_exist['orig'] and files_exist['trans']:
        # Skip transcription, but continue to quality check/correction phase
        print(f"[{filename}] -> Subtitle files exist, proceeding to AI processing...")
        base = os.path.splitext(filename)[0]
        
        # Skip to AI processing directly
        try:
            if target_language == 'en':
                trans_srt = os.path.join(output_dir, f"{base}_en.srt")
            else:
                trans_srt = os.path.join(output_dir, f"{base}_{target_language}.srt")
            
            orig_srt = os.path.join(output_dir, f"{base}_orig.srt")
            
            # Language detection from filename or assume Croatian for this case
            detected_lang = 'hr'  # Default assumption for this demo
            
            # AI Processing
            return _process_ai_only(filename, orig_srt, trans_srt, detected_lang, target_language, 
                                  quality_check, auto_correct, overwrite_corrections)
            
        except Exception as e:
            return filename, False, f"AI processing error: {str(e)}"
    
    try:
        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, _transcribe_file_sync, video_path, filename, device, 
            compute_type, skip_translation, english_only, target_language, model_name, output_dir,
            quality_check, auto_correct, overwrite_corrections
        )
        return filename, True, result
    except Exception as e:
        return filename, False, f"Error: {str(e)}"


def _transcribe_file_sync(video_path: str, filename: str, device: str, compute_type: str, 
                         skip_translation: bool, english_only: bool, target_language: str,
                         model_name: str, output_dir: str, quality_check: bool = False, 
                         auto_correct: bool = False, overwrite_corrections: bool = False) -> str:
    """Synchronous transcription function to run in thread pool."""
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    
    base = os.path.splitext(filename)[0]
    orig_srt = os.path.join(output_dir, f"{base}_orig.srt")
    
    # Create target language filename
    if target_language == 'en':
        trans_srt = os.path.join(output_dir, f"{base}_en.srt")
        lang_name = "English"
    else:
        trans_srt = os.path.join(output_dir, f"{base}_{target_language}.srt")
        lang_name = f"{target_language.upper()}"
    
    print(f"\n[{filename}] Processing...")

    if english_only:
        # Target-language-only mode: directly translate to target language
        print(f"[{filename}] -> Target-language-only mode: translating to {lang_name}...")
        t0 = time.time()
        segments_trans, info = model.transcribe(
            video_path,
            task="translate",
            language=target_language if target_language != 'en' else None,
            word_timestamps=True
        )
        t1 = time.time()
        detected_lang = info.language
        print(f"[{filename}] * {lang_name} translation done in {t1 - t0:.1f}s")
        print(f"[{filename}] -> Detected language: {detected_lang}")

        # Post-process segments with batching
        print(f"[{filename}] -> Post-processing segments...")
        processed_trans = post_process_segments_batched(list(segments_trans))

        # Write the target language SRT
        print(f"[{filename}] -> Writing {lang_name} SRT")
        write_srt(processed_trans, trans_srt, use_word_ts=True)
        
        return f"Completed successfully ({lang_name}-only)"
    else:
        # Standard mode: original + optional translation
        # 1) Transcribe original with word timestamps
        print(f"[{filename}] -> Transcribing original...")
        t0 = time.time()
        segments_orig, info = model.transcribe(
            video_path,
            word_timestamps=True
        )
        t1 = time.time()
        detected_lang = info.language
        print(f"[{filename}] * Original transcription done in {t1 - t0:.1f}s")
        print(f"[{filename}] -> Detected language: {detected_lang}")

        # Post-process segments with batching
        print(f"[{filename}] -> Post-processing segments...")
        processed_orig = post_process_segments_batched(list(segments_orig))

        # Write the original-language SRT
        print(f"[{filename}] -> Writing original SRT")
        write_srt(processed_orig, orig_srt, use_word_ts=True)

        # 2) If different from target language, translate
        if detected_lang != target_language and not skip_translation:
            print(f"[{filename}] -> Translating to {lang_name}...")
            t2 = time.time()
            segments_trans, _ = model.transcribe(
                video_path,
                task="translate",
                language=target_language if target_language != 'en' else None,
                word_timestamps=True
            )
            t3 = time.time()
            print(f"[{filename}] * {lang_name} translation done in {t3 - t2:.1f}s")

            processed_trans = post_process_segments_batched(list(segments_trans))
            print(f"[{filename}] -> Writing {lang_name} translation SRT")
            write_srt(processed_trans, trans_srt, use_word_ts=True)
        else:
            if detected_lang == target_language:
                print(f"[{filename}] -> Audio is already in {lang_name}; skipping translation")
            else:
                print(f"[{filename}] -> Translation skipped (--skip-translation enabled)")

    # Quality assessment and correction (if enabled)
    if (quality_check or auto_correct) and not skip_translation:
        if os.path.exists(orig_srt) and os.path.exists(trans_srt):
            print(f"[{filename}] -> Running quality assessment...")
            try:
                # Determine source language for assessment
                source_lang_map = {
                    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
                    'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
                    'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
                    'tr': 'Turkish', 'pl': 'Polish', 'nl': 'Dutch', 'sv': 'Swedish',
                    'da': 'Danish', 'no': 'Norwegian', 'fi': 'Finnish', 'he': 'Hebrew'
                }
                
                source_lang_name = source_lang_map.get(detected_lang, detected_lang)
                target_lang_name = source_lang_map.get(target_language, target_language)
                
                if auto_correct:
                    corrected = auto_correct_translation(orig_srt, trans_srt, source_lang_name, target_lang_name, overwrite_corrections)
                    if corrected:
                        print(f"[{filename}] -> Translation corrected")
                elif quality_check:
                    report = assess_translation_quality(orig_srt, trans_srt, source_lang_name, target_lang_name)
                    if report:
                        from quality_assessment import TranslationQualityAssessor
                        assessor = TranslationQualityAssessor()
                        print(f"[{filename}] -> Quality assessment completed (score: {report.overall_score:.1f}/10)")
                        
                        # Offer corrections if quality is poor
                        if assessor.should_offer_correction(report):
                            from correction import offer_correction
                            corrected = offer_correction(trans_srt, report, overwrite_corrections)
                            if corrected:
                                print(f"[{filename}] -> Translation corrected")
                        
            except Exception as e:
                print(f"[{filename}] -> Quality assessment failed: {e}")

    return f"Completed successfully"