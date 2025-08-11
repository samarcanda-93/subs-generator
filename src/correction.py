"""Translation correction feature with user confirmation."""

import os
import re
from typing import List, Dict, Optional
from quality_assessment import QualityReport, TranslationQualityAssessor

class TranslationCorrector:
    """Apply AI-suggested corrections to translation files."""
    
    def __init__(self):
        self.assessor = TranslationQualityAssessor()
    
    def apply_corrections_interactive(self, srt_path: str, report: QualityReport) -> bool:
        """Apply corrections interactively with user confirmation."""
        if not report.corrected_segments:
            print("ℹ️ No specific corrections suggested.")
            return False
        
        print(f"\n🔧 CORRECTION SUGGESTIONS for {os.path.basename(srt_path)}")
        print("=" * 60)
        
        corrections_to_apply = []
        
        for i, correction in enumerate(report.corrected_segments, 1):
            print(f"\n{i}. Suggested correction:")
            print(f"   Original:  {correction['original']}")
            print(f"   Suggested: {correction['corrected']}")
            
            while True:
                choice = input(f"   Apply this correction? [y/n/s/q/a] (y=yes, n=no, s=skip all, q=quit, a=accept all): ").lower().strip()
                
                if choice in ['y', 'yes']:
                    corrections_to_apply.append(correction)
                    print("   ✓ Correction will be applied")
                    break
                elif choice in ['n', 'no']:
                    print("   ✗ Correction skipped")
                    break
                elif choice in ['s', 'skip']:
                    print("   ⏭️ Skipping all remaining corrections")
                    return self._apply_corrections_to_file(srt_path, corrections_to_apply)
                elif choice in ['a', 'accept', 'accept all']:
                    print("   ✅ Accepting all remaining corrections")
                    # Add current correction and all remaining ones
                    corrections_to_apply.append(correction)
                    corrections_to_apply.extend(report.corrected_segments[i:])  # Add remaining corrections
                    return self._apply_corrections_to_file(srt_path, corrections_to_apply, create_new_file=True)
                elif choice in ['q', 'quit']:
                    print("   ❌ Correction process cancelled")
                    return False
                else:
                    print("   Please enter 'y', 'n', 's', 'q', or 'a'")
        
        if corrections_to_apply:
            return self._apply_corrections_to_file(srt_path, corrections_to_apply, create_new_file=True)
        else:
            print("\n   ℹ️ No corrections applied")
            return False
    
    def _apply_corrections_to_file(self, srt_path: str, corrections: List[Dict[str, str]], create_new_file: bool = True) -> bool:
        """Apply approved corrections to the SRT file."""
        if not corrections:
            return False
        
        try:
            # Always create backup first
            backup_path = srt_path + '.backup'
            with open(srt_path, 'r', encoding='utf-8') as original:
                original_content = original.read()
                with open(backup_path, 'w', encoding='utf-8') as backup:
                    backup.write(original_content)
            
            # Determine output file path
            if create_new_file:
                # Create new corrected file instead of overwriting
                base_name = srt_path.replace('.srt', '')
                corrected_path = base_name + '_ai_corrected.srt'
            else:
                corrected_path = srt_path
            
            # Apply corrections to original content
            corrected_content = original_content
            applied_count = 0
            
            for correction in corrections:
                original_text = correction['original']
                corrected_text = correction['corrected']
                
                # Try exact match first
                if original_text in corrected_content:
                    corrected_content = corrected_content.replace(original_text, corrected_text, 1)
                    applied_count += 1
                else:
                    # Try fuzzy matching for minor variations
                    if self._apply_fuzzy_correction(corrected_content, original_text, corrected_text):
                        applied_count += 1
            
            # Write corrected content to appropriate file
            with open(corrected_path, 'w', encoding='utf-8') as f:
                f.write(corrected_content)
            
            if applied_count > 0:
                if create_new_file:
                    print(f"Applied {applied_count}/{len(corrections)} corrections")
                    print(f"Original file: {os.path.basename(srt_path)}")
                    print(f"Backup saved: {os.path.basename(backup_path)}")
                    print(f"AI-corrected: {os.path.basename(corrected_path)}")
                else:
                    print(f"Applied {applied_count}/{len(corrections)} corrections to {os.path.basename(srt_path)}")
                    print(f"Backup saved: {os.path.basename(backup_path)}")
                return True
            else:
                # Remove backup if no changes made
                os.remove(backup_path)
                print(f"Could not apply any corrections (text not found in file)")
                return False
                
        except Exception as e:
            print(f"Error applying corrections: {e}")
            return False
    
    def _apply_fuzzy_correction(self, content: str, original: str, corrected: str) -> bool:
        """Apply correction with fuzzy matching for minor text variations."""
        # This is a simple implementation - could be enhanced with more sophisticated matching
        
        # Try with normalized whitespace
        original_normalized = re.sub(r'\s+', ' ', original.strip())
        
        # Look for similar text in content
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line_normalized = re.sub(r'\s+', ' ', line.strip())
            if original_normalized in line_normalized:
                lines[i] = line.replace(original_normalized, corrected)
                return True
        
        return False

def offer_correction(srt_path: str, report: QualityReport, overwrite_mode: bool = False) -> bool:
    """Offer correction to user and apply if accepted."""
    if not report.corrected_segments:
        return False
    
    print(f"\nAI has identified potential improvements for the translation.")
    print(f"Quality score: {report.overall_score:.1f}/10")
    
    while True:
        choice = input("Would you like to review and apply corrections? [y/n]: ").lower().strip()
        
        if choice in ['y', 'yes']:
            corrector = TranslationCorrector()
            if overwrite_mode:
                return corrector._apply_corrections_to_file(srt_path, report.corrected_segments, create_new_file=False)
            else:
                return corrector.apply_corrections_interactive(srt_path, report)
        elif choice in ['n', 'no']:
            print("Corrections skipped")
            return False
        else:
            print("Please enter 'y' or 'n'")

def auto_correct_translation(original_srt: str, translated_srt: str, 
                           source_lang: str, target_lang: str, overwrite_mode: bool = False) -> bool:
    """Automatically assess and offer correction for a translation."""
    try:
        assessor = TranslationQualityAssessor()
        report = assessor.assess_translation_quality(original_srt, translated_srt, source_lang, target_lang)
        
        # Display quality report
        print("\n" + assessor.format_quality_report(report, source_lang, target_lang))
        
        # Offer correction if quality is below threshold
        if assessor.should_offer_correction(report):
            return offer_correction(translated_srt, report, overwrite_mode)
        else:
            print(f"Translation quality is acceptable (score: {report.overall_score:.1f}/10)")
            return False
            
    except Exception as e:
        print(f"Auto-correction error: {e}")
        return False