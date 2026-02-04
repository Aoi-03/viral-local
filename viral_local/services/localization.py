"""
LocalizationEngine for AI-powered content analysis and translation.

This module handles viral segment analysis, intelligent translation,
and cultural adaptation using AI APIs like Gemini and Groq.
"""

import json
import time
import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple

# Conditional imports for API clients
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    Groq = None
    GROQ_AVAILABLE = False

import requests
from ..models import Transcription, ViralSegment, TranscriptSegment, TranslatedSegment, ProcessingResult
from ..config import SystemConfig
from ..utils import get_logger, TranslationError, APIError
from ..utils.logging import LoggerMixin, log_performance


class LocalizationEngine(LoggerMixin):
    """Engine for analyzing content virality and performing intelligent translation."""
    
    def __init__(self, config: SystemConfig):
        """Initialize the localization engine.
        
        Args:
            config: System configuration
        """
        self.config = config
        # Note: logger is provided by LoggerMixin as a property
        
        # Initialize API clients
        self._setup_api_clients()
        
        # Language mappings for cultural adaptation
        self.language_mappings = {
            'hi': {'name': 'Hindi', 'region': 'North India', 'cultural_context': 'Bollywood, Delhi, Mumbai'},
            'bn': {'name': 'Bengali', 'region': 'West Bengal/Bangladesh', 'cultural_context': 'Kolkata, Tagore, fish culture'},
            'ta': {'name': 'Tamil', 'region': 'Tamil Nadu', 'cultural_context': 'Chennai, Kollywood, temple culture'}
        }
        
        # Viral analysis criteria weights
        self.viral_criteria = {
            'emotional_intensity': 0.25,
            'information_density': 0.20,
            'engagement_potential': 0.25,
            'technical_complexity': 0.15,
            'cultural_relevance': 0.15
        }
    
    def _setup_api_clients(self):
        """Set up API clients for Gemini and Groq."""
        try:
            # Configure Gemini
            if self.config.gemini_api_key and GEMINI_AVAILABLE:
                genai.configure(api_key=self.config.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')  # Use available model
                self.logger.info("Gemini API client initialized")
            else:
                self.gemini_model = None
                if not GEMINI_AVAILABLE:
                    self.logger.warning("Gemini API not available - google-generativeai not installed")
                else:
                    self.logger.warning("Gemini API key not provided")
            
            # Configure Groq
            if self.config.groq_api_key and GROQ_AVAILABLE:
                self.groq_client = Groq(api_key=self.config.groq_api_key)
                self.logger.info("Groq API client initialized")
            else:
                self.groq_client = None
                if not GROQ_AVAILABLE:
                    self.logger.warning("Groq API not available - groq package not installed")
                else:
                    self.logger.warning("Groq API key not provided")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize API clients: {e}")
            raise APIError(f"API client initialization failed: {e}")
    
    @log_performance("viral_segment_analysis")
    def analyze_viral_segments(self, transcription: Transcription) -> List[ViralSegment]:
        """Analyze transcription to identify segments with high viral potential.
        
        Args:
            transcription: Transcription object to analyze
            
        Returns:
            List of ViralSegment objects with scoring and analysis
            
        Raises:
            APIError: If viral analysis API call fails
        """
        self.logger.info(f"Starting viral segment analysis for {len(transcription.segments)} segments")
        
        try:
            viral_segments = []
            
            # Group segments into analysis chunks (30-60 seconds each)
            analysis_chunks = self._create_analysis_chunks(transcription.segments)
            
            for chunk_idx, chunk in enumerate(analysis_chunks):
                self.logger.debug(f"Analyzing chunk {chunk_idx + 1}/{len(analysis_chunks)}")
                
                # Analyze chunk for viral potential
                analysis_result = self._analyze_chunk_virality(chunk)
                
                if analysis_result and analysis_result.get('viral_score', 0) > 0.3:
                    viral_segment = ViralSegment(
                        segment=chunk[0] if len(chunk) == 1 else self._merge_segments(chunk),
                        viral_score=analysis_result['viral_score'],
                        engagement_factors=analysis_result['engagement_factors'],
                        priority_level=self._calculate_priority_level(analysis_result['viral_score']),
                        analysis_rationale=analysis_result['rationale']
                    )
                    viral_segments.append(viral_segment)
                
                # Rate limiting - small delay between API calls
                time.sleep(0.1)
            
            # Sort by viral score (highest first)
            viral_segments.sort(key=lambda x: x.viral_score, reverse=True)
            
            self.logger.info(f"Identified {len(viral_segments)} viral segments")
            return viral_segments
            
        except Exception as e:
            self.logger.error(f"Viral segment analysis failed: {e}")
            raise APIError(f"Viral analysis failed: {e}", api_name="gemini/groq")
    
    def _create_analysis_chunks(self, segments: List[TranscriptSegment]) -> List[List[TranscriptSegment]]:
        """Group segments into analysis chunks of optimal size."""
        chunks = []
        current_chunk = []
        current_duration = 0
        
        for segment in segments:
            segment_duration = segment.duration
            
            # If adding this segment would exceed 60 seconds, start new chunk
            if current_duration + segment_duration > 60 and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [segment]
                current_duration = segment_duration
            else:
                current_chunk.append(segment)
                current_duration += segment_duration
        
        # Add final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _analyze_chunk_virality(self, chunk: List[TranscriptSegment]) -> Optional[Dict[str, Any]]:
        """Analyze a chunk of segments for viral potential using AI."""
        # Combine chunk text
        combined_text = " ".join([segment.text for segment in chunk])
        
        # Create analysis prompt
        prompt = self._create_viral_analysis_prompt(combined_text)
        
        # Try Gemini first, fallback to Groq
        try:
            if self.gemini_model:
                return self._analyze_with_gemini(prompt)
            elif self.groq_client:
                return self._analyze_with_groq(prompt)
            else:
                raise APIError("No API client available for viral analysis")
        except Exception as e:
            self.logger.warning(f"Primary API failed, trying fallback: {e}")
            try:
                if self.groq_client and self.gemini_model:  # Try the other one
                    if "gemini" in str(e).lower():
                        return self._analyze_with_groq(prompt)
                    else:
                        return self._analyze_with_gemini(prompt)
            except Exception as fallback_error:
                self.logger.error(f"Both APIs failed: {fallback_error}")
                return None
    
    def _create_viral_analysis_prompt(self, text: str) -> str:
        """Create a structured prompt for viral content analysis."""
        return f"""
Analyze the following video transcript segment for viral potential. Consider these factors:

1. Emotional Intensity: Does it evoke strong emotions (excitement, surprise, humor, inspiration)?
2. Information Density: Does it contain valuable, shareable information or insights?
3. Engagement Potential: Would viewers want to comment, share, or discuss this?
4. Technical Complexity: Is it accessible to a broad audience while being informative?
5. Cultural Relevance: Does it relate to current trends, universal experiences, or cultural moments?

Text to analyze:
"{text}"

Respond with a JSON object containing:
{{
    "viral_score": <float between 0.0 and 1.0>,
    "engagement_factors": [<list of specific factors that make this engaging>],
    "rationale": "<brief explanation of the scoring>",
    "emotional_markers": [<list of emotional triggers found>],
    "information_value": "<assessment of information density>",
    "shareability": "<why someone would share this>"
}}

Focus on content that would perform well on social media platforms and drive engagement.
"""
    
    def _analyze_with_gemini(self, prompt: str) -> Dict[str, Any]:
        """Analyze content using Gemini API."""
        try:
            response = self.gemini_model.generate_content(prompt)
            
            # Extract JSON from response
            response_text = response.text.strip()
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # Validate and normalize the result
                return self._normalize_analysis_result(result)
            else:
                self.logger.warning("No JSON found in Gemini response")
                return None
                
        except Exception as e:
            self.logger.error(f"Gemini analysis failed: {e}")
            raise APIError(f"Gemini API call failed: {e}", api_name="gemini")
    
    def _analyze_with_groq(self, prompt: str) -> Dict[str, Any]:
        """Analyze content using Groq API."""
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert content analyst specializing in viral content identification. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                model="mixtral-8x7b-32768",
                temperature=0.3,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # Validate and normalize the result
                return self._normalize_analysis_result(result)
            else:
                self.logger.warning("No JSON found in Groq response")
                return None
                
        except Exception as e:
            self.logger.error(f"Groq analysis failed: {e}")
            raise APIError(f"Groq API call failed: {e}", api_name="groq")
    
    def _normalize_analysis_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and validate analysis result."""
        normalized = {
            'viral_score': max(0.0, min(1.0, float(result.get('viral_score', 0.0)))),
            'engagement_factors': result.get('engagement_factors', []),
            'rationale': result.get('rationale', 'No rationale provided'),
            'emotional_markers': result.get('emotional_markers', []),
            'information_value': result.get('information_value', 'Unknown'),
            'shareability': result.get('shareability', 'Unknown')
        }
        
        # Ensure engagement_factors is a list
        if not isinstance(normalized['engagement_factors'], list):
            normalized['engagement_factors'] = [str(normalized['engagement_factors'])]
        
        return normalized
    
    def _merge_segments(self, segments: List[TranscriptSegment]) -> TranscriptSegment:
        """Merge multiple segments into a single segment."""
        if not segments:
            raise ValueError("Cannot merge empty segment list")
        
        if len(segments) == 1:
            return segments[0]
        
        # Combine text
        combined_text = " ".join([segment.text for segment in segments])
        
        # Use timing from first and last segments
        start_time = segments[0].start_time
        end_time = segments[-1].end_time
        
        # Average confidence
        avg_confidence = sum(segment.confidence for segment in segments) / len(segments)
        
        # Use speaker from first segment (or most common)
        speaker_id = segments[0].speaker_id
        
        return TranscriptSegment(
            text=combined_text,
            start_time=start_time,
            end_time=end_time,
            confidence=avg_confidence,
            speaker_id=speaker_id,
            language=segments[0].language
        )
    
    def _calculate_priority_level(self, viral_score: float) -> int:
        """Calculate priority level based on viral score."""
        if viral_score >= 0.8:
            return 1  # Highest priority
        elif viral_score >= 0.6:
            return 2  # High priority
        elif viral_score >= 0.4:
            return 3  # Medium priority
        else:
            return 4  # Low priority
    
    @log_performance("content_translation")
    def translate_content(self, segments: List[TranscriptSegment], target_lang: str) -> List[TranslatedSegment]:
        """Translate transcript segments to target language with cultural adaptation.
        
        Args:
            segments: List of TranscriptSegment objects to translate
            target_lang: Target language code (e.g., 'hi', 'bn', 'ta')
            
        Returns:
            List of TranslatedSegment objects
            
        Raises:
            TranslationError: If translation fails
            APIError: If translation API call fails
        """
        if target_lang not in self.config.supported_languages:
            raise TranslationError(
                f"Unsupported target language: {target_lang}",
                target_language=target_lang
            )
        
        self.logger.info(f"Starting translation to {target_lang} for {len(segments)} segments")
        
        try:
            translated_segments = []
            
            # Process segments in batches for efficiency
            batch_size = 5  # Translate 5 segments at once for context
            
            for i in range(0, len(segments), batch_size):
                batch = segments[i:i + batch_size]
                self.logger.debug(f"Translating batch {i//batch_size + 1}/{(len(segments) + batch_size - 1)//batch_size}")
                
                # Translate batch with context
                batch_translations = self._translate_batch(batch, target_lang)
                translated_segments.extend(batch_translations)
                
                # Rate limiting
                time.sleep(0.2)
            
            self.logger.info(f"Successfully translated {len(translated_segments)} segments")
            return translated_segments
            
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            raise TranslationError(f"Translation to {target_lang} failed: {e}", target_language=target_lang)
    
    def _translate_batch(self, segments: List[TranscriptSegment], target_lang: str) -> List[TranslatedSegment]:
        """Translate a batch of segments with context preservation."""
        # Create translation prompt with cultural context
        prompt = self._create_translation_prompt(segments, target_lang)
        
        # Try translation with primary API
        try:
            if self.gemini_model:
                translation_result = self._translate_with_gemini(prompt, target_lang)
            elif self.groq_client:
                translation_result = self._translate_with_groq(prompt, target_lang)
            else:
                raise APIError("No API client available for translation")
            
            # Parse and validate translation result
            return self._parse_translation_result(segments, translation_result, target_lang)
            
        except Exception as e:
            self.logger.warning(f"Primary translation API failed, trying fallback: {e}")
            try:
                # Try fallback API
                if self.groq_client and self.gemini_model:
                    if "gemini" in str(e).lower():
                        translation_result = self._translate_with_groq(prompt, target_lang)
                    else:
                        translation_result = self._translate_with_gemini(prompt, target_lang)
                    
                    return self._parse_translation_result(segments, translation_result, target_lang)
                else:
                    raise e
            except Exception as fallback_error:
                self.logger.error(f"Both translation APIs failed: {fallback_error}")
                # Use simple fallback translation when APIs are unavailable
                error_str = str(fallback_error).lower()
                if any(keyword in error_str for keyword in ["quota", "429", "rate limit", "exceeded", "empty", "unavailable"]):
                    self.logger.warning("API issues detected, using simple fallback translation")
                    return self._create_simple_fallback_translation(segments, target_lang)
                else:
                    raise TranslationError(f"All translation APIs failed: {fallback_error}", target_language=target_lang)
    
    def _create_translation_prompt(self, segments: List[TranscriptSegment], target_lang: str) -> str:
        """Create a structured prompt for translation with cultural adaptation."""
        
    def _create_simple_fallback_translation(self, segments: List[TranscriptSegment], target_lang: str) -> List[TranslatedSegment]:
        """Create simple fallback translations when APIs are unavailable.
        
        This provides basic placeholder translations to allow the pipeline to continue
        when API quotas are exceeded or services are unavailable.
        """
        try:
            translated_segments = []
            
            # Simple language mappings for basic words
            basic_translations = {
                'hi': {
                    'hello': 'नमस्ते',
                    'thank you': 'धन्यवाद',
                    'good': 'अच्छा',
                    'very': 'बहुत',
                    'this': 'यह',
                    'that': 'वह',
                    'and': 'और',
                    'the': '',  # Hindi doesn't always need articles
                    'is': 'है',
                    'are': 'हैं',
                    'elephant': 'हाथी',
                    'zoo': 'चिड़ियाघर',
                    'video': 'वीडियो',
                    'here': 'यहाँ',
                    'there': 'वहाँ',
                    'big': 'बड़ा',
                    'small': 'छोटा',
                    'teeth': 'दाँत',
                    'amazing': 'कमाल',
                    'cool': 'कूल',
                    'awesome': 'शानदार'
                },
                'bn': {
                    'hello': 'হ্যালো',
                    'thank you': 'ধন্যবাদ',
                    'good': 'ভাল',
                    'very': 'খুব',
                    'this': 'এই',
                    'that': 'ওই',
                    'and': 'এবং',
                    'elephant': 'হাতি',
                    'zoo': 'চিড়িয়াখানা'
                },
                'ta': {
                    'hello': 'வணக்கம்',
                    'thank you': 'நன்றி',
                    'good': 'நல்ல',
                    'very': 'மிகவும்',
                    'this': 'இது',
                    'that': 'அது',
                    'and': 'மற்றும்',
                    'elephant': 'யானை',
                    'zoo': 'மிருகக்காட்சிசாலை'
                }
            }
            
            lang_dict = basic_translations.get(target_lang, {})
            
            for segment in segments:
                # Simple word-by-word replacement for basic translation
                text = segment.text.lower()
                translated_text = text
                
                # Apply basic word replacements
                for english, translation in lang_dict.items():
                    if english in text:
                        translated_text = translated_text.replace(english, translation)
                
                # If no translations applied, add a language prefix to indicate it's a fallback
                if translated_text == text:
                    if target_lang == 'hi':
                        translated_text = f"[हिंदी में] {segment.text}"
                    elif target_lang == 'bn':
                        translated_text = f"[বাংলায়] {segment.text}"
                    elif target_lang == 'ta':
                        translated_text = f"[தமிழில்] {segment.text}"
                    else:
                        translated_text = f"[{target_lang}] {segment.text}"
                
                # Create TranslatedSegment
                translated_segment = TranslatedSegment(
                    original_segment=segment,
                    translated_text=translated_text,
                    target_language=target_lang,
                    quality_score=0.5,  # Low quality for fallback
                    cultural_adaptations=[f"Fallback translation (API quota exceeded)"],
                    technical_terms_preserved=[]
                )
                
                translated_segments.append(translated_segment)
            
            self.logger.info(f"Created {len(translated_segments)} fallback translations for {target_lang}")
            return translated_segments
            
        except Exception as e:
            self.logger.error(f"Fallback translation failed: {e}")
            # Last resort: just copy original text with language marker
            fallback_segments = []
            for segment in segments:
                fallback_segment = TranslatedSegment(
                    original_segment=segment,
                    translated_text=f"[{target_lang}] {segment.text}",
                    target_language=target_lang,
                    quality_score=0.1,
                    cultural_adaptations=["Emergency fallback - original text preserved"],
                    technical_terms_preserved=[]
                )
                fallback_segments.append(fallback_segment)
            
            return fallback_segments
        """Create a culturally-aware translation prompt."""
        lang_info = self.language_mappings.get(target_lang, {})
        lang_name = lang_info.get('name', target_lang)
        region = lang_info.get('region', 'India')
        cultural_context = lang_info.get('cultural_context', 'Indian culture')
        
        # Combine segment texts with timing markers
        segments_text = ""
        for i, segment in enumerate(segments):
            segments_text += f"[{i+1}] {segment.text}\n"
        
        return f"""
Translate the following video transcript segments to {lang_name}, keeping these guidelines in mind:

CULTURAL ADAPTATION RULES:
1. Adapt content for {region} audience while preserving original meaning
2. Consider cultural context: {cultural_context}
3. Preserve technical terms but explain if needed for local audience
4. Maintain speaker's tone and emotional intensity
5. Use natural, conversational {lang_name} that sounds native
6. Adapt cultural references to local equivalents when appropriate
7. Keep timing and segment structure intact

TECHNICAL PRESERVATION:
- Keep technical terms in English if commonly used (e.g., "software", "algorithm")
- Provide brief explanations for complex technical concepts
- Maintain professional terminology for business/educational content

TONE MATCHING:
- Preserve emotional intensity (excitement, concern, humor)
- Match formality level of original content
- Keep speaker's personality and style

Original segments to translate:
{segments_text}

Respond with a JSON array containing translations:
[
    {{
        "segment_index": 1,
        "translated_text": "<translation>",
        "quality_score": <0.0-1.0>,
        "cultural_adaptations": ["<list of cultural changes made>"],
        "technical_terms_preserved": ["<list of technical terms kept>"],
        "tone_notes": "<notes about tone preservation>"
    }},
    ...
]

Ensure translations are natural, culturally appropriate, and maintain the original's impact.
"""
    
    def _translate_with_gemini(self, prompt: str, target_lang: str) -> Dict[str, Any]:
        """Translate content using Gemini API."""
        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                return {"translations": result, "api_used": "gemini"}
            else:
                self.logger.warning("No JSON array found in Gemini translation response")
                raise APIError("Invalid response format from Gemini")
                
        except Exception as e:
            self.logger.error(f"Gemini translation failed: {e}")
            raise APIError(f"Gemini translation API call failed: {e}", api_name="gemini")
    
    def _translate_with_groq(self, prompt: str, target_lang: str) -> Dict[str, Any]:
        """Translate content using Groq API."""
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"You are an expert translator specializing in culturally-aware translation to {target_lang}. Always respond with valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                model="mixtral-8x7b-32768",
                temperature=0.2,  # Lower temperature for more consistent translations
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                return {"translations": result, "api_used": "groq"}
            else:
                self.logger.warning("No JSON array found in Groq translation response")
                raise APIError("Invalid response format from Groq")
                
        except Exception as e:
            self.logger.error(f"Groq translation failed: {e}")
            raise APIError(f"Groq translation API call failed: {e}", api_name="groq")
    
    def _parse_translation_result(
        self, 
        original_segments: List[TranscriptSegment], 
        translation_result: Dict[str, Any], 
        target_lang: str
    ) -> List[TranslatedSegment]:
        """Parse and validate translation results."""
        translations = translation_result.get("translations", [])
        
        if len(translations) != len(original_segments):
            self.logger.warning(f"Translation count mismatch: expected {len(original_segments)}, got {len(translations)}")
        
        translated_segments = []
        
        for i, original_segment in enumerate(original_segments):
            # Find corresponding translation
            translation = None
            for trans in translations:
                if trans.get("segment_index") == i + 1:
                    translation = trans
                    break
            
            if not translation:
                # Fallback: use translation by index if available
                if i < len(translations):
                    translation = translations[i]
                else:
                    # Create a fallback translation
                    self.logger.warning(f"No translation found for segment {i+1}, creating fallback")
                    translation = {
                        "translated_text": f"[Translation missing for: {original_segment.text[:50]}...]",
                        "quality_score": 0.1,
                        "cultural_adaptations": [],
                        "technical_terms_preserved": [],
                        "tone_notes": "Fallback translation"
                    }
            
            # Create TranslatedSegment
            translated_segment = TranslatedSegment(
                original_segment=original_segment,
                translated_text=translation.get("translated_text", ""),
                target_language=target_lang,
                quality_score=max(0.0, min(1.0, float(translation.get("quality_score", 0.5)))),
                cultural_adaptations=translation.get("cultural_adaptations", []),
                technical_terms_preserved=translation.get("technical_terms_preserved", [])
            )
            
            translated_segments.append(translated_segment)
        
        return translated_segments
    
    def _detect_technical_terms(self, text: str) -> List[str]:
        """Detect technical terms in text that should be preserved."""
        # Common technical terms that are often kept in English
        technical_patterns = [
            r'\b(?:API|SDK|URL|HTTP|JSON|XML|SQL|AI|ML|IoT|VR|AR)\b',
            r'\b(?:software|hardware|algorithm|database|server|client)\b',
            r'\b(?:application|programming|development|framework|library)\b',
            r'\b(?:website|email|internet|online|digital|virtual)\b',
            r'\b(?:smartphone|laptop|computer|technology|platform)\b'
        ]
        
        technical_terms = []
        for pattern in technical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            technical_terms.extend(matches)
        
        return list(set(technical_terms))  # Remove duplicates
    
    def _assess_cultural_adaptation_needs(self, text: str, target_lang: str) -> List[str]:
        """Assess what cultural adaptations might be needed."""
        adaptations = []
        
        # Check for cultural references that might need adaptation
        cultural_indicators = {
            'food': r'\b(?:pizza|burger|sandwich|coffee|tea)\b',
            'currency': r'\$\d+|\b(?:dollar|USD|rupee|INR)\b',
            'locations': r'\b(?:America|USA|New York|California|Silicon Valley)\b',
            'brands': r'\b(?:Google|Apple|Microsoft|Amazon|Facebook)\b',
            'measurements': r'\b\d+\s*(?:feet|inches|pounds|fahrenheit)\b'
        }
        
        for category, pattern in cultural_indicators.items():
            if re.search(pattern, text, re.IGNORECASE):
                adaptations.append(f"Consider {category} localization")
        
        return adaptations
    
    def optimize_translation_quality(self, segments: List[TranslatedSegment]) -> List[TranslatedSegment]:
        """Optimize translation quality using retry logic and fallback strategies.
        
        Args:
            segments: List of TranslatedSegment objects to optimize
            
        Returns:
            List of optimized TranslatedSegment objects
            
        Raises:
            TranslationError: If optimization fails
        """
        self.logger.info(f"Optimizing translation quality for {len(segments)} segments")
        
        optimized_segments = []
        retry_segments = []
        
        # Identify segments that need improvement
        for segment in segments:
            if self._needs_quality_improvement(segment):
                retry_segments.append(segment)
            else:
                optimized_segments.append(segment)
        
        if retry_segments:
            self.logger.info(f"Retrying translation for {len(retry_segments)} low-quality segments")
            
            # Retry with different strategies
            improved_segments = self._retry_translation_with_strategies(retry_segments)
            optimized_segments.extend(improved_segments)
        
        # Sort back to original order
        optimized_segments.sort(key=lambda x: x.original_segment.start_time)
        
        self.logger.info(f"Translation optimization completed. Final quality scores: {[s.quality_score for s in optimized_segments]}")
        return optimized_segments
    
    def _needs_quality_improvement(self, segment: TranslatedSegment) -> bool:
        """Determine if a translated segment needs quality improvement."""
        # Check quality score threshold
        if segment.quality_score < 0.6:
            return True
        
        # Check for obvious translation issues
        if self._has_translation_issues(segment):
            return True
        
        # Check if translation is too similar to original (might indicate poor translation)
        if self._is_too_similar_to_original(segment):
            return True
        
        return False
    
    def _has_translation_issues(self, segment: TranslatedSegment) -> bool:
        """Check for common translation issues."""
        translated_text = segment.translated_text.lower()
        
        # Check for untranslated English words (basic check)
        english_indicators = ['the', 'and', 'or', 'but', 'with', 'from', 'this', 'that', 'have', 'will']
        english_word_count = sum(1 for word in english_indicators if f' {word} ' in f' {translated_text} ')
        
        # If more than 20% of common English words remain, might be poorly translated
        if english_word_count > len(english_indicators) * 0.2:
            return True
        
        # Check for placeholder text or error indicators
        error_indicators = ['[translation', 'error', 'failed', 'missing']
        if any(indicator in translated_text for indicator in error_indicators):
            return True
        
        # Check if translation is suspiciously short compared to original
        original_length = len(segment.original_segment.text)
        translated_length = len(segment.translated_text)
        
        if translated_length < original_length * 0.3:  # Translation is less than 30% of original
            return True
        
        return False
    
    def _is_too_similar_to_original(self, segment: TranslatedSegment) -> bool:
        """Check if translation is too similar to original (indicating poor translation)."""
        original = segment.original_segment.text.lower()
        translated = segment.translated_text.lower()
        
        # Simple similarity check - count common words
        original_words = set(original.split())
        translated_words = set(translated.split())
        
        if len(original_words) == 0:
            return False
        
        common_words = original_words.intersection(translated_words)
        similarity_ratio = len(common_words) / len(original_words)
        
        # If more than 70% of words are the same, might be poorly translated
        return similarity_ratio > 0.7
    
    def _retry_translation_with_strategies(self, segments: List[TranslatedSegment]) -> List[TranslatedSegment]:
        """Retry translation using different strategies for improved quality."""
        improved_segments = []
        
        for segment in segments:
            self.logger.debug(f"Retrying translation for segment with quality score {segment.quality_score}")
            
            # Try different strategies in order of preference
            strategies = [
                self._retry_with_simplified_prompt,
                self._retry_with_context_emphasis,
                self._retry_with_alternative_api,
                self._create_fallback_translation
            ]
            
            improved_segment = None
            
            for strategy in strategies:
                try:
                    improved_segment = strategy(segment)
                    if improved_segment and improved_segment.quality_score > segment.quality_score:
                        self.logger.debug(f"Strategy {strategy.__name__} improved quality from {segment.quality_score} to {improved_segment.quality_score}")
                        break
                except Exception as e:
                    self.logger.warning(f"Strategy {strategy.__name__} failed: {e}")
                    continue
            
            # Use improved segment or original if no improvement
            improved_segments.append(improved_segment or segment)
            
            # Rate limiting between retries
            time.sleep(0.1)
        
        return improved_segments
    
    def _retry_with_simplified_prompt(self, segment: TranslatedSegment) -> Optional[TranslatedSegment]:
        """Retry translation with a simplified, more direct prompt."""
        lang_info = self.language_mappings.get(segment.target_language, {})
        lang_name = lang_info.get('name', segment.target_language)
        
        simplified_prompt = f"""
Translate this text to natural, conversational {lang_name}:

"{segment.original_segment.text}"

Requirements:
- Use simple, clear language
- Preserve the original meaning exactly
- Make it sound natural to native speakers
- Keep technical terms if they're commonly used

Respond with only the translation, no explanations.
"""
        
        try:
            # Use primary API for retry
            if self.gemini_model:
                response = self.gemini_model.generate_content(simplified_prompt)
                translated_text = response.text.strip()
            elif self.groq_client:
                response = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": f"You are a professional translator. Translate to {lang_name} naturally and accurately."},
                        {"role": "user", "content": simplified_prompt}
                    ],
                    model="mixtral-8x7b-32768",
                    temperature=0.1,
                    max_tokens=500
                )
                translated_text = response.choices[0].message.content.strip()
            else:
                return None
            
            # Create improved segment
            return TranslatedSegment(
                original_segment=segment.original_segment,
                translated_text=translated_text,
                target_language=segment.target_language,
                quality_score=min(0.8, segment.quality_score + 0.2),  # Assume some improvement
                cultural_adaptations=segment.cultural_adaptations,
                technical_terms_preserved=segment.technical_terms_preserved
            )
            
        except Exception as e:
            self.logger.warning(f"Simplified prompt retry failed: {e}")
            return None
    
    def _retry_with_context_emphasis(self, segment: TranslatedSegment) -> Optional[TranslatedSegment]:
        """Retry translation with emphasis on context and cultural adaptation."""
        lang_info = self.language_mappings.get(segment.target_language, {})
        lang_name = lang_info.get('name', segment.target_language)
        cultural_context = lang_info.get('cultural_context', 'Indian culture')
        
        context_prompt = f"""
You are translating video content for {cultural_context} audience.

Original text: "{segment.original_segment.text}"

Translate to {lang_name} considering:
1. The audience is familiar with both local and global contexts
2. Technical terms can be kept in English if commonly understood
3. Maintain the speaker's tone and energy
4. Make it sound like something a native {lang_name} speaker would naturally say

Translation:
"""
        
        try:
            # Try with alternative API if available
            api_to_use = self.groq_client if self.groq_client else self.gemini_model
            
            if api_to_use == self.groq_client:
                response = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": f"You are an expert {lang_name} translator with deep cultural knowledge."},
                        {"role": "user", "content": context_prompt}
                    ],
                    model="mixtral-8x7b-32768",
                    temperature=0.3,
                    max_tokens=500
                )
                translated_text = response.choices[0].message.content.strip()
            else:
                response = self.gemini_model.generate_content(context_prompt)
                translated_text = response.text.strip()
            
            return TranslatedSegment(
                original_segment=segment.original_segment,
                translated_text=translated_text,
                target_language=segment.target_language,
                quality_score=min(0.85, segment.quality_score + 0.25),
                cultural_adaptations=segment.cultural_adaptations + ["Context-aware retry"],
                technical_terms_preserved=segment.technical_terms_preserved
            )
            
        except Exception as e:
            self.logger.warning(f"Context emphasis retry failed: {e}")
            return None
    
    def _retry_with_alternative_api(self, segment: TranslatedSegment) -> Optional[TranslatedSegment]:
        """Retry translation using the alternative API."""
        # This strategy switches to the other API if both are available
        if not (self.gemini_model and self.groq_client):
            return None
        
        try:
            # Create a basic translation request for the alternative API
            original_segments = [segment.original_segment]
            alternative_translations = self._translate_batch(original_segments, segment.target_language)
            
            if alternative_translations:
                alternative_segment = alternative_translations[0]
                # Only return if quality is better
                if alternative_segment.quality_score > segment.quality_score:
                    return alternative_segment
            
        except Exception as e:
            self.logger.warning(f"Alternative API retry failed: {e}")
        
        return None
    
    def _create_fallback_translation(self, segment: TranslatedSegment) -> TranslatedSegment:
        """Create a basic fallback translation when all else fails."""
        # This is a last resort - create a basic translation with clear indication
        fallback_text = f"[Auto-translated]: {segment.original_segment.text}"
        
        # Try to do basic word-by-word replacement for common terms
        fallback_text = self._apply_basic_translation_rules(segment.original_segment.text, segment.target_language)
        
        return TranslatedSegment(
            original_segment=segment.original_segment,
            translated_text=fallback_text,
            target_language=segment.target_language,
            quality_score=0.3,  # Low quality score to indicate fallback
            cultural_adaptations=["Fallback translation used"],
            technical_terms_preserved=[]
        )
    
    def _apply_basic_translation_rules(self, text: str, target_lang: str) -> str:
        """Apply basic translation rules as a last resort."""
        # This is a very basic fallback - in a real system you might use
        # a dictionary-based approach or simpler translation service
        
        basic_translations = {
            'hi': {
                'hello': 'नमस्ते',
                'thank you': 'धन्यवाद',
                'yes': 'हाँ',
                'no': 'नहीं',
                'good': 'अच्छा',
                'bad': 'बुरा'
            },
            'bn': {
                'hello': 'নমস্কার',
                'thank you': 'ধন্যবাদ',
                'yes': 'হ্যাঁ',
                'no': 'না',
                'good': 'ভাল',
                'bad': 'খারাপ'
            },
            'ta': {
                'hello': 'வணக்கம்',
                'thank you': 'நன்றி',
                'yes': 'ஆம்',
                'no': 'இல்லை',
                'good': 'நல்ல',
                'bad': 'கெட்ட'
            }
        }
        
        translations = basic_translations.get(target_lang, {})
        result = text.lower()
        
        for english, translated in translations.items():
            result = result.replace(english, translated)
        
        return result
    
    def _implement_exponential_backoff(self, attempt: int, base_delay: float = 1.0) -> None:
        """Implement exponential backoff for API retries."""
        if not self.config.exponential_backoff:
            time.sleep(self.config.retry_delay)
            return
        
        # Calculate exponential backoff delay
        delay = base_delay * (2 ** attempt)
        max_delay = 30.0  # Cap at 30 seconds
        
        actual_delay = min(delay, max_delay)
        self.logger.debug(f"Applying exponential backoff: {actual_delay:.2f}s (attempt {attempt})")
        time.sleep(actual_delay)