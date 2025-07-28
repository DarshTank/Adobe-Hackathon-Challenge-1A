import os
import json
import fitz  # PyMuPDF
import re
import unicodedata
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import math
from statistics import mean, median, mode, StatisticsError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_DIR = os.environ.get("PDF_INPUT_DIR", "./sample_dataset/pdfs")
OUTPUT_DIR = os.environ.get("PDF_OUTPUT_DIR", "./sample_dataset/outputs")

class EnhancedPDFExtractor:
    def __init__(self):
        self.min_heading_candidates = 3
        self.max_analysis_pages = 15  # Analyze more pages for better pattern learning
        self.title_area_threshold = 0.4  # Top 40% of first page for title
        self.max_title_length = 200  # Maximum characters for title
        
    def extract_pdf_metadata(self, pdf_path):
        """Enhanced metadata extraction with better span processing"""
        doc = fitz.open(pdf_path)
        
        all_spans = []
        for page_number, page in enumerate(doc, start=1):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get('type') == 0:  # text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "")
                            if not text.strip():
                                continue
                                
                            font = span.get("font", "")
                            size = span.get("size", 0)
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            flags = span.get("flags", 0)
                            
                            is_bold = "Bold" in font or bool(flags & 16)
                            is_italic = "Italic" in font or bool(flags & 2)
                            
                            all_spans.append({
                                "page": page_number,
                                "text": text.strip(),
                                "font": font,
                                "size": size,
                                "bbox": bbox,
                                "bold": is_bold,
                                "italic": is_italic,
                                "flags": flags
                            })
        
        doc.close()
        return all_spans

    def analyze_document_structure(self, spans):
        """Enhanced document structure analysis"""
        structure_data = {
            'font_analysis': defaultdict(list),
            'size_distribution': defaultdict(int),
            'position_analysis': defaultdict(list),
            'text_patterns': [],
            'page_analysis': defaultdict(list),
            'line_spacing': [],
            'formatting_patterns': defaultdict(int)
        }
        
        # Analyze all spans
        for span in spans:
            font_key = (span["font"], span["size"], span["flags"])
            structure_data['font_analysis'][font_key].append(span)
            structure_data['size_distribution'][span["size"]] += 1
            structure_data['page_analysis'][span["page"]].append(span)
            
            # Extract text patterns
            patterns = self.extract_text_patterns(span["text"])
            structure_data['text_patterns'].extend(patterns)
            
            # Formatting analysis
            if span["bold"]:
                structure_data['formatting_patterns']['bold'] += 1
            if span["italic"]:
                structure_data['formatting_patterns']['italic'] += 1
            if span["text"].isupper():
                structure_data['formatting_patterns']['uppercase'] += 1
        
        # Calculate size statistics
        all_sizes = list(structure_data['size_distribution'].keys())
        if all_sizes:
            structure_data['size_stats'] = {
                'min': min(all_sizes),
                'max': max(all_sizes),
                'median': median(all_sizes),
                'mean': mean(all_sizes),
                'unique_sizes': len(all_sizes)
            }
        
        return structure_data

    def extract_text_patterns(self, text):
        """Enhanced pattern extraction with more comprehensive rules"""
        patterns = []
        
        if not text or not text.strip():
            return patterns
        
        text_clean = text.strip()
        
        # Numbering patterns
        numbering_patterns = [
            (r'^\d+\.?\s*', 'decimal_number'),
            (r'^\d+\.\d+\.?\s*', 'decimal_subsection'),
            (r'^\d+\.\d+\.\d+\.?\s*', 'decimal_subsubsection'),
            (r'^\(\d+\)\s*', 'parenthetical_number'),
            (r'^\[\d+\]\s*', 'bracketed_number'),
            (r'^[IVXLCDMivxlcdm]+\.?\s*', 'roman_numeral'),
            (r'^[A-Za-z]\.?\s*', 'letter_enumeration'),
            (r'^\([A-Za-z]\)\s*', 'parenthetical_letter')
        ]
        
        for pattern, pattern_type in numbering_patterns:
            match = re.match(pattern, text_clean)
            if match:
                patterns.append((pattern_type, match.group()))
        
        # Bullet patterns
        bullet_chars = ['•', '·', '▪', '▫', '◦', '‣', '⁃', '-', '–', '—', '○', '●', '□', '■']
        if text_clean and text_clean[0] in bullet_chars:
            patterns.append(('bullet_point', text_clean[0]))
        
        # Formatting patterns
        if text_clean.isupper() and len(text_clean.split()) <= 10:
            patterns.append(('all_uppercase', 'CAPS'))
        
        if text_clean.endswith(':') and len(text_clean) < 100:
            patterns.append(('colon_ending', 'COLON'))
        
        # Special heading indicators
        heading_indicators = ['chapter', 'section', 'part', 'appendix', 'introduction', 'conclusion']
        text_lower = text_clean.lower()
        for indicator in heading_indicators:
            if text_lower.startswith(indicator):
                patterns.append(('heading_keyword', indicator))
        
        return patterns

    def learn_heading_patterns(self, structure_data):
        """Enhanced pattern learning with better font analysis"""
        font_analysis = structure_data['font_analysis']
        size_stats = structure_data['size_stats']
        
        # Analyze each font combination
        font_characteristics = {}
        
        for font_key, spans in font_analysis.items():
            font, size, flags = font_key
            
            if not spans:
                continue
                
            texts = [span['text'] for span in spans]
            avg_length = mean([len(text) for text in texts])
            avg_words = mean([len(text.split()) for text in texts])
            unique_ratio = len(set(texts)) / len(texts) if texts else 0
            
            # Position analysis
            positions = [(span['bbox'][0], span['bbox'][1]) for span in spans]
            pages = [span['page'] for span in spans]
            
            font_characteristics[font_key] = {
                'count': len(spans),
                'avg_length': avg_length,
                'avg_words': avg_words,
                'unique_ratio': unique_ratio,
                'size': size,
                'is_bold': bool(flags & 16),
                'is_italic': bool(flags & 2),
                'texts': texts,
                'positions': positions,
                'pages': pages,
                'size_percentile': self.calculate_size_percentile(size, structure_data['size_distribution'])
            }
        
        # Score fonts for heading likelihood
        heading_candidates = []
        
        for font_key, char in font_characteristics.items():
            score = self.calculate_heading_score(char, size_stats)
            
            if score >= 5:  # Minimum threshold for heading consideration
                heading_candidates.append((font_key, char, score))
        
        # Sort by score
        heading_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Pattern analysis
        pattern_stats = defaultdict(int)
        for pattern_type, pattern_value in structure_data['text_patterns']:
            pattern_stats[pattern_type] += 1
        
        return {
            'heading_fonts': heading_candidates,
            'pattern_frequency': pattern_stats,
            'size_stats': size_stats,
            'total_fonts': len(font_characteristics)
        }

    def calculate_size_percentile(self, size, size_distribution):
        """Calculate what percentile this font size is"""
        total_occurrences = sum(size_distribution.values())
        smaller_sizes = sum(count for s, count in size_distribution.items() if s < size)
        return (smaller_sizes / total_occurrences) * 100 if total_occurrences > 0 else 0

    def calculate_heading_score(self, characteristics, size_stats):
        """Enhanced scoring algorithm for heading detection"""
        score = 0
        
        # Size scoring (more important factor)
        size_percentile = characteristics['size_percentile']
        if size_percentile >= 80:
            score += 5
        elif size_percentile >= 60:
            score += 3
        elif size_percentile >= 40:
            score += 1
        
        # Style scoring
        if characteristics['is_bold']:
            score += 3
        if characteristics['is_italic']:
            score += 1
        
        # Text characteristics
        if characteristics['avg_words'] <= 8:  # Short text
            score += 2
        elif characteristics['avg_words'] <= 15:
            score += 1
        elif characteristics['avg_words'] > 25:
            score -= 2
        
        # Uniqueness (headings are usually unique)
        if characteristics['unique_ratio'] > 0.8:
            score += 3
        elif characteristics['unique_ratio'] > 0.5:
            score += 1
        elif characteristics['unique_ratio'] < 0.2:
            score -= 2
        
        # Frequency analysis (headings appear less frequently)
        if characteristics['count'] <= 5:
            score += 2
        elif characteristics['count'] <= 15:
            score += 1
        elif characteristics['count'] >= 50:
            score -= 3
        
        # Length analysis
        if characteristics['avg_length'] <= 50:
            score += 1
        elif characteristics['avg_length'] >= 150:
            score -= 2
        
        return score

    def extract_title_enhanced(self, spans, learned_patterns):
        """Enhanced title extraction combining both approaches"""
        # Filter to first page spans
        page1_spans = [s for s in spans if s["page"] == 1 and len(s["text"].strip()) > 2]
        
        if not page1_spans:
            return "Untitled Document"
        
        # Define title area (top portion of first page)
        max_y = max(s["bbox"][3] for s in page1_spans)
        title_threshold = max_y * self.title_area_threshold
        title_area_spans = [s for s in page1_spans if s["bbox"][1] < title_threshold]
        
        if not title_area_spans:
            title_area_spans = page1_spans[:10]  # Fallback to first 10 spans
        
        # Get title candidates using multiple methods
        title_candidates = []
        
        # Method 1: Largest font sizes
        size_groups = defaultdict(list)
        for span in title_area_spans:
            size_key = round(span["size"] * 2) / 2
            size_groups[size_key].append(span)
        
        # Take top 3 font sizes
        sorted_sizes = sorted(size_groups.keys(), reverse=True)[:3]
        
        for size in sorted_sizes:
            spans_for_size = size_groups[size]
            lines = self.group_spans_into_lines(spans_for_size)
            
            for line_spans in lines:
                # Filter out labels and colons
                content_spans = [s for s in line_spans if not s["text"].strip().endswith(":")]
                if content_spans:
                    merged_text = self.merge_line_spans(content_spans)
                    if merged_text and len(merged_text.strip()) >= 5:
                        score = self.score_title_candidate(merged_text, size, spans_for_size[0])
                        title_candidates.append((merged_text.strip(), score, size))
        
        # Method 2: Use learned patterns for validation
        if learned_patterns and 'heading_fonts' in learned_patterns:
            for font_key, char, font_score in learned_patterns['heading_fonts'][:3]:
                for span in title_area_spans:
                    span_key = (span["font"], span["size"], span["flags"])
                    if span_key == font_key:
                        text = span["text"].strip()
                        if len(text) >= 5:
                            score = self.score_title_candidate(text, span["size"], span)
                            title_candidates.append((text, score + font_score/2, span["size"]))
        
        if not title_candidates:
            return "Untitled Document"
        
        # Sort by score and select best candidate
        title_candidates.sort(key=lambda x: x[1], reverse=True)
        best_title = title_candidates[0][0]
        
        # Clean and validate title
        cleaned_title = self.clean_title(best_title)
        
        # Check length constraint
        if len(cleaned_title) > self.max_title_length:
            # Try to get a shorter version
            words = cleaned_title.split()
            if len(words) > 3:
                cleaned_title = ' '.join(words[:int(len(words)*0.7)])
        
        return cleaned_title if cleaned_title else "Untitled Document"

    def score_title_candidate(self, text, font_size, span):
        """Score a title candidate"""
        score = 0
        
        # Length scoring
        word_count = len(text.split())
        if 3 <= word_count <= 12:
            score += 5
        elif word_count <= 20:
            score += 2
        elif word_count > 25:
            score -= 5
        
        # Font size (assume larger is better for title)
        score += font_size / 5
        
        # Style scoring
        if span.get("bold", False):
            score += 3
        if span.get("italic", False):
            score += 1
        
        # Position scoring (higher on page is better)
        y_pos = span["bbox"][1]
        if y_pos < 100:  # Very top of page
            score += 5
        elif y_pos < 200:
            score += 3
        
        # Content quality
        if not any(exclude in text.lower() for exclude in ['page', 'figure', 'table', 'www', 'http', '©']):
            score += 2
        
        # Capitalization patterns
        if text.istitle():
            score += 2
        elif text.isupper() and len(text) < 100:
            score += 1
        
        return score

    def extract_headings_enhanced(self, spans, learned_patterns, title_text=""):
        """Enhanced heading extraction using learned patterns and positioning"""
        if not spans:
            return []
        
        # Get heading font candidates from learned patterns
        heading_fonts = {}
        if learned_patterns and 'heading_fonts' in learned_patterns:
            for font_key, char, score in learned_patterns['heading_fonts']:
                heading_fonts[font_key] = score
        
        # Extract title font sizes to exclude from headings
        title_font_sizes = self.get_title_font_sizes(spans, title_text)
        
        heading_candidates = []
        
        # Process each page
        for page_num in range(1, max(s["page"] for s in spans) + 1):
            page_spans = [s for s in spans if s["page"] == page_num]
            if not page_spans:
                continue
            
            # Sort by vertical position
            page_spans.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
            
            i = 0
            while i < len(page_spans):
                span = page_spans[i]
                text = span["text"].strip()
                
                if len(text) < 2:
                    i += 1
                    continue
                
                # Check if this could be a heading
                font_key = (span["font"], span["size"], span["flags"])
                is_heading, confidence = self.is_potential_heading_enhanced(
                    span, page_spans, heading_fonts, title_font_sizes, learned_patterns
                )
                
                if is_heading:
                    # Try to build complete heading (handle multi-line)
                    complete_heading = self.build_complete_heading(page_spans, i)
                    
                    if complete_heading:
                        heading_candidates.append({
                            "text": complete_heading["text"],
                            "size": span["size"],
                            "page": page_num,
                            "confidence": confidence,
                            "bbox": span["bbox"],
                            "font_key": font_key,
                            "bold": span["bold"],
                            "spans_used": complete_heading["spans_count"]
                        })
                        
                        # Skip the spans we used
                        i += complete_heading["spans_count"]
                    else:
                        i += 1
                else:
                    i += 1
        
        if not heading_candidates:
            return []
        
        # Remove duplicates and assign levels
        unique_headings = self.remove_duplicate_headings(heading_candidates)
        final_headings = self.assign_heading_levels_enhanced(unique_headings)
        
        return final_headings

    def is_potential_heading_enhanced(self, span, page_spans, heading_fonts, title_font_sizes, learned_patterns):
        """Enhanced heading detection with multiple criteria"""
        text = span["text"].strip()
        font_key = (span["font"], span["size"], span["flags"])
        
        # Basic validation
        if len(text) < 2 or len(text) > 300:
            return False, 0
        
        confidence = 0
        
        # Check against learned heading fonts
        if font_key in heading_fonts:
            confidence += heading_fonts[font_key]
        
        # Size analysis
        size = span["size"]
        if any(abs(size - title_size) <= 1.0 for title_size in title_font_sizes):
            return False, 0  # Too similar to title
        
        size_percentile = self.calculate_span_size_percentile(span, page_spans)
        if size_percentile >= 70:
            confidence += 4
        elif size_percentile >= 50:
            confidence += 2
        
        # Style analysis
        if span["bold"]:
            confidence += 3
        if span["italic"]:
            confidence += 1
        
        # Text pattern analysis
        patterns = self.extract_text_patterns(text)
        for pattern_type, pattern_value in patterns:
            if pattern_type in ['decimal_number', 'roman_numeral', 'heading_keyword']:
                confidence += 2
            elif pattern_type in ['all_uppercase', 'colon_ending']:
                confidence += 1
        
        # Length analysis
        word_count = len(text.split())
        if word_count <= 10:
            confidence += 2
        elif word_count <= 20:
            confidence += 1
        elif word_count > 30:
            confidence -= 3
        
        # Position and context analysis
        following_content = self.analyze_following_content(span, page_spans)
        if following_content["has_substantial_content"]:
            confidence += 3
        if following_content["content_smaller_font"]:
            confidence += 2
        
        # Content validation
        if self.is_likely_non_heading_content(text):
            confidence -= 5
        
        return confidence >= 6, confidence

    def calculate_span_size_percentile(self, span, page_spans):
        """Calculate size percentile for a span within page context"""
        sizes = [s["size"] for s in page_spans]
        if not sizes:
            return 50
        
        smaller_count = sum(1 for s in sizes if s < span["size"])
        return (smaller_count / len(sizes)) * 100

    def analyze_following_content(self, heading_span, page_spans):
        """Analyze content following a potential heading"""
        heading_y = heading_span["bbox"][3]  # Bottom of heading
        heading_size = heading_span["size"]
        
        following_spans = []
        for span in page_spans:
            if span["bbox"][1] > heading_y + (heading_size * 0.5):  # Below heading with some gap
                following_spans.append(span)
        
        # Sort by position
        following_spans.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
        
        # Analyze first few spans
        analysis = {
            "has_substantial_content": False,
            "content_smaller_font": False,
            "total_chars": 0,
            "unique_texts": 0
        }
        
        if len(following_spans) >= 3:
            first_spans = following_spans[:8]
            texts = [s["text"].strip() for s in first_spans if len(s["text"].strip()) > 2]
            
            analysis["total_chars"] = sum(len(t) for t in texts)
            analysis["unique_texts"] = len(set(texts))
            analysis["has_substantial_content"] = analysis["total_chars"] >= 50
            
            # Check if following content has smaller font
            following_sizes = [s["size"] for s in first_spans if s["text"].strip()]
            if following_sizes:
                avg_following_size = mean(following_sizes)
                analysis["content_smaller_font"] = avg_following_size < heading_size
        
        return analysis

    def is_likely_non_heading_content(self, text):
        """Check if text is likely NOT a heading"""
        text_lower = text.lower()
        
        # Skip patterns
        skip_patterns = [
            'copyright', '©', 'page', 'figure', 'table', 'www', 'http', '.com',
            'email', '@', 'phone', 'tel:', 'fax:', 'address', 'signature',
            'name:', 'date:', 'time:', 'location:'
        ]
        
        if any(pattern in text_lower for pattern in skip_patterns):
            return True
        
        # Skip if mostly numbers or symbols
        if re.match(r'^[\d\s\-_\.,:;()]+$', text):
            return True
        
        # Skip very repetitive text
        words = text.split()
        if len(words) > 1 and len(set(words)) == 1:
            return True
        
        return False

    def build_complete_heading(self, page_spans, start_index):
        """Build complete heading that might span multiple lines"""
        if start_index >= len(page_spans):
            return None
        
        start_span = page_spans[start_index]
        heading_text = start_span["text"].strip()
        spans_used = 1
        
        # Check for continuation on same line
        current_y = start_span["bbox"][1]
        current_size = start_span["size"]
        
        i = start_index + 1
        while i < len(page_spans):
            next_span = page_spans[i]
            y_diff = abs(next_span["bbox"][1] - current_y)
            size_diff = abs(next_span["size"] - current_size)
            
            # Same line continuation
            if y_diff < current_size * 0.4 and size_diff < 1:
                heading_text = self.merge_with_overlap_removal(heading_text, next_span["text"].strip())
                spans_used += 1
                i += 1
            else:
                break
        
        # Check for multi-line continuation
        if i < len(page_spans):
            next_line_span = page_spans[i]
            y_gap = next_line_span["bbox"][1] - current_y
            
            # Next line with same size might be continuation
            if (current_size * 0.8 < y_gap < current_size * 2.5 and 
                abs(next_line_span["size"] - current_size) < 1 and
                len(heading_text.split()) < 8):  # Only for shorter headings
                
                continuation_text = next_line_span["text"].strip()
                if continuation_text and not continuation_text.startswith(('•', '-', '1.', 'a.', 'i.')):
                    heading_text += " " + continuation_text
                    spans_used += 1
        
        return {
            "text": heading_text,
            "spans_count": spans_used
        }

    def remove_duplicate_headings(self, candidates):
        """Remove duplicate headings while preserving the best ones"""
        seen_texts = {}
        unique_candidates = []
        
        # Sort by confidence first
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        
        for candidate in candidates:
            text_key = candidate["text"].lower().strip()
            
            # Check for exact match
            if text_key in seen_texts:
                continue
            
            # Check for substantial overlap
            is_duplicate = False
            for existing_text in seen_texts:
                similarity = self.calculate_text_similarity(text_key, existing_text)
                if similarity > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_texts[text_key] = candidate
                unique_candidates.append(candidate)
        
        return unique_candidates

    def calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    def assign_heading_levels_enhanced(self, candidates):
        """Enhanced level assignment with better hierarchy detection"""
        if not candidates:
            return []
        
        # Sort by size (descending) then by page order
        candidates.sort(key=lambda x: (-x["size"], x["page"], x["bbox"][1]))
        
        # Get unique sizes
        unique_sizes = sorted(list(set(c["size"] for c in candidates)), reverse=True)
        
        # Assign levels (maximum 6 levels)
        size_to_level = {}
        for i, size in enumerate(unique_sizes[:6]):
            size_to_level[size] = f"H{i+1}"
        
        # Build final outline
        outline = []
        for candidate in candidates:
            level = size_to_level.get(candidate["size"], "H6")
            
            outline.append({
                "level": level,
                "text": candidate["text"],
                "page": candidate["page"],
                "confidence": candidate.get("confidence", 0)
            })
        
        # Sort by page order for final output
        outline.sort(key=lambda x: (x["page"], x["text"]))
        
        return outline

    def get_title_font_sizes(self, spans, title_text=""):
        """Get font sizes used in title area"""
        page1_spans = [s for s in spans if s["page"] == 1]
        if not page1_spans:
            return []
        
        max_y = max(s["bbox"][3] for s in page1_spans)
        title_threshold = max_y * self.title_area_threshold
        title_spans = [s for s in page1_spans if s["bbox"][1] < title_threshold and s["size"] > 8]
        
        title_sizes = []
        if title_text:
            # Find spans that match title text
            for span in title_spans:
                if title_text.lower() in span["text"].lower() or span["text"].lower() in title_text.lower():
                    title_sizes.append(span["size"])
        
        if not title_sizes:
            # Fallback: get largest sizes from title area
            if title_spans:
                sizes = [s["size"] for s in title_spans]
                title_sizes = sorted(set(sizes), reverse=True)[:2]
        
        return title_sizes

    def group_spans_into_lines(self, spans):
        """Group spans into lines based on vertical position"""
        if not spans:
            return []
        
        spans.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
        
        lines = []
        current_line = [spans[0]]
        
        for span in spans[1:]:
            prev_y = current_line[-1]["bbox"][1]
            curr_y = span["bbox"][1]
            y_diff = abs(curr_y - prev_y)
            
            font_size = span["size"]
            line_threshold = font_size * 0.4
            
            if y_diff < line_threshold:
                current_line.append(span)
            else:
                lines.append(current_line)
                current_line = [span]
        
        if current_line:
            lines.append(current_line)
        
        return lines

    def merge_line_spans(self, spans):
        """Merge spans on same line with overlap removal"""
        if not spans:
            return ""
        
        if len(spans) == 1:
            return spans[0]["text"]
        
        spans.sort(key=lambda s: s["bbox"][0])  # Sort by x position
        
        merged_text = spans[0]["text"].strip()
        
        for span in spans[1:]:
            next_text = span["text"].strip()
            merged_text = self.merge_with_overlap_removal(merged_text, next_text)
        
        return merged_text

    def merge_with_overlap_removal(self, existing_text, new_text):
        """Merge text while removing overlaps"""
        if not existing_text:
            return new_text
        if not new_text:
            return existing_text
        
        # Check complete containment
        if new_text.lower() in existing_text.lower():
            return existing_text
        if existing_text.lower() in new_text.lower():
            return new_text
        
        # Find character-level overlap
        max_overlap_len = min(len(existing_text), len(new_text))
        best_overlap_len = 0
        
        for i in range(1, max_overlap_len + 1):
            end_part = existing_text[-i:].lower()
            start_part = new_text[:i].lower()
            
            if end_part == start_part:
                best_overlap_len = i
        
        if best_overlap_len > 0:
            return existing_text + new_text[best_overlap_len:]
        
        # Check word-level overlap
        existing_words = existing_text.split()
        new_words = new_text.split()
        
        if existing_words and new_words:
            if existing_words[-1].lower() == new_words[0].lower():
                return existing_text + " " + " ".join(new_words[1:])
        
        # No overlap found
        if existing_text and new_text:
            if existing_text[-1].isalnum() and new_text[0].isalnum():
                return existing_text + " " + new_text
            else:
                return existing_text + new_text
        
        return existing_text + new_text

    def clean_title(self, title):
        """Clean and validate extracted title"""
        if not title:
            return "Untitled Document"
        
        # Remove extra whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Remove leading/trailing artifacts
        title = re.sub(r'^[\s\-_]+|[\s\-_]+$', '', title)
        title = title.replace('\n', ' ').replace('\r', ' ')
        
        # Remove excessive repeated characters
        title = re.sub(r'(.)\1{4,}', r'\1', title)
        
        # Remove common artifacts
        title = re.sub(r'\s*\|\s*', ' ', title)  # Remove pipe separators
        title = re.sub(r'\s*[•·▪]\s*', ' ', title)  # Remove bullets
        
        if len(title.strip()) < 3:
            return "Untitled Document"
        
        return title.strip()

    def extract_outline(self, pdf_path):
        """Main method to extract title and outline from PDF"""
        try:
            logger.info(f"Processing PDF: {os.path.basename(pdf_path)}")
            
            # First, try embedded TOC
            try:
                doc = fitz.open(pdf_path)
                toc = doc.get_toc()
                if toc and len(toc) >= 3:  # Only use if substantial TOC
                    outline = []
                    for level, title, page in toc:
                        clean_title = re.sub(r'\s+', ' ', title.strip())
                        if clean_title and len(clean_title) > 2:
                            outline.append({
                                "level": f"H{min(level, 6)}",
                                "text": clean_title,
                                "page": page,
                            })
                    
                    # Extract title separately
                    spans = self.extract_pdf_metadata(pdf_path)
                    structure_data = self.analyze_document_structure(spans)
                    learned_patterns = self.learn_heading_patterns(structure_data)
                    title = self.extract_title_enhanced(spans, learned_patterns)
                    
                    doc.close()
                    return {
                        "title": title,
                        "outline": outline,
                        "source": "embedded_toc",
                        "method": "toc_with_enhanced_title"
                    }
                doc.close()
            except Exception as e:
                logger.warning(f"Could not read embedded TOC: {e}")
            
            # Extract spans and analyze structure
            spans = self.extract_pdf_metadata(pdf_path)
            if not spans:
                return {
                    "title": "Empty Document",
                    "outline": [],
                    "error": "No text content found"
                }
            
            logger.info(f"Extracted {len(spans)} text spans")
            
            # Analyze document structure and learn patterns
            structure_data = self.analyze_document_structure(spans)
            learned_patterns = self.learn_heading_patterns(structure_data)
            
            logger.info(f"Found {len(learned_patterns['heading_fonts'])} potential heading fonts")
            
            # Extract title using enhanced method
            title = self.extract_title_enhanced(spans, learned_patterns)
            
            # Extract headings using enhanced method
            headings = self.extract_headings_enhanced(spans, learned_patterns, title)
            
            logger.info(f"Extracted title: '{title}'")
            logger.info(f"Extracted {len(headings)} headings")
            
            return {
                "title": title,
                "outline": headings,
                "source": "content_analysis",
                "method": "enhanced_extraction",
                "stats": {
                    "total_spans": len(spans),
                    "heading_fonts_found": len(learned_patterns['heading_fonts']),
                    "unique_font_sizes": structure_data['size_stats']['unique_sizes'] if 'size_stats' in structure_data else 0,
                    "pages_analyzed": max(s["page"] for s in spans) if spans else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {
                "title": "Error - Could not process",
                "outline": [],
                "error": str(e)
            }

    def debug_extraction(self, pdf_path, show_details=True):
        """Debug method to show detailed extraction process"""
        print(f"\n{'='*80}")
        print(f"DEBUG EXTRACTION: {os.path.basename(pdf_path)}")
        print('='*80)
        
        try:
            spans = self.extract_pdf_metadata(pdf_path)
            print(f"📊 Total spans extracted: {len(spans)}")
            
            if not spans:
                print("❌ No spans found!")
                return
            
            structure_data = self.analyze_document_structure(spans)
            learned_patterns = self.learn_heading_patterns(structure_data)
            
            print(f"📈 Document Statistics:")
            print(f"  - Pages: {max(s['page'] for s in spans)}")
            print(f"  - Unique fonts: {len(structure_data['font_analysis'])}")
            print(f"  - Font sizes: {structure_data['size_stats']['unique_sizes']}")
            print(f"  - Size range: {structure_data['size_stats']['min']:.1f} - {structure_data['size_stats']['max']:.1f}")
            
            print(f"\n🎯 Learned Heading Patterns:")
            for i, (font_key, char, score) in enumerate(learned_patterns['heading_fonts'][:5]):
                font, size, flags = font_key
                print(f"  {i+1}. Font: {font}, Size: {size:.1f}, Score: {score:.1f}")
                print(f"     Bold: {bool(flags & 16)}, Avg Words: {char['avg_words']:.1f}")
                if show_details and char['texts']:
                    sample_texts = char['texts'][:3]
                    print(f"     Samples: {sample_texts}")
            
            title = self.extract_title_enhanced(spans, learned_patterns)
            headings = self.extract_headings_enhanced(spans, learned_patterns, title)
            
            print(f"\n📌 Extracted Title: '{title}'")
            print(f"\n📋 Extracted Headings ({len(headings)}):")
            
            level_counts = defaultdict(int)
            for heading in headings:
                level_counts[heading['level']] += 1
                confidence = heading.get('confidence', 0)
                print(f"  {heading['level']}: '{heading['text']}' (Page {heading['page']}, Conf: {confidence:.1f})")
            
            print(f"\n📊 Heading Distribution:")
            for level in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']:
                count = level_counts.get(level, 0)
                if count > 0:
                    print(f"  {level}: {count} headings")
            
        except Exception as e:
            print(f"❌ Debug error: {e}")
            import traceback
            traceback.print_exc()

def ensure_directory(path):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)

def process_single_pdf(pdf_path, output_dir, extractor, debug_mode=False):
    """Process a single PDF file"""
    filename = os.path.basename(pdf_path)
    base = os.path.splitext(filename)[0]
    json_path = os.path.join(output_dir, f'{base}.json')
    
    try:
        if debug_mode:
            extractor.debug_extraction(pdf_path)
        
        outline_data = extractor.extract_outline(pdf_path)
        
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(outline_data, f, indent=2, ensure_ascii=False)
        
        method = outline_data.get('method', 'unknown')
        outline_count = len(outline_data.get('outline', []))
        stats = outline_data.get('stats', {})
        
        logger.info(f'✅ {filename} -> {outline_count} headings (Method: {method})')
        return True, filename, None, outline_data
        
    except Exception as e:
        error_msg = f"Failed to process {filename}: {e}"
        logger.error(error_msg)
        return False, filename, str(e), None

def process_all_pdfs(input_dir, output_dir, max_workers=4, debug_mode=False):
    """Process all PDFs in directory"""
    ensure_directory(output_dir)
    
    pdf_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            pdf_files.append(pdf_path)
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return
    
    logger.info(f"🚀 Processing {len(pdf_files)} PDFs with enhanced extraction")
    
    extractor = EnhancedPDFExtractor()
    
    successful = 0
    failed = 0
    method_stats = defaultdict(int)
    total_headings = 0
    
    results = []
    
    if debug_mode or len(pdf_files) <= 3:
        # Process sequentially for debugging or small batches
        for pdf_path in pdf_files:
            success, filename, error, data = process_single_pdf(
                pdf_path, output_dir, extractor, debug_mode
            )
            results.append((success, filename, error, data))
    else:
        # Process in parallel for larger batches
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pdf = {
                executor.submit(process_single_pdf, pdf_path, output_dir, extractor, False): pdf_path 
                for pdf_path in pdf_files
            }
            
            for future in as_completed(future_to_pdf):
                result = future.result()
                results.append(result)
    
    # Analyze results
    for success, filename, error, data in results:
        if success:
            successful += 1
            if data:
                method = data.get('method', 'unknown')
                method_stats[method] += 1
                total_headings += len(data.get('outline', []))
        else:
            failed += 1
    
    # Print summary
    logger.info(f"🎯 Processing complete: {successful} successful, {failed} failed")
    logger.info(f"📊 Total headings extracted: {total_headings}")
    
    if method_stats:
        logger.info("📋 Extraction methods used:")
        for method, count in sorted(method_stats.items()):
            logger.info(f"  {method}: {count} documents")
    
    # Create summary report
    summary_path = os.path.join(output_dir, "extraction_summary.json")
    summary = {
        "total_files": len(pdf_files),
        "successful": successful,
        "failed": failed,
        "total_headings_extracted": total_headings,
        "methods_used": dict(method_stats),
        "processing_timestamp": json.dumps({"timestamp": "now"}, default=str)
    }
    
    with open(summary_path, "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 Summary saved to: {summary_path}")

def test_single_file(pdf_path, debug=True):
    """Test function for single file processing"""
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    
    extractor = EnhancedPDFExtractor()
    
    print(f"🔍 Testing: {os.path.basename(pdf_path)}")
    
    if debug:
        extractor.debug_extraction(pdf_path)
    
    result = extractor.extract_outline(pdf_path)
    
    print(f"\n{'='*60}")
    print("FINAL RESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print('='*60)

if __name__ == "__main__":
    # Example usage:
    
    # Test single file
    # test_single_file("path/to/your/test.pdf", debug=True)
    
    # Process all PDFs in directory
    process_all_pdfs(INPUT_DIR, OUTPUT_DIR, max_workers=4, debug_mode=False)
    
    # Debug mode for smaller batches
    # process_all_pdfs(INPUT_DIR, OUTPUT_DIR, max_workers=1, debug_mode=True)