#!/usr/bin/env python3
"""
Automated Detection of Biblical Allusions in Toni Morrison's Novels
Felix Oke - University of Illinois at Urbana-Champaign

Processes Morrison novels from HathiTrust Digital Library (.txt format)
using Extended Features API for computational text analysis.
"""

import re
import spacy
from typing import List, Dict, Tuple, Set
import json
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Allusion:
    """Data class for storing detected allusions"""
    text: str
    start_pos: int
    end_pos: int
    allusion_type: str
    confidence: float
    biblical_source: str = ""
    functional_category: str = ""

class BiblicalAllusionDetector:
    """Main class for detecting biblical allusions in text"""
    
    def __init__(self):
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            raise
        
        # Biblical books (Old and New Testament)
        self.biblical_books = {
            # Old Testament
            'genesis', 'exodus', 'leviticus', 'numbers', 'deuteronomy', 'joshua', 'judges', 'ruth',
            '1 samuel', '2 samuel', '1 kings', '2 kings', '1 chronicles', '2 chronicles', 'ezra',
            'nehemiah', 'esther', 'job', 'psalms', 'proverbs', 'ecclesiastes', 'song of songs',
            'isaiah', 'jeremiah', 'lamentations', 'ezekiel', 'daniel', 'hosea', 'joel', 'amos',
            'obadiah', 'jonah', 'micah', 'nahum', 'habakkuk', 'zephaniah', 'haggai', 'zechariah',
            'malachi',
            # New Testament
            'matthew', 'mark', 'luke', 'john', 'acts', 'romans', '1 corinthians', '2 corinthians',
            'galatians', 'ephesians', 'philippians', 'colossians', '1 thessalonians', '2 thessalonians',
            '1 timothy', '2 timothy', 'titus', 'philemon', 'hebrews', 'james', '1 peter', '2 peter',
            '1 john', '2 john', '3 john', 'jude', 'revelation'
        }
        
        # Biblical names (characters, places)
        self.biblical_names = {
            # Major characters
            'adam', 'eve', 'noah', 'abraham', 'isaac', 'jacob', 'moses', 'david', 'solomon',
            'jesus', 'mary', 'joseph', 'peter', 'paul', 'john', 'matthew', 'mark', 'luke',
            'hagar', 'sarah', 'rebecca', 'rachel', 'leah', 'ruth', 'esther', 'judith',
            'pilate', 'pontius pilate', 'herod', 'pharaoh', 'goliath', 'samson', 'delilah',
            'cain', 'abel', 'seth', 'enoch', 'methuselah', 'lot', 'ishmael', 'esau',
            'aaron', 'miriam', 'joshua', 'caleb', 'gideon', 'samuel', 'saul', 'jonathan',
            'bathsheba', 'absalom', 'elijah', 'elisha', 'isaiah', 'jeremiah', 'ezekiel',
            'daniel', 'job', 'jonah', 'hosea', 'amos', 'micah', 'habakkuk', 'malachi',
            # Places
            'eden', 'babylon', 'jerusalem', 'bethlehem', 'nazareth', 'galilee', 'jordan',
            'sinai', 'mount sinai', 'calvary', 'golgotha', 'gethsemane', 'jericho',
            'sodom', 'gomorrah', 'egypt', 'canaan', 'israel', 'judah', 'samaria',
            # Morrison-specific names
            'first corinthians', 'magdalene', 'ruth dead', 'pilate dead', 'hagar dead'
        }
        
        # Biblical phrases and concepts
        self.biblical_phrases = {
            'promised land', 'garden of eden', 'tree of knowledge', 'forbidden fruit',
            'forty days and forty nights', 'parting of the sea', 'burning bush',
            'ten commandments', 'golden calf', 'ark of the covenant', 'holy grail',
            'last supper', 'sermon on the mount', 'good samaritan', 'prodigal son',
            'loaves and fishes', 'water into wine', 'resurrection', 'crucifixion',
            'second coming', 'judgment day', 'book of life', 'lamb of god',
            'alpha and omega', 'born again', 'baptism', 'communion', 'trinity',
            'original sin', 'redemption', 'salvation', 'grace', 'faith', 'hope', 'charity'
        }
        
        # Allusion types
        self.allusion_types = [
            'direct_quote', 'paraphrase', 'thematic_reference', 
            'character_reference', 'structural_echo', 'no_biblical_reference'
        ]
        
        # Functional categories
        self.functional_categories = [
            'characterization', 'thematic_development', 'narrative_structure',
            'cultural_commentary', 'ironic_contrast', 'spiritual_dimension'
        ]
    
    def rule_based_detection(self, text: str) -> List[Allusion]:
        """Rule-based detection of biblical allusions"""
        allusions = []
        text_lower = text.lower()
        
        # Detect biblical book references
        for book in self.biblical_books:
            pattern = r'\b' + re.escape(book) + r'\b'
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                allusion = Allusion(
                    text=text[match.start():match.end()],
                    start_pos=match.start(),
                    end_pos=match.end(),
                    allusion_type='thematic_reference',
                    confidence=0.9,
                    biblical_source=book.title()
                )
                allusions.append(allusion)
        
        # Detect biblical phrases
        for phrase in self.biblical_phrases:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                allusion = Allusion(
                    text=text[match.start():match.end()],
                    start_pos=match.start(),
                    end_pos=match.end(),
                    allusion_type='thematic_reference',
                    confidence=0.8,
                    biblical_source=phrase.title()
                )
                allusions.append(allusion)
        
        # Detect chapter:verse patterns (e.g., "Romans 9:25")
        verse_pattern = r'\b([1-3]?\s*[a-zA-Z]+)\s+(\d+):(\d+)\b'
        matches = re.finditer(verse_pattern, text)
        for match in matches:
            book_name = match.group(1).strip().lower()
            if book_name in self.biblical_books:
                allusion = Allusion(
                    text=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    allusion_type='direct_quote',
                    confidence=0.95,
                    biblical_source=f"{book_name.title()} {match.group(2)}:{match.group(3)}"
                )
                allusions.append(allusion)
        
        return allusions
    
    def ner_detection(self, text: str) -> List[Allusion]:
        """Named Entity Recognition for biblical names"""
        allusions = []
        doc = self.nlp(text)
        text_lower = text.lower()
        
        # Check for biblical names in the text
        for name in self.biblical_names:
            pattern = r'\b' + re.escape(name) + r'\b'
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Get confidence based on NER recognition
                confidence = 0.7
                
                # Check if spaCy recognizes it as a PERSON
                span_start = match.start()
                span_end = match.end()
                for ent in doc.ents:
                    if (ent.start_char <= span_start < ent.end_char or 
                        ent.start_char < span_end <= ent.end_char) and ent.label_ == "PERSON":
                        confidence = 0.85
                        break
                
                allusion = Allusion(
                    text=text[match.start():match.end()],
                    start_pos=match.start(),
                    end_pos=match.end(),
                    allusion_type='character_reference',
                    confidence=confidence,
                    biblical_source=name.title()
                )
                allusions.append(allusion)
        
        return allusions
    
    def detect_allusions(self, text: str) -> List[Allusion]:
        """Main method to detect all types of biblical allusions"""
        rule_based_allusions = self.rule_based_detection(text)
        ner_allusions = self.ner_detection(text)
        
        # Combine and deduplicate allusions
        all_allusions = rule_based_allusions + ner_allusions
        
        # Remove overlapping allusions (keep highest confidence)
        filtered_allusions = []
        for allusion in sorted(all_allusions, key=lambda x: x.confidence, reverse=True):
            overlap = False
            for existing in filtered_allusions:
                if (allusion.start_pos < existing.end_pos and 
                    allusion.end_pos > existing.start_pos):
                    overlap = True
                    break
            if not overlap:
                filtered_allusions.append(allusion)
        
        return sorted(filtered_allusions, key=lambda x: x.start_pos)
    
    def analyze_text(self, text: str, title: str = "") -> Dict:
        """Analyze text and return structured results"""
        allusions = self.detect_allusions(text)
        
        # Group by type
        by_type = defaultdict(list)
        for allusion in allusions:
            by_type[allusion.allusion_type].append(allusion)
        
        results = {
            'title': title,
            'total_allusions': len(allusions),
            'allusions': [
                {
                    'text': a.text,
                    'start_pos': a.start_pos,
                    'end_pos': a.end_pos,
                    'type': a.allusion_type,
                    'confidence': a.confidence,
                    'biblical_source': a.biblical_source,
                    'context': text[max(0, a.start_pos-50):a.end_pos+50]
                }
                for a in allusions
            ],
            'by_type': {k: len(v) for k, v in by_type.items()},
            'statistics': {
                'avg_confidence': sum(a.confidence for a in allusions) / len(allusions) if allusions else 0,
                'high_confidence_count': len([a for a in allusions if a.confidence > 0.8])
            }
        }
        
        return results

def analyze_morrison_novel(detector, filename, title):
    """Analyze Morrison novel for biblical allusions"""
    
    # Load text file
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"✅ Loaded {title}: {len(text):,} characters")
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return None
    
    print(f"🔍 Analyzing biblical allusions in {title}...")
    
    # Analyze allusions using detector
    results = detector.analyze_text(text, title)
    
    # Add metadata
    from datetime import datetime
    results['metadata'] = {
        'filename': filename,
        'analysis_date': datetime.now().isoformat(),
        'text_length': len(text),
        'analyzer': 'BiblicalAllusionDetector v1.0'
    }
    
    # Print summary
    print(f"\n📊 RESULTS FOR {title}")
    print(f"Total allusions: {results['total_allusions']}")
    print(f"Average confidence: {results['statistics']['avg_confidence']:.2f}")
    print(f"High confidence allusions: {results['statistics']['high_confidence_count']}")
    print(f"Breakdown by type: {results['by_type']}")
    
    # Show sample allusions
    print(f"\n📝 Sample allusions (first 10):")
    for i, a in enumerate(results['allusions'][:10], 1):
        print(f"{i:2d}. '{a['text']}' ({a['type']}) - {a['confidence']:.2f}")
        print(f"     Context: ...{a['context'][:80]}...")
    
    if len(results['allusions']) > 10:
        print(f"     ... and {len(results['allusions']) - 10} more allusions")
    
    # Save to JSON
    import json
    json_file = f"allusions_{filename.replace('.txt', '.json').replace(' ', '_')}"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 JSON saved to: {json_file}")
    
    # Save to CSV for HathiTrust compatibility
    csv_file = f"allusions_{filename.replace('.txt', '.csv').replace(' ', '_')}"
    export_to_csv(results, csv_file)
    
    return results

def export_to_csv(results, output_filename):
    """Export allusions to CSV format for HathiTrust compatibility"""
    import csv
    
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['novel_title', 'allusion_text', 'start_pos', 'end_pos', 'allusion_type', 
                     'confidence', 'biblical_source', 'context']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for allusion in results['allusions']:
            writer.writerow({
                'novel_title': results['title'],
                'allusion_text': allusion['text'],
                'start_pos': allusion['start_pos'],
                'end_pos': allusion['end_pos'],
                'allusion_type': allusion['type'],
                'confidence': allusion['confidence'],
                'biblical_source': allusion['biblical_source'],
                'context': allusion['context'].replace('\n', ' ').strip()
            })
    
    print(f"📊 CSV exported to: {output_filename}")

def export_all_to_csv(all_results, output_filename="morrison_allusions_combined.csv"):
    """Export all novels' allusions to single CSV file"""
    import csv
    
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['novel_title', 'allusion_text', 'start_pos', 'end_pos', 'allusion_type', 
                     'confidence', 'biblical_source', 'context']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for title, results in all_results.items():
            for allusion in results['allusions']:
                writer.writerow({
                    'novel_title': title,
                    'allusion_text': allusion['text'],
                    'start_pos': allusion['start_pos'],
                    'end_pos': allusion['end_pos'],
                    'allusion_type': allusion['type'],
                    'confidence': allusion['confidence'],
                    'biblical_source': allusion['biblical_source'],
                    'context': allusion['context'].replace('\n', ' ').strip()
                })
    
    print(f"📊 Combined CSV exported to: {output_filename}")

def main():
    """Example usage"""
    detector = BiblicalAllusionDetector()
    
    # Example text from Morrison's work
    sample_text = """
    Pilate was the daughter of the dead man. She had been named by her father, 
    who had pointed to a word in the Bible with his finger. Like First Corinthians, 
    she carried her name with dignity. The Romans 9:25 passage spoke of calling 
    those who were not beloved, beloved. In the garden of Eden, there was no shame.
    """
    
    results = detector.analyze_text(sample_text, "Sample Text")
    
    print("Biblical Allusion Detection Results:")
    print(f"Total allusions found: {results['total_allusions']}")
    print("\nDetected allusions:")
    
    for allusion in results['allusions']:
        print(f"- '{allusion['text']}' ({allusion['type']}) - Confidence: {allusion['confidence']:.2f}")
        print(f"  Source: {allusion['biblical_source']}")
        print(f"  Context: ...{allusion['context']}...")
        print()
    
    print("Summary by type:")
    for allusion_type, count in results['by_type'].items():
        print(f"- {allusion_type}: {count}")
    
    # Analyze all 8 Morrison novels
    # Update the filenames in the list to match your actual .txt files
    morrison_novels = [
        {"filename": "bluest_eye_1970.txt", "title": "The Bluest Eye (1970)"},
        {"filename": "sula_1973.txt", "title": "Sula (1973)"},
        {"filename": "song_of_solomon_1977.txt", "title": "Song of Solomon (1977)"},
        {"filename": "tar_baby_1981.txt", "title": "Tar Baby (1981)"},
        {"filename": "beloved_1987.txt", "title": "Beloved (1987)"},
        {"filename": "jazz_1992.txt", "title": "Jazz (1992)"},
        {"filename": "paradise_1998.txt", "title": "Paradise (1998)"},
        {"filename": "mercy_2008.txt", "title": "A Mercy (2008)"}
    ]
    
    print("\n" + "=" * 60)
    print("ANALYZING ALL MORRISON NOVELS")
    print("=" * 60)
    
    all_results = {}
    total_allusions = 0
    
    for novel in morrison_novels:
        print(f"\n{'-' * 40}")
        results = analyze_morrison_novel(detector, novel["filename"], novel["title"])
        if results:
            all_results[novel["title"]] = results
            total_allusions += results['total_allusions']
    
    # Save combined results
    if all_results:
        import json
        with open("morrison_all_novels_combined.json", 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Export combined CSV for HathiTrust
        export_all_to_csv(all_results)
        
        print(f"\n" + "=" * 60)
        print("FINAL SUMMARY - ALL MORRISON NOVELS")
        print("=" * 60)
        print(f"Total novels processed: {len(all_results)}")
        print(f"Total allusions found: {total_allusions}")
        print(f"\nPer novel breakdown:")
        for title, results in all_results.items():
            print(f"  {title}: {results['total_allusions']} allusions")
        print(f"\n💾 Combined JSON saved to: morrison_all_novels_combined.json")
        print(f"📊 Combined CSV saved to: morrison_allusions_combined.csv")

if __name__ == "__main__":
    main()