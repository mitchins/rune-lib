#!/usr/bin/env python3
"""
Augmentation Utilities for NER Training Data

This module provides utilities for augmenting NER training data using the diverse
name inventory system to break bias patterns and improve model generalization.

Key Features:
- Character name replacement with cultural consistency
- Role randomization to prevent bias
- Frequency balancing to avoid overuse
- Data validation and quality checks
- Integration with existing story datasets

Addresses specific issues from the analysis:
- "Aria Nguyen" = 91.8% PROTAGONIST bias
- "Wounded Hawk" used 723 times (catastrophic repetition)
- 79.5% names classified as "Other/Mixed" (poor diversity)
"""

import json
import random
import logging
from typing import Dict, List, Set, Optional, Tuple, Union
from collections import Counter, defaultdict
from pathlib import Path
from dataclasses import dataclass

from .name_inventory import NameInventory

logger = logging.getLogger(__name__)


@dataclass
class CharacterData:
    """Structured representation of a character in a story."""
    name: str
    role: str
    culture: Optional[str] = None
    gender: Optional[str] = None
    original_name: Optional[str] = None


class AugmentUtils:
    """
    Utilities for augmenting NER training data with diverse names
    to prevent model bias and improve generalization.
    """

    def __init__(self, name_inventory: Optional[NameInventory] = None, seed: Optional[int] = None):
        """
        Initialize augmentation utilities.

        Args:
            name_inventory: NameInventory instance or None to create new one
            seed: Random seed for reproducible results
        """
        if seed is not None:
            random.seed(seed)

        self.name_inventory = name_inventory or NameInventory(seed=seed)

        # Bias prevention settings
        self.max_name_uses = 50  # Prevent "Wounded Hawk" scenario (723 uses)
        self.max_role_bias = 0.65  # Prevent "Aria Nguyen" scenario (91.8% PROTAGONIST)
        self.role_shuffle_probability = 0.3  # 30% chance to shuffle roles

        # Quality tracking
        self.augmentation_stats = {
            'stories_processed': 0,
            'characters_replaced': 0,
            'roles_shuffled': 0,
            'bias_prevented': 0,
            'overuse_prevented': 0
        }

        logger.info("AugmentUtils initialized with bias prevention mechanisms")

    def augment_story_data(
        self,
        story_data: Dict,
        replacement_probability: float = 0.8,
        role_shuffle_probability: Optional[float] = None,
        preserve_cultural_consistency: bool = True
    ) -> Dict:
        """
        Augment a single story's character data with diverse names.

        Args:
            story_data: Story dictionary with 'characters' list
            replacement_probability: Chance to replace each character name
            role_shuffle_probability: Chance to shuffle roles (overrides instance setting)
            preserve_cultural_consistency: Try to maintain cultural backgrounds

        Returns:
            Augmented story data with diverse character names
        """

        if 'characters' not in story_data:
            logger.warning("Story data missing 'characters' field")
            return story_data

        # Rule 1: Skip entire story if ANY character has "The X" name
        for char in story_data['characters']:
            if char.get('name', '').strip().lower().startswith('the '):
                logger.debug(f"Skipping entire story {story_data.get('story_id', 'unknown')} due to 'The X' character: {char.get('name')}")
                return story_data

        shuffle_prob = role_shuffle_probability if role_shuffle_probability is not None else self.role_shuffle_probability

        # Extract characters
        characters = [CharacterData(
            name=char.get('name', 'Unknown'),
            role=char.get('role', 'UNKNOWN'),
            culture=char.get('culture'),
            gender=char.get('gender')
        ) for char in story_data['characters']]

        # Apply role shuffling if enabled (prevents role-name associations)
        if random.random() < shuffle_prob and len(characters) > 1:
            roles = [char.role for char in characters]
            random.shuffle(roles)
            for char, new_role in zip(characters, roles):
                char.role = new_role
            self.augmentation_stats['roles_shuffled'] += 1

        # Rule 2: Set theory for unique names per story
        # Track names used in this story (original + replacements)
        used_names_in_story = {char.name for char in characters}

        # Replace character names
        augmented_characters = []
        for char in characters:
            if random.random() < replacement_probability:
                # Check if current name is overused
                current_usage = self.name_inventory.name_usage_counter.get(char.name, 0)

                # Check if current name has role bias
                role_distribution = self.name_inventory.role_tracking.get(char.name, {})
                has_role_bias = self._check_role_bias(role_distribution, char.role)

                # Replace if overused, biased, or random selection
                if (current_usage >= self.max_name_uses or
                    has_role_bias or
                    random.random() < replacement_probability):

                    # Generate unique name for this story
                    new_name_data = self._generate_unique_replacement_name(
                        char, preserve_cultural_consistency, used_names_in_story
                    )

                    if new_name_data:  # Only replace if we found a unique name
                        # Track original name for analysis
                        char.original_name = char.name
                        char.name = new_name_data['full_name']
                        char.culture = new_name_data['culture']
                        char.gender = new_name_data['gender']

                        # Add new name to used set for this story
                        used_names_in_story.add(char.name)

                        # Update tracking
                        self.name_inventory.track_role_assignment(char.name, char.role)
                        self.augmentation_stats['characters_replaced'] += 1

                        if current_usage >= self.max_name_uses:
                            self.augmentation_stats['overuse_prevented'] += 1
                        if has_role_bias:
                            self.augmentation_stats['bias_prevented'] += 1

            # Convert back to dictionary format
            char_dict = {
                'name': char.name,
                'role': char.role
            }
            if char.culture:
                char_dict['culture'] = char.culture
            if char.gender:
                char_dict['gender'] = char.gender
            if char.original_name:
                char_dict['original_name'] = char.original_name

            augmented_characters.append(char_dict)

        # Update story text with new names
        augmented_text = story_data.get('text', '')
        name_mapping = {}

        # Build mapping of original names to new names
        for original_char, new_char in zip(story_data['characters'], augmented_characters):
            if 'original_name' in new_char:
                original_name = new_char['original_name']
                new_name = new_char['name']
                name_mapping[original_name] = new_name

        # Apply text replacements using lemmatization-safe strategy
        if name_mapping:
            augmented_text = self._replace_names_in_text(augmented_text, name_mapping)

        # Update story data
        augmented_story = story_data.copy()
        augmented_story['text'] = augmented_text
        augmented_story['characters'] = augmented_characters

        # Add augmentation metadata
        augmented_story['augmentation_metadata'] = {
            'characters_replaced': sum(1 for char in augmented_characters if 'original_name' in char),
            'roles_shuffled': shuffle_prob < random.random() and len(characters) > 1,
            'augmentation_version': '1.0'
        }

        self.augmentation_stats['stories_processed'] += 1

        return augmented_story

    def _generate_replacement_name(
        self,
        char: CharacterData,
        preserve_cultural_consistency: bool
    ) -> Dict[str, str]:
        """
        Generate a replacement name for a character.

        Args:
            char: Character data
            preserve_cultural_consistency: Try to maintain cultural background

        Returns:
            Generated name data
        """

        # Determine cultural preference
        preferred_culture = None
        if preserve_cultural_consistency and char.culture:
            # Try to maintain the same culture
            if random.random() < 0.7:  # 70% chance to preserve culture
                preferred_culture = char.culture

        # Generate new name
        return self.name_inventory.generate_name(
            gender=char.gender,
            culture=preferred_culture,
            avoid_overused=True
        )

    def _generate_unique_replacement_name(
        self,
        char: CharacterData,
        preserve_cultural_consistency: bool,
        used_names_in_story: set
    ) -> Optional[Dict]:
        """
        Generate a replacement name that's unique within the current story.

        Args:
            char: Character to replace
            preserve_cultural_consistency: Try to maintain cultural backgrounds
            used_names_in_story: Set of names already used in this story

        Returns:
            Name data dict or None if no unique name found after max attempts
        """
        max_attempts = 50  # Prevent infinite loops

        for attempt in range(max_attempts):
            new_name_data = self._generate_replacement_name(char, preserve_cultural_consistency)

            if new_name_data['full_name'] not in used_names_in_story:
                return new_name_data

            # If we can't find a unique name, try different culture/gender
            if attempt > 20:
                preserve_cultural_consistency = False

        logger.warning(f"Could not generate unique name for {char.name} after {max_attempts} attempts")
        return None

    def _check_role_bias(self, role_distribution: Dict[str, int], current_role: str) -> bool:
        """
        Check if a name has problematic role bias.

        Args:
            role_distribution: Dictionary of role -> count
            current_role: The role being assigned

        Returns:
            True if the name has concerning bias for this role
        """

        if not role_distribution or sum(role_distribution.values()) < 5:
            return False  # Not enough data to determine bias

        total_uses = sum(role_distribution.values())
        current_role_uses = role_distribution.get(current_role, 0)
        bias_percentage = current_role_uses / total_uses

        return bias_percentage > self.max_role_bias

    def augment_dataset(
        self,
        input_path: str,
        output_path: str,
        replacement_probability: float = 0.8,
        role_shuffle_probability: Optional[float] = None,
        batch_size: int = 1000
    ) -> Dict:
        """
        Augment an entire JSONL dataset with diverse names.

        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file
            replacement_probability: Chance to replace each character name
            role_shuffle_probability: Chance to shuffle roles per story
            batch_size: Process in batches for memory efficiency

        Returns:
            Summary statistics of the augmentation process
        """

        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info(f"Starting dataset augmentation: {input_path} -> {output_path}")

        # Reset statistics
        self.augmentation_stats = {
            'stories_processed': 0,
            'characters_replaced': 0,
            'roles_shuffled': 0,
            'bias_prevented': 0,
            'overuse_prevented': 0
        }

        # Process dataset
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            batch = []
            for line_num, line in enumerate(infile, 1):
                try:
                    story = json.loads(line.strip())
                    batch.append(story)

                    # Process batch when full
                    if len(batch) >= batch_size:
                        self._process_batch(batch, outfile, replacement_probability, role_shuffle_probability)
                        batch = []

                    # Progress updates
                    if line_num % 2000 == 0:
                        logger.info(f"Processed {line_num:,} stories...")

                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error at line {line_num}: {e}")
                    continue

            # Process remaining batch
            if batch:
                self._process_batch(batch, outfile, replacement_probability, role_shuffle_probability)

        # Generate summary report
        summary = self._generate_augmentation_summary()
        summary['input_file'] = str(input_path)
        summary['output_file'] = str(output_path)

        logger.info(f"Dataset augmentation complete: {summary['augmentation_stats']['stories_processed']:,} stories processed")
        return summary

    def _process_batch(
        self,
        batch: List[Dict],
        outfile,
        replacement_probability: float,
        role_shuffle_probability: Optional[float]
    ):
        """Process a batch of stories."""

        for story in batch:
            augmented_story = self.augment_story_data(
                story,
                replacement_probability=replacement_probability,
                role_shuffle_probability=role_shuffle_probability
            )
            outfile.write(json.dumps(augmented_story, ensure_ascii=False) + '\n')

    def _generate_augmentation_summary(self) -> Dict:
        """Generate comprehensive summary of augmentation process."""

        # Get bias report from name inventory
        bias_report = self.name_inventory.get_bias_report()

        # Calculate improvement metrics
        replacement_rate = (
            self.augmentation_stats['characters_replaced'] /
            max(self.augmentation_stats['stories_processed'], 1)
        )

        return {
            'augmentation_stats': dict(self.augmentation_stats),
            'replacement_rate_per_story': replacement_rate,
            'bias_report': bias_report,
            'settings': {
                'max_name_uses': self.max_name_uses,
                'max_role_bias': self.max_role_bias,
                'role_shuffle_probability': self.role_shuffle_probability
            },
            'quality_metrics': {
                'names_with_usage_over_limit': len([
                    name for name, count in bias_report['overused_names'].items()
                    if count > self.max_name_uses
                ]),
                'names_with_role_bias': len(bias_report['severely_biased_names']),
                'cultural_diversity_score': self._calculate_diversity_score(bias_report['culture_distribution'])
            }
        }

    def _calculate_diversity_score(self, culture_distribution: Dict[str, float]) -> float:
        """
        Calculate a diversity score based on cultural distribution.

        Args:
            culture_distribution: Percentage distribution across cultures

        Returns:
            Diversity score (0.0 = no diversity, 1.0 = perfect diversity)
        """

        if not culture_distribution:
            return 0.0

        # Calculate entropy-based diversity score
        num_cultures = len(culture_distribution)
        ideal_percentage = 100.0 / num_cultures

        # Calculate how far each culture deviates from ideal
        deviations = [
            abs(percentage - ideal_percentage) / 100.0
            for percentage in culture_distribution.values()
        ]

        # Average deviation (0 = perfect, 1 = worst possible)
        avg_deviation = sum(deviations) / len(deviations)

        # Convert to score (1 = perfect diversity, 0 = no diversity)
        diversity_score = 1.0 - avg_deviation

        return max(0.0, min(1.0, diversity_score))

    def _replace_names_in_text(self, text: str, name_mapping: Dict[str, str]) -> str:
        """
        Replace names in text using lemmatization-safe strategy.

        Handles edge cases:
        - Possessive forms ('s)
        - First name only occurrences
        - Title consistency (Miss/Mr./Dr. etc.)
        - Partial name matches (avoid replacing "Johnson" when replacing "John")
        - Case sensitivity
        - Quoted names

        Args:
            text: Original story text
            name_mapping: Dict mapping original names to new names

        Returns:
            Text with names replaced safely
        """
        import re

        modified_text = text

        for original_name, new_name in name_mapping.items():
            # Parse original and new names
            original_parts = self._parse_name_components(original_name)
            new_parts = self._parse_name_components(new_name)

            # Replace full names first (most specific)
            modified_text = self._replace_full_name(modified_text, original_name, new_name)

            # Replace title + last name combinations (e.g., "Dr. Smith" → "Ms. Garcia")
            if original_parts['title'] and original_parts['last_name']:
                original_title_lastname = f"{original_parts['title']} {original_parts['last_name']}"
                new_title_lastname = f"{new_parts['title']} {new_parts['last_name']}"
                modified_text = self._replace_full_name(modified_text, original_title_lastname, new_title_lastname)

            # Replace last name only (with word boundaries)
            if original_parts['last_name']:
                modified_text = self._replace_name_component(
                    modified_text,
                    original_parts['last_name'],
                    new_parts['last_name']
                )

            # Replace first name only (with word boundaries)
            if original_parts['first_name']:
                modified_text = self._replace_name_component(
                    modified_text,
                    original_parts['first_name'],
                    new_parts['first_name']
                )

        return modified_text

    def _parse_name_components(self, full_name: str) -> Dict[str, str]:
        """Parse name into components (title, first, last)."""
        import re

        # Common titles
        title_pattern = r'^(Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.|Sir|Miss|Chancellor|Chief|Sergeant|Captain|Colonel)\s*'

        # Common articles and words that should NOT be treated as names
        INVALID_NAME_PARTS = {
            'the', 'a', 'an', 'and', 'or', 'of', 'in', 'at', 'to', 'for', 'with', 'by'
        }

        title = ""
        remaining = full_name

        # Extract title if present
        title_match = re.match(title_pattern, full_name, re.IGNORECASE)
        if title_match:
            title = title_match.group(1)
            remaining = full_name[title_match.end():].strip()

        # Split remaining into parts and filter out invalid name components
        name_parts = [part for part in remaining.split()
                     if part.lower() not in INVALID_NAME_PARTS and len(part) > 1]

        if len(name_parts) >= 2:
            first_name = name_parts[0]
            last_name = " ".join(name_parts[1:])  # Handle multi-part last names
        elif len(name_parts) == 1:
            first_name = name_parts[0]
            last_name = ""
        else:
            first_name = ""
            last_name = ""

        return {
            'title': title,
            'first_name': first_name,
            'last_name': last_name
        }

    def _replace_full_name(self, text: str, original: str, replacement: str) -> str:
        """Replace full name occurrences including possessive forms."""
        import re

        # Escape special regex characters in names
        escaped_original = re.escape(original)

        # Pattern for exact matches with word boundaries, including possessive
        pattern = rf'\b{escaped_original}(\'s|\u2019s)?\b'

        def replace_func(match):
            possessive = match.group(1) if match.group(1) else ""
            return replacement + possessive

        return re.sub(pattern, replace_func, text, flags=re.IGNORECASE)

    def _replace_name_component(self, text: str, original: str, replacement: str) -> str:
        """Replace individual name component with word boundaries to avoid partial matches."""
        import re

        # Escape special regex characters
        escaped_original = re.escape(original)

        # Pattern with word boundaries and possessive handling
        pattern = rf'\b{escaped_original}(\'s|\u2019s)?\b'

        def replace_func(match):
            possessive = match.group(1) if match.group(1) else ""
            # Preserve case pattern of original
            original_text = match.group(0).replace(match.group(1) or "", "")
            if original_text.isupper():
                return replacement.upper() + possessive
            elif original_text.islower():
                return replacement.lower() + possessive
            elif original_text[0].isupper():
                return replacement.capitalize() + possessive
            else:
                return replacement + possessive

        return re.sub(pattern, replace_func, text, flags=re.IGNORECASE)

    def validate_augmented_data(
        self,
        filepath: str,
        sample_size: int = 1000
    ) -> Dict:
        """
        Validate the quality of augmented data.

        Args:
            filepath: Path to augmented JSONL file
            sample_size: Number of stories to sample for validation

        Returns:
            Validation report with quality metrics
        """

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info(f"Validating augmented data: {filepath}")

        # Sample stories for validation
        stories = []
        with open(filepath, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            sample_lines = random.sample(all_lines, min(sample_size, len(all_lines)))

            for line in sample_lines:
                try:
                    story = json.loads(line.strip())
                    stories.append(story)
                except json.JSONDecodeError:
                    continue

        # Analyze sampled data
        validation_results = self._analyze_sample(stories)

        logger.info(f"Validation complete: {len(stories)} stories analyzed")
        return validation_results

    def _analyze_sample(self, stories: List[Dict]) -> Dict:
        """Analyze a sample of stories for quality metrics."""

        character_names = []
        role_assignments = defaultdict(list)
        cultural_distribution = Counter()
        original_names = Counter()

        for story in stories:
            characters = story.get('characters', [])

            for char in characters:
                name = char.get('name', 'Unknown')
                role = char.get('role', 'UNKNOWN')
                culture = char.get('culture')
                original_name = char.get('original_name')

                character_names.append(name)
                role_assignments[role].append(name)

                if culture:
                    cultural_distribution[culture] += 1
                if original_name:
                    original_names[original_name] += 1

        # Calculate metrics
        total_characters = len(character_names)
        unique_names = len(set(character_names))
        name_usage = Counter(character_names)

        # Role bias analysis
        role_bias_issues = 0
        for role, names in role_assignments.items():
            name_counts = Counter(names)
            for name, count in name_counts.items():
                if count > 0.8 * len(names) and len(names) >= 5:
                    role_bias_issues += 1

        # Overuse analysis
        overused_names = sum(1 for count in name_usage.values() if count > self.max_name_uses)

        return {
            'total_characters_analyzed': total_characters,
            'unique_names': unique_names,
            'name_reuse_rate': total_characters / unique_names if unique_names > 0 else 0,
            'overused_names_count': overused_names,
            'role_bias_issues': role_bias_issues,
            'cultural_distribution': dict(cultural_distribution),
            'replacement_rate': len(original_names) / total_characters if total_characters > 0 else 0,
            'most_used_names': dict(name_usage.most_common(10)),
            'diversity_score': self._calculate_diversity_score({
                culture: (count / sum(cultural_distribution.values())) * 100
                for culture, count in cultural_distribution.items()
            }) if cultural_distribution else 0.0
        }

    def export_augmentation_report(
        self,
        filepath: str,
        include_detailed_stats: bool = True
    ):
        """
        Export comprehensive augmentation report.

        Args:
            filepath: Output file path
            include_detailed_stats: Include detailed statistics
        """

        report = {
            'augmentation_summary': self._generate_augmentation_summary(),
            'name_inventory_stats': self.name_inventory.get_inventory_stats(),
            'bias_prevention_settings': {
                'max_name_uses': self.max_name_uses,
                'max_role_bias': self.max_role_bias,
                'role_shuffle_probability': self.role_shuffle_probability
            }
        }

        if include_detailed_stats:
            report['detailed_bias_report'] = self.name_inventory.get_bias_report()

        filepath = Path(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Augmentation report exported to {filepath}")

    def reset_tracking(self):
        """Reset all tracking statistics."""
        self.augmentation_stats = {
            'stories_processed': 0,
            'characters_replaced': 0,
            'roles_shuffled': 0,
            'bias_prevented': 0,
            'overuse_prevented': 0
        }
        self.name_inventory.reset_usage_tracking()
        logger.info("Tracking statistics reset")


def main():
    """Demonstration of augmentation utilities."""

    print("🔧 Initializing Augmentation Utilities...")

    # Create name inventory and augmentation utils
    inventory = NameInventory(seed=42)
    augmenter = AugmentUtils(name_inventory=inventory, seed=42)

    # Create sample story data (simulating the problematic patterns)
    sample_stories = [
        {
            'story_id': 'test_001',
            'text': 'Aria Nguyen was the protagonist...',
            'characters': [
                {'name': 'Aria Nguyen', 'role': 'PROTAGONIST'},
                {'name': 'Marcus Thorne', 'role': 'ANTAGONIST'},
                {'name': 'Wounded Hawk', 'role': 'SUPPORTING'}
            ],
            'metadata': {'genre': 'thriller'}
        },
        {
            'story_id': 'test_002',
            'text': 'Marcus Thorne emerged from the shadows...',
            'characters': [
                {'name': 'Marcus Thorne', 'role': 'ANTAGONIST'},
                {'name': 'Aria Nguyen', 'role': 'PROTAGONIST'},
                {'name': 'Wounded Hawk', 'role': 'SUPPORTING'}
            ],
            'metadata': {'genre': 'mystery'}
        }
    ]

    print(f"\n📚 SAMPLE STORY DATA (Before Augmentation):")
    for story in sample_stories:
        print(f"   Story: {story['story_id']}")
        for char in story['characters']:
            print(f"     {char['name']:<15} -> {char['role']}")

    # Simulate overuse by pre-tracking
    print(f"\n⚠️  Simulating overuse patterns...")
    for _ in range(50):  # Simulate "Wounded Hawk" overuse
        inventory.track_role_assignment("Wounded Hawk", "SUPPORTING")
    for _ in range(45):  # Simulate "Aria Nguyen" bias
        inventory.track_role_assignment("Aria Nguyen", "PROTAGONIST")
    for _ in range(5):
        inventory.track_role_assignment("Aria Nguyen", "SUPPORTING")

    # Augment sample stories
    print(f"\n🔄 AUGMENTING SAMPLE STORIES:")
    augmented_stories = []
    for story in sample_stories:
        augmented = augmenter.augment_story_data(
            story,
            replacement_probability=0.9,  # High probability for demonstration
            role_shuffle_probability=0.5
        )
        augmented_stories.append(augmented)

    print(f"\n📚 SAMPLE STORY DATA (After Augmentation):")
    for story in augmented_stories:
        print(f"   Story: {story['story_id']}")
        metadata = story.get('augmentation_metadata', {})
        print(f"     Replaced: {metadata.get('characters_replaced', 0)} characters")
        for char in story['characters']:
            original = f" (was: {char['original_name']})" if 'original_name' in char else ""
            culture = f" [{char.get('culture', 'unknown')}]" if 'culture' in char else ""
            print(f"     {char['name']:<20} -> {char['role']}{culture}{original}")

    # Generate statistics
    print(f"\n📊 AUGMENTATION STATISTICS:")
    summary = augmenter._generate_augmentation_summary()
    stats = summary['augmentation_stats']
    print(f"   Stories processed: {stats['stories_processed']}")
    print(f"   Characters replaced: {stats['characters_replaced']}")
    print(f"   Roles shuffled: {stats['roles_shuffled']}")
    print(f"   Bias prevented: {stats['bias_prevented']}")
    print(f"   Overuse prevented: {stats['overuse_prevented']}")

    # Show bias report
    bias_report = inventory.get_bias_report()
    print(f"\n🎭 BIAS ANALYSIS:")
    print(f"   Names generated: {bias_report['total_names_generated']}")
    print(f"   Unique names: {bias_report['unique_names']}")
    print(f"   Avg uses per name: {bias_report['average_uses_per_name']:.1f}")
    print(f"   Overused names: {len(bias_report['overused_names'])}")
    print(f"   Biased names: {len(bias_report['severely_biased_names'])}")

    # Cultural diversity
    print(f"\n🌍 CULTURAL DIVERSITY:")
    for culture, percentage in bias_report['culture_distribution'].items():
        print(f"   {culture:<20} {percentage:>5.1f}%")

    print(f"\n✅ Augmentation utilities demonstration complete!")
    print(f"   System ready for large-scale dataset augmentation")



if __name__ == "__main__":
    main()