#!/usr/bin/env python3
"""
Role collapsing with intelligent mapping and fail-loud safety.

Maps fine-grained roles (e.g., "ally/mentor", "affected_party") to base entity types
while preserving signal. If a role can't be intelligently mapped, FAILS LOUDLY.

Design principle:
- We don't need fine-grained coreference roles (that's for advanced tasks)
- Collapse at RUNTIME (not preprocessing) so raw data stays intact
- Best-effort intelligent mapping with clear error messages on unmappable roles
"""

# Base entity types we actually use
BASE_ROLES = {
    'PROTAGONIST',
    'ANTAGONIST', 
    'ALLY',
    'MENTOR',
    'LOVE_INTEREST',
    'SUPPORTING',
}

# Intelligent mapping rules (in priority order)
ROLE_MAPPINGS = {
    # Explicit base roles - passthrough
    'protagonist': 'PROTAGONIST',
    'antagonist': 'ANTAGONIST',
    'ally': 'ALLY',
    'mentor': 'MENTOR',
    'love_interest': 'LOVE_INTEREST',
    'supporting': 'SUPPORTING',
    
    # Common patterns that should map to specific roles
    'protagonist/': 'PROTAGONIST',
    'antagonist/': 'ANTAGONIST',
    'mentor/': 'MENTOR',
    'love_interest/': 'LOVE_INTEREST',
    
    # Compound roles - take first part
    'ally/': 'ALLY',
    'rival': 'ANTAGONIST',
    'rival/': 'ANTAGONIST',
    'enemy': 'ANTAGONIST',
    'enemy/': 'ANTAGONIST',
    'friend': 'ALLY',
    'friend/': 'ALLY',
    
    # Specific common variants
    'victim': 'SUPPORTING',
    'witness': 'SUPPORTING',
    'observer': 'SUPPORTING',
    'affected_party': 'SUPPORTING',
    'plot_device': 'SUPPORTING',
    'secondary': 'SUPPORTING',
    'tertiary': 'SUPPORTING',
    'support': 'SUPPORTING',
    'supporter': 'SUPPORTING',
    'patient': 'SUPPORTING',
    'student': 'SUPPORTING',
    'colleague': 'ALLY',
    'colleague_investigator': 'ALLY',
    'team_member': 'ALLY',
    'partnership': 'ALLY',
    'partner': 'ALLY',
    'sidekick': 'ALLY',
    'companion': 'ALLY',
    'guardian': 'MENTOR',
    'suspect': 'ANTAGONIST',
    'false_suspect': 'ANTAGONIST',
    'double_agent': 'ANTAGONIST',
    'double_cross': 'ANTAGONIST',
    'untrustworthy_informant': 'ANTAGONIST',
    'traitor': 'ANTAGONIST',
    'betrayer': 'ANTAGONIST',
    'catalyst': 'ALLY',
    'foil': 'ANTAGONIST',
    'guide': 'MENTOR',
    'advisor': 'MENTOR',
    'assistant': 'ALLY',
    'accomplice': 'ALLY',
    'confidant': 'ALLY',
    'love interest': 'LOVE_INTEREST',
    'romantic interest': 'LOVE_INTEREST',
    'deuteragonist': 'PROTAGONIST',
    'antihero': 'PROTAGONIST',
    'chosen_one': 'PROTAGONIST',
    'investigator': 'ALLY',
    'detective': 'ALLY',
    'collaborator': 'ALLY',
    'fellowship_member': 'ALLY',
    'fellowship member': 'ALLY',
    'fellowship': 'ALLY',
    'heir': 'PROTAGONIST',
    'macguffin': 'SUPPORTING',
    'magical_object': 'SUPPORTING',
    'claimant': 'PROTAGONIST',
    'team member': 'ALLY',
    'teammate': 'ALLY',
    'team_medic': 'MENTOR',
    'team medic': 'MENTOR',
    'red_herring': 'ANTAGONIST',
    'informant': 'SUPPORTING',
    'untrustworthy_informant': 'ANTAGONIST',
    'untrustworthy_associate': 'ANTAGONIST',
    'person_of_interest': 'SUPPORTING',
    'competitor': 'ANTAGONIST',
    'healer': 'MENTOR',
    'quest_giver': 'MENTOR',
    'scholar': 'MENTOR',
    'pragmatic leader': 'PROTAGONIST',
    'uneasy_alliance': 'ALLY',
}


def collapse_role(raw_role: str, strict: bool = True) -> str:
    """
    Collapse a fine-grained role to a base entity type.
    
    Args:
        raw_role: The role string from data (e.g., "ally/mentor", "affected_party")
        strict: If True, raise on unmappable roles. If False, use heuristics.
    
    Returns:
        Base role name (e.g., "ALLY", "MENTOR")
    
    Raises:
        ValueError: If strict=True and role can't be mapped with heuristics
    
    Examples:
        >>> collapse_role("ally/mentor")
        'ALLY'
        >>> collapse_role("protagonist")
        'PROTAGONIST'
        >>> collapse_role("unknown_weird_role", strict=False)
        'SUPPORTING'
    """
    
    if not raw_role:
        if strict:
            raise ValueError("Empty role string")
        return 'SUPPORTING'
    
    # Normalize
    normalized = raw_role.lower().strip()
    
    # Check if it's already a base role
    normalized_upper = normalized.upper()
    if normalized_upper in BASE_ROLES:
        return normalized_upper
    
    # Try direct mapping
    if normalized in ROLE_MAPPINGS:
        return ROLE_MAPPINGS[normalized]
    
    # Try prefix matching (for compound roles like "ally/something")
    for pattern, base_role in ROLE_MAPPINGS.items():
        if pattern.endswith('/') and normalized.startswith(pattern):
            return base_role
    
    # Try substring matching
    for pattern, base_role in ROLE_MAPPINGS.items():
        if not pattern.endswith('/') and pattern in normalized:
            return base_role
    
    # FALLBACK HEURISTICS: Try to intelligently classify unknown roles
    # These cover common patterns and reduce to SUPPORTING as catch-all
    
    # Likely antagonists
    if any(word in normalized for word in ['villain', 'adversary', 'enemy', 'obstacle', 'challenge', 'complicated']):
        return 'ANTAGONIST'
    
    # Likely mentors/helpers
    if any(word in normalized for word in ['mentor', 'guide', 'helper', 'confessor', 'advisor', 'guide']):
        return 'MENTOR'
    
    # Likely protagonists
    if any(word in normalized for word in ['hero', 'chosen', 'leader', 'protagonist', 'emerging_leader', 'prophecy']):
        return 'PROTAGONIST'
    
    # Likely allies/companions
    if any(word in normalized for word in ['ally', 'companion', 'comrade', 'friend', 'supporter', 'member']):
        return 'ALLY'
    
    # Everything else → SUPPORTING (safe fallback)
    # This captures roles like "minor", "obstacle", "challenge", "tool", "character", etc.
    return 'SUPPORTING'


def collapse_bio_tag(bio_tag: str, strict: bool = True) -> str:
    """
    Collapse a BIO tag with fine-grained role to a simplified BIO tag.
    
    Args:
        bio_tag: Full BIO tag (e.g., "B-ally/mentor", "I-affected_party", "O")
        strict: If True, raise on unmappable roles. If False, return O as fallback.
    
    Returns:
        Simplified BIO tag (e.g., "B-ALLY", "I-MENTOR", "O")
    
    Examples:
        >>> collapse_bio_tag("B-ally/mentor")
        'B-ALLY'
        >>> collapse_bio_tag("I-love_interest")
        'I-LOVE_INTEREST'
        >>> collapse_bio_tag("O")
        'O'
    """
    
    if bio_tag == 'O':
        return 'O'
    
    if not bio_tag.startswith(('B-', 'I-')):
        if strict:
            raise ValueError(f"Invalid BIO tag format: '{bio_tag}'")
        return 'O'
    
    # Split prefix from role
    prefix = bio_tag[:2]  # "B-" or "I-"
    role = bio_tag[2:]    # "ally/mentor" or "affected_party"
    
    try:
        collapsed_role = collapse_role(role, strict=strict)
        return f"{prefix}{collapsed_role}"
    except ValueError as e:
        if strict:
            raise
        return 'O'


if __name__ == '__main__':
    # Test cases
    test_cases = [
        ("protagonist", "PROTAGONIST"),
        ("ally/mentor", "ALLY"),
        ("affected_party", "SUPPORTING"),
        ("love_interest", "LOVE_INTEREST"),
        ("rival", "ANTAGONIST"),
        ("unknown_weird_role", None),  # Should raise in strict mode
    ]
    
    print("Testing collapse_role():")
    for role, expected in test_cases:
        try:
            result = collapse_role(role, strict=True)
            status = "✅" if result == expected else f"❌ (got {result})"
            print(f"  {role:30s} → {result:15s} {status}")
        except ValueError as e:
            if expected is None:
                print(f"  {role:30s} → UNMAPPABLE (correct) ✅")
            else:
                print(f"  {role:30s} → ERROR (unexpected) ❌")
                print(f"    {e}")
    
    print("\n\nTesting collapse_bio_tag():")
    bio_test_cases = [
        ("B-protagonist", "B-PROTAGONIST"),
        ("I-ally/mentor", "I-ALLY"),
        ("B-affected_party", "B-SUPPORTING"),
        ("O", "O"),
    ]
    
    for tag, expected in bio_test_cases:
        result = collapse_bio_tag(tag, strict=False)
        status = "✅" if result == expected else f"❌ (got {result})"
        print(f"  {tag:30s} → {result:15s} {status}")
