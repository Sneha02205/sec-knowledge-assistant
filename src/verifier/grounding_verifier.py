import re
import time
from typing import Dict, List, Any, Tuple


class GroundingVerifier:
    """Phase 5: Post-Generation Citation Grounding & Hallucination Verifier."""

    def __init__(self, confidence_threshold: float = 0.70):
        self.confidence_threshold = confidence_threshold

    def extract_numerical_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract individual sentences containing financial numbers or metric claims."""
        # Strip citation brackets before extracting claim numbers
        clean_text_no_citations = re.sub(r'\[.*?\]', '', text)
        sentences = re.split(r'(?<=[.!?])\s+', clean_text_no_citations)
        claims = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            # Look for financial numbers, percentages, or figures with 2 or more digits
            numbers = re.findall(r'\b\$?\d+(?:,\d{3})*(?:\.\d+)?%?\b', s_clean)
            # Filter out single digits (like 1, 7) that are section/item labels
            financial_numbers = [n for n in numbers if len(re.sub(r'\D', '', n)) >= 2 or '$' in n or '%' in n]
            if financial_numbers:
                claims.append({
                    'sentence': s_clean,
                    'numbers': financial_numbers
                })
        return claims

    def verify_numerical_entailment(self, claim_numbers: List[str], context_text: str) -> Tuple[bool, float]:
        """Check if numbers in a generated claim exist in the retrieved context text."""
        if not claim_numbers:
            return True, 1.0

        matches = 0
        context_clean = context_text.replace(',', '').replace('$', '')

        for num in claim_numbers:
            num_clean = num.replace(',', '').replace('$', '').replace('%', '')
            # Check for exact or normalized number match
            if num_clean in context_clean:
                matches += 1
            elif len(num_clean) > 4 and num_clean[:4] in context_clean:
                matches += 0.8

        support_ratio = matches / len(claim_numbers)
        is_supported = support_ratio >= 0.75
        return is_supported, support_ratio

    def verify_answer(self, rag_output: Dict[str, Any]) -> Dict[str, Any]:
        """Verify grounding of answer against cited retrieved chunks."""
        start_time = time.time()
        
        answer = rag_output.get('answer', '')
        chunks = rag_output.get('retrieved_chunks', [])
        is_refused = rag_output.get('is_refused', False)

        if is_refused:
            return {
                'is_verified': True,
                'confidence_score': 1.0,
                'status': 'PASSED_REFUSAL',
                'supported_claims': [],
                'unsupported_claims': [],
                'verification_latency_ms': (time.time() - start_time) * 1000,
                'verification_note': 'Answer is an explicit out-of-scope refusal. No grounding required.'
            }

        combined_context = "\n".join([c['text'] for c in chunks])
        claims = self.extract_numerical_claims(answer)

        if not claims:
            return {
                'is_verified': True,
                'confidence_score': 0.95,
                'status': 'PASSED_PROSE',
                'supported_claims': [answer],
                'unsupported_claims': [],
                'verification_latency_ms': (time.time() - start_time) * 1000,
                'verification_note': 'Qualitative prose answer. No specific numerical claims detected.'
            }

        supported_claims = []
        unsupported_claims = []
        total_score = 0.0

        for claim in claims:
            is_supported, score = self.verify_numerical_entailment(claim['numbers'], combined_context)
            total_score += score
            if is_supported:
                supported_claims.append(claim['sentence'])
            else:
                unsupported_claims.append(claim['sentence'])

        confidence_score = total_score / len(claims) if claims else 1.0
        is_verified = confidence_score >= self.confidence_threshold and len(unsupported_claims) == 0

        status = "PASSED_VERIFIED" if is_verified else "FLAGGED_LOW_CONFIDENCE"
        note = (
            f"Verified {len(supported_claims)}/{len(claims)} numerical claims directly against cited context. "
            f"Confidence Score: {confidence_score*100:.1f}%."
        )

        return {
            'is_verified': is_verified,
            'confidence_score': confidence_score,
            'status': status,
            'supported_claims': supported_claims,
            'unsupported_claims': unsupported_claims,
            'verification_latency_ms': (time.time() - start_time) * 1000,
            'verification_note': note
        }


if __name__ == '__main__':
    verifier = GroundingVerifier()

    sample_output = {
        'answer': "In fiscal year 2023, Apple reported total net sales of $383,285 million compared to $394,328 million in 2022 [AAPL 10-K 2023 Item 7].",
        'retrieved_chunks': [
            {'text': "Total net sales for fiscal year 2023 were $383,285 million, down from $394,328 million in fiscal year 2022."}
        ],
        'is_refused': False
    }

    result = verifier.verify_answer(sample_output)

    print("=" * 60)
    print("PHASE 5: CITATION GROUNDING & VERIFICATION TEST")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Confidence Score: {result['confidence_score']*100:.1f}%")
    print(f"Verification Note: {result['verification_note']}")
    print(f"Supported Claims ({len(result['supported_claims'])}): {result['supported_claims']}")
    print(f"Unsupported Claims ({len(result['unsupported_claims'])}): {result['unsupported_claims']}")
    print("=" * 60 + "\n")
