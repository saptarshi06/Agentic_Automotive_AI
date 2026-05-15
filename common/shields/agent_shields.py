"""Centralised shield definitions for the application."""

from llama_stack_api import RegisterShieldRequest

# Shield for business analysis output
BUSINESS_ANALYSIS_SHIELD = RegisterShieldRequest(
    shield_id="business_analysis_shield",
    provider_id="inline::llama-shields",
    provider_shield_id="business_analysis_validator",
    params={
        "validation_rules": [
            "check_completeness",      # ensure all required sections present
            "check_actionable_items",  # must contain concrete recommendations
            "check_no_hallucination_red_flags"  # optional custom rule
        ]
    }
)

# Optional – additional shields for other domains can be added here