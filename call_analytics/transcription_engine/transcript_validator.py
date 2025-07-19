"""
Enhanced transcript validation for healthcare calls with LLM-based quality assurance.
Validates medical terminology, conversation coherence, and compliance requirements.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import re
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from shared.config import config
from shared.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    type: str  # 'error', 'warning', 'suggestion'
    category: str  # 'medical_terminology', 'coherence', 'compliance', 'quality'
    description: str
    severity: int  # 1-5, 5 being most severe
    location: Optional[Dict[str, Any]] = None  # segment index, timestamp, etc.
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Complete validation result."""
    is_valid: bool
    confidence_score: float
    overall_quality: str  # 'excellent', 'good', 'fair', 'poor'
    issues: List[ValidationIssue]
    medical_terms_validated: int
    compliance_score: float
    processing_time: float
    corrections: Dict[str, str]  # original -> corrected


class ITranscriptValidator(ABC):
    """Abstract interface for transcript validators."""
    
    @abstractmethod
    async def validate(
        self,
        transcript: str,
        segments: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate transcript and return detailed results."""
        pass


class LLMTranscriptValidator(ITranscriptValidator):
    """LLM-based transcript validator using OpenAI GPT models."""
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.1):
        """
        Initialize LLM validator.
        
        Args:
            model: OpenAI model to use
            temperature: Temperature for LLM responses
        """
        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key required for LLM validation")
        
        self.llm = ChatOpenAI(
            api_key=config.OPENAI_API_KEY,
            model=model,
            temperature=temperature,
            max_tokens=2000
        )
        
        # Healthcare-specific term lists
        self.medical_terms = self._load_medical_terms()
        self.compliance_keywords = self._load_compliance_keywords()
        
        logger.info(f"LLMTranscriptValidator initialized with model: {model}")
    
    def _load_medical_terms(self) -> Set[str]:
        """Load medical terminology for validation."""
        # In production, this would load from a comprehensive medical dictionary
        return {
            "prior authorization", "preauthorization", "pre-auth", "auth",
            "diagnosis", "icd", "cpt", "hcpcs", "ndc", "procedure",
            "formulary", "tier", "copay", "deductible", "coinsurance",
            "provider", "physician", "doctor", "nurse", "clinic",
            "insurance", "payer", "plan", "coverage", "benefit",
            "medical necessity", "criteria", "guideline", "protocol",
            "appeal", "denial", "approval", "pending", "review"
        }
    
    def _load_compliance_keywords(self) -> Set[str]:
        """Load compliance-related keywords."""
        return {
            "hipaa", "privacy", "confidential", "protected health information",
            "phi", "consent", "authorization", "disclosure", "breach",
            "minimum necessary", "covered entity", "business associate",
            "patient rights", "medical record", "health information"
        }
    
    async def validate(
        self,
        transcript: str,
        segments: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Comprehensive transcript validation.
        
        Args:
            transcript: Full transcript text
            segments: List of transcript segments with metadata
            context: Additional context (call type, participants, etc.)
            
        Returns:
            Detailed validation results
        """
        import time
        start_time = time.time()
        
        logger.info("Starting comprehensive transcript validation")
        
        try:
            # Run multiple validation tasks in parallel
            validation_tasks = [
                self._validate_medical_terminology(transcript, segments),
                self._validate_conversation_coherence(transcript, segments),
                self._validate_compliance_requirements(transcript, segments),
                self._validate_transcription_quality(transcript, segments)
            ]
            
            results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Combine results
            all_issues = []
            medical_terms_count = 0
            compliance_score = 1.0
            corrections = {}
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Validation task failed: {result}")
                    all_issues.append(ValidationIssue(
                        type="error",
                        category="system",
                        description=f"Validation error: {str(result)}",
                        severity=3
                    ))
                elif isinstance(result, dict):
                    all_issues.extend(result.get("issues", []))
                    medical_terms_count += result.get("medical_terms_count", 0)
                    compliance_score = min(compliance_score, result.get("compliance_score", 1.0))
                    corrections.update(result.get("corrections", {}))
            
            # Calculate overall scores
            confidence_score = self._calculate_confidence_score(all_issues)
            overall_quality = self._assess_overall_quality(all_issues, confidence_score)
            is_valid = self._determine_validity(all_issues)
            
            processing_time = time.time() - start_time
            
            result = ValidationResult(
                is_valid=is_valid,
                confidence_score=confidence_score,
                overall_quality=overall_quality,
                issues=all_issues,
                medical_terms_validated=medical_terms_count,
                compliance_score=compliance_score,
                processing_time=processing_time,
                corrections=corrections
            )
            
            logger.info(f"Validation completed in {processing_time:.2f}s - Quality: {overall_quality}")
            return result
            
        except Exception as e:
            logger.error(f"Transcript validation failed: {e}")
            raise
    
    async def _validate_medical_terminology(
        self,
        transcript: str,
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate medical terminology accuracy."""
        logger.debug("Validating medical terminology")
        
        issues = []
        medical_terms_found = 0
        corrections = {}
        
        # Create focused prompt for medical terminology
        system_prompt = """You are a medical terminology expert reviewing a healthcare prior authorization call transcript.
        
Your task:
1. Identify medical terms, codes, and healthcare-specific language
2. Check for obvious errors in medical terminology
3. Suggest corrections for misheard medical terms
4. Validate procedure codes (CPT, ICD, HCPCS) format
5. Check medication names and dosages

Focus on clear errors, not stylistic preferences. Be conservative with corrections."""
        
        user_prompt = f"""Review this healthcare call transcript for medical terminology accuracy:

{transcript[:2000]}...

Respond with JSON only:
{{
  "medical_terms_count": <number>,
  "errors": [
    {{
      "term": "<incorrect term>",
      "correction": "<suggested correction>",
      "confidence": <0.0-1.0>,
      "category": "<medication|procedure|diagnosis|code|other>"
    }}
  ],
  "warnings": [
    {{
      "term": "<questionable term>",
      "reason": "<explanation>",
      "category": "<category>"
    }}
  ]
}}"""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.agenerate([messages])
            result_text = response.generations[0][0].text
            
            # Parse LLM response
            result_data = self._parse_json_response(result_text)
            
            medical_terms_found = result_data.get("medical_terms_count", 0)
            
            # Convert errors to issues
            for error in result_data.get("errors", []):
                issues.append(ValidationIssue(
                    type="error",
                    category="medical_terminology",
                    description=f"Medical term error: '{error['term']}'",
                    severity=4,
                    suggestion=error.get("correction")
                ))
                if error.get("correction"):
                    corrections[error["term"]] = error["correction"]
            
            # Convert warnings to issues
            for warning in result_data.get("warnings", []):
                issues.append(ValidationIssue(
                    type="warning",
                    category="medical_terminology",
                    description=f"Questionable term: '{warning['term']}' - {warning['reason']}",
                    severity=2
                ))
            
        except Exception as e:
            logger.error(f"Medical terminology validation failed: {e}")
            issues.append(ValidationIssue(
                type="error",
                category="medical_terminology",
                description="Failed to validate medical terminology",
                severity=3
            ))
        
        return {
            "issues": issues,
            "medical_terms_count": medical_terms_found,
            "corrections": corrections
        }
    
    async def _validate_conversation_coherence(
        self,
        transcript: str,
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate conversation flow and coherence."""
        logger.debug("Validating conversation coherence")
        
        issues = []
        
        system_prompt = """You are analyzing a healthcare prior authorization call for conversation coherence.
        
Evaluate:
1. Logical flow of conversation
2. Speaker transitions make sense
3. Topics are relevant to prior authorization
4. No obvious gaps or jumps in conversation
5. Professional tone maintained

Look for clear issues, not minor imperfections."""
        
        # Sample segments for coherence check
        sample_segments = segments[:10] if len(segments) > 10 else segments
        segments_text = "\n".join([
            f"[{seg.get('speaker', 'Unknown')}] {seg.get('text', '')}"
            for seg in sample_segments
        ])
        
        user_prompt = f"""Analyze conversation coherence:

{segments_text}

Respond with JSON only:
{{
  "coherence_score": <0.0-1.0>,
  "issues": [
    {{
      "type": "<gap|jump|irrelevant|tone|other>",
      "description": "<issue description>",
      "severity": <1-5>,
      "location": "<segment reference if applicable>"
    }}
  ],
  "flow_quality": "<excellent|good|fair|poor>"
}}"""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.agenerate([messages])
            result_text = response.generations[0][0].text
            
            result_data = self._parse_json_response(result_text)
            
            # Convert to validation issues
            for issue in result_data.get("issues", []):
                issues.append(ValidationIssue(
                    type="warning" if issue.get("severity", 3) <= 3 else "error",
                    category="coherence",
                    description=issue["description"],
                    severity=issue.get("severity", 3),
                    location={"reference": issue.get("location")}
                ))
            
        except Exception as e:
            logger.error(f"Coherence validation failed: {e}")
            issues.append(ValidationIssue(
                type="error",
                category="coherence",
                description="Failed to validate conversation coherence",
                severity=3
            ))
        
        return {"issues": issues}
    
    async def _validate_compliance_requirements(
        self,
        transcript: str,
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate HIPAA and healthcare compliance requirements."""
        logger.debug("Validating compliance requirements")
        
        issues = []
        compliance_score = 1.0
        
        # Check for potential HIPAA violations
        phi_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email pattern
        ]
        
        for pattern in phi_patterns:
            matches = re.findall(pattern, transcript)
            if matches:
                issues.append(ValidationIssue(
                    type="error",
                    category="compliance",
                    description=f"Potential PHI detected: {len(matches)} instances",
                    severity=5,
                    suggestion="Review for HIPAA compliance"
                ))
                compliance_score *= 0.5
        
        # Check for required compliance language
        required_elements = {
            "authorization": ["authorization", "consent", "permission"],
            "privacy": ["privacy", "confidential", "protected"],
            "purpose": ["prior authorization", "preauthorization", "medical necessity"]
        }
        
        transcript_lower = transcript.lower()
        for element, keywords in required_elements.items():
            if not any(keyword in transcript_lower for keyword in keywords):
                issues.append(ValidationIssue(
                    type="warning",
                    category="compliance",
                    description=f"Missing {element} language",
                    severity=3,
                    suggestion=f"Consider including {element} terminology"
                ))
                compliance_score *= 0.9
        
        return {
            "issues": issues,
            "compliance_score": compliance_score
        }
    
    async def _validate_transcription_quality(
        self,
        transcript: str,
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate overall transcription quality."""
        logger.debug("Validating transcription quality")
        
        issues = []
        
        # Check for common transcription errors
        error_patterns = {
            "repeated_words": r'\b(\w+)\s+\1\b',
            "excessive_filler": r'\b(um|uh|er|ah)\b',
            "incomplete_sentences": r'[.!?]\s*[a-z]',
            "missing_punctuation": r'[a-zA-Z]\s+[A-Z][a-z]'  # Simplified check
        }
        
        for error_type, pattern in error_patterns.items():
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            if len(matches) > 5:  # Threshold for concern
                severity = 2 if error_type == "excessive_filler" else 3
                issues.append(ValidationIssue(
                    type="warning",
                    category="quality",
                    description=f"High frequency of {error_type.replace('_', ' ')}: {len(matches)} instances",
                    severity=severity
                ))
        
        # Check segment consistency
        if segments:
            avg_confidence = sum(seg.get('confidence', 1.0) for seg in segments) / len(segments)
            if avg_confidence < 0.7:
                issues.append(ValidationIssue(
                    type="warning",
                    category="quality",
                    description=f"Low average confidence score: {avg_confidence:.2f}",
                    severity=3,
                    suggestion="Consider re-processing with higher quality audio"
                ))
        
        return {"issues": issues}
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response with error handling."""
        try:
            # Clean response text
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            return json.loads(response_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Response text: {response_text}")
            return {}
    
    def _calculate_confidence_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate overall confidence score based on issues."""
        if not issues:
            return 1.0
        
        error_count = sum(1 for issue in issues if issue.type == "error")
        warning_count = sum(1 for issue in issues if issue.type == "warning")
        
        # Weight errors more heavily than warnings
        penalty = (error_count * 0.2) + (warning_count * 0.05)
        confidence = max(0.0, 1.0 - penalty)
        
        return confidence
    
    def _assess_overall_quality(self, issues: List[ValidationIssue], confidence: float) -> str:
        """Assess overall transcript quality."""
        high_severity_issues = sum(1 for issue in issues if issue.severity >= 4)
        
        if confidence >= 0.9 and high_severity_issues == 0:
            return "excellent"
        elif confidence >= 0.7 and high_severity_issues <= 1:
            return "good"
        elif confidence >= 0.5:
            return "fair"
        else:
            return "poor"
    
    def _determine_validity(self, issues: List[ValidationIssue]) -> bool:
        """Determine if transcript is valid for use."""
        critical_issues = sum(1 for issue in issues if issue.severity >= 4)
        compliance_issues = sum(1 for issue in issues if issue.category == "compliance" and issue.type == "error")
        
        # Invalid if critical issues or compliance violations
        return critical_issues == 0 and compliance_issues == 0


class TranscriptValidator:
    """Main transcript validator with multiple validation strategies."""
    
    def __init__(self, validator_type: str = "llm"):
        """
        Initialize transcript validator.
        
        Args:
            validator_type: Type of validator to use ("llm")
        """
        self.validator = self._initialize_validator(validator_type)
        logger.info(f"TranscriptValidator initialized with {type(self.validator).__name__}")
    
    def _initialize_validator(self, validator_type: str) -> ITranscriptValidator:
        """Initialize specific validator implementation."""
        if validator_type == "llm":
            return LLMTranscriptValidator()
        else:
            raise ValueError(f"Unsupported validator type: {validator_type}")
    
    async def validate_healthcare_transcript(
        self,
        transcript: str,
        segments: List[Dict[str, Any]],
        call_context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate healthcare transcript with context-aware analysis.
        
        Args:
            transcript: Full transcript text
            segments: List of transcript segments
            call_context: Context about the call (type, participants, etc.)
            
        Returns:
            Comprehensive validation results
        """
        logger.info("Starting healthcare transcript validation")
        
        try:
            # Add healthcare-specific context processing
            enhanced_context = self._enhance_context_for_healthcare(call_context)
            
            # Perform validation
            result = await self.validator.validate(transcript, segments, enhanced_context)
            
            # Post-process for healthcare-specific requirements
            enhanced_result = await self._post_process_healthcare_validation(result, call_context)
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Healthcare transcript validation failed: {e}")
            raise
    
    def _enhance_context_for_healthcare(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhance context with healthcare-specific information."""
        enhanced_context = context.copy() if context else {}
        
        # Add healthcare validation settings
        enhanced_context.update({
            "validation_mode": "healthcare",
            "require_medical_terminology": True,
            "require_compliance_language": True,
            "prior_auth_context": True,
            "timestamp": datetime.now().isoformat()
        })
        
        return enhanced_context
    
    async def _post_process_healthcare_validation(
        self,
        result: ValidationResult,
        context: Optional[Dict[str, Any]]
    ) -> ValidationResult:
        """Post-process validation results for healthcare requirements."""
        
        # Add healthcare-specific quality checks
        additional_issues = []
        
        # Check for required prior authorization elements
        if context and context.get("call_type") == "prior_authorization":
            required_elements = self._check_prior_auth_requirements(result)
            additional_issues.extend(required_elements)
        
        # Adjust compliance score based on healthcare standards
        adjusted_compliance = self._adjust_compliance_for_healthcare(result.compliance_score, result.issues)
        
        # Create enhanced result
        enhanced_result = ValidationResult(
            is_valid=result.is_valid and len([i for i in additional_issues if i.type == "error"]) == 0,
            confidence_score=result.confidence_score,
            overall_quality=result.overall_quality,
            issues=result.issues + additional_issues,
            medical_terms_validated=result.medical_terms_validated,
            compliance_score=adjusted_compliance,
            processing_time=result.processing_time,
            corrections=result.corrections
        )
        
        return enhanced_result
    
    def _check_prior_auth_requirements(self, result: ValidationResult) -> List[ValidationIssue]:
        """Check for required prior authorization elements."""
        issues = []
        
        # This would check for specific prior auth requirements
        # Implementation depends on healthcare organization standards
        
        return issues
    
    def _adjust_compliance_for_healthcare(self, base_score: float, issues: List[ValidationIssue]) -> float:
        """Adjust compliance score for healthcare-specific requirements."""
        adjusted_score = base_score
        
        # Penalize heavily for any compliance issues in healthcare context
        compliance_errors = [i for i in issues if i.category == "compliance" and i.type == "error"]
        if compliance_errors:
            adjusted_score *= 0.3  # Severe penalty for compliance issues
        
        return max(0.0, adjusted_score)
    
    async def generate_validation_report(
        self,
        result: ValidationResult,
        include_suggestions: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            result: Validation results
            include_suggestions: Include improvement suggestions
            
        Returns:
            Formatted validation report
        """
        logger.info("Generating validation report")
        
        # Group issues by category
        issues_by_category = {}
        for issue in result.issues:
            if issue.category not in issues_by_category:
                issues_by_category[issue.category] = []
            issues_by_category[issue.category].append(issue)
        
        # Generate summary statistics
        summary = {
            "total_issues": len(result.issues),
            "errors": len([i for i in result.issues if i.type == "error"]),
            "warnings": len([i for i in result.issues if i.type == "warning"]),
            "suggestions": len([i for i in result.issues if i.type == "suggestion"]),
            "avg_severity": sum(i.severity for i in result.issues) / len(result.issues) if result.issues else 0
        }
        
        # Create detailed report
        report = {
            "validation_summary": {
                "is_valid": result.is_valid,
                "overall_quality": result.overall_quality,
                "confidence_score": result.confidence_score,
                "compliance_score": result.compliance_score,
                "medical_terms_validated": result.medical_terms_validated,
                "processing_time": result.processing_time
            },
            "issue_summary": summary,
            "issues_by_category": {},
            "recommendations": []
        }
        
        # Format issues by category
        for category, issues in issues_by_category.items():
            report["issues_by_category"][category] = [
                {
                    "type": issue.type,
                    "description": issue.description,
                    "severity": issue.severity,
                    "suggestion": issue.suggestion,
                    "location": issue.location
                }
                for issue in issues
            ]
        
        # Generate recommendations
        if include_suggestions:
            report["recommendations"] = await self._generate_improvement_recommendations(result)
        
        # Add corrections if available
        if result.corrections:
            report["suggested_corrections"] = result.corrections
        
        logger.info("Validation report generated")
        return report
    
    async def _generate_improvement_recommendations(self, result: ValidationResult) -> List[str]:
        """Generate recommendations for improving transcript quality."""
        recommendations = []
        
        # Analyze patterns in issues
        error_categories = set(i.category for i in result.issues if i.type == "error")
        warning_categories = set(i.category for i in result.issues if i.type == "warning")
        
        # Quality-based recommendations
        if result.overall_quality == "poor":
            recommendations.append("Consider re-processing audio with noise reduction and higher quality settings")
        
        if "medical_terminology" in error_categories:
            recommendations.append("Review medical terminology accuracy - consider using medical dictionary validation")
        
        if "compliance" in error_categories:
            recommendations.append("Address HIPAA compliance issues before using transcript")
        
        if "coherence" in warning_categories:
            recommendations.append("Review conversation flow - may indicate audio quality or speaker separation issues")
        
        if result.confidence_score < 0.7:
            recommendations.append("Low confidence score - consider manual review before final use")
        
        if result.compliance_score < 0.8:
            recommendations.append("Compliance score below threshold - healthcare review required")
        
        # Add general recommendations
        if len(result.issues) > 10:
            recommendations.append("High number of issues detected - consider improving audio quality or transcription settings")
        
        return recommendations
    
    def export_validation_results(self, result: ValidationResult, format: str = "json") -> str:
        """
        Export validation results in specified format.
        
        Args:
            result: Validation results
            format: Export format ("json", "csv", "txt")
            
        Returns:
            Formatted export string
        """
        if format == "json":
            return self._export_json(result)
        elif format == "csv":
            return self._export_csv(result)
        elif format == "txt":
            return self._export_text(result)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, result: ValidationResult) -> str:
        """Export results as JSON."""
        export_data = {
            "validation_result": {
                "is_valid": result.is_valid,
                "confidence_score": result.confidence_score,
                "overall_quality": result.overall_quality,
                "medical_terms_validated": result.medical_terms_validated,
                "compliance_score": result.compliance_score,
                "processing_time": result.processing_time
            },
            "issues": [
                {
                    "type": issue.type,
                    "category": issue.category,
                    "description": issue.description,
                    "severity": issue.severity,
                    "suggestion": issue.suggestion,
                    "location": issue.location
                }
                for issue in result.issues
            ],
            "corrections": result.corrections,
            "timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(export_data, indent=2)
    
    def _export_csv(self, result: ValidationResult) -> str:
        """Export issues as CSV."""
        lines = ["Type,Category,Description,Severity,Suggestion"]
        
        for issue in result.issues:
            line = f'"{issue.type}","{issue.category}","{issue.description}",{issue.severity},"{issue.suggestion or ""}"'
            lines.append(line)
        
        return "\n".join(lines)
    
    def _export_text(self, result: ValidationResult) -> str:
        """Export results as human-readable text."""
        lines = [
            "TRANSCRIPT VALIDATION REPORT",
            "=" * 40,
            f"Overall Quality: {result.overall_quality.upper()}",
            f"Valid: {'YES' if result.is_valid else 'NO'}",
            f"Confidence Score: {result.confidence_score:.2f}",
            f"Compliance Score: {result.compliance_score:.2f}",
            f"Medical Terms Validated: {result.medical_terms_validated}",
            f"Processing Time: {result.processing_time:.2f}s",
            "",
            f"ISSUES FOUND ({len(result.issues)} total):",
            "-" * 40
        ]
        
        # Group issues by type
        errors = [i for i in result.issues if i.type == "error"]
        warnings = [i for i in result.issues if i.type == "warning"]
        suggestions = [i for i in result.issues if i.type == "suggestion"]
        
        for issue_type, issues in [("ERRORS", errors), ("WARNINGS", warnings), ("SUGGESTIONS", suggestions)]:
            if issues:
                lines.append(f"\n{issue_type}:")
                for i, issue in enumerate(issues, 1):
                    lines.append(f"  {i}. [{issue.category}] {issue.description}")
                    if issue.suggestion:
                        lines.append(f"     Suggestion: {issue.suggestion}")
        
        if result.corrections:
            lines.extend([
                "",
                "SUGGESTED CORRECTIONS:",
                "-" * 20
            ])
            for original, corrected in result.corrections.items():
                lines.append(f"  '{original}' -> '{corrected}'")
        
        return "\n".join(lines)


# Synchronous wrapper for backward compatibility
def validate_transcript_sync(
    transcript: str,
    segments: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """Synchronous wrapper for transcript validation."""
    validator = TranscriptValidator()
    return asyncio.run(validator.validate_healthcare_transcript(transcript, segments, context))