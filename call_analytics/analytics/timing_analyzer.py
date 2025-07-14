import logging
from typing import Dict, List, Any, Optional
import statistics
import json

from call_analytics.config.settings import settings

logger = logging.getLogger(__name__)

class TimingAnalyzer:
    """
    Analyzes timing metrics: pacing, durations, response times.
    Calculates metrics like average response time, pauses, total duration.
    Uses transcript segments with timestamps.
    Identifies bottlenecks or effective timing patterns.
    """
    
    def __init__(self):
        """
        Initialize the timing analyzer.
        """
        logger.info("TimingAnalyzer initialized")
    
    def analyze_timing(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze timing from transcript segments.
        
        :param segments: List of segments with 'start', 'end', 'speaker'.
        :return: Dict of timing metrics.
        """
        if not segments:
            raise ValueError("No segments provided")
        
        segments.sort(key=lambda s: s['start'])
        
        total_duration = segments[-1]['end'] - segments[0]['start']
        speaker_durations = {}
        response_times = []
        pauses = []
        
        prev_end = segments[0]['start']
        prev_speaker = None
        
        for seg in segments:
            # Pause before this segment
            pause = seg['start'] - prev_end
            if pause > 0:
                pauses.append(pause)
            
            # Segment duration
            dur = seg['end'] - seg['start']
            speaker = seg.get('speaker', 'UNKNOWN')
            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + dur
            
            # Response time if speaker change
            if prev_speaker and speaker != prev_speaker:
                response_time = seg['start'] - prev_end
                response_times.append(response_time)
            
            prev_end = seg['end']
            prev_speaker = speaker
        
        metrics = {
            "total_duration": total_duration,
            "speaker_durations": speaker_durations,
            "average_response_time": statistics.mean(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "average_pause": statistics.mean(pauses) if pauses else 0,
            "pause_count": len(pauses),
            "response_time_count": len(response_times)
        }
        
        logger.info("Timing analysis completed")
        return metrics
    
    def identify_patterns(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Identify timing patterns or issues.
        
        :param metrics: Timing metrics dict.
        :return: List of pattern descriptions.
        """
        patterns = []
        if metrics["average_response_time"] > 5:  # Arbitrary threshold
            patterns.append("High average response time; potential hesitation.")
        if metrics["pause_count"] > 10:
            patterns.append("Frequent pauses; may indicate uncertainty.")
        if "REP" in metrics["speaker_durations"] and "INS" in metrics["speaker_durations"]:
            rep_ratio = metrics["speaker_durations"]["REP"] / metrics["total_duration"]
            if rep_ratio > 0.6:
                patterns.append("Rep speaking majority of time; good control.")
        
        logger.info(f"Identified {len(patterns)} timing patterns")
        return patterns

# Example usage (for testing)
if __name__ == "__main__":
    analyzer = TimingAnalyzer()
    try:
        mock_segments = [
            {"start": 0, "end": 5, "speaker": "REP"},
            {"start": 5.5, "end": 10, "speaker": "INS"},
            {"start": 11, "end": 15, "speaker": "REP"}
        ]
        metrics = analyzer.analyze_timing(mock_segments)
        print(json.dumps(metrics, indent=2))
        
        patterns = analyzer.identify_patterns(metrics)
        print(patterns)
    except Exception as e:
        print(f"Error: {e}")