#!/usr/bin/env python3
"""
Optimization Pipeline
=====================
Modular pipeline architecture for cassette placement optimization.
Production-ready, clean, and simple.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Cassette:
    """Simple cassette representation."""
    x: float
    y: float
    width: float
    height: float

    @property
    def size(self) -> str:
        return f"{int(self.width)}x{int(self.height)}"

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def weight(self) -> float:
        return self.area * 10.4  # 10.4 lbs per sq ft

    def get_corners(self) -> List[Tuple[float, float]]:
        """Get four corner coordinates."""
        return [
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height),
            (self.x, self.y + self.height)
        ]

    def overlaps(self, other: 'Cassette') -> bool:
        """Check if this cassette overlaps with another."""
        return not (self.x >= other.x + other.width or
                   self.x + self.width <= other.x or
                   self.y >= other.y + other.height or
                   self.y + self.height <= other.y)


@dataclass
class PipelineContext:
    """Context passed through pipeline stages."""
    polygon: List[Tuple[float, float]]
    cassettes: List[Cassette]
    metadata: Dict[str, Any]

    def __init__(self, polygon: List[Tuple[float, float]]):
        self.polygon = polygon
        self.cassettes = []
        self.metadata = {
            'original_polygon': polygon.copy(),
            'stage_results': {},
            'total_area': self._calculate_area(polygon)
        }

    def _calculate_area(self, polygon: List[Tuple[float, float]]) -> float:
        """Calculate polygon area using shoelace formula."""
        n = len(polygon)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        return abs(area) / 2.0

    def get_coverage(self) -> float:
        """Calculate current coverage percentage."""
        if not self.cassettes:
            return 0.0
        covered = sum(c.area for c in self.cassettes)
        return (covered / self.metadata['total_area']) * 100


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""

    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """Process the context and return updated context."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Stage name for logging."""
        pass


class OptimizationPipeline:
    """
    Main optimization pipeline.
    Processes polygon through a series of stages to optimize cassette placement.
    """

    def __init__(self):
        self.stages: List[PipelineStage] = []

    def add_stage(self, stage: PipelineStage):
        """Add a processing stage to the pipeline."""
        self.stages.append(stage)
        logger.info(f"Added stage: {stage.name}")

    def optimize(self, polygon: List[Tuple[float, float]]) -> Dict:
        """
        Run optimization pipeline on polygon.

        Args:
            polygon: List of (x, y) vertices in feet

        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting optimization pipeline")
        logger.info(f"Input polygon: {len(polygon)} vertices")

        # Create context
        context = PipelineContext(polygon)

        # Process through stages
        for stage in self.stages:
            logger.info(f"Processing stage: {stage.name}")
            initial_coverage = context.get_coverage()

            try:
                context = stage.process(context)
                final_coverage = context.get_coverage()

                # Record stage results
                context.metadata['stage_results'][stage.name] = {
                    'cassettes_added': len(context.cassettes),
                    'coverage_before': initial_coverage,
                    'coverage_after': final_coverage,
                    'coverage_gain': final_coverage - initial_coverage
                }

                logger.info(f"  Stage complete: {len(context.cassettes)} cassettes, "
                          f"{final_coverage:.1f}% coverage (+{final_coverage-initial_coverage:.1f}%)")

            except Exception as e:
                logger.error(f"Stage {stage.name} failed: {e}")
                # Continue with next stage even if one fails
                continue

        # Calculate final statistics
        return self._calculate_results(context)

    def _calculate_results(self, context: PipelineContext) -> Dict:
        """Calculate final optimization results."""
        total_area = context.metadata['total_area']
        covered_area = sum(c.area for c in context.cassettes)
        gap_area = total_area - covered_area
        coverage_percent = (covered_area / total_area * 100) if total_area > 0 else 0

        # Size distribution
        size_dist = {}
        for c in context.cassettes:
            size_dist[c.size] = size_dist.get(c.size, 0) + 1

        # Convert cassettes to dictionaries for compatibility
        cassette_dicts = []
        for c in context.cassettes:
            cassette_dicts.append({
                'x': c.x,
                'y': c.y,
                'width': c.width,
                'height': c.height,
                'size': c.size,
                'area': c.area,
                'weight': c.weight
            })

        return {
            'success': True,
            'cassettes': cassette_dicts,
            'num_cassettes': len(context.cassettes),
            'total_area': total_area,
            'covered_area': covered_area,
            'gap_area': gap_area,
            'coverage': coverage_percent / 100,
            'coverage_percent': coverage_percent,
            'gap_percent': (gap_area / total_area * 100) if total_area > 0 else 0,
            'size_distribution': size_dist,
            'total_weight': sum(c.weight for c in context.cassettes),
            'avg_cassette_area': covered_area / len(context.cassettes) if context.cassettes else 0,
            'meets_requirement': coverage_percent >= 94.0,
            'stage_results': context.metadata.get('stage_results', {})
        }


class ValidationStage(PipelineStage):
    """Validates and cleans the input polygon."""

    @property
    def name(self) -> str:
        return "ValidationStage"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Validate and clean polygon."""
        polygon = context.polygon

        # Remove duplicate consecutive vertices
        cleaned = []
        for i, vertex in enumerate(polygon):
            if i == 0 or vertex != polygon[i-1]:
                cleaned.append(vertex)

        # Remove last vertex if it duplicates first
        if len(cleaned) > 1 and cleaned[-1] == cleaned[0]:
            cleaned = cleaned[:-1]

        if len(cleaned) < 3:
            logger.warning("Polygon has less than 3 unique vertices")
            return context

        context.polygon = cleaned
        logger.info(f"  Cleaned polygon: {len(polygon)} -> {len(cleaned)} vertices")

        return context


def create_standard_pipeline() -> OptimizationPipeline:
    """Create the standard optimization pipeline with all stages."""
    from edge_cleaner_stage import EdgeCleaner
    from corner_placer_stage import CornerPlacer
    from perimeter_tracer_stage import PerimeterTracer

    pipeline = OptimizationPipeline()

    # Add validation first
    pipeline.add_stage(ValidationStage())

    # Clean edges
    pipeline.add_stage(EdgeCleaner())

    # Place corner cassettes
    pipeline.add_stage(CornerPlacer())

    # Trace perimeter
    pipeline.add_stage(PerimeterTracer())

    # Fill interior with concentric layers
    from concentric_filler_stage import ConcentricFiller
    pipeline.add_stage(ConcentricFiller())

    # Other stages will be added as they are implemented
    # pipeline.add_stage(GapFiller())
    # pipeline.add_stage(MathematicalVerifier())

    return pipeline


def test_pipeline():
    """Test the pipeline architecture."""

    # Test polygon (simple rectangle)
    test_polygon = [
        (0, 0),
        (20, 0),
        (20, 15),
        (0, 15)
    ]

    # Create pipeline
    pipeline = create_standard_pipeline()

    # Run optimization
    results = pipeline.optimize(test_polygon)

    print("\n" + "="*60)
    print("PIPELINE TEST RESULTS")
    print("="*60)
    print(f"Success: {results['success']}")
    print(f"Total area: {results['total_area']:.1f} sq ft")
    print(f"Cassettes placed: {results['num_cassettes']}")
    print(f"Coverage: {results['coverage_percent']:.1f}%")
    print(f"Meets requirement: {results['meets_requirement']}")

    if results['stage_results']:
        print("\nStage Results:")
        for stage, data in results['stage_results'].items():
            print(f"  {stage}: +{data['coverage_gain']:.1f}% coverage")

    return results


if __name__ == "__main__":
    test_pipeline()