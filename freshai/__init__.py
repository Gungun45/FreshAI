from .detector import FreshAIDetector, FreshAIDetectionResult
from .freshness_detector import (
    ConvNeXtFreshnessDetector,
    FreshnessResult,
    FRESHNESS_STAGES,
    RIPENESS_STAGES,
    FRESHNESS_COLORS,
    FRESHNESS_SHELF_LIFE,
    FRESHNESS_EMOJI,
    estimate_freshness_heuristic,
)
from .defect_detector import (
    FreshAIDefectDetector,
    DefectResult,
    DefectItem,
    DEFECT_CLASSES,
    DEFECT_COLORS,
    DEFECT_SEVERITY_WEIGHTS,
)
from .feature_extractor import (
    ProduceFeatureExtractor,
    ProduceExtractedFeatures,
    ColorFeatures,
    TextureFeatures,
)
from .fusion import (
    ProduceFeatureVector,
    assemble_feature_vector,
)
from .fusion_predictor import (
    FreshAIFusionPredictor,
    MultimodalFreshnessAssessment,
)
from .structured_features import (
    EnvironmentalData,
    TimelineStorageData,
    StructuredProduceInput,
    STRUCTURED_FEATURE_NAMES,
    SUPPORTED_PRODUCE_CLASSES,
    STORAGE_CONDITIONS,
)
from .structured_predictor import (
    FreshAIStructuredPredictor,
    StructuredPredictionResult,
)
from .train_structured_layer import (
    train_structured_models,
)
from .rag_engine import (
    FreshAIRAGEngine,
    ProduceRAGRecommendations,
    PRODUCE_KNOWLEDGE_BASE,
)

__all__ = [
    'FreshAIDetector',
    'FreshAIDetectionResult',
    'ConvNeXtFreshnessDetector',
    'FreshnessResult',
    'FRESHNESS_STAGES',
    'RIPENESS_STAGES',
    'FRESHNESS_COLORS',
    'FRESHNESS_SHELF_LIFE',
    'FRESHNESS_EMOJI',
    'estimate_freshness_heuristic',
    'FreshAIDefectDetector',
    'DefectResult',
    'DefectItem',
    'DEFECT_CLASSES',
    'DEFECT_COLORS',
    'DEFECT_SEVERITY_WEIGHTS',
    'ProduceFeatureExtractor',
    'ProduceExtractedFeatures',
    'ColorFeatures',
    'TextureFeatures',
    'ProduceFeatureVector',
    'assemble_feature_vector',
    'FreshAIFusionPredictor',
    'MultimodalFreshnessAssessment',
    'EnvironmentalData',
    'TimelineStorageData',
    'StructuredProduceInput',
    'STRUCTURED_FEATURE_NAMES',
    'SUPPORTED_PRODUCE_CLASSES',
    'STORAGE_CONDITIONS',
    'FreshAIStructuredPredictor',
    'StructuredPredictionResult',
    'train_structured_models',
    'FreshAIRAGEngine',
    'ProduceRAGRecommendations',
    'PRODUCE_KNOWLEDGE_BASE',
]
