"""Main automation pipeline modules."""

from .automation import (
    FacelessAutomation,
    ContentPipeline,
    AutomationScheduler,
    PipelineResult,
    create_automation
)

__all__ = [
    'FacelessAutomation',
    'ContentPipeline',
    'AutomationScheduler',
    'PipelineResult',
    'create_automation'
]
