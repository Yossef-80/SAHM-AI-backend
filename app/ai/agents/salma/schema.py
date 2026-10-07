from pydantic import BaseModel, Field

class OptimizationPlan(BaseModel):
    """Call this tool to submit your final salma output."""
    summary: str = Field(description="Summary of your findings/strategy")
    details: list[str] = Field(description="Detailed bullet points")
