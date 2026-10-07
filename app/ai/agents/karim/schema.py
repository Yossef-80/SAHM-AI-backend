from pydantic import BaseModel, Field

class PerformanceReport(BaseModel):
    """Call this tool to submit your final karim output."""
    summary: str = Field(description="Summary of your findings/strategy")
    details: list[str] = Field(description="Detailed bullet points")
