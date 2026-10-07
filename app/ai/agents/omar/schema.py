from pydantic import BaseModel, Field

class StrategyReport(BaseModel):
    """Call this tool to submit your final omar output."""
    summary: str = Field(description="Summary of your findings/strategy")
    details: list[str] = Field(description="Detailed bullet points")
