from pydantic import BaseModel, Field

class CreativeBrief(BaseModel):
    """Call this tool to submit your final layla output."""
    summary: str = Field(description="Summary of your findings/strategy")
    details: list[str] = Field(description="Detailed bullet points")
