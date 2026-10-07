from langchain_core.tools import tool
import json
import sys
import os

# Add mcp_servers to path so we can import brand_mcp directly
mcp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../mcp_servers"))
if mcp_path not in sys.path:
    sys.path.append(mcp_path)

from brand_mcp import brand_report as _brand_report, full_analysis as _full_analysis, route_file_to_tool as _route_file_to_tool

@tool
def brand_report(brand: str, url: str = "", country: str = "Egypt", country_code: str = "EG") -> str:
    """
    Collects a comprehensive brand report including website metadata, social media links, posts, and Google Maps reviews.
    
    Args:
        brand: The exact name of the brand.
        url: The URL of the brand's website.
        country: The country of the brand (default: Egypt).
        country_code: The 2-letter country code (default: EG).
    """
    try:
        res = _brand_report(brand=brand, website=url, country=country, country_code=country_code)
        return json.dumps(res, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error running brand_report: {str(e)}"

@tool
def full_analysis(brand: str) -> str:
    """
    Runs full analysis (engagement stats, sentiment analysis, product insights) on the collected brand data.
    You MUST call brand_report for this brand first before calling this tool.
    
    Args:
        brand: The name of the brand to analyze.
    """
    try:
        res = _full_analysis(brand=brand)
        return json.dumps(res, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error running full_analysis: {str(e)}"
        
@tool
def route_file_to_tool(file_path: str, brand: str = "") -> str:
    """
    Read a local file precisely, analyze its structure and content, and match it with the most suitable MCP tool to process it.
    
    Args:
        file_path: Path to the local file (e.g., brand_data/report.json).
        brand: Optional brand name.
    """
    try:
        res = _route_file_to_tool(file_path=file_path, brand=brand)
        return json.dumps(res, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error running route_file_to_tool: {str(e)}"
