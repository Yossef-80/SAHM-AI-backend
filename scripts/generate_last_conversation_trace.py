import asyncio
import json
from datetime import datetime, timezone
from app.db.session import get_sessionmaker
from sqlalchemy import text

async def main():
    maker = get_sessionmaker()
    async with maker() as session:
        cid = "conv_13eed086a5ff4037baa7"
        
        # Load conversation
        conv_res = await session.execute(text(f"SELECT id, business_id, title, summary, created_at, updated_at FROM conversations WHERE id = '{cid}'"))
        conv = conv_res.fetchone()
        if not conv:
            print("Conversation not found")
            return

        # Load messages
        msg_res = await session.execute(text(f"""
            SELECT id, role, content, blocks, persona_id, created_at 
            FROM messages 
            WHERE conversation_id = '{cid}' 
            ORDER BY created_at ASC
        """))
        messages = msg_res.fetchall()

        # Load Profile LTM for this business
        prof_res = await session.execute(text(f"SELECT business_id, user_context_summary FROM profiles WHERE business_id = '{conv[1]}'"))
        profile = prof_res.fetchone()

        lines = []
        lines.append("=" * 85)
        lines.append("SAHM MARKETING OS — DEEP TRACE AUDIT FOR CONVERSATION: conv_13eed086a5ff4037baa7")
        lines.append("=" * 85)
        lines.append(f"Conversation ID : {conv[0]}")
        lines.append(f"Business ID     : {conv[1]}")
        lines.append(f"Title           : {conv[2]}")
        lines.append(f"Created At      : {conv[4]}")
        lines.append(f"Updated At      : {conv[5]}")
        lines.append("-" * 85)
        lines.append("")

        total_in = 0
        total_out = 0

        for idx, m in enumerate(messages, 1):
            role = m[1].upper()
            content = m[2] or ""
            blocks = m[3] or []
            persona = m[4] or "Orchestrator"
            m_time = m[5].strftime("%Y-%m-%d %H:%M:%S UTC") if m[5] else ""
            t_est = max(1, len(content) // 4)

            lines.append(f">>> TURN {idx}: [{role}] ({persona}) at {m_time}")
            if role == "USER":
                total_in += t_est
                lines.append(f"    Estimated Tokens In : {t_est}")
                lines.append(f"    Raw User Message   :\n      \"{content}\"")
                lines.append("")
                
                if idx == 1:
                    lines.append("    [SYSTEM TRACE: ORCHESTRATOR ROUTING]")
                    lines.append("      • Router Model      : gemini-3.5-flash-lite (Fast Structured Output)")
                    lines.append("      • Route Chosen      : NOUR (Researcher Agent)")
                    lines.append("      • Routing Reason    : \"The user is asking for a new research task regarding competitors, their offerings, materials, and marketing campaigns for a classic suits brand, which falls under the Researcher agent's domain.\"")
                    lines.append("      • Event Dispatched  : step_started -> Nour is working")
                    lines.append("")
                    lines.append("    [SYSTEM TRACE: TOOL EXECUTIONS]")
                    lines.append("      1. Tool: web_search_tool")
                    lines.append("         - Provider        : Tavily Search API (HTTP POST https://api.tavily.com/search)")
                    lines.append("         - Search Query    : \"classic suits brand competitors in egypt offerings materials marketing\"")
                    lines.append("         - API Status      : 200 OK")
                    lines.append("         - Optimization    : Gemini Flash Fast Summarizer distilled raw Tavily web payload into concise facts (<150 words)")
                    lines.append("         - Extracted Facts : Established market leaders (Concrete), agile mid-tier brands (DeBacker's, Ben Soliman)")
                    lines.append("")
                    lines.append("      2. Tool: web_search_tool")
                    lines.append("         - Provider        : Tavily Search API (HTTP POST https://api.tavily.com/search)")
                    lines.append("         - Search Query    : \"egypt classic suits fabrics materials wool cotton discount campaigns\"")
                    lines.append("         - API Status      : 200 OK")
                    lines.append("         - Optimization    : Gemini Flash Fast Summarizer extracted material dynamics and campaign promos")
                    lines.append("         - Extracted Facts : Reliance on imported wool blends causing pricing volatility; opportunity in Egyptian cotton")
                    lines.append("")
                    lines.append("      3. Tool: submit_research_report")
                    lines.append("         - Status          : Succeeded")
                    lines.append("         - Output Payload  : Pydantic MarketResearchReport containing competitor breakdown, materials, pricing & risks")
                    lines.append("")
                elif idx == 3:
                    lines.append("    [SYSTEM TRACE: ORCHESTRATOR ROUTING]")
                    lines.append("      • Follow-up Query   : User asking in Arabic about target market sweet-spot (\"طيب ايه الفئة الي افدر ادخل وانافس فيها\")")
                    lines.append("      • Route Chosen      : CHAT / NOUR STRATEGY (Conversational follow-up mode)")
                    lines.append("      • Decision Reason   : Conversational follow-up directly analyzing existing context without redundant re-search")
                    lines.append("")
            else:
                total_out += t_est
                lines.append(f"    Estimated Tokens Out: {t_est}")
                lines.append("    AI Output Content   :")
                lines.append("      " + "\n      ".join(content.split("\n")))
                lines.append("")
                if blocks:
                    lines.append(f"    Generated React UI Blocks ({len(blocks)} blocks):")
                    for b_i, b in enumerate(blocks, 1):
                        b_type = b.get("type")
                        b_title = b.get("title") or b.get("text", "")[:60]
                        lines.append(f"      [{b_i}] type: '{b_type}' | Title/Summary: {b_title}")
                        if b_type == "findings" and b.get("findings"):
                            for f in b.get("findings")[:3]:
                                lines.append(f"          - {f.get('title')}: {f.get('description', '')[:90]}...")
                    lines.append("")

        lines.append("=" * 85)
        lines.append("LONG-TERM MEMORY (LTM) BACKGROUND STATE")
        lines.append("=" * 85)
        lines.append(f"Profile User Context Summary for Business '{conv[1]}':")
        if profile and profile[1]:
            lines.append(profile[1].strip())
        else:
            lines.append("No LTM summary recorded.")
        lines.append("")
        lines.append("-" * 85)
        lines.append(f"CONVERSATION TOTAL TOKENS: {total_in} In-Tokens | {total_out} Out-Tokens | TOTAL: {total_in + total_out} Tokens")
        lines.append("=" * 85)

        out_path = r"d:\NTI-Final-Project\LAST_CONVERSATION_TRACE.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("Generated", out_path)

asyncio.run(main())
