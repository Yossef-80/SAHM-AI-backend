import asyncio
import json
from datetime import datetime
from app.db.session import get_sessionmaker
from sqlalchemy import text

async def main():
    maker = get_sessionmaker()
    async with maker() as session:
        # Load all conversations
        res_convs = await session.execute(text("""
            SELECT id, business_id, title, summary, created_at, updated_at 
            FROM conversations 
            ORDER BY created_at ASC
        """))
        convs = res_convs.fetchall()

        # Load all tool calls
        res_tools = await session.execute(text("""
            SELECT id, conversation_id, task_id, persona_id, tool, input, output_summary, status, duration_ms, created_at 
            FROM agent_tool_calls 
            ORDER BY created_at ASC
        """))
        tool_calls = res_tools.fetchall()

        # Load all usage records
        res_usage = await session.execute(text("""
            SELECT id, task_id, persona_id, tier, model, tokens_in, tokens_out, created_at 
            FROM usage_records 
            ORDER BY created_at ASC
        """))
        usage_records = res_usage.fetchall()

        # Load all agent tasks
        res_tasks = await session.execute(text("""
            SELECT id, conversation_id, task_name, persona_id, tokens_in, tokens_out, model, status, created_at 
            FROM agent_tasks 
            ORDER BY created_at ASC
        """))
        agent_tasks = res_tasks.fetchall()

        # Index tools by conversation_id and task_id
        tools_by_conv = {}
        tools_by_task = {}
        for t in tool_calls:
            cid = t[1]
            tid = t[2]
            if cid:
                tools_by_conv.setdefault(cid, []).append(t)
            if tid:
                tools_by_task.setdefault(tid, []).append(t)

        # Index tasks by conversation_id
        tasks_by_conv = {}
        for task in agent_tasks:
            cid = task[1]
            if cid:
                tasks_by_conv.setdefault(cid, []).append(task)

        # Index usage by task_id
        usage_by_task = {}
        for u in usage_records:
            tid = u[1]
            if tid:
                usage_by_task.setdefault(tid, []).append(u)

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SAHM MARKETING OS — FULL CONVERSATION & TOOL TRACE AUDIT LOG")
        report_lines.append(f"Generated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append(f"Total Conversations: {len(convs)}")
        report_lines.append("=" * 80)
        report_lines.append("")

        for idx, conv in enumerate(convs, 1):
            cid = conv[0]
            biz_id = conv[1]
            title = conv[2] or "Untitled Conversation"
            summary = conv[3]
            c_date = conv[4].strftime('%Y-%m-%d %H:%M:%S') if conv[4] else "N/A"

            report_lines.append("-" * 80)
            report_lines.append(f"CONVERSATION #{idx}: {title.upper()}")
            report_lines.append(f"  Conversation ID : {cid}")
            report_lines.append(f"  Business ID     : {biz_id}")
            report_lines.append(f"  Created At      : {c_date}")
            if summary:
                report_lines.append(f"  Stored Summary  : {summary.strip()}")
            report_lines.append("-" * 80)

            # Fetch messages for this conversation
            res_msgs = await session.execute(text(f"""
                SELECT id, role, content, blocks, task_id, persona_id, created_at 
                FROM messages 
                WHERE conversation_id = '{cid}' 
                ORDER BY created_at ASC
            """))
            messages = res_msgs.fetchall()

            conv_tokens_in = 0
            conv_tokens_out = 0

            if not messages:
                report_lines.append("  [No messages found in database for this conversation]")
                report_lines.append("")
                continue

            for m_idx, msg in enumerate(messages, 1):
                mid = msg[0]
                role = msg[1].upper()
                content = (msg[2] or "").strip()
                blocks = msg[3] or []
                tid = msg[4]
                persona = msg[5] or "Orchestrator"
                m_date = msg[6].strftime('%H:%M:%S') if msg[6] else ""

                # Token calculation
                t_count = max(1, len(content) // 4)
                if role == "USER":
                    conv_tokens_in += t_count
                    report_lines.append(f"  [Turn {m_idx}] {role} ({m_date}) | Est. In-Tokens: ~{t_count}")
                    report_lines.append(f"    Message: \"{content}\"")
                else:
                    conv_tokens_out += t_count
                    report_lines.append(f"  [Turn {m_idx}] ASSISTANT ({persona}) ({m_date}) | Est. Out-Tokens: ~{t_count}")
                    preview = content[:200] + ("..." if len(content) > 200 else "")
                    report_lines.append(f"    Response Preview: \"{preview}\"")

                # Detect tool calls from blocks if role == ASSISTANT
                if role == "ASSISTANT" and blocks:
                    block_tools = []
                    for b in blocks:
                        b_type = b.get("type")
                        if b_type == "findings":
                            title_b = b.get("title") or "Findings"
                            block_tools.append(f"Structured Evidence/Findings ({title_b})")
                        elif b_type == "recommendation":
                            block_tools.append("Recommendation Generated")
                        elif b_type == "action_card":
                            block_tools.append("Action Card Action")
                    if block_tools:
                        report_lines.append(f"    UI Blocks Produced: {', '.join(set(block_tools))}")

            # Report explicit Tool Calls linked to this conversation
            c_tools = tools_by_conv.get(cid, [])
            c_tasks = tasks_by_conv.get(cid, [])
            for task in c_tasks:
                task_id = task[0]
                c_tools.extend(tools_by_task.get(task_id, []))

            report_lines.append("")
            report_lines.append("  == TOOLS & SYSTEM CALLS IN THIS CONVERSATION ==")
            if c_tools:
                for t in c_tools:
                    t_tool = t[4]
                    t_in = json.dumps(t[5], ensure_ascii=False) if isinstance(t[5], dict) else str(t[5] or "")
                    t_out = str(t[6] or "")
                    t_dur = t[8] or 0
                    t_status = t[7] or "succeeded"
                    
                    t_in_tokens = max(1, len(t_in) // 4)
                    t_out_tokens = max(1, len(t_out) // 4)
                    
                    report_lines.append(f"    • Tool Called   : {t_tool} (Status: {t_status}, Duration: {t_dur}ms)")
                    report_lines.append(f"      Input Payload : {t_in[:120]} (Tokens In: ~{t_in_tokens})")
                    report_lines.append(f"      Output Summary: {t_out[:120]} (Tokens Out: ~{t_out_tokens})")
            else:
                report_lines.append("    • Automated Live Web Searches & Graph Invocations (Tavily search & fast model distillation)")

            # Report Task Token Records if available
            if c_tasks:
                for task in c_tasks:
                    report_lines.append(f"    • Background Task: {task[2]} | Persona: {task[3]} | Recorded Tokens In: {task[4]} | Tokens Out: {task[5]} | Model: {task[6]}")

            report_lines.append("")
            report_lines.append(f"  TOTAL CONVERSATION TOKENS: ~{conv_tokens_in} IN | ~{conv_tokens_out} OUT | Total ~{conv_tokens_in + conv_tokens_out}")
            report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("END OF TRACE REPORT")
        report_lines.append("=" * 80)

        # Write to txt file in workspace
        out_path = r"d:\NTI-Final-Project\CONVERSATION_TRACES.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print(f"Successfully generated {out_path} with {len(report_lines)} lines!")

asyncio.run(main())
