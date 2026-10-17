# Reply-language policy

> **Audience:** Anyone who has ever seen Sahm answer in the "wrong" language.
> **Scope:** Every user-facing string the backend produces: chat answers, agent
> reports, block labels, interviews, error text and queued background tasks.
> **One-line version:** *the language the user picked in the UI wins.*

---

## 1. The rule

| Situation | Answer language |
| :--- | :--- |
| UI is Arabic, user writes Arabic | Arabic |
| UI is Arabic, user pastes an English article or link | **Arabic** |
| UI is English, user writes Arabic | **English** |
| UI switched from Arabic to English mid-conversation | **English from that turn on** |
| Client sends no language at all | the last explicit choice, else an Arabic project language, else the language of the message, else English |

There is no per-message language guessing once a choice exists. A conversation has
one answer language at a time, and the user changes it by changing the UI.

---

## 2. Where the decision is made

`app/core/i18n.py::resolve_reply_language` is the only place that decides. It is
called once per turn and the result is threaded everywhere; no downstream branch
re-derives it.

```python
resolve_reply_language(
    requested=request.language,          # the UI's current choice
    conversation=context.get("language"),# last explicit choice, persisted
    project=business.language,           # workspace project language
    prompt=request.message,              # only a fallback signal
)
```

Precedence, in order:

1. **`request.language`** — sent by the client on every message. This is what makes
   a mid-conversation switch take effect on the next answer.
2. **`conversation.context["language"]`** — the last *explicit* choice, so a client
   that stops sending the field does not lose the setting.
3. **`businesses.language == "ar"`** — Arabic is never a default column value, so
   it can only have been chosen deliberately. English is the column default and is
   therefore treated as "not chosen" (otherwise every tenant with no client-side
   language would get English over Arabic content).
4. **Detection from the message** — Arabic letters anywhere → Arabic; Latin letters
   → English; neither → keep looking.
5. **`en`**.

### Why detection is not first

The old behaviour inferred the answer language from the prompt text on every
branch (`_contains_arabic(prompt)`). Two failure modes followed:

- an Arabic-first user pasted one English link and the conversation "switched";
- each output path decided independently, so a single turn could produce Arabic
  prose with English headings and English error text.

Detection is now a *fallback*, used only when nothing was ever chosen.

---

## 3. What the persisted choice looks like

`conversations.context` (JSON) holds at most one small key:

```json
{ "language": "ar", "persona_id": "layla", "workflow": "creative_prompt" }
```

Only an **explicit** choice is written. A language that was merely detected for a
single turn is never persisted — otherwise one Arabic sentence would trap an
English-speaking user in Arabic for the rest of the conversation.

Writing it happens in `app/core/orchestrator/service.py`, in the same block that
merges the rest of the sticky context, so it survives like any other per-turn
state.

---

## 4. How the instruction reaches the model

Every turn prepends one system message, built by
`app/core/i18n.py::reply_language_directive`:

```
[RESPONSE_LANGUAGE]
The user chose Arabic as the application language. Write every user-facing
sentence in Arabic. This holds even if the user writes in English, pastes English
text, or if earlier turns in this conversation were in English. Keep brand names,
product names, code, numbers, URLs and quoted evidence exactly as they are; do not
translate them. Structured JSON keys stay in English exactly as the schema
requires.
```

That message is prepended to `chat_history`, and `agent_history = chat_history +
…`, so **every specialist graph receives it** as part of its input. The specialist
prompts no longer say "respond in the user's language" — they defer to this block
(`SUBAGENT_LANGUAGE_RULE`), because five agents each deciding for themselves is
how mixed-language answers were produced.

---

## 5. Every path that had to change

| Path | Before | Now |
| :--- | :--- | :--- |
| Orchestrator directive | Arabic forced, English ignored | both languages forced |
| Specialist prompts (`nour/omar/layla/karim/salma`) | "respond in the user's language" | defer to `[RESPONSE_LANGUAGE]` |
| Nour's report fields | "if the prompt is Arabic, write Arabic" | the resolved language |
| Report formatters (`presentation.py`) | `_contains_arabic(prompt)` | resolved language |
| Creative-prompt interview labels | `_contains_arabic(prompt)` | resolved language |
| Error / empty-state strings | Arabic or English by prompt script | resolved language |
| Mock image preview labels | Arabic only | resolved language |
| Final block-forming prompt | "Arabic" or "same as result" | explicit language, with translate instruction |
| Queued agent tasks (`runtime.py`) | persona's workspace language | resolved per task: explicit input → project → message |
| Layla text endpoint (`/creatives/generate-text`) | per-request `language` (already correct) | unchanged, same resolver semantics |

---

## 6. Tests

`tests/test_reply_language.py` is the contract:

- **Resolver:** region codes (`ar-EG`), native names (`العربية`), `auto`, unknown
  values; explicit choice beats the message script in both directions; remembered
  choice beats detection; Arabic project beats detection, English project does not.
- **Brain wiring:** a specialist receives `[RESPONSE_LANGUAGE]` for both
  directions; creative-prompt labels follow the language, not the prompt; the
  report formatter emits no Arabic characters when English is chosen.
- **Transport:** the UI language reaches `orchestrate()`; switching mid-conversation
  switches the next turn and updates the stored choice; the last explicit choice
  survives a client that stops sending it; a detected language is not persisted.

---

## 7. Adding a new user-facing surface

1. Get the turn's language from the caller — do not read the prompt yourself.
2. Call `resolve_reply_language` if you are at a boundary that has the request.
3. Pass `arabic=(language == "ar")` to formatters, or use
   `reply_language_directive(language)` in a prompt.
4. Add a case to `tests/test_reply_language.py`.

If a new surface re-derives the language from the text, this bug comes back.
