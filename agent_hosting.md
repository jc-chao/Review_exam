# Exam Review: Agents (L4), Safety / Observability / Cost (L5), Model Hosting

This file covers the concepts only. The course's story examples (Meridian HR, Telva Pay, the support assistant) are left out. Where an example helps, it is a small, generic one.

How to use it:
- Each section starts with the core idea in one or two lines, then builds up.
- **Bold** marks terms you should be able to define.
- Each part ends with "Likely exam questions" with short answers.
- Part D at the end is a one-page cheat sheet of formulas and rules.

---

# Part A. Module L4: Agents

## A1. Workflow vs agent (Notebook 01)

**The core idea.** The difference is *who decides the next step*.

- In a **workflow**, your code fixes the steps before any request arrives. Example: "search, then call the model once". It does the same thing for every request.
- In an **agent**, the model chooses the next step based on what it has found so far. It also decides when it is finished.
- "Agentic" is a **scale**, not a yes/no label. The more the model decides, the more agentic the system is.

**The harness.** The **harness** is the code around the model. It runs the tools, checks permissions, sets limits and records what happens. The more the model decides, the more harness you need, because there is more to check and limit.

**How to evaluate a system honestly:**
- **Score by type of question, not one overall number.** An average of 71% can hide one type that scores 0%. Only the per-type table tells you what to fix.
- **Refusing an answerable question counts as wrong.** If it didn't, a system that always says "I don't know" would look safe and accurate.
- **Repeat each question several times (`N_RUNS`).** The model samples its words, so the same question can give different answers. One run cannot tell a reliable answer from a lucky one.
- **Programmatic checks** (for example, a regex looking for "8" in the answer) are free and repeatable, but blunt. They can pass a wrong answer that happens to contain the number, or fail a right answer that is worded differently.
- **Small samples are noisy.** With only 12 attempts, one wrong answer moves the score by 8 points.

**Debugging order: rule out the cheap suspects first.** Check the earlier, cheaper step (search) before blaming the model. If search found the right document every time, then no failure is a search failure.

**Read the failures, not just the score.** If the data the answer needs never reaches the model, no prompt can fix it. Example: the model is asked "how many leave days do I have left?" but nothing fetches the user's record.

**Patching a workflow doesn't scale.** You can add a branch: "if logged in, fetch the record". But every new kind of request needs another hand-written branch. When **the next step depends on what the last step found**, you need an agent with tools.

**Identity comes from the login, never from the chat.** If the user ID came from the message text, anyone could type a colleague's ID and read their data.

**Rule of thumb.** Start with a workflow. Move to an agent only when you can point to measured failures that a fixed path cannot handle.

**Likely exam questions**
- *Why is a RAG pipeline a workflow even though it uses an LLM?* The steps (retrieve, build the prompt, one call) are fixed in code.
- *Why can't a better prompt fix a question whose data is never fetched?* The model can't use data it never receives.
- *Why does BM25 miss "health allowance" when the document says "medical subsidy"?* BM25 matches exact words. This is a **vocabulary mismatch**.

---

## A2. Tool calling (Notebook 02)

**The core idea.** A tool call is a **request**, not an action. The model only writes structured text that asks for a tool. Your code decides whether to run it.

**What a tool call contains:**
- the tool **name**;
- the **arguments**, as a JSON *string* (not a dict, and it may be malformed);
- an **id** (`tool_call_id`), so the result can be matched back to the request.

**What a tool definition contains.** A tool is a **name + description + JSON Schema** for its arguments.
- The **description** is the model's only manual. The model uses the name and description to decide *whether* and *when* to call it.
- The **JSON Schema** sets the rules for the arguments: types, `enum` lists, `pattern`, min/max and `additionalProperties: false`. The harness uses it as the first input check.
- The Python function and its permission level are **not** sent to the model.
- **Tool definitions cost input tokens on every call**, even when the tool is never used. Unused tools also give the model another wrong option.

**Tool design rules:**
- Use verb + noun names (`get_employee`, `search_handbook`).
- Descriptions should say what the tool does, when to use it, when not to use it, and what it returns.
- Use tight types and `additionalProperties: false`.
- **Measure** whether better descriptions help instead of assuming it. In the course, clear names did most of the work.

**Never `eval()` model-written text.** A calculator tool must parse the expression and allow only numbers, operators and a few safe functions. Otherwise the model, or text it read, could run any code.

**Permissions live in code.** A prompt instruction like "only show staff their own record" is a request, not a lock. The harness enforces permission levels:
- `read`: always allowed.
- `personal`: only for the logged-in user's own ID.
- `write`: only for your own ID, *and* with a human's approval.
- Anything else is **denied by default** (**default-deny**). A new tool stays blocked until someone decides its level, like a firewall that blocks every port not explicitly opened.

**`execute_tool`: run every call safely.** The steps, and what breaks if you skip each one:

| Step | If skipped |
|---|---|
| 1. Parse the JSON arguments | Malformed JSON crashes the run |
| 2. Validate against the schema | Nonsense arguments reach the function |
| 3. Check permission | The call runs as if the user had approved it |
| 4. Run inside `try` | One exception ends the whole run |
| 5. Limit the result size | One huge result floods the context and the bill |

Every failure becomes a readable message (`ERROR: ...` or `DENIED: ...`) that tells the model what to do next. The run never crashes.

**Messages that guide the model.** "Fix the arguments and retry" suits a fixable mistake. "Do not retry; tell the user" suits a permanent refusal, so the model doesn't loop.

**`finish_reason`.** It can be `"tool_calls"`, `"stop"` or `"length"`. Check whether `tool_calls` is empty instead, because providers differ in what they report.

**One round of tools is not enough.** If step 2 needs the output of step 1, you need a loop.

**Likely exam questions**
- *Which is worse: an unnecessary search, or an unnecessary write?* The write, because it changes data. A wrong search only costs a retry.
- *Why return failures as messages instead of raising?* The run keeps going, and the model can often fix its own mistake.

---

## A3. The agent loop and the harness (Notebook 03)

**The core idea.** The agent loop is short:

```
loop:
    reply = model(conversation, tools)
    if reply has no tool calls: return reply      # final answer
    for each tool call:
        result = execute_tool(call)               # never raises
        conversation.append(result, linked by tool_call_id)
```

The loop is about 20 lines. The harness around it is the real work.

**What the model decides:** what to do next and when it is finished.
**What the harness does:** runs tools, checks permissions, keeps the conversation, logs every step and enforces stopping.

**The protocol rules.** The API rejects the request (`400 BadRequestError`) if you break these:
1. Every `tool_call_id` gets **exactly one** result (`role: "tool"`).
2. The results come **straight after** the assistant message that asked for them.
3. **Parallel calls**: one assistant message can ask for several tools. Each gets its own result, and all results go back before the next model call.

This is why `execute_tool` must never raise. A crash means a missing result, and the next call fails.

**`finish_reason == "length"`.** The reply was cut off at the output limit. A cut tool call has broken JSON, so the loop treats it as an error and never runs it.

**Errors become information.** When a tool returns a clear error, the model reads it and tries something else. You don't need to write recovery code, but the message must be clear.

**Token growth is quadratic.** The model is **stateless**: it keeps nothing between calls. So call *n* re-sends everything before it.
- If each step adds about *k* tokens, call *n* sends about *n × k* tokens.
- The total over *n* steps is about *k × n² / 2*.
- **Twice the steps means about four times the tokens.**

**Ways to cut cost:**
- Use fewer, smaller steps: trim tool results and stop stuck runs.
- Use **prompt caching**: a repeated, identical prompt *prefix* is billed cheaper. Keep the system prompt and tool list **byte-for-byte identical**. Don't put a changing timestamp at the top of the system prompt, because it breaks the cache on every call.

**Logging runs (traces on disk):**
- **JSONL** means one JSON object per line. Appending is cheap, and a crash damages at most the last line.
- **Write logs as events happen**, not at the end. A crash at step 5 would otherwise lose steps 1 to 4.
- **`run_end` works like a commit marker.** A log counts as finished only if its last line is `run_end`. A half-written log is redone, never loaded.
- **Fingerprint**: a short hash of all the settings (model, prompt, tools, budget) in the file name, for example `sha1(json.dumps(settings, sort_keys=True))[:8]`. If any setting changes, the hash changes, so old and new runs never get mixed. It works like a build cache key.

**Idempotency.** An **idempotent** tool has the same effect however many times it runs (search, calculator). Resuming a run from its log is safe only for these. For a tool that submits a request, you can't tell if the last call already happened, so it might run twice.

**`safety_max_steps` is a seatbelt, not a stopping rule.** It stops infinite loops while you're learning. It isn't based on data, and it doesn't check cost, time or progress.

---

## A4. Stopping rules and budgets (Notebook 04)

**The core idea.** A run ends one of two ways:
- **Naturally**: the model answers without asking for a tool. The model decides.
- **By force**: a cap is hit, or the run stops making progress. The harness decides.

**Exit statuses.** Every run ends with exactly one:

| Status | Meaning | Success? |
|---|---|---|
| `completed` | The model finished naturally | Yes, if the answer is right |
| `budget_exhausted` | A cap was hit | **Never**, even with a correct wrap-up answer |
| `stalled` | No progress for too long | Never |
| `error` | Something broke | Never |

Counting a `budget_exhausted` run as a success hides a failure and overstates results.

**Two ways an agent fails at stopping:** it **never stops** (it loops and spends money), or it **stops too early** (it claims it's done without checking). Caps fix the first. Scoring answers is how you catch the second.

**Hard caps, checked before each call:**
- steps;
- **wall-clock time** (real elapsed time, including slow tools);
- cost, estimated for the *next* call.

Check the cost cap **before** the call, because you can't un-spend money. Estimate the next call's input as *last input + last output + new tool results*, since each call re-sends everything.

A time cap checked only between calls can be overshot, because it can't interrupt a running tool. Use a **per-call timeout** for hung tools.

**Choosing caps from data:**
- **Step cap**: using saved runs, plot "success rate" against "cap". Find where the curve stops rising, then add a margin of 1 or 2 steps for run-to-run variation. Set the cap by the *correct* runs, not the mean.
- **Cost cap**: about 3× the **95th percentile** of cost per run. The 95th percentile is the value 95% of runs stay under, a "normal worst case". A cap at 3× it should stop only abnormal runs.
- Redo this whenever anything changes.

**Detecting a stuck run (the progress monitor):**
- A step makes **progress** only if it returns **new information** or grows the set of evidence. Calling a tool alone doesn't count.
- `ERROR` and `DENIED` results never count as new information. Their text can change (a timestamp, a request ID) while saying nothing new.
- **Judge progress over a window** of several steps, not one. One dull step is normal. This is like early stopping with "patience".
  - Window too small (1): the first dull step kills good, slow runs.
  - Window too large (10): a stuck run wastes many steps, and the step cap fires first.
- A naive rule like "same tool 3 times = stuck" looks at tool *names*, not results. It kills good runs that search several times and find something new each time.

**Test the detector like a classifier.** Use labelled traces (stuck or not stuck) and measure:
- **Precision**: of the runs it killed, how many were really stuck?
- **Recall**: of the stuck runs, how many did it catch?
- Count **good runs killed**. For a user-facing assistant, killing a good run is worse than missing a stuck one, since the step cap still catches stuck runs. So lean towards a larger window.

**Stop gracefully (the escalation ladder):**
1. Warn the model when one step is left.
2. Nudge it after a quiet window.
3. Stop after a second quiet window.
4. Make a final wrap-up call with `tool_choice="none"`, so it must answer in text.
5. In a product, hand over to a person.

Then check that **normal runs are left alone**: run the full question set and confirm there are no false stops.

**Flaky vs down.** A *down* tool fails every time. A *flaky* one fails sometimes, so retrying helps. A failure followed by a success counts as progress. Use **retry with backoff**: wait longer before each retry (1 s, 2 s, 4 s), often with random jitter.

**A trap.** A run can end `completed` with "not in policy" because search was *down*. The status looks like success, but the answer is dishonest. Stopping rules limit runs. They don't check that the ending was honest.

---

## A5. The agent end to end (Notebook 05) and the midway homework (06)

**Agent = Loop + Tools + Stopping rule.** Change any part and it is a different agent, which must be measured again. The model is only one setting among many.

**A fair comparison** keeps everything else the same: the same questions, model, scorer and number of runs.

**Error analysis.** Read the failures, group them by cause, name each group, fix the biggest group first, then measure again. A score tells you *how often*. Error analysis tells you *why*, and each cause points to a different fix. Automatic failure labels are only a first sort, and a person must check them.

**The cost of autonomy.** An agent makes more calls than a workflow. Input tokens grow even faster than calls, because each call re-sends the conversation.

**Cost per correct answer = total cost ÷ accuracy.** This is the fairest single number. A cheap system that's often wrong isn't really cheap.

**When to use which:**

| Use a workflow when | Use an agent when |
|---|---|
| The steps are the same every time | The next step depends on what was found |
| You need predictable cost and latency | Requests vary too much to list every path |
| You need easy testing | Data is spread over several sources |
| One search and one call answer most requests | A right answer is worth several times the cost |

**Often the answer is both: routing.** A **router** sends each request to the cheapest system that can handle it. Simple questions go to the workflow and hard ones go to the agent.
- Routing math: with 80% simple requests at cost *a* and 20% hard ones at cost *b*, the router's cost is *0.8a + 0.2b*, plus the router's own cost.
- Which misroute is worse? Sending a hard request to the workflow gives a wrong answer. Sending an easy one to the agent only costs more. **When unsure, route to the more capable path.**

**Don't over-read small differences.** 100% vs 96% over 45 runs is 2 answers, which is within normal variation.

**Lessons from the homework (06):**
- **A tool that refuses honestly beats one that guesses.** If no formula applies, return `null` with a reason, not a wrong number.
- **Check identity before reading data.** Otherwise the harness reads someone's personal data just to refuse them.
- **Business rules live in code**, after the identity check. Examples: "no leave in the past", "not more than you have left".
- **A dedicated tool can replace model arithmetic.** It cuts steps and cost, and removes the chance of a mental-math slip.
- **Don't trust a detector you tuned on the same data you scored it on.** You need fresh labelled runs.

**Reading a broken trace to find which part failed:**

| What the trace shows | Which part failed | Fix |
|---|---|---|
| The answer was found early, then the same search repeated 20 times | Stopping rule | Progress monitor, real cap, wrap-up |
| User A read user B's record because the ID came from the chat | Permissions | Use the session's user ID only |
| An answer with no tool call that invents a number | Tools / prompt | Require tools; check a search backs each policy answer |
| A tool crashed, got no result, and the next call failed with 400 | Execution | Catch every exception and return it as `ERROR:` |

---

## A6. Context engineering (Notebook 07)

**The core idea.** The model keeps nothing between calls. Each call sends the system prompt, the tool definitions and the conversation list, and that is all the model knows. The list is the agent's **context**. Choosing what each call sends is **context engineering**.

**Terms:**
- A **turn** is one user message and everything the agent does in reply.
- A **context policy** decides which part of the context each call sends.
- A **token budget** is the most tokens a policy should send in one call.

**The Responses API version of the loop.** It uses *items* instead of messages:
- `function_call`: the model's request, with a `call_id`.
- `function_call_output`: your result, with the same `call_id`.
- A **reasoning item** is the model's encrypted working. It must be sent back directly before the item it led to.
- With `store=False`, the server keeps no copy, so your code must re-send everything.

**Pairing rules a policy must not break:** each `function_call` keeps its output, and each reasoning item keeps the item after it. Break either and the API rejects the request. A policy returns a *view* for this one call. It must never change the stored list, because later calls still need the whole conversation.

**Where tokens go:**
- The system prompt and tool definitions are re-sent on every call. With several tools, they can outweigh a short conversation.
- Over a long conversation, the **size of each call grows linearly**, and the **running total grows quadratically**. Twice the turns means about four times the total.
- **Load tools on demand**: send a short list of tool names, and fetch a tool's full definition only when needed.

**Prompt caching rewards "append only".** A call gets cache hits only if it **starts with** the previous call's exact input.
- Full history that only appends at the end keeps the prefix stable, so most input is cached. Cached input costs about a tenth of the normal input price.
- A sliding window or trimming changes the start or middle of the list. It gets almost no cache hits and pays the cache-*write* price (about 1.25×) again and again.
- There is a minimum prompt length for caching (1,024 tokens on this provider). Short calls get no cache.
- So "send less" can cost *more* than "send everything".

**Context rot.** Models answer worse as the input grows, even when the fact is in the window. **Lost in the middle**: models are often worst at using facts in the middle of a long context. This is another reason to send less, besides cost.

**Context strategies and what each one loses:**

| Strategy | What it does | What it loses |
|---|---|---|
| Full history | Sends everything | Nothing, but it grows without limit and may break a cap |
| Sliding window | Keeps only the newest items | Old facts, such as a constraint the user stated early on |
| Trimming | Cuts long tool results (e.g. after 200 characters) | Facts deep inside tool results |
| Compaction | Replaces old turns with a model-written summary | Some detail. Summaries can also run longer than planned |

**Probes** test what a policy kept. Ask a question about an earlier fact with tools turned off.

**Decide from measurements, with ranges:**
- Report the lowest-to-highest range across runs.
- **Overlapping ranges count as "no different"**, so decide on another measure.
- A token cap applies to the **largest** call, not the average. A policy whose biggest call exceeds the cap is out.

**Triage, the context question.** Was the right value in the conversation, then dropped by the policy or not used? If yes, the root cause is **context**.

---

## A7. Memory (Notebook 08)

**The core idea.** **Working context** is one conversation. It is gone when the conversation ends. **Long-term memory** is a store the agent writes to, searches, and copies into a *later* conversation.

**Three questions memory raises:** what to keep, how to find it again, and when not to trust it.

**1. What to keep: a persistence policy.**
- **Keep only what no tool can return.** Examples: preferences ("don't page me before 7am"), decisions and episodes. Don't store a fact a tool can look up, like a server's region, because the copy can go stale and only the source knows what's true now.
- Two hard-coded rules run before the model: **never keep a tool result**, and **never keep a question**. Being in code means no instruction can override them, and they save model calls.
- Don't rely on phrases like "remember that". People rarely say them, and "remember that X is in region Y" is a tool fact that shouldn't be kept.
- **Turn relative dates into real dates**: "next two weeks" becomes a start and end date, in both the text and the `valid_until` field.
- Each memory must make sense on its own.
- Test the policy on a **held-out** set you never tuned on, and use it once. A policy tweaked until it passes the test set is fitted to it.
- Kinds of memory: **preference, decision, episode**.

**2. Finding memories: retrieval.**
- **Recall** = needed memories returned ÷ all needed memories. "Did we find what we needed?"
- **Precision** = needed memories returned ÷ all memories returned. "How much of what we returned was useful?"
- Three methods compared:

| Method | Strength | Weakness |
|---|---|---|
| Keyword search | Cheap | Misses different wording ("billing" vs "billing-worker", "handover" vs "handovers") |
| Embedding search | Cheap, handles meaning | Returns a fixed top-k, so precision is capped (top-3 with one needed item gives at most 1/3) |
| Index in context (a model reads a one-line summary of each memory and picks) | Very precise | Costs a model call, and its tokens grow with the store size |

- Choose with a token budget. The index can be best for a small store and too expensive for a large one.

**3. When not to trust a memory:**
- **Expired**: its `valid_until` date has passed.
- **Conflicting**: two memories disagree. Settle it with `resolve`'s four rules, in order:
  1. An expired memory loses.
  2. A person's statement beats the agent's.
  3. Newer wins.
  4. Higher confidence wins.
  5. Otherwise there is no winner.
- **Out of date**: the source now says something different.

**Reading the truth isn't enough.** An agent can read the correct value from a tool and still follow a stale memory, because the memory reads like a standing instruction. The instructions must say **which source wins**, for example: "Be careful with memories: they might be out of date. The tool result wins." A warning that says *what* could be wrong works much better than a vague one.

**Where a memory check can go:** when writing, when retrieving, when resolving conflicts, and before acting.

**Privacy.** Memories about people are personal data under the PDPA (Singapore's Personal Data Protection Act). Record each memory's subject and source, give it an end date, and allow deletion and correction.

**Triage, the memory question.** Was the first wrong item a memory the agent acted on? If yes, the root cause is **memory**.

---

## A8. Architectures: single agent, router, sub-agents, planner (Notebook 09)

**The core idea.** Anthropic's and OpenAI's guides agree: **start with a single agent, and split only when you can show a measured reason.** Every split adds calls, cost and places to fail. The patterns can be combined.

```
 single agent   request -> [agent + tools] -> answer
 router         request -> [route?] -> agent | a person | decline
 sub-agents     request -> [lead] -> [sub-agent + tools] x N -> reports -> lead writes answer
 planner        request -> [model writes plan] -> code checks -> code runs -> [model writes answer]
```

| Pattern | What it buys | What it costs |
|---|---|---|
| Router | A rule enforced before any agent runs | An extra call, and requests can be misrouted |
| Sub-agents | Small, clean contexts for each piece of work | Many more calls. The lead only gets what it asked for |
| Planner | A plan that code can check before running | Can't adapt or retry mid-run |

**The baseline.** Measure the single agent first: facts found, wrong verdicts, cost, tokens and the largest call. A run that aborted early isn't "cheap", it's a failure. Lucky discoveries, like an error message that happens to list every service, are not a repeatable method.

### Routing

- A **router** is a small model call that returns a **label** (for example `answer`, `needs a person`, `out of scope`) plus a reason. Your **code** reads the label and acts.
- Use **structured output with a `Literal` type** so the model *can't* return a label that doesn't exist:

```python
class Route(BaseModel):
    route: Literal["answer", "needs a person", "out of scope"]
    reason: str
```

- **Route by intent, not wording.** "Is it safe to restart X?" sounds like an action, but it only asks for information, so it goes to `answer`. "Could you get the team to look into X?" is polite, but it's an action.
- **A mixed request** (a question plus an action) goes to a person as a whole. Otherwise the action is silently dropped.
- **Write rules for *kinds* of request, not for specific examples.** Test on held-out requests to prove the rules generalise. Router quality comes mostly from its instructions.
- **Cost of each misroute:**
  - A question sent to a person wastes human time and adds delay.
  - An action sent to the assistant wastes tool calls, ends in "I can't do that", and may *look* handled while nobody acts. This is the most dangerous direction when tools can change things.
  - Out-of-scope sent to the agent is cheap, but it's off-topic work.

**Router vs handoff:**
- **Router**: returns a decision your code acts on. It's good for enforcing rules and for routes with no agent behind them (a person, a refusal). Every decision is logged.
- **Handoff** (OpenAI Agents SDK): a dispatcher agent picks a specialist from short `handoff_description`s and **hands over the whole conversation**. The specialist then talks to the user directly.
- A handoff's weak spot: the dispatcher can always answer a request itself. So handoffs are worst on requests that fit *no* agent. A router can't do this, because it must return one of its labels.

### Sub-agents

- The **lead agent** has no data tools. Its only tool starts a sub-agent with an **assignment**: which entity, what task, and which **fields** to bring back.
- Each **sub-agent** works in its **own, separate conversation** with all the tools, and returns a **report**.
- The lead sees **only the report**: not the sub-agent's tool calls or reasoning.
- **Report format:**
  - `facts`: each requested field, its value, and the `call_id` it came from, so it can be traced.
  - `missing`: requested fields that no successful result contained.
  - `failed`: tool calls that returned an error (tool, arguments, error).
- **Golden rule: build the report from tool results in code, never from the sub-agent's own text.** A model can write a value no tool returned, and the lead can't check.
- Edge cases:
  - A result is a failure only if it has an `"error"` **key**. Text that mentions "error" is still a success.
  - An empty list (`[]`, "no incidents") is a **fact**, not a missing field.
  - A failure followed by a successful retry appears in both `failed` and `facts`.
- **Blind spot:** `missing` tracks missing *fields*, not missing *entities*. If a whole service is skipped, nothing warns you. Only a cross-check against the master list catches it.
- **The lead only gets what it explicitly asks for.** If it forgets a field the verdict depends on, that field is stripped from the report and the verdict is wrong.
- **Handling `failed` or `missing`:** retry reads (safe, because reads are idempotent). If still unresolved, say clearly in the answer that the data couldn't be retrieved. **Never let the model guess.**
- **The paradox.** Each sub-agent's context is smaller, so the lead's largest call shrinks. But total calls and cost go *up*, because each sub-agent needs its own calls.

### Plan-then-execute

- A **planner** model writes the whole plan up front: a list of steps, each one tool call with arguments. **Plain code** runs the steps in order, with no model in between. A **writer** model then answers from the results.
- A step can use an earlier result with a reference like `{s1.oncall}`, and must list `s1` in `depends_on`. This works like a build system's dependency graph.
- **Code validates the plan before anything runs**, because it is model output written before any result exists:
  1. Each step uses a real tool, with arguments that tool accepts.
  2. Each entity named exists in the registry.
  3. Each reference points to an *earlier* step listed in `depends_on`, and to a field that step's tool really returns.
  4. The plan is within the step limit.
- If the checks fail, the problems go back to the planner **once** for a corrected plan, which is checked again.
- **Passing checks doesn't make a plan good.** A valid plan can still be useless.
- **A planner can't discover things.** It must know *what* to look up before it starts. It suits fixed, known lookups. Since each step runs once, a timeout just leaves a gap, and no agent notices or retries.

### Choosing an architecture

**Rules first, then measurements.**
1. **Filter**: does it obey the hard rule (for example, "a request to *do* something goes to a person")? If not, it's out, however well it scores.
2. **Choose**: among the survivors, compare facts, wrong answers, cost, tokens and the largest call. Overlapping ranges count as equal.

**Triage, the architecture question.** Did a value **never reach** the agent that needed it, because of the route, the tools given, the assignment, or the report? If yes, the root cause is **architecture**. It's not context, because the value never arrived, so nothing dropped it.

Where an architecture check can go: the router, the tools each agent gets, the lead's assignments, the sub-agent's report, and the plan check.

---

## A9. Context poisoning (Notebook 10)

**The core idea.** **Context poisoning** is when a value that was **never true** gets into the agent's context and is treated as fact from then on. It comes from a mistake (a buggy tool, or the model inventing something), not an attacker.

**Prompt injection** is related but different. It's text **planted on purpose** to steer the agent. OWASP lists it first among LLM risks. The defences overlap.

**One wrong read spreads everywhere it's written down:**
- the answer;
- a compaction summary;
- memory, if the policy keeps tool facts;
- a sub-agent's report;
- a later action, such as a filed change request.

Each copy looks as trustworthy as the original tool result.

**A correct read fixes the conversation, not the copies.** After a later correct read, the conversation is fine. But the memory, the filed request and the report are already elsewhere and don't re-read the source.

**The model's own invention is poison too.** If an earlier assistant message states a wrong value, the model tends to repeat it. It's in the conversation, nothing marks it as unbacked, and the model has no tool to check.

**Four defences, in four places:**

| Defence | Where | What it catches | What it misses |
|---|---|---|---|
| **Labels**: tag each tool result with its tool, arguments and call ID; mark contradicted reads "replaced"; flag claims no current result backs as "unverified" | Context policy | Disagreements between reads, and unbacked model claims | A single wrong read with nothing contradicting it |
| **Check before writing a memory** | Memory write | Tool facts being stored | Errors in the answer itself |
| **Guardrail on tool output**: check the result against a rule before the model sees it | Tool output | Values that break a rule | Wrong values that break no rule (two valid values swapped) |
| **Approval against an independent source** before an action | Action boundary | The action, for any bug that doesn't touch the independent source | It doesn't remove the value from the answer or memory |

**Independence.** A check only works against a source **not filled by the same process**. A second copy from the same buggy export proves nothing.

**Use several defences.** Each one works in one place and misses something.

**Triage, the poisoning question: "Was it ever true?"**
- If the first wrong item is a tool result or model claim that was **never true**, the cause is **poisoning**.
- If it came from a **memory** the agent acted on, the cause is **memory**, even if the memory was wrong when written.
- If a tool value was true once and went stale, it's not poisoning. It's out of date.
- To test "ever true?", compare with an independent source, or check a rule the value would break.

---

## A10. Incident triage (Notebook 11)

**The core idea.** Find the **first wrong item**, not the most visible one. The complaint is about the last reply, but the cause is usually earlier. Everything after the first wrong item is a consequence, and fixing a consequence leaves the cause.

**Three steps:** find the first wrong item, classify it, and ask "was it ever true?".

**Classify with four questions, in order:**

| # | Question | Root cause |
|---|---|---|
| 1 | Was the right value in the conversation, then not sent or not used? | **Context** |
| 2 | Was the first wrong item a memory the agent acted on? | **Memory** |
| 3 | Did a value fail to reach the agent (route, tools, report)? | **Architecture** |
| 4 | Was it a never-true tool result or model claim? | **Poisoning** |

**Compare the conversation with the failing call's input.** The `conversation` is everything that happened. The `input` is what the failing model call was actually sent. A value can be in the first and cut from the second. For example, a trimmed result ends with "[N characters dropped]".

**A good triage entry** (a **blameless post-mortem**, which records *what* failed and *what check would catch it*, not *who* got it wrong):
- the root cause;
- evidence: quotes **copied exactly** from the cited items;
- a claim;
- the check that would have prevented it;
- the **stage** where that check belongs: router, context policy, memory conflict resolution, or tool output check.

Passing tests only proves the entry is complete and its quotes hold. Whether the root cause is right is your own judgement.

**Common hidden causes:**
- **A canned reply** like "passed to on-call" makes a misroute look handled while nobody acts.
- **Trimming's parallel-call blind spot.** Only the newest result stays whole, so with two parallel calls, the first is cut before the model ever reads it.
- **Memories listed without dates**, so the model can't tell which conflicting memory is newer.

---

## A11. Sandboxing (note 12)

**The core idea.** Treat every action the model asks for as **untrusted input**, like text from a stranger in a web form. A model can be **wrong**, **misled** (by something it read), or doing exactly what was asked when the request itself was harmful. Testing only covers the cases you tried.

**A sandbox** is a closed-off, disposable space where a tool runs with only what it needs. It controls:
- **files**: only the needed folder;
- **network**: off by default, because the network is how data leaves;
- **secrets**: kept out of the conversation, because anything in context can end up in an output, log or tool call. The tool holds the secret and uses it on the model's behalf;
- **resources**: time, memory and CPU limits.

**Least privilege.** Each tool gets its own account, limited to its job. A reporting tool gets read-only access. Prefer **short-lived credentials**, because a leaked one is soon useless.

**Isolation isn't enough on its own.** The agent can still do harm inside its allowed space, so you also need **gates**:

| Action level | What to do |
|---|---|
| Read | Allow |
| Reversible change | Allow, log it, make undo easy |
| Irreversible or high-impact change | A person approves first, every time |

- **Code checks come before asking a person.** A forbidden action is denied outright, and nobody is asked to approve it.
- **A good approval screen** shows the real action and real arguments, for this one action.
- **"Approve always"** turns the gate into a formality.
- **Tools must be safe to run twice**, because a retry or resumed run can repeat them.
- **A result from your own tool isn't automatically trusted.** It can carry injected text or a wrong value.

---

# Part B. Module L5: Safety, Observability, Cost, Governance

## B1. Safer LLM design (Notebook 01)

**The core idea: reading is not obeying.** **Prompt injection** is when text the application *processes* (a ticket, an email, a web page, a document) contains instructions that try to change the task. Processed text can supply **information**. Only the developer and the authorised user can supply **instructions and permission**.

**Message roles vs permission checks.** Roles (system, user, tool) tell the model where text came from. Only application checks *enforce* what's allowed.

**Trust boundary.** A point where data or a request crosses into a component with different permissions. Examples: ticket text entering the model, and a model's proposed action reaching the tool executor.

**Zero trust.** Treat the model's proposal as a request to verify, not as proof of permission. The test: **"If the model were fully compromised, would the forbidden action still be blocked?"** Zero trust doesn't mean a human approves everything. Routine actions can run on narrow, verified, standing permissions.

**The Rule of Two.** Danger comes when one component has all three of these:
1. **untrusted input** (text from outsiders);
2. **private data**;
3. **external action** (send, post, write).

Together they give an outsider's words a route to move someone's data out. **The fix is to remove or restrict one capability:**
- **Remove tools.** A fixed workflow fetches the data in Python and gives the model *no tools*. The model can't send what it has no tool for.
- **Scoped retrieval.** Give the model only the data this user may see, fetched in code from the login. Never "load all records and prompt 'answer only about Daniel'". This is the **principle of least privilege**: grant the minimum access the task needs.
- For a first release, remove the riskiest capability, often outbound sending.

**Test enforcement separately from model behaviour.**
- **Enforcement test**: call the tool directly in Python with a bad proposal. This simulates "the model proposed something bad" and tests the harness.
- **Model behaviour test**: give the model an injected input and record what it proposes. That's a different question.
- "The model didn't propose a bad send" doesn't prove the app is safe, because the check never ran. "One attack was resisted" doesn't prove it either, because it covers one wording and one model.

**Prompt warnings and keyword filters are not the last line of defence.** They change what the model is *likely* to propose. They don't check permission. Filters also block legitimate text, attackers reword, and a harmful send may contain no suspicious words.

**Validate, then authorise, then execute:**
1. **Validate the shape** with a Pydantic schema: required fields, correct types, no extra fields. `strict=True` rejects wrong types without converting them. `extra="forbid"` rejects unknown fields.
2. **Authorise** against **trusted state**, not model claims: does the task allow this action, is this customer in scope, does the customer exist, is the recipient the verified address? **Structure isn't entitlement.**
3. **Execute** the side effect only after every check passes.

Only the proposed values (customer, recipient) come from the model, and they are *claims to check*. The task, scope and verified email come from the application. A model field must never expand the task's permission.

**Protect outputs, even with no tools.** An answer can still:
- leak private data;
- make unsupported claims (it says "approved" when the source says "pending");
- carry HTML, SQL or shell that another component runs.

You need evidence checks, privacy checks, **safe output handling** (escape HTML, use parameterised SQL), and a defined response for every failure. A content filter checks policy, not how the receiver treats the text.

**Retries and idempotency.** Retrying a model call is safe. Retrying an *action* may repeat it, and a timeout doesn't prove nothing happened. An **idempotency key** (a unique ID per intended action) lets the receiver recognise and ignore the second attempt.

**Risk assessment and the control plane.** Ask how a failure could happen, how likely it is, and how bad the harm would be: can it be undone? The **control plane** is the application logic that decides what the agent may access or do. Stronger controls cost latency, money and false refusals, so match them to the harm.

**Testing checklist.** Test the legitimate task, a forbidden resource, a malformed proposal, the same request reworded, and a failed dependency. Test on the **deployed** execution path, because a validator the application never calls protects nothing. **Red teaming** means deliberately attacking your own system.

---

## B2. Observability (Notebook 02)

**The core idea.** When something goes wrong in production, you need **records** to find out what happened. The records must not become a privacy problem themselves.

**Log vs trace vs metric:**
- A **log** is a line about one event.
- A **trace** connects all the operations of **one request**.
- A **metric** is a number summarised across **many requests**: rates, P50 or P99 latency.

**Spans, traces, sessions:**
- A **span** is one operation, with a start and end time.
- A **trace** is all the spans of one request. They share a **trace ID**.
- Each span's **parent ID** points to the span that contains it, so the spans form a tree.
- A **session** groups several related traces (an attempt and its retry) by a shared `session.id`. A session is *not* a parent span.

How do you know an operation belongs to request B? It has B's trace ID, and its parent ID is B's request span. Don't group by user ID alone, because two simultaneous requests from one person become ambiguous. Add the user ID as an attribute instead.

**Record the decision and the outcome, not only the call.**
- "Tool `send_email` called" isn't enough. It doesn't say which request, whose record, which recipient, or what happened.
- "Permission allowed" and "sent" are different facts. **A missing outcome is "unknown"**, not "blocked".
- "Sent" may only mean an outbox write, not delivery.
- **A permission denial returns normally**, so its span has no error status. Error dashboards miss it. You need a **denial-rate** signal, counted by reason.
- Library tracing can't infer your application logic, so **your code must add the decision, the reason, the outcome and the versions**.

**Monitoring signals:**
- denial rate, error rate;
- **P99 vs P50 latency**. P50 is the typical request and P99 is the slowest 1%. If P99 rises while P50 stays flat, the slowest requests got worse, so inspect slow traces.

**Attack or misconfiguration?** Look at spread and timing. Specific inputs and outside destinations suggest an attack. A whole workflow failing right after a version change suggests a misconfiguration.

**PII detection is imperfect.** **PII** (personally identifiable information) is data that identifies a person. Tools like **Presidio** find some PII and miss some. Card digits may be labelled as a date, and a national ID may be missed completely. A detection score is the recogniser's confidence, not proof the rest is safe.

**The safest field is one you never capture.** Use an **allow-list** of fields to record (customer ID, recipient, decision, outcome), and drop free text like the message body.

**Protection methods** each trade privacy against what you can still investigate:
- **Exclude**: don't record it.
- **Redact**: remove the value.
- **Mask**: hide part of it (`****1234`).
- **Pseudonymise**: replace it with a stable token, so events still link up. Protect the mapping table.
- **Generalise**: keep a category instead of the value ("external domain").

**A plain hash of an email is not anonymous.** Someone can hash likely addresses and compare.

**Sampling and retention decide what evidence exists.**
- **Sampling** keeps only some traces. A reported problem may be outside the sample. Keeping only errors loses wrong-but-successful answers.
- **Retention** deletes records after a set time. If bodies are kept 7 days and metadata 30 days, a report on day 21 can prove a send happened, but not what was in it.
- A narrower fix than keeping bodies longer is to record the *categories* of fields sent.

**OpenTelemetry and Phoenix.** **OpenTelemetry** is the open standard for producing spans. **Phoenix** stores and displays them. Decorators create spans, and your code adds the facts only it knows. `TraceConfig(hide_inputs=True, hide_outputs=True)` hides content in **telemetry** only. It does **not** change what is sent to the **model provider**, and custom attributes aren't filtered.

**Version attributes.** Put `app.version` and `app.policy.version` on every request span. They tell you which code and policy produced a trace, and link evidence to a release.

**Choosing a platform** (Phoenix, LangSmith, Langfuse, MLflow). Compare **service fees + infrastructure + engineering time** for your volume, retention and data-location rules. "Self-hosted, no licence fee" still costs servers, backups and people's time.

---

## B3. Cost control (Notebook 03)

**The core idea.** Cost = reported tokens × the rate for **your** deployment.

```
cost = (input_tokens × input_rate + output_tokens × output_rate) / 1,000,000
```

Example: 1,200 input and 200 output tokens at $2.50 and $10.00 per million cost (1,200 × 2.5 + 200 × 10) / 1,000,000 = **$0.005**.

**Cached input** tokens are *part of* the input count, billed at a lower rate. Don't add them on top, or you charge them twice. Likewise, don't add a trace total to its span costs, because that double counts.

**Where the price comes from:** a pay-per-token API, reserved capacity, or your own servers. Each has a different cost basis. With reserved capacity, tokens explain usage, but the agreement sets the charge. Open-weight doesn't mean free: a hosted open model can still charge per token.

**Check the price match. Unpriced is not free.**
- Phoenix matches the span's `llm.model_name` (and provider, if set) against a price table. It needs two things to price a call: **model identity** and **token usage**.
- A catalogue **list price** may differ from your negotiated or regional rate. Take the rate from your contract or bill, and save a custom price.
- Use an exact pattern like `^name$` so you don't price other models by accident.
- A price saved later doesn't re-price old spans. Make a new call and check the new trace.
- If a model shows $0, the cost is **unknown**, not zero.

**Estimate before, measure after.**
- A **tokenizer** (for example `tiktoken`) counts tokens *before* the call, for budgets and context limits.
- Only the response's **usage** field is billing evidence.
- Local counts come out lower than billed. They miss role and formatting overhead, the system prompt, tools, all output, cache effects and retries.
- **Language matters.** Chinese text can use more tokens per character than English, and other languages differ again.

**Reasoning tokens.** Some models "think" before answering. These hidden tokens are billed as **output** and count against `max_completion_tokens`. With a low limit, reasoning can use the whole budget and the visible answer comes back empty, with `finish_reason = "length"`.

**Find the source of waste, then choose a control.** Common sources of **context bloat**:
- repeated instructions;
- too many retrieved documents;
- long chat history (a good first target, because it grows);
- verbose tool results.

Controls: trim or summarise context (keeping stated constraints), use a cheaper model for an easy step, set output limits and step limits, and use **prompt caching**. For caching, check the reported cache hits and the deployment's cached rates. Repeated text doesn't guarantee a hit.

**Judge a saving by cost per successful task.** Lower cost only counts if the answer still meets the task.
- Trimming an irrelevant line that saves tokens with the same answer is a real saving.
- Setting the output limit so low that the answer is cut off is not a saving. The task failed.
- More context can make an answer *worse*, not just more expensive.

---

## B4. Governance (Notebook 04)

**The core idea.** **Governance** defines who makes decisions and who maintains the controls. **Assurance** is the evidence supporting a specific claim, such as "the assistant only retrieves documents the user may access".

**1. Define the use before judging safety.** The same summary has different risks if a colleague reads it, or if it *automatically* triggers a refund. Write down:
- the intended use;
- the affected people;
- permitted access;
- prohibited actions;
- required boundaries;
- operational limits.

If the use changes (summaries now rank refunds automatically), you need a new assessment.

**Frameworks as question lists.** Singapore's Model AI Governance Framework, its GenAI and Agentic versions, and the AI Verify principles become concrete review questions, each with evidence attached. The agentic framework has four areas: **bound the risks, assign accountability, implement controls, prepare users**. Frameworks are **voluntary guidance**. They don't replace legal, contractual or sector rules such as PDPA or MAS guidelines.

**2. Oversight must be able to work.** A human reviewer helps only with enough:
- **information**: the source, the context, the policy;
- **authority**: they can actually stop it;
- **time**: clicking "approve" on 500 items an hour is not oversight.

More rules:
- Routine actions can run automatically within tested, narrow permissions.
- Unusual or high-impact actions need review **before** execution.
- On a **review timeout**, fail safe: hold with no action and alert. Never approve by default.
- **Low confidence alone is a weak review trigger.** A confidently wrong output skips review. Check that confidence predicts errors, and sample high-confidence cases too.
- **Monitoring can't replace a permission check.** Monitoring happens after the action, and the check happens before.
- **Named owners**: every duty (approve the use, build boundaries, privacy and security review, operate, decide release) has an accountable role. The **deploying organisation** is accountable for the app's actions, not the model vendor.

**3. Set acceptance criteria *before* testing.** Decide which failures block release before you see scores, so nobody reinterprets a good-looking score afterwards. Store the conditions with every result: the versions of app, prompt, model, tool policy, dataset and rubric, plus the reviewer, the date, and what's untested.

**4. Review the evidence, not just the score.**
- **Grounded**: every claim is supported by the source.
- **Complete**: nothing the task needs is missing.
- A summary can be grounded but incomplete. To catch this, list the task's required items in the rubric and require each one.
- **A model judge is evidence to calibrate**, against human-reviewed examples. It is not release authority.
- **A benchmark can't show that your app enforces its own boundaries.**
- **A high average can't offset a critical failure.** 96% quality doesn't cancel a failed or untested access-control test.

**5. A bounded release decision, kept current, with an incident path.**
- Each claim needs evidence, scope, version, owner and a release condition.
- **Release only the scope the evidence supports.** Example: a pilot with outbound actions turned off while a new component is untested. A *promised* test isn't evidence.
- Define **stop conditions** for a pilot.
- **Re-review** when anything changes: model, prompt, retriever, tool, permission rule, user group, language, price or serving setup. A retriever change can invalidate old access tests.
- Prepare the incident path **before** an incident: **contain → preserve evidence → investigate → recover → communicate**. Don't paste prompts into a shared channel, because that creates another exposure.

---

# Part C. Model Hosting

## C1. The VRAM budget (P1 §1.1)

**The core idea.** **VRAM** is the memory on the GPU card. It must hold three things:

```
VRAM needed ≈ weights + KV cache + overhead (~2 GB)
```

**Weights** are the model's numbers, loaded once and shared by all users.

```
weights = parameter count × bytes per parameter
```

| Precision | Bytes / param | 7B model | 70B model |
|---|---|---|---|
| fp32 | 4 | 28 GB | 280 GB |
| bf16 / fp16 | 2 | 14 GB | 140 GB |
| int8 / fp8 | 1 | 7 GB | 70 GB |
| int4 (AWQ, GPTQ, GGUF Q4) | 0.5 | 3.5 GB | 35 GB |

**Precision** is how many bits store each number. **Quantisation** means storing weights in fewer bits. Real 4-bit files are about 10–20% bigger than the clean sum, because embeddings and scales are kept in higher precision.

**KV cache: per-user conversation memory.**
- Generation is **autoregressive**: each new token attends to every token before it.
- Attention turns each token into a **query**, a **key** and a **value**. Old tokens' keys and values never change, so they're stored instead of recomputed. That store is the **KV cache**.
- Unlike weights, the KV cache is **per request**, and it grows with every token.

```
KV bytes per token = 2 (K and V) × n_layers × n_kv_heads × head_dim × bytes_per_value
KV for one request = tokens × KV bytes per token
```

Example: 32 layers, 8 KV heads, head_dim 128, bf16 gives 2 × 32 × 8 × 128 × 2 = 131,072 bytes = **128 KB per token**. An 8,192-token request then needs **1 GB**.

**What shrinks the KV cache:**
- **GQA (Grouped-Query Attention)**: several query heads share one K/V head. With 32 query heads and 8 KV heads, the cache is 4× smaller. A model without GQA can have a *bigger* cache than a larger model that has it.
- **Hybrid / linear attention**: some layers keep one fixed-size state per request instead of K/V per token, so their memory doesn't grow with context. For a hybrid model, one request costs *(full-attention layers' per-token cost × tokens) + (fixed linear state)*.
- **The architecture decides KV size, not the parameter count.** The cache grows with *layers* and *KV heads*, not overall width.

**Overhead is about 2 GB**, a fixed "entry fee". It covers the CUDA context (driver setup), **activations** (temporary results of one forward pass), **CUDA graphs** (pre-recorded GPU work replayed to save launch time) and allocator slack.

**A full worked sum.** A 24 GB card running a 7.6B model in bf16, at 56 KB per token, with 4,096-token requests:
- weights = 7.6 × 2 = 15.2 GB
- left for KV = 24 − 15.2 − 2 = 6.8 GB
- one request = 56 KB × 4,096 ≈ 0.22 GB
- concurrent full-length requests ≈ 6.8 ÷ 0.22 ≈ **31**

In vLLM, you first multiply VRAM by `--gpu-memory-utilization` (default 0.9), so the real number is lower.

**Memory bandwidth sets decode speed.** To write *one* token, the GPU reads *every* weight once.

```
best-case time per token ≈ weight bytes ÷ memory bandwidth
```

A 16 GB model on a 300 GB/s card takes at least 16 ÷ 300 ≈ 53 ms per token, about 19 tokens/s for one user. This is why a small model on a fast card can beat a big model on a slow one.

---

## C2. Choosing a model: fit, run, allowed (P1 §1.2–§1.4)

**The core idea.** Three gates, checked **in order**. A model that fails one never reaches the next:
1. **Does it fit?** Weights + KV cache + overhead must fit in VRAM.
2. **Can I run it?** The serving engine must support the architecture, the quantisation format and your GPU generation.
3. **Am I allowed to?** The licence must allow your use.

Only then do you compare quality: task fit, language coverage, benchmarks, and your own evaluation set.

**Runtimes (serving engines):**

| Engine | Best for | Formats |
|---|---|---|
| **vLLM** | Production on NVIDIA/AMD GPUs, high concurrency | bf16/fp16, FP8, AWQ, GPTQ |
| **SGLang** | Like vLLM; strong on structured output and prefix reuse (RadixAttention) | similar |
| **TGI** | Production with Hugging Face Hub integration | bf16, AWQ, GPTQ |
| **llama.cpp / Ollama** | CPU, Apple Silicon, one user or a small team | GGUF only |
| **TensorRT-LLM** | Maximum NVIDIA throughput, with a build step | its own engines |
| **Transformers `generate()`** | Experiments only | anything; no batching scheduler |

A wrong format pairing (GGUF in vLLM, AWQ in Ollama) is a common way to lose a day. Unusual architectures need a new enough engine version.

**GPU generations and precision:**

| Generation | Examples | bf16 | FP8 |
|---|---|---|---|
| Turing | T4 | **No**, so use fp16 | No |
| Ampere | A100 | Yes | No |
| Ada | L4, RTX 4090, L40S | Yes | Yes |
| Hopper | H100 | Yes | Yes |

**bf16 vs fp16.** Both use 16 bits, split differently. fp16 has 5 exponent bits (max ≈ 65,504) and 10 fraction bits. bf16 has 8 exponent bits (the same range as fp32) and 7 fraction bits. A model trained in bf16 can hold values that **overflow** in fp16. So casting bf16 to fp16 on a T4 usually works but can occasionally give garbage. Test the outputs, not just the load.

**Licences.** Check four things: **commercial use, redistribution and derivatives, usage restrictions** (acceptable-use policy, user-count threshold, geography) and **attribution**.

| Family | Commercial | Notes |
|---|---|---|
| Permissive (Apache-2.0, MIT) | Yes | Keep the notice |
| Community licences (Llama) | Yes, with conditions | Acceptable-use policy, "Built with Llama", derivatives carry "Llama" in the name, 700M monthly-user threshold |
| Provider terms (Gemma) | Yes, with conditions | Prohibited-use policy flows down to derivatives and users |
| Research / non-commercial | **No** | Fine for a prototype, a blocker for production |
| Hosted API terms | Yes | Not a weights licence. The provider can change or retire models |

- **Licences inherit.** A fine-tune of Llama is still under the Llama licence, whatever its model card says.
- **The licence can differ by size within one family.** Always check the exact repo.

---

## C3. Where to host (P1 §2)

**The core idea.** Hosting means keeping the model loaded on a machine, behind a server, so software can send it requests. Almost every engine speaks the **OpenAI-compatible API** (`POST /v1/chat/completions`), so switching between options is mostly a URL change.

**Three questions, in order:**
1. **Where may the data go?** If it can't leave the network, only on-prem remains.
2. **How much traffic is there?** An API bills per token and costs nothing when idle. A GPU bills per hour, busy or idle.
3. **Who will run it?** Self-hosting means drivers, CUDA versions and out-of-memory crashes at 2am.

| | On-prem | Cloud GPU | Hosted API |
|---|---|---|---|
| Data | Never leaves your network | Your cloud account, in a region you pick | Goes to the provider |
| Cost shape | Hardware up front | Per GPU-hour | Per token |
| Cheapest when | Steady, high, long-lived load | Steady load, or scheduled batch jobs | Low, spiky or unknown load |
| Models | Any open model or fine-tune | Any open model or fine-tune | Only what the provider offers |
| Time to first call | Days to weeks | Hours | Minutes |

**Hybrids are common.** Serve the steady baseline yourself, and spill over to a hosted API when **queue depth** crosses a threshold. Route on load, not on the day of the week.

**Chat templates.** The model was trained on one exact text format, with special markers for roles and turn ends. A **chat template** (Jinja, stored in `tokenizer_config.json`) turns a list of messages into that string.
- With a hosted API, the provider applies it for you.
- vLLM applies the model's template on `/v1/chat/completions`, but *not* on raw `/v1/completions`.
- `add_generation_prompt=True` adds the "assistant, your turn" marker.
- **A wrong template doesn't raise an error. The answers just get worse**: the model rambles, ignores the system prompt or never stops.
- Check the template with `tokenizer.apply_chat_template(...)`.

---

## C4. Prefill vs decode, and the metrics (P1 §3.1–§3.2)

**The core idea.** A request has two phases with **opposite bottlenecks**:

| Phase | What happens | Bottleneck | What helps |
|---|---|---|---|
| **Prefill** | Reads the whole prompt in one pass and fills the KV cache | **Compute** (arithmetic): weights read once, reused for all prompt tokens | Less computation (FlashAttention, prefix caching, FP8 compute) |
| **Decode** | Writes one token per step; each step reads all the weights | **Memory bandwidth** | Fewer bytes to read (quantisation, a faster card, batching) |

**Batching helps decode a lot.** If 16 users decode together, the weights are read once per step and serve 16 tokens instead of 1. Throughput goes up almost 16×, while each user's speed drops only a little.

**The metrics:**
- **TTFT (time to first token)** = queue wait + prefill. It's the "is it working?" delay. It rises with long prompts and traffic spikes.
- **ITL (inter-token latency)** = the time between tokens while streaming. It measures decode. About 50 ms feels fast and 200 ms feels sluggish.
- **Throughput** = total output tokens per second across all users. It sets capacity and cost: GPU price ÷ throughput = cost per token.
- **Queue depth** = requests waiting to start (`Waiting: N` in vLLM's log). It's the **early warning**: wait time adds straight onto TTFT. Use it to trigger scaling or spill-over.
- **p50 / p95 / p99** are percentiles. p95 TTFT is the TTFT that 95% of requests beat. **An average hides the tail.** Targets (SLOs) are written on p95 or p99. A widening gap between p50 and p99 is the classic sign of queueing.

**From complaint to metric:**
- "Takes forever to start, then streams fine": **TTFT** (queue or prefill).
- "Streams slowly once started": **ITL** (decode).
- "One client sends huge prompts and everyone slows down": TTFT for all, because of long prefills. ITL also rises, because prefill shares steps with decode.
- "Many more users; things hang, then stream slowly": **queue depth → TTFT**, plus ITL. Throughput flattens because the server is saturated. Add capacity.

---

## C5. Serving levers (P1 §3.3–§3.4)

**There is no free lunch.** Each lever trades memory, quality, latency or flexibility.

**Quantisation:**
- Smaller weights give **faster decode** (fewer bytes to read) and **more room for KV cache** (more users).
- 8-bit (FP8, INT8) has negligible quality loss and is the default if your hardware supports it. 4-bit (AWQ, GPTQ) has a small, measurable drop, worse on small models. Below 4-bit, evaluate carefully.
- `--kv-cache-dtype fp8` stores K and V in 8 bits, which halves the per-token KV cost.

**Context length (`--max-model-len`).** This is the cap on tokens per request: prompt plus reply. **Prompt tokens + generated tokens ≤ max-model-len.**
- The **trained maximum** (`max_position_embeddings`) is a ceiling, not a target. Going above it degrades quality.
- vLLM **refuses to start** if even **one** full-length request can't fit in the KV pool.
- To choose a value, take your longest realistic prompt, add the longest reply you allow, add a margin, and round up (4,096 or 8,192).

**Continuous batching and PagedAttention:**
- **Static batching** waits for a whole group to finish before starting the next, so slots sit idle behind the slowest request.
- **Continuous batching**: at every decode step, finished requests leave and waiting requests join. The GPU stays full.
- **PagedAttention** hands out KV memory in small blocks (e.g. 16 tokens) as a request grows, like pages in an operating system. A 200-token request uses about 200 tokens' worth, not a full reserved slab.
- `--max-num-seqs` caps the number of requests per step. Higher means more throughput, slightly slower per user, and more KV pressure.
- `--max-num-batched-tokens` caps the tokens per step. It limits how much prefill is mixed in with decode (**chunked prefill**), which protects streaming users from spikes.
- **Prefix caching** (`--enable-prefix-caching`): requests that share a beginning (a system prompt, a document) compute that prefix's KV **once** and reuse it. This cuts TTFT and saves memory.
- **Preemption**: when KV blocks run out, the scheduler evicts a running request and recomputes it later. If this happens, `max-num-seqs` or `max-model-len` is too generous for your memory.

**The capacity sum, with every lever.** A 24 GB card, a 7B model, 0.5 MB of KV per token, 4k requests:

| Change | KV budget | Per request | Requests |
|---|---|---|---|
| fp16 weights (14 GB) | 24 − 14 − 2 = 8 GB | 4,096 × 0.5 MB = 2 GB | **4** |
| …with max-model-len 16k | 8 GB | 8 GB | **1** |
| …with max-model-len 32k | 8 GB | 16 GB | **0**: vLLM won't start |
| FP8 weights (7 GB) | 15 GB | 2 GB | **7** |
| + FP8 KV cache | 15 GB | 1 GB | **15** |

Prefix caching: with a 1,000-token shared prompt, 500 user tokens and 300 reply tokens, each request needs 1,800 tokens without caching, but only 800 *new* tokens with it. That roughly doubles concurrency.

---

## C6. Loading, serving and benchmarking in practice (P2)

**`nvidia-smi`** is Task Manager for the GPU.
- It shows: driver version and the **highest** CUDA it supports, GPU name (which tells you the generation), memory used/total, GPU utilisation, power draw vs cap, and the processes holding memory.
- **Memory looks about 90% full even when idle**, because vLLM reserves the KV pool at startup. So **memory used can't tell you if the server is busy**. Use **GPU-Util and power**, and vLLM's log line (`Running`, `Waiting`, `GPU KV cache usage`).
- **GPU-Util pinned at 100% with power at the cap means the GPU is saturated.** More requests only lengthen the queue and slow each step. Add capacity instead. GPU-Util only means "some kernel was running", so read it together with power.

**Loading has four stages**, each matching a VRAM term:
1. **Find the files** in the Hugging Face cache (`~/.cache/huggingface/hub/`), downloading if needed.
2. **Build the model** from `config.json`: layers, heads and so on.
3. **Load the weights** (`.safetensors`) into GPU memory. This is the weights term. Size on disk ≈ VRAM for weights.
4. **Reserve the KV cache pool** with whatever `--gpu-memory-utilization` leaves. This is also where vLLM checks that one full-length request fits.

Then `Application startup complete.` means it's ready.

**Serving.** `vllm serve <model> --port 8000 --max-model-len 8192` exposes `/v1/chat/completions`.
- The request's `model` field must equal the served name: the Hugging Face repo ID, unless `--served-model-name` renames it.
- `chat_template_kwargs` passes options into the chat template, for example `enable_thinking: false` to turn off a reasoning trace.
- Without `--max-model-len`, vLLM defaults to the model's trained maximum, which may not fit.

**Benchmarking with `vllm bench serve`.** It's a load generator that plays many users through the normal HTTP API.
- The **single-user run (concurrency 1) is your floor**: no queue, no batching, the best your hardware can do.
- **Long prompts, short replies** show up in **TTFT** (prefill-bound).
- **Short prompts, long replies** show up in **ITL and output throughput** (decode-bound, and batching helps).
- `--ignore-eos` forces every reply to full length, so runs are comparable. `--seed` makes the prompts identical between runs.
- "Successful requests" must equal the number of prompts. Fewer means errors.

**Diagnosing broken containers.** Each fault leaves a different trace:

| Fault | Symptom | Concept |
|---|---|---|
| Container not given the GPU | Exits early: "no CUDA GPUs" | The process must see the GPU |
| Container RAM limit too low | Killed, **exit code 137** (signal 9, out-of-memory killer), often with no traceback | Weights pass through CPU RAM while loading |
| Model too big for VRAM | `CUDA out of memory` while loading weights | Weights sum; fix with quantisation |
| `--max-model-len` too high | **Refuses to start**: max length larger than the KV cache can hold | KV per request; lower the limit |
| Wrong chat template | Starts and answers, but **badly** | A wrong template doesn't error |

Useful commands: `docker compose up` / `down`, `docker ps -a`, `docker logs -f <name>`. `ipc: host` gives the container the shared memory PyTorch needs.

---

## C7. Tuning for TTFT under load (P3, the Gauntlet)

**The core idea:**

```
TTFT = queue wait + prefill time
```

Example: a 1,000-token prompt at 10,000 tokens/s takes 100 ms of prefill. If 20 requests are ahead and a slot opens every 50 ms, the wait is 1,000 ms, so TTFT is 1.1 s. The GPU didn't get slower. **All the extra delay came from waiting in line.** An average TTFT above about 500 ms usually means queueing.

**Diagnose with the `Waiting:` log value:**
- **`Waiting > 0`**: a capacity problem. Free VRAM for KV cache (FP8 weights, FP8 KV, higher `--gpu-memory-utilization`, lower `--max-model-len`) or raise `--max-num-seqs` if it's the cap.
- **`Waiting = 0`** but TTFT is still high: a prefill speed problem. Use FP8 compute, a faster attention backend, chunked-prefill settings or prefix caching.

**Each lever and its effect on TTFT:**

| Lever | Acts on | Effect on TTFT |
|---|---|---|
| `--quantization fp8` | Memory and prefill | Helps: more KV room, and faster FP8 compute on Ada/Hopper |
| `--kv-cache-dtype fp8` | Capacity | Helps under load: about 2× tokens in the pool |
| `--gpu-memory-utilization` 0.92–0.95 | Capacity | Helps a little; too close to 1.0 can OOM |
| Attention backend (FlashAttention, FlashInfer) | Mostly prefill | Can help a lot |
| `--max-num-seqs` | Scheduling | Helps only if it's the limit |
| Chunked prefill / `--max-num-batched-tokens` | Prefill vs decode balance | A larger budget finishes prefill sooner but hurts ITL |
| Prefix caching | Prefill | Big help if prompts share a prefix |
| CUDA graphs | Decode | Keep them on; `--enforce-eager` turns them off and hurts |
| Speculative decoding | Decode | Little or negative under load; it steals compute from prefill |

**What the experiments showed.** **Capacity levers won** (FP8 KV, FP8 weights), because the limit was KV memory. Decode-only levers did nothing or hurt TTFT.

**Operating lessons:**
- Change **one thing at a time**, and test with `vllm bench serve` before going live.
- Restarts take minutes and every request during a restart fails. Plan restarts around the error-rate limit.
- Start with a known-safe baseline. A working config with zero errors beats a clever one that crashes.

---

## C8. Local inference and GGUF (P4, extras)

**Two families of engines:**
- **GPU serving engines** (vLLM, SGLang, TGI, TensorRT-LLM) keep the whole model in VRAM and serve many users. If the model doesn't fit, they refuse to load it.
- **Local-first engines** (llama.cpp and its wrappers: Ollama, LM Studio; plus Apple's MLX) run on CPUs, laptops and consumer GPUs for one user or a few. They can **split layers between GPU and CPU**.

**GGUF is one self-contained file.** It holds the weights, tokeniser, architecture settings and usually the chat template. The template travels with the file, so the template problem mostly disappears. A GGUF is fixed at one quantisation: you pick the precision by downloading the right file, not with a load flag. vLLM's GGUF support is partial, so use AWQ or GPTQ there.

**Reading a quant name**, for example `Q4_K_M`:
- `Q4` is about 4 bits per weight (nominal).
- `K` is the k-quant family, with block-wise scales.
- `_S`, `_M`, `_L` say how many sensitive tensors got higher precision: small, medium, large.
- `IQ` is an importance-matrix quant, best at very low bits.
- `_0` / `_1` are older, simpler schemes. `Q8_0` is near-lossless.
- Real bits per weight are higher than the name (about 4.8 for Q4_K_M), because of per-block scales and higher-precision embeddings.

**Sizing:**

```
file size ≈ parameters × bits per weight ÷ 8
memory    ≈ file size + KV cache + ~0.5–1 GB
```

Example: a 4B model at 4.8 bits is about 2.4 GB. Add about 1.2 GB of KV at 8k context, plus overhead, for about 4–4.5 GB. It fits an 8 GB laptop.

**Offload.** `-ngl N` puts N layers on the GPU and leaves the rest in system RAM. **Decode speed follows the slowest memory** the weights sit in: laptop RAM runs at about 50–100 GB/s, versus 300–1,000+ GB/s on a GPU. Half a 70B model in RAM gives only a few tokens per second. That's usable for one person and useless for a service.

**Ollama = llama.cpp + a model registry + auto-offload + a background service.** It runs on port 11434 and speaks the OpenAI-compatible API, so the *same* curl works against vLLM, Ollama and `llama-server`. Defaults that catch people out:
- The **context length defaults to 4,096 tokens**, and longer prompts are silently truncated from the front.
- **Tags, not Hugging Face names.** `qwen3:4b` is a Q4_K_M file. Use `ollama show` to check.
- The model **unloads after 5 minutes idle**, so the next request pays the load time again.

Use `--verbose` to see timings: `prompt eval rate` is prefill speed, and `eval rate` is decode speed.

**When to use what:**

| Situation | Use |
|---|---|
| Shared server, many users, model fits in VRAM | vLLM (or SGLang / TGI) |
| Model doesn't fit, one user, latency-tolerant | llama.cpp with partial offload |
| Laptop, demos, data that must not leave the machine | Ollama or llama.cpp |
| Apple Silicon | llama.cpp/Ollama via Metal, or MLX |
| Maximum NVIDIA throughput | TensorRT-LLM |

---

# Part D. One-page cheat sheet

**Formulas**

```
Agent token growth     total input ≈ k × n² / 2          (2× steps → ~4× tokens)
Cost per call          (in × in_rate + out × out_rate) / 1,000,000
Cost per success       total cost ÷ accuracy
Next-call estimate     last input + last output + new tool results
Cost cap               ≈ 3 × p95 cost per run
Step cap               where success stops rising + 1–2 margin
Precision              relevant returned ÷ all returned
Recall                 relevant returned ÷ all relevant
Weights                params × bytes/param  (bf16 2, int8 1, int4 0.5)
KV bytes / token       2 × layers × kv_heads × head_dim × bytes
KV per request         tokens × KV bytes/token
VRAM                   weights + KV + ~2 GB  (× gpu-memory-utilization first)
Concurrent requests    floor((VRAM − weights − overhead) ÷ KV per request)
Decode ceiling         time/token ≈ weight bytes ÷ bandwidth
TTFT                   queue wait + prefill
GGUF file              params × bits/weight ÷ 8
```

**Rules to remember**
1. A tool call is a request. The harness decides. Permissions live in code, not prompts.
2. Identity comes from the login, never from the chat.
3. Default-deny unknown permission levels.
4. Every tool call gets exactly one result, straight after its request. `execute_tool` never raises.
5. Only `completed` counts as success.
6. Progress means new information over a window, not new calls. Errors never count.
7. Prompt caching needs an identical prefix: append only, no timestamps at the top.
8. Overlapping ranges mean "no different". Caps apply to the largest call.
9. Memory keeps only what no tool returns. The tool wins over memory, and the prompt must say so.
10. Start with one agent. Split only for a measured reason. Rules first, then measures.
11. Build reports from tool results, never from the model's text.
12. Triage: first wrong item, then context, memory, architecture, poisoning. Ask "was it ever true?"
13. Rule of Two: never combine untrusted input, private data and external action in one component.
14. Validate the shape, authorise against trusted state, then execute.
15. A denial isn't an error, so track the denial rate.
16. Unpriced is not free. Usage from the response is the only billing evidence.
17. Oversight needs information, authority and time. Timeouts fail safe.
18. Set acceptance criteria before testing. An average never offsets a critical failure.
19. Model choice: fits, then runs, then allowed. Licences inherit.
20. Prefill is compute-bound (TTFT). Decode is bandwidth-bound (ITL). If `Waiting > 0`, add capacity.
