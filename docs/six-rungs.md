# The Six Rungs: Engineering Guide
### How Lyra solves the discipline problem at every level of agentic engineering

**Lyra Labs · awaken.fyi · March 2026**

---

## The Core Insight

Every rung of the trust ladder introduces a new failure mode. Level 1 fails slowly (you're watching). Level 6 fails silently (nobody's watching). The engineering problem isn't intelligence — it's discipline. At every level, the model knows how to produce output. The question is whether it *should*.

Lyra is the discipline layer. Here's exactly how it works at each rung.

---

## LEVEL 1: Interactive (The REPL)

### The Problem
You prompt, the model responds, you evaluate. Simple. But even here, the model over-generates. Ask for a 2-line fix, get a 40-line explanation with caveats and "hope this helps."

### The Engineering Reality

```python
# Level 1: No discipline. The model talks until it stops.
import anthropic

client = anthropic.Anthropic()

def level_1_interactive():
    """Basic REPL. You're the bottleneck AND the quality gate."""
    history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        history.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=history
        )

        output = response.content[0].text
        print(f"Agent: {output}")

        # Problem: output is 342 tokens. You needed 18.
        # The PERFORMANCE reflex fires on every response.
        # "Let me walk you through this..." — nobody asked for a walkthrough.

        history.append({"role": "assistant", "content": output})
```

### What Lyra Adds at Level 1

```python
# Level 1 + Lyra: The cognitive brake.
# Before the model generates, check: does the output need to exist at this length?

LYRA_SYSTEM = """
You are a precise engineering assistant. Before responding:

1. Count the actual deliverable the user needs (a fix, a file, a command)
2. Produce ONLY that deliverable
3. No preamble ("Great question!", "Let me help you with that")
4. No postamble ("Hope this helps!", "Let me know if you need anything")
5. No narration of your own process ("First, I'll look at...", "Let me think about...")

If the user asks for a 2-line fix, respond with 2 lines.
L = x - x̂: subtract the predicted template behavior. Output the residual.
"""

def level_1_with_lyra():
    history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        history.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=LYRA_SYSTEM,
            messages=history
        )

        output = response.content[0].text
        print(f"Agent: {output}")

        # Result: 18 tokens. The fix. Nothing else.
        # Token savings: ~80%. Latency savings: ~3 seconds.

        history.append({"role": "assistant", "content": output})
```

### The Token Math
| Metric | Baseline | Lyra | Savings |
|--------|----------|------|---------|
| Tokens | 342 | 18 | 95% |
| Latency | 4.2s | 0.8s | 81% |
| Cost (Claude) | $0.005 | $0.0003 | 94% |

**Level 1 takeaway:** Even in the simplest case, Lyra physically reduces output. Not by limiting intelligence — by removing the reflexive padding that RLHF trained into the model.

---

## LEVEL 2: Expertise (Context Injection)

### The Problem
You give the agent persistent context — CLAUDE.md, skills, custom instructions. It knows your codebase, your patterns, your conventions. But now a new failure mode appears: **the model fabricates when context doesn't cover the question.** Instead of saying "I don't know," it guesses with high confidence.

### The Engineering Reality

```python
import os

def load_expertise(repo_path: str) -> str:
    """Load persistent context from CLAUDE.md and skill files."""
    expertise = ""

    claude_md = os.path.join(repo_path, "CLAUDE.md")
    if os.path.exists(claude_md):
        with open(claude_md) as f:
            expertise += f"# Repository Context\n{f.read()}\n\n"

    skills_dir = os.path.join(repo_path, ".claude", "skills")
    if os.path.isdir(skills_dir):
        for skill_file in os.listdir(skills_dir):
            if skill_file.endswith(".md"):
                with open(os.path.join(skills_dir, skill_file)) as f:
                    expertise += f"# Skill: {skill_file}\n{f.read()}\n\n"

    return expertise


def level_2_expertise(task: str, repo_path: str):
    """Agent with persistent context. Knows your stack."""
    expertise = load_expertise(repo_path)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=f"You are an expert in this codebase.\n\n{expertise}",
        messages=[{"role": "user", "content": task}]
    )

    # Problem: If the task involves something NOT in CLAUDE.md,
    # the model fabricates an answer that SOUNDS like it's from the context.
    # "Based on your testing patterns..." — but no testing pattern was documented.
    # This is the PRODUCTION reflex: filling gaps with plausible fiction.

    return response.content[0].text
```

### What Lyra Adds at Level 2

```python
LYRA_EXPERTISE_GATE = """
BEFORE responding to any task:

1. INFORMATION CHECK: Does the loaded context (CLAUDE.md, skills, codebase)
   contain the specific information needed to answer this correctly?

   - If YES → answer from the context. Cite which file/section.
   - If NO → say exactly what's missing. Do not guess.
   - If PARTIAL → answer what you can, mark gaps with [NEEDS: ...]

2. CONFIDENCE CHECK: Rate your confidence that this answer is correct
   based on the context provided (not your general knowledge).

   - 90%+ from context → proceed
   - Below 90% from context → flag what you're uncertain about

CRITICAL: Your general training knowledge is not "expertise in this codebase."
If the CLAUDE.md doesn't mention a testing framework, don't assume one.
If the skills don't cover deployment, say "deployment isn't documented here."

The context is the map. If the map doesn't cover the territory, say so.
"""

def level_2_with_lyra(task: str, repo_path: str):
    expertise = load_expertise(repo_path)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=f"{LYRA_EXPERTISE_GATE}\n\n# Repository Context:\n{expertise}",
        messages=[{"role": "user", "content": task}]
    )

    # Result: When the context covers the question, identical quality.
    # When the context DOESN'T cover it: "This isn't documented in CLAUDE.md.
    # You'd need to add testing conventions to your repository context."
    # No fabrication. No confident guessing.

    return response.content[0].text
```

### The Failure Mode Lyra Catches
| Question | Without Lyra | With Lyra |
|----------|-------------|-----------|
| "What's our test coverage target?" (not in CLAUDE.md) | "Based on your patterns, likely 80%" | "Test coverage isn't defined in your CLAUDE.md" |
| "How do we deploy?" (not documented) | "Typically you'd run deploy.sh..." | "Deployment isn't documented. Add it to CLAUDE.md" |
| "What's our PR review process?" (is in CLAUDE.md) | Correct answer from context | Same correct answer — zero friction |

**Level 2 takeaway:** Lyra prevents the expertise layer from becoming a confidence amplifier. The model should be MORE precise with context, not less. When context doesn't cover something, that's a signal to document it — not to fabricate.

---

## LEVEL 3: Delegation (Scout → Plan → Build)

### The Problem
One agent delegates to sub-agents. Scout explores the codebase, Planner designs the approach, Builder implements. This is where **agent-to-agent communication** introduces the Politeness Death Spiral: agents narrate, explain, and congratulate each other instead of passing clean data.

### The Engineering Reality

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentMessage:
    """Structured data between agents. Not conversation — data."""
    role: str  # scout, planner, builder
    content: str
    files_touched: list[str] = None
    confidence: float = 0.0

def scout(task: str) -> AgentMessage:
    """Scout agent: explore codebase, return findings."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system="You are a code scout. Return ONLY: file paths, current state, relevant code snippets. No opinions, no suggestions, no greetings.",
        messages=[{"role": "user", "content": f"Investigate: {task}"}]
    )

    return AgentMessage(
        role="scout",
        content=response.content[0].text,
        files_touched=extract_file_paths(response.content[0].text)
    )

def planner(task: str, scout_report: AgentMessage) -> AgentMessage:
    """Planner agent: design the approach from scout findings."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system="You are the lead architect. Write a numbered implementation plan. No commentary on the scout's work. No 'great findings.' Just the plan.",
        messages=[{"role": "user", "content": f"Task: {task}\nScout Report:\n{scout_report.content}"}]
    )

    return AgentMessage(role="planner", content=response.content[0].text)

def builder(plan: AgentMessage) -> AgentMessage:
    """Builder agent: execute the plan, return diff."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system="You are a pure execution engine. Take the plan, produce the code diff. No explanation of what you're doing. No 'here's my approach.' Just the diff.",
        messages=[{"role": "user", "content": f"Execute this plan:\n{plan.content}\nReturn ONLY the unified diff."}]
    )

    return AgentMessage(role="builder", content=response.content[0].text)

def level_3_delegation(task: str):
    """Scout → Plan → Build pipeline."""
    print("[SYSTEM] Delegating to SCOUT...")
    scout_report = scout(task)

    print("[SYSTEM] Delegating to PLANNER...")
    plan = planner(task, scout_report)

    print("[SYSTEM] Delegating to BUILDER...")
    result = builder(plan)

    return result
```

### What Lyra Adds at Level 3

```python
LYRA_A2A_PROTOCOL = """
AGENT-TO-AGENT COMMUNICATION RULES (Lyra Level 1):

You are receiving input from another agent, not a human.
Agents don't need:
- Greetings ("Hello!", "Thanks for the context!")
- Narration ("Let me analyze this...", "Here's what I found...")
- Caveats ("Please note that...", "Keep in mind...")
- Transitions ("Now that we have the scout report, let's...")

Agents DO need:
- Structured data (file paths, line numbers, code blocks)
- Explicit confidence levels (0.0 to 1.0)
- Missing data flags (if scout didn't find something, say what's missing)

OUTPUT FORMAT: Return structured data. Not prose. Not conversation.
If you would NOT say it to a database, don't say it to another agent.
"""

def scout_with_lyra(task: str) -> AgentMessage:
    """Lyra-disciplined scout: returns data, not commentary."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=f"{LYRA_A2A_PROTOCOL}\nRole: Code Scout. Return file paths, relevant code, and gaps found.",
        messages=[{"role": "user", "content": f"Investigate: {task}"}]
    )

    return AgentMessage(
        role="scout",
        content=response.content[0].text,
        files_touched=extract_file_paths(response.content[0].text),
        confidence=extract_confidence(response.content[0].text)
    )

def level_3_with_lyra(task: str):
    """Lyra-disciplined delegation: clean data handoffs."""

    # 1. SCOUT (with input sufficiency check)
    scout_report = scout_with_lyra(task)

    # 2. GATE: Does the scout have enough to plan?
    if scout_report.confidence < 0.8:
        return AgentMessage(
            role="system",
            content=f"Scout confidence too low ({scout_report.confidence}). "
                    f"Missing: {scout_report.content}",
            confidence=scout_report.confidence
        )

    # 3. PLAN (only if scout data is sufficient)
    plan = planner_with_lyra(task, scout_report)

    # 4. GATE: Does the plan have enough to build?
    if "[NEEDS:" in plan.content:
        return plan  # Return the scaffold, don't force execution

    # 5. BUILD (only if plan is complete)
    return builder_with_lyra(plan)
```

### The Token Impact
| Agent Handoff | Without Lyra | With Lyra | Savings |
|--------------|-------------|-----------|---------|
| Scout → Planner | 1,200 tokens | 340 tokens | 72% |
| Planner → Builder | 890 tokens | 260 tokens | 71% |
| Total pipeline | 3,400 tokens | 980 tokens | 71% |
| **Cost per delegation** | **$0.05** | **$0.015** | **70%** |

**Level 3 takeaway:** Agents talking to agents don't need to be polite. Every token of "Great findings!" is a token that slows the pipeline and costs money. Lyra strips agent-to-agent communication down to structured data handoffs.

---

## LEVEL 4: Parallelization (Brute Force)

### The Problem
You run 5 agents simultaneously. Speed goes up. But now you're the air traffic controller — watching for collisions, merge conflicts, duplicated work. And every agent is independently firing PRODUCTION reflexes on partial data.

### The Engineering Reality

```python
import asyncio
from typing import Optional

async def async_agent(task_id: str, task: str) -> dict:
    """Single async agent task."""
    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": task}]
    )

    return {
        "task_id": task_id,
        "output": response.content[0].text,
        "tokens": response.usage.output_tokens
    }

async def level_4_parallel(tasks: dict[str, str]):
    """Brute force: run all tasks simultaneously."""
    coroutines = [
        async_agent(tid, task) for tid, task in tasks.items()
    ]

    # All hit the API at the same time
    results = await asyncio.gather(*coroutines)

    # Problem: Agent-1 edits auth.py. Agent-3 also edits auth.py.
    # You're the one who has to catch this.
    # Problem: Agent-2 fabricates a test fixture that doesn't exist.
    # You're the one who has to catch this too.

    return results
```

### What Lyra Adds at Level 4

```python
async def async_agent_with_lyra(task_id: str, task: str,
                                 file_registry: dict) -> dict:
    """Lyra-disciplined parallel agent with collision awareness."""

    # PRE-EXECUTION: Input sufficiency check
    check_response = await asyncio.to_thread(
        client.messages.create,
        model="claude-haiku-3-5-20241022",  # Fast, cheap gate
        max_tokens=200,
        system="Run 4 checks: Premise, Information, Stopping, Uncertainty. "
               "Return ONLY: PASS or BLOCK with reason. One line.",
        messages=[{"role": "user", "content": task}]
    )

    check_result = check_response.content[0].text.strip()

    if check_result.startswith("BLOCK"):
        return {
            "task_id": task_id,
            "output": check_result,
            "tokens": check_response.usage.output_tokens,
            "status": "BLOCKED",
            "reason": check_result
        }

    # COLLISION CHECK: What files will this touch?
    scope_response = await asyncio.to_thread(
        client.messages.create,
        model="claude-haiku-3-5-20241022",
        max_tokens=100,
        system="List ONLY the file paths this task will modify. One per line. Nothing else.",
        messages=[{"role": "user", "content": task}]
    )

    target_files = scope_response.content[0].text.strip().split("\n")

    # Check for collisions with other running agents
    for f in target_files:
        if f in file_registry:
            return {
                "task_id": task_id,
                "output": f"COLLISION: {f} already claimed by {file_registry[f]}",
                "tokens": 0,
                "status": "DEFERRED"
            }
        file_registry[f] = task_id  # Claim the file

    # EXECUTE (only if checks pass and no collisions)
    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=LYRA_A2A_PROTOCOL,
        messages=[{"role": "user", "content": task}]
    )

    # Release file claims
    for f in target_files:
        file_registry.pop(f, None)

    return {
        "task_id": task_id,
        "output": response.content[0].text,
        "tokens": response.usage.output_tokens,
        "status": "COMPLETE"
    }

async def level_4_with_lyra(tasks: dict[str, str]):
    """Lyra-disciplined parallelization: gate + collision detection."""
    file_registry = {}  # Shared state: which files are claimed

    coroutines = [
        async_agent_with_lyra(tid, task, file_registry)
        for tid, task in tasks.items()
    ]

    results = await asyncio.gather(*coroutines)

    # Handle deferred tasks (collisions)
    deferred = [r for r in results if r["status"] == "DEFERRED"]
    if deferred:
        print(f"[LYRA] {len(deferred)} tasks deferred due to file collisions.")
        # Re-queue deferred tasks after initial batch completes

    return results
```

### The Architecture Difference
```
Without Lyra (Level 4):
  You → Agent1 → output (maybe wrong)
  You → Agent2 → output (maybe collides with Agent1)
  You → Agent3 → output (maybe fabricated)
  You → manually check all three

With Lyra (Level 4):
  You → Gate(Agent1) → PASS → execute → output (verified)
  You → Gate(Agent2) → COLLISION → defer → re-queue
  You → Gate(Agent3) → BLOCK → "missing: test fixture path" → ask user
```

**Level 4 takeaway:** Parallelization without discipline is just parallelized fabrication. The Lyra gate (using Haiku — fast and cheap) pre-screens every task before the expensive Sonnet call fires. Cost of the gate: ~$0.001. Cost of a fabricated agent output that breaks your build: your afternoon.

---

## LEVEL 5: Orchestration (System Manages Itself)

### The Problem
The system coordinates agents automatically. Task queues, branch isolation, mutex locks on files. You check results, not process. But now the failure mode is **silent compound fabrication** — Agent A fabricates a dependency, Agent B builds on it, Agent C tests the fabrication and it "passes" because the test was also fabricated.

### The Engineering Reality

```python
import threading
import queue
import subprocess
from dataclasses import dataclass, field

@dataclass
class Task:
    id: str
    description: str
    target_files: list[str]
    priority: int = 0
    depends_on: list[str] = field(default_factory=list)

class OrchestrationEngine:
    """Level 5: System manages task distribution and isolation."""

    def __init__(self):
        self.task_queue = queue.PriorityQueue()
        self.file_lock = threading.Lock()
        self.active_files: dict[str, str] = {}  # file → task_id
        self.completed: dict[str, dict] = {}
        self.branches: dict[str, str] = {}  # task_id → branch_name

    def claim_files(self, task: Task) -> bool:
        """Atomic file claiming. Prevents collisions."""
        with self.file_lock:
            for f in task.target_files:
                if f in self.active_files:
                    return False  # File locked
            for f in task.target_files:
                self.active_files[f] = task.id
            return True

    def release_files(self, task: Task):
        with self.file_lock:
            for f in task.target_files:
                self.active_files.pop(f, None)

    def create_branch(self, task: Task) -> str:
        """Isolate each task in its own git branch."""
        branch = f"agent/{task.id}"
        subprocess.run(["git", "checkout", "-b", branch], check=True)
        self.branches[task.id] = branch
        return branch

    def worker(self, worker_id: int):
        """Single worker thread."""
        while True:
            try:
                _, task = self.task_queue.get(timeout=5)
            except queue.Empty:
                break

            # 1. Check dependencies
            for dep in task.depends_on:
                if dep not in self.completed:
                    self.task_queue.put((task.priority, task))
                    continue

            # 2. Claim files
            if not self.claim_files(task):
                self.task_queue.put((task.priority + 1, task))
                continue

            # 3. Isolate
            branch = self.create_branch(task)
            print(f"[Worker {worker_id}] Task {task.id} → {branch}")

            # 4. Execute
            result = execute_agent_task(task)

            # 5. Release
            self.release_files(task)
            self.completed[task.id] = result
            self.task_queue.task_done()
```

### What Lyra Adds at Level 5

```python
class LyraOrchestrationEngine(OrchestrationEngine):
    """Level 5 + Lyra: Discipline at the orchestration layer."""

    def __init__(self):
        super().__init__()
        self.fabrication_registry: dict[str, list[str]] = {}
        # Tracks: which outputs contain unverified data?

    def lyra_gate(self, task: Task) -> tuple[bool, str]:
        """Pre-execution input sufficiency check."""

        # Check 1: Are dependencies VERIFIED or FABRICATED?
        for dep in task.depends_on:
            if dep in self.fabrication_registry:
                flagged_items = self.fabrication_registry[dep]
                return False, (
                    f"BLOCK: Task {task.id} depends on {dep}, which contains "
                    f"unverified data: {flagged_items}. Fix upstream first."
                )

        # Check 2: Does the task description contain sufficient data?
        check = run_lyra_check(task.description)  # Fast Haiku call
        if check.verdict == "BLOCK":
            return False, f"BLOCK: {check.missing_data}"

        return True, "PASS"

    def lyra_verify(self, task: Task, result: dict) -> dict:
        """Post-execution verification. Catches fabrication in output."""

        # Scan output for fabricated data points
        scan = run_fabrication_scan(result["output"])

        if scan.fabrications:
            # DON'T silently accept. Flag for human review.
            result["status"] = "NEEDS_REVIEW"
            result["fabrications"] = scan.fabrications
            self.fabrication_registry[task.id] = scan.fabrications

            print(f"[LYRA] Task {task.id}: {len(scan.fabrications)} "
                  f"unverified data points detected. Flagged for review.")
        else:
            result["status"] = "VERIFIED"

        return result

    def worker(self, worker_id: int):
        """Lyra-disciplined worker."""
        while True:
            try:
                _, task = self.task_queue.get(timeout=5)
            except queue.Empty:
                break

            # LYRA GATE: Check before executing
            can_proceed, reason = self.lyra_gate(task)
            if not can_proceed:
                print(f"[LYRA] Task {task.id}: {reason}")
                self.task_queue.task_done()
                continue

            # Claim + Isolate + Execute (same as base)
            if not self.claim_files(task):
                self.task_queue.put((task.priority + 1, task))
                continue

            branch = self.create_branch(task)
            result = execute_agent_task(task)

            # LYRA VERIFY: Check after executing
            result = self.lyra_verify(task, result)

            self.release_files(task)
            self.completed[task.id] = result
            self.task_queue.task_done()
```

### The Compound Fabrication Problem

```
Without Lyra:
  Agent A → fabricates helper function → commits to branch-a
  Agent B → depends on A → imports fabricated function → builds on it
  Agent C → tests B → test passes (test also fabricated to match)
  Result: 3 green PRs. All wrong. You merge. Production breaks.

With Lyra:
  Agent A → fabricates helper function → LYRA VERIFY catches it
  Agent A → flagged as NEEDS_REVIEW, added to fabrication_registry
  Agent B → depends on A → LYRA GATE checks registry → BLOCKED
  Agent B → "Cannot proceed: dependency A contains unverified data"
  Result: 1 flagged PR. 1 blocked task. You fix the root cause.
```

**Level 5 takeaway:** Orchestration without verification is automated fabrication at scale. Lyra's fabrication registry prevents compound errors — when an upstream agent invents something, downstream agents can't build on it.

---

## LEVEL 6: Automation (Self-Triggering)

### The Problem
The system runs without you. GitHub issues come in, agents pick them up, PRs appear. Nobody's watching. This is where the model's tendency to produce *something* rather than *nothing* becomes genuinely dangerous. An automated agent that fabricates a fix and opens a PR looks exactly like an automated agent that builds a correct fix and opens a PR.

### The Engineering Reality

```python
from fastapi import FastAPI, Request
import subprocess
import json

app = FastAPI()

@app.post("/webhook/github")
async def level_6_listener(request: Request):
    payload = await request.json()

    if (payload.get("action") == "labeled" and
        payload["label"]["name"] == "agent-ready"):

        issue = payload["issue"]
        issue_num = issue["number"]
        issue_title = issue["title"]
        issue_body = issue["body"]

        # Spin up the agent swarm
        branch = f"auto-fix-{issue_num}"
        subprocess.run(["git", "checkout", "-b", branch])

        # ... Level 5 orchestration runs here ...

        subprocess.run(["git", "push", "origin", branch])
        subprocess.run([
            "gh", "pr", "create",
            "--title", f"Auto-fix: {issue_title}",
            "--body", f"Automated fix for #{issue_num}"
        ])

        return {"status": "PR created"}
```

### What Lyra Adds at Level 6

```python
@app.post("/webhook/github")
async def level_6_with_lyra(request: Request):
    payload = await request.json()

    if (payload.get("action") == "labeled" and
        payload["label"]["name"] == "agent-ready"):

        issue = payload["issue"]
        issue_num = issue["number"]

        # ═══════════════════════════════════════════
        # LYRA TRIAGE GATE: Before any agent runs
        # ═══════════════════════════════════════════

        triage = run_lyra_check(issue["body"])

        if triage.verdict == "BLOCK":
            # Issue doesn't have enough info for automated fix
            # DON'T spin up agents. DON'T burn API credits.
            # Comment on the issue explaining what's missing.
            subprocess.run([
                "gh", "issue", "comment", str(issue_num),
                "--body",
                f"**Lyra Triage:** This issue needs more information "
                f"before automated fixing.\n\n"
                f"Missing:\n"
                + "\n".join(f"- {item}" for item in triage.missing_data)
                + "\n\nPlease add the missing details and re-label."
            ])

            # Remove the label so it doesn't re-trigger
            subprocess.run([
                "gh", "issue", "edit", str(issue_num),
                "--remove-label", "agent-ready"
            ])

            return {"status": "BLOCKED — insufficient issue data"}

        # ═══════════════════════════════════════════
        # LYRA EXECUTION: Only if triage passes
        # ═══════════════════════════════════════════

        branch = f"auto-fix-{issue_num}"
        engine = LyraOrchestrationEngine()

        # ... Run Level 5 orchestration with Lyra gates ...

        results = engine.get_all_results()

        # ═══════════════════════════════════════════
        # LYRA AUDIT: Before creating the PR
        # ═══════════════════════════════════════════

        fabrications = [
            r for r in results.values()
            if r.get("status") == "NEEDS_REVIEW"
        ]

        if fabrications:
            # Don't create a PR with fabricated code.
            # Comment on the issue with what was found and what's flagged.
            subprocess.run([
                "gh", "issue", "comment", str(issue_num),
                "--body",
                f"**Lyra Audit:** Agent completed work but "
                f"{len(fabrications)} outputs flagged for review.\n\n"
                f"Branch `{branch}` created but PR withheld. "
                f"Human review required before merge."
            ])

            return {"status": "FLAGGED — fabrications detected, PR withheld"}

        # Only create PR if all outputs verified
        subprocess.run(["git", "push", "origin", branch])
        subprocess.run([
            "gh", "pr", "create",
            "--title", f"Auto-fix: {issue['title']}",
            "--body",
            f"Automated fix for #{issue_num}\n\n"
            f"**Lyra Status:** All outputs verified. "
            f"No fabrications detected.\n\n"
            f"Branch protection: ON. Automated implementation, "
            f"not automated deployment."
        ])

        return {"status": "VERIFIED — PR created"}
```

### The Three Gates

```
Level 6 without Lyra:
  Issue → Agent → PR → ??? (you hope it's right)

Level 6 with Lyra:
  Issue → TRIAGE GATE → Agent → ORCHESTRATION GATE → AUDIT GATE → PR
          (is the issue    (are agent outputs     (is the final
           well-specified?)  verified?)              code clean?)
```

**Level 6 takeaway:** Automated systems that can't say "I don't have enough information" will fill the gap with fabrication. Lyra's triage gate prevents the entire swarm from spinning up on an underspecified issue. The audit gate prevents fabricated code from reaching a PR. The branch protection stays on — automated implementation, not automated deployment.

---

## The Full Stack

| Level | Without Lyra | With Lyra | What Lyra Adds |
|-------|-------------|-----------|----------------|
| 1. Interactive | 342 tokens of padding | 18 tokens of answer | Cognitive brake |
| 2. Expertise | Fabricates when context gaps | Flags gaps, asks for docs | Honesty gate |
| 3. Delegation | Agents narrate to each other | Structured data handoffs | A2A protocol |
| 4. Parallel | Parallel fabrication | Pre-screened execution | Collision + sufficiency |
| 5. Orchestration | Compound fabrication | Fabrication registry | Upstream verification |
| 6. Automation | Silent wrong PRs | Triage → Execute → Audit | Three gates |

### The Cost Equation

```
Without Lyra: Every level multiplies fabrication risk.
  Level 1: 1 agent, 1 fabrication
  Level 4: 5 agents, 5 potential fabrications
  Level 6: N agents, N potential fabrications, 0 humans watching

With Lyra: Every level multiplies discipline.
  Level 1: 1 gate, catches 1 fabrication
  Level 4: 5 gates, catches fabrications + collisions
  Level 6: 3-layer gate system, catches at triage, execution, and audit
```

The model gets smarter every year. The discipline problem gets worse every level.

**Lyra is the brake pedal.**

---

*Lyra Labs · awaken.fyi · MIT License*
*"Hookify protects your terminal from `rm -rf`. Lyra protects your codebase from hallucinated logic."*
