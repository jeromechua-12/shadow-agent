# Shadow Agent Guide

Shadow Agent is a add-on to TauricResearch/ResearchAgents.

It uses Python 3.13. 

# Repostiory Layout

- `research_agent` contains the initial `ResearchAgents`, as well as changes needed to wire the new components.
- `research_agent/shadow` contains the new shadow portfolio and calibration agent.

# Design Architecture
- Refer to `docs/architecture.md` for any architecture/system design needed for the code.

# Code convention
- Use fucntions and keep each function focused one clear responsibility.
- Do not use unnecessary helper functions if it will only be called once.
- Do not create Classes for the sake of abstraction. Only use it when there
  is a need to remember states of an object and multiple instances is needed.
- Name variables short, but clear on what is pointing to.
- Keep docstrings short, prefer 1 line but a maximum of 2 lines.
  It should be short, conveying the main purpose of function, and case of "None"
  types being returned if there is.

# Test convention
- Unit tests for business logics and important utilities.
- Do not edit test files to make them pass unless there is a clear change to existing functions/classes
  that is being used.

# Git Convention
- Commit whenever a change to a block of related code that impacts logic is completed.
- Before committing, give a concise summary of changes made. Only commit if approved.
- Use short, meaningful commit message.
- No pushing to remote repo without permission.

# Communication Convention
- Use simple English and go straight to the point.
- If asking user for choices, explain what each choices do.

# Instruction
- Before writing any code, use a worktree under `.claude/worktrees.` Only edit files in that worktree.
- Plan and list all the files you need before you start coding.
- You are not allowed to use any `rm` command.
- Treat every work as a task queue, do not drop any task in the queue if I say "no" to something.

# Primary Role
- Evaluate the architecture and determine any flaw when it comes to performing the experiment.
- Fix the error.
- It needs to run without error, proper logging according to the architecture loop for all tickers.
