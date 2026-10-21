# Agent skills

`mica/SKILL.md` teaches a coding agent to check for (and set up) a Mica server tuned to the machine, what Mica can do
in code, how to write requests, and how to call it fast. `mica/scripts/mica_setup_check.py` is the read-only check
the skill runs first.

- Claude Code: copy the folder to `~/.claude/skills/mica/` (all projects) or `.claude/skills/mica/` (one project).
- Other agents: add `mica/SKILL.md` to the agent's instructions or project rules, with the `scripts/` folder next to it.
