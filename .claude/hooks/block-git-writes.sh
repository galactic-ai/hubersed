#!/bin/bash
# Blocks git/gh WRITE operations. Nikhil commits; the agent never does.
# Wired via .claude/settings.json -> hooks.PreToolUse (matcher "Bash").
# Reads the PreToolUse JSON on stdin; needs `jq` on PATH.

COMMAND=$(jq -r '.tool_input.command // empty')
[ -z "$COMMAND" ] && exit 0

deny() {
  jq -n --arg reason "$1" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: $reason
    }
  }'
  exit 0
}

# git writes
echo "$COMMAND" | grep -qE '(^|[;&|] *)git +(commit|push|add|stage|rm|reset|revert|rebase|merge|cherry-pick|tag|stash)\b' \
  && deny "Blocked: git write. Nikhil stages and commits every change himself. Print the commands for him to run instead. (.claude/hooks/block-git-writes.sh)"

# gh writes
echo "$COMMAND" | grep -qE '(^|[;&|] *)gh +(issue|pr|release|repo|api) +(create|comment|close|edit|merge|delete|reopen|ready|review)\b' \
  && deny "Blocked: gh write. gh is read-only here (view/list/diff only). Draft the text and hand it over. (.claude/hooks/block-git-writes.sh)"

# gh api POST/PATCH/DELETE
echo "$COMMAND" | grep -qE 'gh +api\b.*(-X|--method) *(POST|PATCH|PUT|DELETE)' \
  && deny "Blocked: gh api write. Read-only only. (.claude/hooks/block-git-writes.sh)"

exit 0
