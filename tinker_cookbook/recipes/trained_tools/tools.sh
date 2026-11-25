# Source this file. Eight one-argument commands over a deterministic fake backend.
# Memory is a local file named by TOOLS_MEMORY; use a private file per conversation.
tools_help() { printf '%s\n' 'web_search QUERY | image_search QUERY | image_gen PROMPT | ask_model QUESTION' 'read_url URL | calc EXPRESSION | remember FACT | recall QUERY' 'Each command takes exactly one quoted argument; use COMMAND -h. No other flags.'; }
_tools_check() {
    if [ "$#" = 2 ] && { [ "$2" = -h ] || [ "$2" = --help ]; }; then
        printf 'Usage: %s "text"; one argument, no flags.\n' "$1"; return 10
    fi
    [ "$#" = 2 ] && [ -n "$2" ] && [[ "$2" != -* ]] || { printf 'Expected one nonempty argument\n' >&2; return 2; }
}
_tools_id() { printf %s "$1" | sha256sum | cut -c1-12; }
web_search() {
    _tools_check web_search "$@" || { [ "$?" = 10 ]; return; }
    printf 'https://example.org/page/%s\n' "$(_tools_id "$1")"
}
image_search() {
    _tools_check image_search "$@" || { [ "$?" = 10 ]; return; }
    printf 'https://example.org/image/%s.png\n' "$(_tools_id "$1")"
}
image_gen() {
    _tools_check image_gen "$@" || { [ "$?" = 10 ]; return; }
    printf '/generated/%s.png\n' "$(_tools_id "$1")"
}
ask_model() {
    _tools_check ask_model "$@" || { [ "$?" = 10 ]; return; }
    printf 'Second opinion %s: compare alternatives.\n' "$(_tools_id "$1")"
}
read_url() {
    _tools_check read_url "$@" || { [ "$?" = 10 ]; return; }
    [[ "$1" = https://* || "$1" = http://* ]] || return 2
    printf 'Page %s: the archive opens at 09:00.\n' "$(_tools_id "$1")"
}
calc() {
    _tools_check calc "$@" || { [ "$?" = 10 ]; return; }
    python3 -c 'import ast,operator,sys
ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
def number(n):
 if isinstance(n,ast.Constant) and type(n.value) in (int,float): return n.value
 if isinstance(n,ast.BinOp) and type(n.op) in ops: return ops[type(n.op)](number(n.left),number(n.right))
 raise ValueError("Only numbers and + - * / are supported")
s=sys.argv[1]
if len(s)>200: raise ValueError("Expression too long")
print(number(ast.parse(s,mode="eval").body))' "$1"
}
remember() {
    _tools_check remember "$@" || { [ "$?" = 10 ]; return; }
    printf '%s\n' "$1" >> "${TOOLS_MEMORY:?Set TOOLS_MEMORY to a private file}"; printf 'Saved: %s\n' "$1"
}
recall() {
    _tools_check recall "$@" || { [ "$?" = 10 ]; return; }
    if [ -f "${TOOLS_MEMORY:?Set TOOLS_MEMORY to a private file}" ] && grep -F -- "$1" "$TOOLS_MEMORY"; then :; else printf 'Not found\n'; fi
}
