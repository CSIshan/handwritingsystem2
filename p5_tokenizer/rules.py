import re

# ── Punctuation attachment rules ─────────────────────────────────────────────

# These stay glued to the PRECEDING word ("hello," not "hello" + ",")
TRAILING_ATTACH = frozenset({ ',', '.', '!', '?', ';', ':' })

# These stay glued to the FOLLOWING word ('"hello' not '"' + 'hello')
LEADING_ATTACH  = frozenset({ '"', '\u201c', '\u2018', '(' })

# These stay glued to the PRECEDING word (closing pairs)
CLOSING_ATTACH  = frozenset({ '"', '\u201d', '\u2019', ')' })

# Standalone symbols — get their own SYMBOL token, not merged into words
STANDALONE_SYMBOLS = frozenset({ '@', '&', '%', '$', '#', '/', '+', '=', '*',
                                  '\\', '|', '^', '~', '<', '>', '[', ']',
                                  '{', '}', '£', '€', '¥' })

# ── Regex patterns ────────────────────────────────────────────────────────────

# Slash markers (must appear as whole tokens after whitespace split)
RE_PARAGRAPH  = re.compile(r'^/p$', re.IGNORECASE)
RE_NEW_PAGE   = re.compile(r'^/o$', re.IGNORECASE)

# Bracket tags — matched BEFORE whitespace splitting
RE_GHOST      = re.compile(r'\[GHOST:\s*(.+?)\]',           re.IGNORECASE)
RE_IMAGE      = re.compile(r'\[IMAGE\s+(.+?)\]',            re.IGNORECASE)
RE_STYLE      = re.compile(r'\[STYLE:\s*(\w+)\]',           re.IGNORECASE)
RE_PEN_OPEN   = re.compile(r'\[PEN:(\w+)\]',                re.IGNORECASE)
RE_PEN_CLOSE  = re.compile(r'\[/PEN\]',                     re.IGNORECASE)
RE_CORRECTION = re.compile(r'~~(.+?)~~(\S*)',               re.IGNORECASE)
RE_MARGIN     = re.compile(r'\[MARGIN:\s*(.+?)\]',          re.IGNORECASE)
RE_SIGNATURE  = re.compile(r'\[SIGNATURE\]',                re.IGNORECASE)

# IMAGE attribute parser
RE_IMG_SRC    = re.compile(r'src=["\']([^"\']+)["\']',      re.IGNORECASE)
RE_IMG_WIDTH  = re.compile(r'width=["\']([^"\']+)["\']',    re.IGNORECASE)
RE_IMG_ALIGN  = re.compile(r'align=["\']([^"\']+)["\']',    re.IGNORECASE)

# Digit detection: pure number, optional decimal, optional leading currency
RE_DIGIT      = re.compile(r'^[£€$¥]?[\d,]+(\.\d+)?[%]?$')

# Digit prefix on word e.g. "42nd" "1st" "3rd"
RE_DIGIT_PREFIX = re.compile(r'^(\d+)([a-zA-Z].*)$')

# Purely standalone symbol (single char in STANDALONE_SYMBOLS after strip)
def is_standalone_symbol(s: str) -> bool:
    return len(s) == 1 and s in STANDALONE_SYMBOLS
