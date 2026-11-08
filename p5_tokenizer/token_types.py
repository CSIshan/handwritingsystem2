from enum import Enum, auto
from dataclasses import dataclass, field


class TokenType(Enum):
    WORD        = auto()   
# regular word, includes attached punctuation
    GHOST       = auto()   
# [GHOST: phrase] — space reserved, nothing drawn, no box
    IMAGE       = auto()   
# [IMAGE src="x" width="Xmm" align="x"]
    STYLE       = auto()   
# [STYLE: neat|casual|tired]
    PEN         = auto()   
# [PEN:color]
    PEN_END     = auto()   
# [/PEN]
    CORRECTION  = auto()
# ~~word~~ or ~~word~~rewrite
    PARAGRAPH   = auto()   
# /p — paragraph break
    NEWLINE     = auto()   
# explicit newline (single \n)
    NEW_PAGE    = auto()   
# /o — force new page
    MARGIN      = auto()   
# [MARGIN: text]
    SIGNATURE   = auto()   
# [SIGNATURE]
    DIGIT       = auto()   
# standalone number "42", "3.14"
    SYMBOL      = auto()   # standalone &, @, %, $, # etc.


@dataclass
class Token:
    type:     TokenType
    value:    str | None        = None   
# main content
    metadata: dict              = field(default_factory=dict)
    raw:      str               = ""     
# original string fragment (for debug)

    def __repr__(self):
        if self.metadata:
            return f"Token({self.type.name}, {self.value!r}, meta={self.metadata})"
        return f"Token({self.type.name}, {self.value!r})"
