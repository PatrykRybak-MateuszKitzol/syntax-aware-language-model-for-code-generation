class _Tags:
    QUOTATION_1= "[QUOT_1]"
    QUOTATION_2= "[QUOT_2]"

    DEF= "[DEF]"
    ASYNC_DEF= "[ASYNC_DEF]"
    CLASS= "[CLASS]"
    IF= "[IF]"
    ELIF= "[ELIF]"
    ELSE= "[ELSE]"

    FOR= "[FOR]"
    ASYNC_FOR= "[ASYNC_FOR]"
    WHILE= "[WHILE]"

    WITH= "[WITH]"
    ASYNC_WITH= "[ASYNC_WITH]"

    TRY= "[TRY]"
    EXCEPT= "[EXCEPT]"
    EXCEPT_STAR= "[EXCEPT*]"
    FINALLY= "[FINALLY]"

    RAISE= "[RAISE]"

    ASSERT= "[ASSERT]"

    BREAK= "[BREAK]"
    CONTINUE= "[CONTINUE]"

    PASS= "[PASS]"

    DEL= "[DEL]"

    RETURN= "[RETURN]"
    YIELD= "[YIELD]"
    YIELD_FROM= "[YIELD_FROM]"

    FROM= "[FROM]"
    AS= "[AS]"

    BLOCK= "[BLOCK]"

    NAMED_EXPR= "[NAMED_EXPR]"
    ASSIGN= "[ASSIGN]"

    AWAIT= "[AWAIT]"

    JOINEDSTR= "[F]"
    U= "[U]"

    ADD= "[ADD]"
    SUB= "[SUB]"
    MULT= "[MULT]"
    MATMULT= "[MATMULT]"
    DIV= "[DIV]"
    MOD= "[MOD]"
    LSHIFT= "[LSHIFT]"
    RSHIFT= "[RSHIFT]"
    BITOR= "[BITOR]"
    BITXOR= "[BITXOR]"
    BITAND= "[BITAND]"
    FLOORDIV= "[FLOORDIV]"
    POW= "[POW]"

    EQ= "[EQ]"
    NOTEQ= "[NOT_EQ]"
    LT= "[LT]"
    LTE= "[LT_E]"
    GT= "[GT]"
    GTE= "[GT_E]"
    IS= "[IS]"
    ISNOT= "[IS_NOT]"
    IN= "[IN]"
    NOTIN= "[NOT_IN]"
    AND= "[AND]"
    OR= "[OR]"

    NOT= "[NOT]"
    INVERT= "[INVERT]"
    UADD= "[UADD]"
    USUB= "[USUB]"

    INDENT= "[INDENT]"
    DEDENT= "[DEDENT]"
    NEW_LINE= "[NEW_LINE]"
    COMMA= "[COMMA]"

    DELIMIT_1_L= "[DELIMIT_1_L]"
    DELIMIT_1_R= "[DELIMIT_1_R]"
    DELIMIT_2_L= "[DELIMIT_2_L]"
    DELIMIT_2_R= "[DELIMIT_2_R]"
    DELIMIT_3_L= "[DELIMIT_3_L]"
    DELIMIT_3_R= "[DELIMIT_3_R]"

    ADD_ASSIGN= "[ADD_ASSIGN]"
    SUB_ASSIGN= "[SUB_ASSIGN]"
    MULT_ASSIGN= "[MULT_ASSIGN]"
    MATMULT_ASSIGN= "[MATMULT_ASSIGN]"
    DIV_ASSIGN= "[DIV_ASSIGN]"
    MOD_ASSIGN= "[MOD_ASSIGN]"
    LSHIFT_ASSIGN= "[LSHIFT_ASSIGN]"
    RSHIFT_ASSIGN= "[RSHIFT_ASSIGN]"
    BITOR_ASSIGN= "[BITOR_ASSIGN]"
    BITXOR_ASSIGN= "[BITXOR_ASSIGN]"
    BITAND_ASSIGN= "[BITAND_ASSIGN]"
    FLOORDIV_ASSIGN= "[FLOORDIV_ASSIGN]"
    POW_ASSIGN= "[POW_ASSIGN]"

    DICT_COLON= "[DICT_COLON]"
    UNPACK= "[UNPACK]"

    ASYNC_FOR_COMP= "[ASYNC_FOR_COMP]"
    FOR_COMP= "[FOR_COMP]"
    IF_COMP= "[IF_COMP]"
    ELSE_COMP= "[ELSE_COMP]"
    IN_COMP= "[IN_COMP]"

    TUPLE_L= "[TUPLE_L]"
    TUPLE_R= "[TUPLE_R]"

    SLICE= "[SLICE]"

    ELLIPSIS= "[ELLIPSIS]"

    REFERENCE= "[REFERENCE]"

    DOT= "[DOT]"

    MATCH= "[MATCH]"
    MATCH_CASE= "[CASE]"
    MATCH_DEFAULT= "[MATCH_DEFAULT]"
    MATCH_AS= "[MATCH_AS]"
    MATCH_GUARD= "[MATCH_GUARD]"
    MATCH_OR= "[MATCH_OR]"
    MATCH_STAR= "[MATCH_STAR]"

    POSONLYARGS= "[POSONLYARGS]"

    KWARGS= "[KWARGS]"

    LAMBDA= "[LAMBDA]"
    LAMBDA_BODDY= "[LAMBDA_BODDY]"

    AS_ITEM= "[AS_ITEM]"

    GUARD= "[GUARD]"

    INF= "[INF]"
    NAN= "[NAN]"

    SEMANTIC_START= "[SEMANTIC_START]"
    SEMANTIC_END= "[SEMANTIC_END]"



tag_to_symbol = {
    _Tags.QUOTATION_1: "'",
    _Tags.QUOTATION_2: '"',
    
    _Tags.DEF: "def ",
    _Tags.ASYNC_DEF: "async def ",
    _Tags.CLASS: "class ",
    _Tags.IF: "if ",
    _Tags.ELIF: "elif ",
    _Tags.ELSE: "else",

    _Tags.FOR: "for ",
    _Tags.ASYNC_FOR: "async for ",
    _Tags.WHILE: "while ",

    _Tags.WITH: "with ",
    _Tags.ASYNC_WITH: "async with ",

    _Tags.TRY: "try",
    _Tags.EXCEPT: "except ",
    _Tags.EXCEPT_STAR: "except* ",
    _Tags.FINALLY: "finally",

    _Tags.RAISE: "raise ",

    _Tags.ASSERT: "assert ",

    _Tags.BREAK: "break",
    _Tags.CONTINUE: "continue",

    _Tags.PASS: "pass",

    _Tags.DEL: "del ",

    _Tags.RETURN: "return ",
    _Tags.YIELD: "yield ",
    _Tags.YIELD_FROM: "yield from ",

    _Tags.FROM: " from ",
    _Tags.AS: " as ",

    _Tags.BLOCK: ":",

    _Tags.ASSIGN: " = ",
    _Tags.NAMED_EXPR: " := ",

    _Tags.AWAIT: "await ",

    _Tags.JOINEDSTR: "f",
    _Tags.U: "u",

    _Tags.ADD: " + ",
    _Tags.SUB: " - ",
    _Tags.MULT: " * ",
    _Tags.MATMULT: " @ ",
    _Tags.DIV: " / ",
    _Tags.MOD: " % ",
    _Tags.LSHIFT: " << ",
    _Tags.RSHIFT: " >> ",
    _Tags.BITOR: " | ",
    _Tags.BITAND: " & ",
    _Tags.BITXOR: " ^ ",
    _Tags.FLOORDIV: " // ",
    _Tags.POW: "**",

    _Tags.EQ: " == ",
    _Tags.NOTEQ: " != ",
    _Tags.GT: " > ",
    _Tags.GTE: " >= ",
    _Tags.LT: " < ",
    _Tags.LTE: " <= ",
    _Tags.IS: " is ",
    _Tags.ISNOT: " is not ",
    _Tags.IN: " in ",
    _Tags.NOTIN: " not in ",
    _Tags.AND: " and ",
    _Tags.OR: " or ",

    _Tags.NOT : "not ",
    _Tags.INVERT: "~",
    _Tags.UADD: "+",
    _Tags.USUB: "-",

    _Tags.INDENT: "    ",
    _Tags.DEDENT: "",
    _Tags.NEW_LINE: "\n",
    _Tags.COMMA: ", ",

    _Tags.DELIMIT_1_L: "(",
    _Tags.DELIMIT_1_R: ")",
    _Tags.DELIMIT_2_L: "[",
    _Tags.DELIMIT_2_R: "]",
    _Tags.DELIMIT_3_L: "{",
    _Tags.DELIMIT_3_R: "}",

    _Tags.ADD_ASSIGN: " += ",
    _Tags.SUB_ASSIGN: " -= ",
    _Tags.MULT_ASSIGN: " *= ",
    _Tags.MATMULT_ASSIGN: " @= ",
    _Tags.DIV_ASSIGN: " /= ",
    _Tags.MOD_ASSIGN: " %= ",
    _Tags.LSHIFT_ASSIGN: " <<= ",
    _Tags.RSHIFT_ASSIGN: " >>= ",
    _Tags.BITOR_ASSIGN: " |= ",
    _Tags.BITXOR_ASSIGN: " ^= ",
    _Tags.BITAND_ASSIGN: " &= ",
    _Tags.FLOORDIV_ASSIGN: " //= ",
    _Tags.POW_ASSIGN: " **= ",

    _Tags.DICT_COLON: ": ",
    _Tags.UNPACK: "**",

    _Tags.ASYNC_FOR_COMP: " async for ",
    _Tags.FOR_COMP: " for ",
    _Tags.IF_COMP: " if ",
    _Tags.ELSE_COMP: " else ",
    _Tags.IN_COMP: " in ",

    _Tags.TUPLE_L: "(",
    _Tags.TUPLE_R: ")",

    _Tags.SLICE: ":",

    _Tags.ELLIPSIS: "...",

    _Tags.REFERENCE: "*",

    _Tags.DOT: ".",

    _Tags.MATCH: "match ",
    _Tags.MATCH_CASE: "case ",
    _Tags.MATCH_DEFAULT: "_",
    _Tags.MATCH_AS: " as ",
    _Tags.MATCH_GUARD: " if ",
    _Tags.MATCH_OR: " | ",
    _Tags.MATCH_STAR: "*",

    _Tags.POSONLYARGS: "/",

    _Tags.KWARGS: "**",

    _Tags.LAMBDA: "lambda ",
    _Tags.LAMBDA_BODDY: ": ",

    _Tags.AS_ITEM: " as ",

    _Tags.GUARD: " if ",

    _Tags.INF: "inf",
    _Tags.NAN: "nan",

    _Tags.SEMANTIC_START: "",
    _Tags.SEMANTIC_END: "",
}