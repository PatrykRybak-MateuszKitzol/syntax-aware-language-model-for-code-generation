import sys

import sys
from ast import _Precedence, _MULTI_QUOTES, _ALL_QUOTES, NodeVisitor, AsyncFunctionDef, FunctionDef, ClassDef, Module, Expr, Name, JoinedStr, Constant, FormattedValue, Tuple, If
from contextlib import contextmanager, nullcontext

from tags import _Tags


class Pretokenizer(NodeVisitor):
    def __init__(self, *, _avoid_backslashes=False, _use_dedent=False, _use_semantics=True):
        self._source = []
        self._precedences = {}
        self._type_ignores = {}
        self._indent = 0
        self._avoid_backslashes = _avoid_backslashes
        self._in_try_star = False
        self._use_dedent = _use_dedent
        self._use_semantics = _use_semantics

    def _add_semantic_tags(self, text):
        if self._use_semantics:
            return _Tags.SEMANTIC_START + text + _Tags.SEMANTIC_END
        return text

    def interleave(self, inter, f, seq):
        seq = iter(seq)
        try:
            f(next(seq))
        except StopIteration:
            pass
        else:
            for x in seq:
                inter()
                f(x)

    def items_view(self, traverser, items):
        if len(items) == 1:
            traverser(items[0])
            self.write(_Tags.COMMA)
        else:
            self.interleave(lambda: self.write(_Tags.COMMA), traverser, items)

    def maybe_newline(self):
        if self._source:
            if not (self._source[-1] == _Tags.INDENT or self._source[-1] == _Tags.DEDENT):
                self.write(_Tags.NEW_LINE)

    def fill(self, text=""):
        self.maybe_newline()
        if self._use_dedent:
            self.write(text)
        else:
            self.write(_Tags.INDENT * self._indent + text)
            

    def write(self, *text):
        self._source.extend(text)

    @contextmanager
    def buffered(self, buffer = None):
        if buffer is None:
            buffer = []

        original_source = self._source
        self._source = buffer
        yield buffer
        self._source = original_source

    @contextmanager
    def block(self, *, extra = None):
        self.write(_Tags.BLOCK)
        if extra:
            self.write(extra)
        self._indent += 1
        if self._use_dedent:
          self.write(_Tags.INDENT)
        yield
        self._indent -= 1
        if self._use_dedent:
            self.write(_Tags.DEDENT)

    @contextmanager
    def delimit(self, start, end):
        self.write(start)
        yield
        self.write(end)

    def delimit_if(self, start, end, condition):
        if condition:
            return self.delimit(start, end)
        else:
            return nullcontext()

    def require_parens(self, precedence, node):
        return self.delimit_if(_Tags.DELIMIT_1_L, _Tags.DELIMIT_1_R, self.get_precedence(node) > precedence)

    def get_precedence(self, node):
        return self._precedences.get(node, _Precedence.TEST)

    def set_precedence(self, precedence, *nodes):
        for node in nodes:
            self._precedences[node] = precedence

    def traverse(self, node):
        if isinstance(node, list):
            for item in node:
                self.traverse(item)
        else:
            super().visit(node)

    def visit(self, node):
        self._source = []
        self.traverse(node)
        return "".join(self._source)

    def visit_Module(self, node):
        self._type_ignores = {
            ignore.lineno: f"ignore{ignore.tag}"
            for ignore in node.type_ignores
        }
        self.traverse(node.body)
        self._type_ignores.clear()

    def visit_Expr(self, node):
        self.fill()
        self.set_precedence(_Precedence.YIELD, node.value)
        self.traverse(node.value)

    def visit_NamedExpr(self, node):
        with self.require_parens(_Precedence.NAMED_EXPR, node):
            self.set_precedence(_Precedence.ATOM, node.target, node.value)
            self.traverse(node.target)
            self.write(_Tags.NAMED_EXPR)
            self.traverse(node.value)

    def visit_Assign(self, node):
        self.fill()
        for target in node.targets:
            self.set_precedence(_Precedence.TUPLE, target)
            self.traverse(target)
            self.write(_Tags.ASSIGN)
        self.traverse(node.value)

    def visit_AugAssign(self, node):
        self.fill()
        self.traverse(node.target)
        self.write(getattr(_Tags, node.op.__class__.__name__.upper() + "_ASSIGN"))
        self.traverse(node.value)

    def visit_Return(self, node):
        self.fill(_Tags.RETURN)
        if node.value:
            self.traverse(node.value)

    def visit_Pass(self, node):
        self.fill(_Tags.PASS)

    def visit_Break(self, node):
        self.fill(_Tags.BREAK)

    def visit_Continue(self, node):
        self.fill(_Tags.CONTINUE)

    def visit_Delete(self, node):
        self.fill(_Tags.DEL)
        self.interleave(lambda: self.write(_Tags.COMMA), self.traverse, node.targets)

    def visit_Assert(self, node):
        self.fill(_Tags.ASSERT)
        self.traverse(node.test)
        if node.msg:
            self.write(_Tags.COMMA)
            self.traverse(node.msg)

    def visit_Await(self, node):
        with self.require_parens(_Precedence.AWAIT, node):
            self.write(_Tags.AWAIT)
            if node.value:
                self.set_precedence(_Precedence.ATOM, node.value)
                self.traverse(node.value)

    def visit_Yield(self, node):
        with self.require_parens(_Precedence.YIELD, node):
            self.write(_Tags.YIELD)
            if node.value:
                self.set_precedence(_Precedence.ATOM, node.value)
                self.traverse(node.value)

    def visit_YieldFrom(self, node):
        with self.require_parens(_Precedence.YIELD, node):
            self.write(_Tags.YIELD_FROM)
            if not node.value:
                raise ValueError("Node can't be used without a value attribute.")
            self.set_precedence(_Precedence.ATOM, node.value)
            self.traverse(node.value)

    def visit_Raise(self, node):
        self.fill(_Tags.RAISE)
        if not node.exc:
            if node.cause:
                raise ValueError(f"Node can't use cause without an exception.")
            return
        self.traverse(node.exc)
        if node.cause:
            self.write(_Tags.FROM)
            self.traverse(node.cause)

    def do_visit_try(self, node):
        self.fill(_Tags.TRY)
        with self.block():
            self.traverse(node.body)
        for ex in node.handlers:
            self.traverse(ex)
        if node.orelse:
            self.fill(_Tags.ELSE)
            with self.block():
                self.traverse(node.orelse)
        if node.finalbody:
            self.fill(_Tags.FINALLY)
            with self.block():
                self.traverse(node.finalbody)

    def visit_Try(self, node):
        prev_in_try_star = self._in_try_star
        try:
            self._in_try_star = False
            self.do_visit_try(node)
        finally:
            self._in_try_star = prev_in_try_star

    def visit_TryStar(self, node):
        prev_in_try_star = self._in_try_star
        try:
            self._in_try_star = True
            self.do_visit_try(node)
        finally:
            self._in_try_star = prev_in_try_star

    def visit_ExceptHandler(self, node):
        self.fill(_Tags.EXCEPT_STAR if self._in_try_star else _Tags.EXCEPT)
        if node.type:
            self.traverse(node.type)
        if node.name:
            self.write(_Tags.AS)
            self.write(node.name)
        with self.block():
            self.traverse(node.body)

    def visit_ClassDef(self, node):
        self.maybe_newline()
        self.fill(_Tags.CLASS + node.name)
        with self.delimit_if(_Tags.DELIMIT_1_L, _Tags.DELIMIT_1_R, condition = node.bases or node.keywords):
            comma = False
            for e in node.bases:
                if comma:
                    self.write(_Tags.COMMA)
                else:
                    comma = True
                self.traverse(e)
            for e in node.keywords:
                if comma:
                    self.write(_Tags.COMMA)
                else:
                    comma = True
                self.traverse(e)

        with self.block():
            self.traverse(node.body)

    def visit_FunctionDef(self, node):
        self._function_helper(node, _Tags.DEF)

    def visit_AsyncFunctionDef(self, node):
        self._function_helper(node, _Tags.ASYNC_DEF)

    def _function_helper(self, node, fill_suffix):
        self.maybe_newline()
        def_str = fill_suffix + node.name
        self.fill(def_str)
        with self.delimit(_Tags.DELIMIT_1_L, _Tags.DELIMIT_1_R):
            self.traverse(node.args)
        with self.block():
            self.traverse(node.body)

    def visit_For(self, node):
        self._for_helper(_Tags.FOR, node)

    def visit_AsyncFor(self, node):
        self._for_helper(_Tags.ASYNC_FOR, node)

    def _for_helper(self, fill, node):
        self.fill(fill)
        self.set_precedence(_Precedence.TUPLE, node.target)
        self.traverse(node.target)
        self.write(_Tags.IN)
        self.traverse(node.iter)
        with self.block():
            self.traverse(node.body)
        if node.orelse:
            self.fill(_Tags.ELSE)
            with self.block():
                self.traverse(node.orelse)

    def visit_If(self, node):
        self.fill(_Tags.IF)
        self.traverse(node.test)
        with self.block():
            self.traverse(node.body)
        while node.orelse and len(node.orelse) == 1 and isinstance(node.orelse[0], If):
            node = node.orelse[0]
            self.fill(_Tags.ELIF)
            self.traverse(node.test)
            with self.block():
                self.traverse(node.body)
        if node.orelse:
            self.fill(_Tags.ELSE)
            with self.block():
                self.traverse(node.orelse)

    def visit_While(self, node):
        self.fill(_Tags.WHILE)
        self.traverse(node.test)
        with self.block():
            self.traverse(node.body)
        if node.orelse:
            self.fill(_Tags.ELSE)
            with self.block():
                self.traverse(node.orelse)

    def visit_With(self, node):
        self.fill(_Tags.WITH)
        self.interleave(lambda: self.write(_Tags.COMMA), self.traverse, node.items)
        with self.block():
            self.traverse(node.body)

    def visit_AsyncWith(self, node):
        self.fill(_Tags.ASYNC_WITH)
        self.interleave(lambda: self.write(_Tags.COMMA), self.traverse, node.items)
        with self.block():
            self.traverse(node.body)

    def _str_literal_helper(
        self, string, *, quote_types=_ALL_QUOTES, escape_special_whitespace=False
    ):
        def escape_char(c):
            if not escape_special_whitespace and c in "\n\t":
                return c
            if c == "\\" or not c.isprintable():
                return c.encode("unicode_escape").decode("ascii")
            return c

        escaped_string = "".join(map(escape_char, string))
        possible_quotes = quote_types
        if "\n" in escaped_string:
            possible_quotes = [q for q in possible_quotes if q in _MULTI_QUOTES]
        possible_quotes = [q for q in possible_quotes if q not in escaped_string]
        if not possible_quotes:
            string = repr(string)
            quote = next((q for q in quote_types if string[0] in q), string[0])
            return string[1:-1], [quote]
        if escaped_string:
            possible_quotes.sort(key=lambda q: q[0] == escaped_string[-1])
            if possible_quotes[0][0] == escaped_string[-1]:
                assert len(possible_quotes[0]) == 3
                escaped_string = escaped_string[:-1] + "\\" + escaped_string[-1]
        return escaped_string, possible_quotes

    def _write_str_avoiding_backslashes(self, string, *, quote_types=_ALL_QUOTES):
        string, quote_types = self._str_literal_helper(string, quote_types=quote_types)
        quote_type = quote_types[0]

        if quote_type == '"':
            quote_type = _Tags.QUOTATION_1
        elif quote_type == "'":
            quote_type = _Tags.QUOTATION_2

        self.write(f"{quote_type}{string}{quote_type}")

    def visit_JoinedStr(self, node):
        self.write(_Tags.JOINEDSTR)

        fstring_parts = []
        for value in node.values:
            with self.buffered() as buffer:
                self._write_fstring_inner(value)
            fstring_parts.append(
                ("".join(buffer), isinstance(value, Constant))
            )

        new_fstring_parts = []
        quote_types = list(_ALL_QUOTES)
        fallback_to_repr = False
        for value, is_constant in fstring_parts:
            if is_constant:
                value, new_quote_types = self._str_literal_helper(
                    value,
                    quote_types=quote_types,
                    escape_special_whitespace=True,
                )
                if set(new_quote_types).isdisjoint(quote_types):
                    fallback_to_repr = True
                    break
                quote_types = new_quote_types
            elif "\n" in value:
                quote_types = [q for q in quote_types if q in _MULTI_QUOTES]
                assert quote_types
            new_fstring_parts.append(value)

        if fallback_to_repr:
            quote_types = ["'''"]
            new_fstring_parts.clear()
            for value, is_constant in fstring_parts:
                if is_constant:
                    value = repr('"' + value)  # force repr to use single quotes
                    expected_prefix = "'\""
                    assert value.startswith(expected_prefix), repr(value)
                    value = value[len(expected_prefix):-1]
                new_fstring_parts.append(value)

        value = "".join(new_fstring_parts)
        quote_type = quote_types[0]

        if quote_type == '"':
            quote_type = _Tags.QUOTATION_1
        elif quote_type == "'":
            quote_type = _Tags.QUOTATION_2

        self.write(f"{quote_type}{value}{quote_type}")

    def _write_fstring_inner(self, node, is_format_spec=False):
        if isinstance(node, JoinedStr):
            for value in node.values:
                self._write_fstring_inner(value, is_format_spec=is_format_spec)
        elif isinstance(node, Constant) and isinstance(node.value, str):
            value = node.value.replace("{", "{{").replace("}", "}}")

            if is_format_spec:
                value = value.replace("\\", "\\\\")
                value = value.replace("'", "\\'")
                value = value.replace('"', '\\"')
                value = value.replace("\n", "\\n")
            self.write(self._add_semantic_tags(value))
        elif isinstance(node, FormattedValue):
            self.visit_FormattedValue(node)
        else:
            raise ValueError(f"Unexpected node inside JoinedStr, {node!r}")

    def visit_FormattedValue(self, node):
        def unparse_inner(inner):
            unparser = type(self)()
            unparser.set_precedence(_Precedence.TEST.next(), inner)
            return unparser.visit(inner)

        with self.delimit(_Tags.DELIMIT_3_L, _Tags.DELIMIT_3_R):
            expr = unparse_inner(node.value)
            if expr.startswith("{"):
                # huh?
                self.write(" ")
            self.write(expr)

    def visit_Name(self, node):
        self.write(self._add_semantic_tags(node.id))

    def _write_constant(self, value):
        if isinstance(value, (float, complex)):
            self.write(
                repr(value)
                .replace("inf", _Tags.INF)
                .replace("nan", f"({_Tags.INF}-{_Tags.INF})")
            )
        elif self._avoid_backslashes and isinstance(value, str):
            self._write_str_avoiding_backslashes(self._add_semantic_tags(value))
        elif isinstance(value, str):
            self.write(_Tags.QUOTATION_1 + self._add_semantic_tags(value) + _Tags.QUOTATION_1)
        else:
            self.write(self._add_semantic_tags(repr(value)))

    def visit_Constant(self, node):
        value = node.value
        if isinstance(value, tuple):
            with self.delimit(_Tags.DELIMIT_1_L, _Tags.DELIMIT_1_R):
                self.items_view(self._write_constant, value)
        elif value is ...:
            self.write(_Tags.ELLIPSIS)
        else:
            if node.kind == "u":
                self.write(_Tags.U)
            self._write_constant(node.value)

    def visit_List(self, node):
        with self.delimit(_Tags.DELIMIT_2_L, _Tags.DELIMIT_2_R):
            self.interleave(lambda: self.write(_Tags.COMMA), self.traverse, node.elts)

    def visit_ListComp(self, node):
        with self.delimit(_Tags.DELIMIT_2_L, _Tags.DELIMIT_2_R):
            self.traverse(node.elt)
            for gen in node.generators:
                self.traverse(gen)
    
    def visit_GeneratorExp(self, node):
        with self.delimit(_Tags.DELIMIT_1_L, _Tags.DELIMIT_1_R):
            self.traverse(node.elt)
            for gen in node.generators:
                self.traverse(gen)

    def visit_SetComp(self, node):
        with self.delimit(_Tags.DELIMIT_3_L, _Tags.DELIMIT_3_R):
            self.traverse(node.elt)
            for gen in node.generators:
                self.traverse(gen)

    def visit_DictComp(self, node):
        with self.delimit(_Tags.DELIMIT_3_L, _Tags.DELIMIT_3_R):
            self.traverse(node.key)
            self.write(_Tags.DICT_COLON)
            self.traverse(node.value)
            for gen in node.generators:
                self.traverse(gen)

    def visit_comprehension(self, node):
        if node.is_async:
            self.write(_Tags.ASYNC_FOR_COMP)
        else:
            self.write(_Tags.FOR_COMP)
        self.set_precedence(_Precedence.TUPLE, node.target)
        self.traverse(node.target)
        self.write(_Tags.IN_COMP)
        self.set_precedence(_Precedence.TEST.next(), node.iter, *node.ifs)
        self.traverse(node.iter)
        for if_clause in node.ifs:
            self.write(_Tags.IF_COMP)
            self.traverse(if_clause)

    def visit_IfExp(self, node):
        with self.require_parens(_Precedence.TEST, node):
            self.set_precedence(_Precedence.TEST.next(), node.body, node.test)
            self.traverse(node.body)
            self.write(_Tags.IF_COMP)
            self.traverse(node.test)
            self.write(_Tags.ELSE_COMP)
            self.set_precedence(_Precedence.TEST, node.orelse)
            self.traverse(node.orelse)

    def visit_Set(self, node):
        if node.elts:
            with self.delimit(_Tags.DELIMIT_3_L, _Tags.DELIMIT_3_R):
                self.interleave(lambda: self.write(_Tags.COMMA), self.traverse, node.elts)
        else:
            self.write(_Tags.DELIMIT_3_L + _Tags.UNPACK + _Tags.TUPLE_L + _Tags.TUPLE_R +_Tags.DELIMIT_3_R)

    def visit_Dict(self, node):
        def write_key_value_pair(k, v):
            self.traverse(k)
            self.write(_Tags.DICT_COLON)
            self.traverse(v)

        def write_item(item):
            k, v = item
            if k is None:
                self.write(_Tags.UNPACK)
                self.set_precedence(_Precedence.EXPR, v)
                self.traverse(v)
            else:
                write_key_value_pair(k, v)

        with self.delimit(_Tags.DELIMIT_3_L, _Tags.DELIMIT_3_R):
            self.interleave(
                lambda: self.write(_Tags.COMMA), write_item, zip(node.keys, node.values)
            )

    def visit_Tuple(self, node):
        with self.delimit_if(
            _Tags.TUPLE_L,
            _Tags.TUPLE_R,
            len(node.elts) == 0 or self.get_precedence(node) > _Precedence.TUPLE
        ):
            self.items_view(self.traverse, node.elts)

    unop = {"Invert": "~", "Not": "not", "UAdd": "+", "USub": "-"}
    unop_precedence = {
        "not": _Precedence.NOT,
        "~": _Precedence.FACTOR,
        "+": _Precedence.FACTOR,
        "-": _Precedence.FACTOR,
    }

    def visit_UnaryOp(self, node):
        name = node.op.__class__.__name__
        operator = self.unop[name]
        operator_precedence = self.unop_precedence[operator]
        with self.require_parens(operator_precedence, node):
            self.write(getattr(_Tags, name.upper()))
            self.set_precedence(operator_precedence, node.operand)
            self.traverse(node.operand)

    binop = {
        "Add": "+",
        "Sub": "-",
        "Mult": "*",
        "MatMult": "@",
        "Div": "/",
        "Mod": "%",
        "LShift": "<<",
        "RShift": ">>",
        "BitOr": "|",
        "BitXor": "^",
        "BitAnd": "&",
        "FloorDiv": "//",
        "Pow": "**",
    }

    binop_precedence = {
        "+": _Precedence.ARITH,
        "-": _Precedence.ARITH,
        "*": _Precedence.TERM,
        "@": _Precedence.TERM,
        "/": _Precedence.TERM,
        "%": _Precedence.TERM,
        "<<": _Precedence.SHIFT,
        ">>": _Precedence.SHIFT,
        "|": _Precedence.BOR,
        "^": _Precedence.BXOR,
        "&": _Precedence.BAND,
        "//": _Precedence.TERM,
        "**": _Precedence.POWER,
    }

    binop_rassoc = frozenset((_Tags.POW,))
    def visit_BinOp(self, node):
        name = node.op.__class__.__name__
        operator = self.binop[name]
        operator_precedence = self.binop_precedence[operator]
        with self.require_parens(operator_precedence, node):
            if operator in self.binop_rassoc:
                left_precedence = operator_precedence.next()
                right_precedence = operator_precedence
            else:
                left_precedence = operator_precedence
                right_precedence = operator_precedence.next()

            self.set_precedence(left_precedence, node.left)
            self.traverse(node.left)
            self.write(f"{getattr(_Tags, name.upper())}")
            self.set_precedence(right_precedence, node.right)
            self.traverse(node.right)

    def visit_Compare(self, node):
        with self.require_parens(_Precedence.CMP, node):
            self.set_precedence(_Precedence.CMP.next(), node.left, *node.comparators)
            self.traverse(node.left)
            for o, e in zip(node.ops, node.comparators):
                self.write(getattr(_Tags, o.__class__.__name__.upper()))
                self.traverse(e)

    boolops = {"And": "and", "Or": "or"}
    boolop_precedence = {"and": _Precedence.AND, "or": _Precedence.OR}

    def visit_BoolOp(self, node):
        operator = self.boolops[node.op.__class__.__name__]
        operator_precedence = self.boolop_precedence[operator]

        def increasing_level_traverse(node):
            nonlocal operator_precedence
            operator_precedence = operator_precedence.next()
            self.set_precedence(operator_precedence, node)
            self.traverse(node)

        with self.require_parens(operator_precedence, node):
            s = f"{getattr(_Tags, operator.upper())}"
            self.interleave(lambda: self.write(s), increasing_level_traverse, node.values)

    def visit_Attribute(self, node):
        self.set_precedence(_Precedence.ATOM, node.value)
        self.traverse(node.value)
        self.write(_Tags.DOT)
        self.write(node.attr)

    def visit_Call(self, node):
        self.set_precedence(_Precedence.ATOM, node.func)
        self.traverse(node.func)
        with self.delimit(_Tags.DELIMIT_1_L, _Tags.DELIMIT_1_R):
            comma = False
            for e in node.args:
                if comma:
                    self.write(_Tags.COMMA)
                else:
                    comma = True
                self.traverse(e)
            for e in node.keywords:
                if comma:
                    self.write(_Tags.COMMA)
                else:
                    comma = True
                self.traverse(e)

    def visit_Subscript(self, node):
        def is_non_empty_tuple(slice_value):
            return (
                isinstance(slice_value, Tuple)
                and slice_value.elts
            )

        self.set_precedence(_Precedence.ATOM, node.value)
        self.traverse(node.value)
        with self.delimit(_Tags.DELIMIT_2_L, _Tags.DELIMIT_2_R):
            if is_non_empty_tuple(node.slice):
                self.items_view(self.traverse, node.slice.elts)
            else:
                self.traverse(node.slice)

    def visit_Starred(self, node):
        self.write(_Tags.REFERENCE)
        self.set_precedence(_Precedence.EXPR, node.value)
        self.traverse(node.value)

    def visit_Ellipsis(self, node):
        self.write(_Tags.ELLIPSIS)

    def visit_Slice(self, node):
        if node.lower:
            self.traverse(node.lower)
        self.write(_Tags.SLICE)
        if node.upper:
            self.traverse(node.upper)
        if node.step:
            self.write(_Tags.SLICE)
            self.traverse(node.step)

    def visit_arg(self, node):
        self.write(self._add_semantic_tags(node.arg))

    def visit_arguments(self, node):
        first = True
        all_args = node.posonlyargs + node.args
        defaults = [None] * (len(all_args) - len(node.defaults)) + node.defaults
        for index, elements in enumerate(zip(all_args, defaults), 1):
            a, d = elements
            if first:
                first = False
            else:
                self.write(_Tags.COMMA)
            self.traverse(a)
            if d:
                self.write(_Tags.ASSIGN)
                self.traverse(d)
            if index == len(node.posonlyargs):
                self.write(f"{_Tags.COMMA}{_Tags.POSONLYARGS}")

        if node.vararg or node.kwonlyargs:
            if first:
                first = False
            else:
                self.write(_Tags.COMMA)
            self.write(_Tags.REFERENCE)
            if node.vararg:
                self.write(self._add_semantic_tags(node.vararg.arg))

        if node.kwonlyargs:
            for a, d in zip(node.kwonlyargs, node.kw_defaults):
                self.write(_Tags.COMMA)
                self.traverse(a)
                if d:
                    self.write(_Tags.ASSIGN)
                    self.traverse(d)

        if node.kwarg:
            if first:
                first = False
            else:
                self.write(_Tags.COMMA)
            self.write(_Tags.KWARGS + self._add_semantic_tags(node.kwarg.arg))

    def visit_keyword(self, node):
        if node.arg is None:
            self.write(_Tags.KWARGS)
        else:
            self.write(self._add_semantic_tags(node.arg))
            self.write(_Tags.ASSIGN)
        self.traverse(node.value)

    def visit_Lambda(self, node):
        with self.require_parens(_Precedence.TEST, node):
            self.write(_Tags.LAMBDA)
            with self.buffered() as buffer:
                self.traverse(node.args)
            if buffer:
                self.write(*buffer)
            self.write(_Tags.LAMBDA_BODDY)
            self.set_precedence(_Precedence.TEST, node.body)
            self.traverse(node.body)

    def visit_withitem(self, node):
        self.traverse(node.context_expr)
        if node.optional_vars:
            self.write(_Tags.AS_ITEM)
            self.traverse(node.optional_vars)

    def visit_Match(self, node):
        self.fill(_Tags.MATCH)
        self.traverse(node.subject)
        with self.block():
            for case in node.cases:
                self.traverse(case)

    def visit_match_case(self, node):
        self.fill(_Tags.MATCH_CASE)
        self.traverse(node.pattern)
        if node.guard:
            self.write(_Tags.GUARD)
            self.traverse(node.guard)
        with self.block():
            self.traverse(node.body)

    def visit_MatchValue(self, node):
        self.traverse(node.value)

    def visit_MatchSingleton(self, node):
        self._write_constant(node.value)

    def visit_MatchSequence(self, node):
        with self.delimit(_Tags.DELIMIT_2_L, _Tags.DELIMIT_2_R):
            self.interleave(
                lambda: self.write(_Tags.COMMA), self.traverse, node.patterns
            )

    def visit_MatchStar(self, node):
        name = node.name
        if name is None:
            name = _Tags.MATCH_DEFAULT
        self.write(_Tags.MATCH_STAR + self._add_semantic_tags(name))

    def visit_MatchMapping(self, node):
        def write_key_pattern_pair(pair):
            k, p = pair
            self.traverse(k)
            self.write(_Tags.DICT_COLON)
            self.traverse(p)

        with self.delimit(_Tags.DELIMIT_3_L, _Tags.DELIMIT_3_R):
            keys = node.keys
            self.interleave(
                lambda: self.write(_Tags.COMMA),
                write_key_pattern_pair,
                zip(keys, node.patterns, strict=True),
            )
            rest = node.rest
            if rest is not None:
                if keys:
                    self.write(_Tags.COMMA)
                self.write(_Tags.UNPACK + self._add_semantic_tags(rest))

    def visit_MatchClass(self, node):
        self.set_precedence(_Precedence.ATOM, node.cls)
        self.traverse(node.cls)
        with self.delimit(_Tags.DELIMIT_1_L, _Tags.DELIMIT_1_R):
            patterns = node.patterns
            self.interleave(
                lambda: self.write(_Tags.COMMA), self.traverse, patterns
            )
            attrs = node.kwd_attrs
            if attrs:
                def write_attr_pattern(pair):
                    attr, pattern = pair
                    self.write(self._add_semantic_tags(attr) + _Tags.ASSIGN)
                    self.traverse(pattern)

                if patterns:
                    self.write(_Tags.COMMA)
                self.interleave(
                    lambda: self.write(_Tags.COMMA),
                    write_attr_pattern,
                    zip(attrs, node.kwd_patterns, strict=True),
                )

    def visit_MatchAs(self, node):
        name = node.name
        pattern = node.pattern
        if name is None:
            self.write(_Tags.MATCH_DEFAULT)
        elif pattern is None:
            self.write(self._add_semantic_tags(node.name))
        else:
            with self.require_parens(_Precedence.TEST, node):
                self.set_precedence(_Precedence.BOR, node.pattern)
                self.traverse(node.pattern)
                self.write(_Tags.MATCH_AS + self._add_semantic_tags(node.name))

    def visit_MatchOr(self, node):
        with self.require_parens(_Precedence.BOR, node):
            self.set_precedence(_Precedence.BOR.next(), *node.patterns)
            self.interleave(lambda: self.write(_Tags.MATCH_OR), self.traverse, node.patterns)

def pretokenize(ast_obj, _use_dedent=False, _use_semantics=True):
    pretokenizer = Pretokenizer(_use_dedent=_use_dedent, _use_semantics=_use_semantics)
    return pretokenizer.visit(ast_obj)
