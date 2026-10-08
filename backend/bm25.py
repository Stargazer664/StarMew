"""Versioned, stateless preprocessing for Milvus native BM25.

Identifier tokens retain digits and internal separators. Chinese 1/2/3-grams
retain adjacent-character information without an external dictionary/model.
"""
import json
import re
import unicodedata

from pymilvus import FunctionType

BM25_SCHEMA_VERSION = "starmew_native_bm25_v2_cjk123_identifiers_nfkc"
_TOKENS = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[a-z0-9]+(?:[._/\-][a-z0-9]+)*(?:\+\+|#)?")
_CHINESE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


class BM25MigrationRequired(RuntimeError):
    pass


def tokenize(text: str) -> list[str]:
    out = []
    for part in _TOKENS.findall(unicodedata.normalize("NFKC", text).lower()):
        if _CHINESE.fullmatch(part):
            for size in (1, 2, 3):
                out.extend(part[i:i + size] for i in range(len(part) - size + 1))
        else:
            out.append(part)
    return out


def lexical_text(text: str) -> str:
    return " ".join(tokenize(text))


def validate_schema(description: dict) -> None:
    fields = {f["name"]: f for f in description.get("fields", [])}
    lexical = fields.get("lexical_text", {}).get("params", {})
    analyzer = lexical.get("analyzer_params", {})
    if isinstance(analyzer, str):
        try:
            analyzer = json.loads(analyzer)
        except ValueError:
            analyzer = {}
    functions = description.get("functions", [])
    valid_function = any(
        f.get("type") == FunctionType.BM25
        and f.get("input_field_names") == ["lexical_text"]
        and f.get("output_field_names") == ["sparse_embedding"]
        for f in functions
    )
    if (description.get("description") != BM25_SCHEMA_VERSION
        or not valid_function or analyzer != {"tokenizer": "whitespace"}):
        raise BM25MigrationRequired(
            "BM25 索引格式不匹配，请先运行 scripts/migrate_bm25.py 重建并切换新集合；旧集合未修改"
        )
