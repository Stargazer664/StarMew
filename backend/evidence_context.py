"""Stable, source-bearing evidence selection independent of retrieval score scales."""
from dataclasses import dataclass, replace
import re


@dataclass(frozen=True)
class EvidenceSnippet:
    citation_id: str
    source: str
    page: str
    text: str
    rank: int
    score: float | None = None


@dataclass(frozen=True)
class EvidenceSelection:
    snippets: tuple[EvidenceSnippet, ...]
    used_tokens: int
    omitted_count: int


def aggregate_evidence(groups):
    """Assign IDs in round/candidate order; scores are descriptive, never compared."""
    result, ids = [], {}
    for group in groups:
        for item in group:
            source = str(item.get('filename') or item.get('source') or item.get('url') or '未知来源')
            if item.get('title') and item.get('url'):
                source = f"{item['title']} ({item['url']})"
            page = str(item.get('page_number') if item.get('page_number') is not None else item.get('page') or '未知')
            text = str(item.get('text') or item.get('content') or '').strip()
            key = (source, page, re.sub(r'\s+', ' ', text))
            citation = ids.setdefault(key, str(len(ids) + 1))
            result.append(EvidenceSnippet(citation, source, page, text, len(result) + 1, item.get('score')))
    return tuple(result)


def render_evidence(snippets):
    return '\n\n'.join(f'[{s.citation_id}] {s.source} (Page {s.page}):\n{s.text}' for s in snippets)


def _sentences(text):
    # Delimited rows are atomic: only the whole-snippet path may retain them.
    # Keep prose blocks separate so text on opposite sides cannot form a sentence.
    blocks, prose = [], []
    for line in text.splitlines(keepends=True):
        if '|' in line or '\t' in line:
            blocks.append(''.join(prose))
            prose = []
        else:
            prose.append(line)
    blocks.append(''.join(prose))
    sentences = []
    for block in blocks:
        # Equal-length masking protects URL punctuation while retaining offsets
        # into the original text; a kept sentence therefore keeps the whole URL.
        masked = re.sub(r'(?:https?://|www\.)[^\s<>"\']+',
                        lambda match: ' ' * len(match.group()), block, flags=re.I)
        start = 0
        for ending in re.finditer(r'[。！？!?][”’"\']*|\.(?=\s|$)', masked):
            sentence = block[start:ending.end()].strip()
            if sentence:
                sentences.append(sentence)
            start = ending.end()
    return sentences


def _metadata_field(text, source):
    if '\n' in text.strip():
        return False
    label = re.match(r'^(?:(?:合成)?(?:内部)?资料|标题|来源文件|文件名|Title|Document|Source(?: file)?|Filename)\s*[:：]', text, re.I)
    provenance = (re.match(r'^(?:版本|Version)\s*[:：]?', text, re.I)
                  and source in text
                  and re.search(r'来源文件|Source(?: file)?\s*[:：]', text, re.I))
    return bool(label or provenance)


def _body_sentences(snippet):
    """Skip only explicit leading title/provenance fields when compressing.

    This is structural, not semantic relevance ranking. A bare Version field
    may be a business fact; only a version tied to this source is provenance.
    The original snippet remains intact whenever it fits in full.
    """
    text = snippet.text.strip()
    # Consume the actual heading boundary, not its whole line: following body
    # text can be valid even without a sentence-ending punctuation mark.
    while text:
        line, separator, rest = text.partition('\n')
        if not _metadata_field(line, snippet.source):
            break
        headings = _sentences(line)
        if headings:
            text = text[len(headings[0]):].strip()
        else:
            text = rest.strip() if separator else ''
    return _sentences(text), bool(text)


def select_evidence(snippets, token_limit, budget):
    ordered, seen, bodies = [], set(), {}
    for snippet in sorted(snippets, key=lambda s: s.rank):
        key = (snippet.source, snippet.page, re.sub(r'\s+', ' ', snippet.text))
        sentences, has_body = _body_sentences(snippet)
        if key not in seen and snippet.text.strip() and has_body:
            ordered.append(snippet)
            seen.add(key)
            bodies[snippet.citation_id] = sentences
    # Highest rank first, then best candidate from each other source, then extras.
    diverse, extra, sources = [], [], set()
    for snippet in ordered:
        if snippet.source in sources:
            extra.append(snippet)
        else:
            diverse.append(snippet)
            sources.add(snippet.source)
    ordered = diverse + extra
    def minimum_body(snippet, prefix):
        sentences = bodies[snippet.citation_id]
        candidates = (replace(snippet, text=sentence) for sentence in sentences) if sentences else (snippet,)
        return next((candidate for candidate in candidates
                     if budget.count_text(render_evidence(prefix + [candidate])) <= token_limit), None)

    selected = []
    for index, snippet in enumerate(ordered):
        sentences = bodies[snippet.citation_id]
        required = minimum_body(snippet, selected)
        if required is None:
            continue
        # Reserve affordable bodies in rank order after the primary fragment.
        # An oversized intervening source must not evict a later affordable one.
        supplements = []
        for other in ordered[index + 1:]:
            candidate = minimum_body(other, selected + [required] + supplements) if other in diverse else None
            if candidate is not None:
                supplements.append(candidate)
        def fits(candidate):
            return budget.count_text(render_evidence(selected + [candidate] + supplements)) <= token_limit
        if fits(snippet):
            selected.append(snippet)
            continue
        retained = ''
        for sentence in sentences:
            candidate = replace(snippet, text=(retained + ' ' + sentence).strip())
            if fits(candidate):
                retained = candidate.text
            elif retained:
                break
        if retained:
            selected.append(replace(snippet, text=retained))
    return EvidenceSelection(tuple(selected), budget.count_text(render_evidence(selected)), len(snippets) - len(selected))


class CitationFilter:
    """Buffer numeric citations across chunks; never publish an unknown ID."""
    def __init__(self, allowed):
        self.allowed = set(allowed)
        self.pending = ''

    def feed(self, text):
        output = []
        for char in text:
            if not self.pending:
                if char == '[':
                    self.pending = char
                else:
                    output.append(char)
            elif char == ']':
                ids = re.findall(r'\d+', self.pending)
                if not ids or ('-' not in self.pending and all(value in self.allowed for value in ids)):
                    output.append(self.pending + char)
                self.pending = ''
            elif char.isdigit() or char in ' ,，、-':
                self.pending += char
            else:
                output.append(self.pending)
                self.pending = ''
                if char == '[':
                    self.pending = char
                else:
                    output.append(char)
        return ''.join(output)

    def finish(self):
        pending, self.pending = self.pending, ''
        return '' if re.search(r'\d', pending) else pending
