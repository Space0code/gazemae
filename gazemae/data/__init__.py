import logging
from typing import Dict, List

from .corpora import *


CORPUS_LIST = {
    'EMVIC2014': EMVIC2014,
    'Cerf2007-FIFA': Cerf2007_FIFA,
    'ETRA2019': ETRA2019,
    'MIT-LowRes': MIT_LowRes,
}


frequencies = {
    1000: ['EMVIC2014', 'Cerf2007-FIFA'],
    500: ['ETRA2019'],
    # the ones from MIT are 240Hz but oh well
    250: ['MIT-LowRes'],
}

_CORPUS_ALIASES = {
    'emvic2014': 'EMVIC2014',
    'emvic': 'EMVIC2014',
    'cerf2007-fifa': 'Cerf2007-FIFA',
    'fifa': 'Cerf2007-FIFA',
    'etra2019': 'ETRA2019',
    'etra': 'ETRA2019',
    'mit-lowres': 'MIT-LowRes',
    'mit-low-res': 'MIT-LowRes',
    'mit': 'MIT-LowRes',
}


def _parse_corpora_arg(corpora_arg: str) -> List[str]:
    """Parse and validate the optional comma-separated `--corpora` argument."""
    if not corpora_arg:
        return []

    parsed: List[str] = []
    for raw_name in corpora_arg.split(','):
        key = raw_name.strip().lower()
        if not key:
            continue
        corpus_name = _CORPUS_ALIASES.get(key)
        if corpus_name is None:
            valid = ', '.join(sorted(CORPUS_LIST.keys()))
            raise ValueError(
                f"Unknown corpus '{raw_name}'. Valid names: {valid}")
        parsed.append(corpus_name)
    return parsed


def _unique(names: List[str]) -> List[str]:
    """Return names preserving first occurrence order."""
    out: List[str] = []
    seen = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def get_corpora(args, additional_corpus=None) -> Dict[str, EyeTrackingCorpus]:
    requested = _parse_corpora_arg(getattr(args, 'corpora', ''))
    if requested:
        corpora = requested
        logging.info('Using explicitly requested corpora: {}'.format(corpora))
    else:
        corpora = []
        for f, c in frequencies.items():
            if args.hz <= f:
                corpora.extend(c)

    # corpora = list(CORPUS_LIST.keys())

    # used to add a corpus to evaluator to test for overfitting during
    # training time
    if isinstance(additional_corpus, str) and additional_corpus not in corpora:
        corpora.append(additional_corpus)
        logging.info('[evaluator] Added an unseen data set: {}'.format(
            additional_corpus))

    corpora = _unique(corpora)
    return {c: CORPUS_LIST[c](args) for c in corpora}
