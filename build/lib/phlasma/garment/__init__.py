"""Garment module: 2D patterns sewn and draped on a mannequin. Verbs: drape."""


def _drape(job, out, log, **o):
    from .drape import drape
    return drape(job, out, log, **o)


VERBS = {"drape": _drape}
