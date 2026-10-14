"""Frictionless column mapping for heterogeneous light-curve tables.

Astronomers' light curves carry the same three physical quantities — time,
brightness, brightness error — under wildly different column names (``HJD`` vs
``BJD_TDB`` vs ``time``; ``Vmag`` vs ``flux`` vs ``m``; ``e_mag`` vs ``dmag`` vs
``flux_err``). :class:`ColumnMap` removes that friction: the common names are
auto-detected, and any column can be pinned explicitly. Detection is
case-insensitive and ordered, so the most specific/standard name wins.

The brightness *domain* (magnitude vs flux) is inferred from the matched value
column when possible (``flux`` → flux, ``mag`` → magnitude) and can always be set
explicitly. Methods that are naturally flux-based (BLS, TLS) convert internally.

The built-in detection lists cover the standard light-curve products of the major
time-domain surveys — ASAS-SN, ASAS-3, ATLAS, CRTS/CSS, ZTF, Pan-STARRS, TESS, Kepler,
Gaia, LSST, and MACHO — so a downloaded table usually needs no column hints. Error
resolution is *domain-aware*: a flux value is paired with a flux error and a magnitude
value with a magnitude error, which disambiguates surveys (e.g. ATLAS) that expose both
``m``/``dm`` and ``uJy``/``duJy`` in one table. Headerless products (OGLE ``.dat``) have
no column names to detect — pass the data as a ``(time, value, error)`` tuple or with an
explicit :class:`ColumnMap`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from cuperiod.core.errors import ColumnResolutionError


class Domain(StrEnum):
    """Brightness domain of a light curve's value column.

    Construction from a string ignores case and surrounding whitespace, so
    ``Domain("Flux")`` is :attr:`FLUX`; anything else raises ``ValueError``.
    """

    MAGNITUDE = "magnitude"
    FLUX = "flux"

    @classmethod
    def _missing_(cls, value: object) -> Domain | None:
        if isinstance(value, str):
            norm = value.strip().lower()
            for member in cls:
                if member.value == norm:
                    return member
        return None


def coerce_domain(domain: Domain | str | None) -> Domain | None:
    """``None`` stays ``None``; a string or :class:`Domain` becomes a :class:`Domain`.

    Every public entry point takes ``domain`` as either spelling (``"flux"`` is
    documented), but identity checks such as ``domain is Domain.FLUX`` only hold
    for the enum: an uncoerced ``"flux"`` used to fall through to the magnitude
    branch. Coerce once, at the entry point, with this.

    Raises
    ------
    ValueError
        If ``domain`` is not a valid domain name.
    """
    return None if domain is None else Domain(domain)


#: Time-column name candidates, most-specific first. Barycentric/heliocentric
#: corrected times precede plain JD so a table carrying both picks the better one.
#: Covers JD/HJD/BJD (ASAS, ASAS-SN, OGLE, Gaia), MJD (ATLAS, CRTS, ZTF, Pan-STARRS),
#: BTJD (TESS), BKJD (Kepler), ``midpointMjdTai`` (LSST), and ``obsTime`` (Pan-STARRS).
TIME_NAMES: Final[tuple[str, ...]] = (
    "bjd_tdb",
    "bjd",
    "hjd",
    "btjd",
    "bkjd",
    "mjd",
    "hmjd",
    "midpointmjdtai",
    "midpointtai",
    "obstime",
    "jd",
    "time",
    "date",
    "t",
)

#: Magnitude value-column candidates. Includes ASAS-3 ``MAG_0`` (first aperture),
#: ``psfMag`` (Pan-STARRS/LSST), and ``Rmag``/``Bmag`` (MACHO red/blue).
MAG_NAMES: Final[tuple[str, ...]] = (
    "mag",
    "magnitude",
    "vmag",
    "gmag",
    "rmag",
    "bmag",
    "imag",
    "psfmag",
    "phot_mag",
    "mag_0",
    "m",
)

#: Flux value-column candidates. Includes ``PDCSAP_FLUX``/``SAP_FLUX``/``KSPSAP_FLUX``
#: (TESS/Kepler), ``psfFlux`` (Pan-STARRS/LSST), and ``uJy``/``mJy`` (ATLAS, ASAS-SN).
#: ``uJy`` precedes the bare ``f`` so ATLAS's flux column wins over its ``F`` filter.
FLUX_NAMES: Final[tuple[str, ...]] = (
    "pdcsap_flux",
    "kspsap_flux",
    "sap_flux",
    "psfflux",
    "psflux",
    "psfdiffflux",
    "norm_flux",
    "rel_flux",
    "flux",
    "ujy",
    "mjy",
    "fnu",
    "f",
)

#: Magnitude-error candidates (used when the value column is a magnitude): ``magerr``
#: (ZTF, CRTS, ASAS-SN), ``dm`` (ATLAS), ``MER_0`` (ASAS-3), ``e_mag``, ``dmag``, ...
MAG_ERR_NAMES: Final[tuple[str, ...]] = (
    "mag_err",
    "magnitude_err",
    "mag_error",
    "magerr",
    "e_mag",
    "merr",
    "dmag",
    "dm",
    "mer_0",
)

#: Flux-error candidates (used when the value column is a flux): ``flux_error`` (Gaia),
#: ``*SAP_FLUX_ERR`` (TESS/Kepler), ``psfFluxErr`` (Pan-STARRS/LSST), ``duJy`` (ATLAS).
FLUX_ERR_NAMES: Final[tuple[str, ...]] = (
    "flux_err",
    "flux_error",
    "sap_flux_err",
    "pdcsap_flux_err",
    "kspsap_flux_err",
    "psfflux_err",
    "psffluxerr",
    "psfluxerr",
    "psfdiffflux_err",
    "psfdifffluxerr",
    "e_flux",
    "dujy",
)

#: Domain-neutral error spellings, tried after the domain-specific lists.
GENERIC_ERR_NAMES: Final[tuple[str, ...]] = (
    "err",
    "error",
    "sigma",
)

#: All error spellings. Resolution is domain-aware (see :meth:`ColumnMap.resolve`): a
#: flux value pairs with a flux error and a magnitude value with a magnitude error.
ERR_NAMES: Final[tuple[str, ...]] = MAG_ERR_NAMES + FLUX_ERR_NAMES + GENERIC_ERR_NAMES

#: Band/filter-column candidates (used only by multi-band ingestion): ``filtercode``
#: (ZTF), ``filterID`` (Pan-STARRS), ``band`` (Gaia/LSST), ``fid`` (ZTF alerts).
BAND_NAMES: Final[tuple[str, ...]] = (
    "band",
    "filter",
    "filtercode",
    "filterid",
    "filtername",
    "phot_filter",
    "passband",
    "fid",
)


@dataclass(frozen=True)
class ResolvedColumns:
    """Concrete column names + domain, the result of resolving a :class:`ColumnMap`."""

    time: str
    value: str
    error: str | None
    band: str | None
    domain: Domain


def _match(role: str, explicit: str | None, candidates: tuple[str, ...],
           available: list[str], lower_to_actual: dict[str, str]) -> str | None:
    """Resolve one role to an actual column name, or ``None`` if optional & absent.

    An explicit override is honored verbatim (raising if absent); otherwise the
    first case-insensitive candidate present in ``available`` wins.
    """
    if explicit is not None:
        if explicit in available:
            return explicit
        if explicit.lower() in lower_to_actual:
            return lower_to_actual[explicit.lower()]
        raise ColumnResolutionError(
            f"{role} column {explicit!r} not found; available columns: {available}"
        )
    for cand in candidates:
        if cand in lower_to_actual:
            return lower_to_actual[cand]
    return None


def infer_domain(value_column: str) -> Domain | None:
    """Infer the brightness domain from a value column's name, or ``None``.

    Parameters
    ----------
    value_column : str
        The resolved value-column name.

    Returns
    -------
    Domain or None
        :attr:`Domain.FLUX` if the name looks flux-like, :attr:`Domain.MAGNITUDE`
        if magnitude-like, otherwise ``None`` (caller should default or require an
        explicit domain). Every name in :data:`FLUX_NAMES` / :data:`MAG_NAMES` maps
        to its own list's domain.
    """
    low = value_column.lower()
    # The detection lists first: a column picked *because* it is a flux name must be
    # labelled flux. The bare ``f`` matches no flux token and used to fall back to
    # magnitude, so BLS/TLS converted a {t, f, err} table with 10**(-0.4 f).
    if low in FLUX_NAMES:
        return Domain.FLUX
    if low in MAG_NAMES:
        return Domain.MAGNITUDE
    flux_tokens = ("flux", "fnu", "ujy", "mjy", "njy", "nmgy", "jansky")
    if any(tok in low for tok in flux_tokens):
        return Domain.FLUX
    if "mag" in low:
        return Domain.MAGNITUDE
    return None


@dataclass(frozen=True)
class ColumnMap:
    """Selects the time/value/error/band columns of a light-curve table.

    Any field left ``None`` is auto-detected from the table's column names using
    the package detection lists (:data:`TIME_NAMES`, :data:`MAG_NAMES`,
    :data:`FLUX_NAMES`, :data:`ERR_NAMES`, :data:`BAND_NAMES`); a non-``None`` field
    pins that column explicitly. Matching is case-insensitive.

    Parameters
    ----------
    time, value, error, band : str, optional
        Explicit column names. ``time`` and ``value`` are required (by detection or
        override); ``error`` and ``band`` are optional. A non-string label (the
        integer columns of ``pandas.read_csv(header=None)``) is taken as its
        string form, which is how table column names are matched.

    Examples
    --------
    >>> ColumnMap(time="HJD", value="Vmag", error="e_Vmag")
    ColumnMap(time='HJD', value='Vmag', error='e_Vmag', band=None)
    """

    time: str | None = None
    value: str | None = None
    error: str | None = None
    band: str | None = None

    def __post_init__(self) -> None:
        for role in ("time", "value", "error", "band"):
            label = getattr(self, role)
            if label is not None and not isinstance(label, str):
                object.__setattr__(self, role, str(label))

    def resolve(
        self, columns: list[str], *, domain: Domain | str | None = None
    ) -> ResolvedColumns:
        """Resolve against a table's actual column names.

        Parameters
        ----------
        columns : list of str
            The table's column names (non-string labels are matched as strings).
        domain : Domain or str, optional
            Explicit domain (``"flux"``/``"magnitude"`` or the enum). It labels the
            value column and, when ``value`` is auto-detected, decides which kind of
            column is preferred: a table carrying both flux and magnitude columns
            gives its magnitude column for ``domain="magnitude"``. If ``None``, the
            domain is inferred from the resolved value-column name, defaulting to
            :attr:`Domain.MAGNITUDE`.

        Returns
        -------
        ResolvedColumns

        Raises
        ------
        ColumnResolutionError
            If the time or value column cannot be resolved.
        ValueError
            If ``domain`` is not a valid domain name.

        Warns
        -----
        UserWarning
            If ``domain`` is given and ``value`` is auto-detected, but the table has
            no column of that domain and the one found looks like the other (e.g.
            only ``flux`` with ``domain="magnitude"``). The override is still
            honoured.
        """
        explicit = coerce_domain(domain)
        available = [str(col) for col in columns]
        lower_to_actual: dict[str, str] = {}
        for col in available:  # first spelling wins on case collisions
            lower_to_actual.setdefault(col.lower(), col)

        # Without an explicit domain flux names are tried first (the long-standing
        # order); with one, that domain's names are, so domain="magnitude" on an
        # ASAS-SN/Gaia table with both `flux` and `mag` reads `mag` instead of
        # stamping the flux column as magnitudes (which made BLS invert transits).
        if self.value is not None:
            value_candidates: tuple[str, ...] = ()
        elif explicit is Domain.MAGNITUDE:
            value_candidates = MAG_NAMES + FLUX_NAMES
        else:
            value_candidates = FLUX_NAMES + MAG_NAMES
        time = _match("time", self.time, TIME_NAMES, available, lower_to_actual)
        value = _match(
            "value", self.value, value_candidates, available, lower_to_actual
        )
        band = _match("band", self.band, BAND_NAMES, available, lower_to_actual)

        if time is None:
            raise ColumnResolutionError(
                "could not resolve a time column; pass "
                "ColumnMap(time=...) explicitly. Available columns: "
                f"{available}"
            )
        if value is None:
            raise ColumnResolutionError(
                "could not resolve a value (magnitude/flux) column; pass "
                "ColumnMap(value=...) explicitly. Available columns: "
                f"{available}"
            )
        inferred = infer_domain(value)
        if (
            explicit is not None
            and self.value is None
            and inferred is not None
            and inferred is not explicit
        ):
            warnings.warn(
                f"domain={explicit.value!r} was given but no {explicit.value} column "
                f"was found; the auto-detected value column {value!r} looks like a "
                f"{inferred.value} column and is read as {explicit.value} as "
                "requested. Pin ColumnMap(value=...) to silence this, or omit "
                "domain= to label the column by its name.",
                UserWarning,
                stacklevel=2,
            )
        resolved_domain = explicit or inferred or Domain.MAGNITUDE
        # Resolve the error for the *same* measurement as the value: first try names
        # derived from the value column (so ``PDCSAP_FLUX`` pairs with
        # ``PDCSAP_FLUX_ERR`` and ``uJy`` with ``duJy``), then the spellings for the
        # value's domain, then domain-neutral spellings. This keeps a flux value with a
        # flux error and a magnitude value with a magnitude error.
        v = value.lower()
        derived = (f"{v}_err", f"{v}err", f"{v}_error", f"{v}error", f"e_{v}", f"d{v}")
        domain_err = FLUX_ERR_NAMES if resolved_domain is Domain.FLUX else MAG_ERR_NAMES
        err_candidates = derived + domain_err + GENERIC_ERR_NAMES
        error = _match("error", self.error, err_candidates, available, lower_to_actual)
        return ResolvedColumns(
            time=time, value=value, error=error, band=band, domain=resolved_domain
        )


__all__ = [
    "BAND_NAMES",
    "ERR_NAMES",
    "FLUX_ERR_NAMES",
    "FLUX_NAMES",
    "GENERIC_ERR_NAMES",
    "MAG_ERR_NAMES",
    "MAG_NAMES",
    "TIME_NAMES",
    "ColumnMap",
    "Domain",
    "ResolvedColumns",
    "coerce_domain",
    "infer_domain",
]
