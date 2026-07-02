"""Domain data types (stdlib dataclasses).

These are the normalised internal representations every data adapter must
produce and every engine consumes. See docs/02 for the normalisation rules
(unified symbol, timezone, currency, corporate-action handling).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional


class Market(str, Enum):
    US = "US"   # United States
    HK = "HK"   # Hong Kong
    CN = "CN"   # Mainland China (A-shares)


class AssetClass(str, Enum):
    EQUITY = "equity"
    BOND = "bond"
    ETF = "etf"
    OPTION = "option"
    FUTURE = "future"
    CASH = "cash"


class OptionRight(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class Instrument:
    """A tradable instrument. ``symbol`` is the internal unified id, e.g.
    ``US:AAPL``, ``HK:00700``, ``CN:600519``."""
    symbol: str
    market: Market
    asset_class: AssetClass
    currency: str
    name: str = ""
    sector: str = ""
    # Derivative-specific fields (None for cash equities).
    underlying: Optional[str] = None
    strike: Optional[float] = None
    expiry: Optional[date] = None  # option expiry OR bond maturity
    right: Optional[OptionRight] = None
    multiplier: float = 1.0  # contract multiplier for options/futures
    # Fixed-income fields (None for non-bonds). ``expiry`` doubles as maturity.
    coupon_rate: Optional[float] = None  # annual coupon as a fraction of face
    coupon_freq: int = 2                 # coupon payments per year

    @property
    def is_derivative(self) -> bool:
        return self.asset_class in (AssetClass.OPTION, AssetClass.FUTURE)


@dataclass
class PriceSeries:
    """Adjusted close prices for one instrument, aligned to ``dates``."""
    symbol: str
    dates: List[date]
    closes: List[float]
    currency: str = "USD"

    def __post_init__(self) -> None:
        if len(self.dates) != len(self.closes):
            raise ValueError(f"{self.symbol}: dates/closes length mismatch")

    def returns(self) -> List[float]:
        """Simple period-over-period returns."""
        c = self.closes
        return [(c[i] / c[i - 1] - 1.0) for i in range(1, len(c))]

    def last(self) -> float:
        return self.closes[-1]


@dataclass
class Position:
    """A holding in the portfolio. ``quantity`` is signed (negative = short)."""
    instrument: Instrument
    quantity: float
    avg_cost: float          # per unit, in instrument currency
    last_price: float        # per unit, in instrument currency

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price * self.instrument.multiplier

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_cost * self.instrument.multiplier

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def is_long(self) -> bool:
        return self.quantity > 0


@dataclass
class Portfolio:
    positions: List[Position] = field(default_factory=list)
    cash: Dict[str, float] = field(default_factory=dict)  # currency -> amount
    base_currency: str = "USD"
    # FX rates: currency -> units of base_currency per 1 unit of currency.
    fx_rates: Dict[str, float] = field(default_factory=dict)
    as_of: Optional[date] = None

    def fx(self, currency: str) -> float:
        if currency == self.base_currency:
            return 1.0
        return self.fx_rates.get(currency, 1.0)

    def position_value_base(self, pos: Position) -> float:
        return pos.market_value * self.fx(pos.instrument.currency)

    def gross_exposure(self) -> float:
        return sum(abs(self.position_value_base(p)) for p in self.positions)

    def net_exposure(self) -> float:
        return sum(self.position_value_base(p) for p in self.positions)

    def cash_base(self) -> float:
        return sum(amt * self.fx(ccy) for ccy, amt in self.cash.items())

    def total_value(self) -> float:
        return self.net_exposure() + self.cash_base()

    def leverage(self) -> float:
        tv = self.total_value()
        return self.gross_exposure() / tv if tv else 0.0

    def weights(self) -> Dict[str, float]:
        """Signed weights of each position relative to total portfolio value."""
        tv = self.total_value()
        if tv == 0:
            return {p.instrument.symbol: 0.0 for p in self.positions}
        return {p.instrument.symbol: self.position_value_base(p) / tv for p in self.positions}

    def equity_positions(self) -> List[Position]:
        return [p for p in self.positions
                if p.instrument.asset_class in (AssetClass.EQUITY, AssetClass.ETF)]

    def option_positions(self) -> List[Position]:
        return [p for p in self.positions if p.instrument.asset_class == AssetClass.OPTION]

    def symbols(self) -> List[str]:
        return [p.instrument.symbol for p in self.positions]
