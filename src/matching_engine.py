from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

BID, ASK = "bid", "ask"


@dataclass
class Order:
    oid: int
    side: str
    tick: int
    size: float
    is_agent: bool = False
    prev: Optional["Order"] = None
    next: Optional["Order"] = None


@dataclass
class Fill:
    maker_oid: int
    taker_is_agent: bool
    maker_is_agent: bool
    tick: int
    price: float
    size: float
    aggressor_side: str
    t: float


class PriceLevel:
    __slots__ = ("head", "tail", "total_size")

    def __init__(self):
        self.head = None; self.tail = None; self.total_size = 0.0

    def append(self, o: Order):
        o.prev = self.tail; o.next = None
        if self.tail: self.tail.next = o
        else: self.head = o
        self.tail = o
        self.total_size += o.size

    def unlink(self, o: Order):
        if o.prev: o.prev.next = o.next
        else: self.head = o.next
        if o.next: o.next.prev = o.prev
        else: self.tail = o.prev
        self.total_size -= o.size
        o.prev = o.next = None

    def empty(self):
        return self.head is None


class OrderBook:
    def __init__(self, cfg, name: str):
        self.cfg = cfg
        self.name = name
        self.levels: list = [None] * (cfg.n_ticks + 1)
        self.registry: dict[int, Order] = {}
        self.best_bid_tick: Optional[int] = None
        self.best_ask_tick: Optional[int] = None
        self._oid = 0
        self.rejects = 0

    def _next_oid(self):
        self._oid += 1
        return self._oid

    def _refresh_best_bid(self):
        t = self.best_bid_tick
        if t is None:
            return
        while t >= 0:
            lv = self.levels[t]
            if lv is not None and not lv.empty():
                self.best_bid_tick = t; return
            t -= 1
        self.best_bid_tick = None

    def _refresh_best_ask(self):
        t = self.best_ask_tick
        if t is None:
            return
        n = self.cfg.n_ticks
        while t <= n:
            lv = self.levels[t]
            if lv is not None and not lv.empty():
                self.best_ask_tick = t; return
            t += 1
        self.best_ask_tick = None

    def add_limit(self, side, price, size, t, is_agent=False):
        tick = self.cfg.price_to_tick(price)
        if not (0 <= tick <= self.cfg.n_ticks):
            self.rejects += 1
            return None, []
        fills = []
        if side == BID and self.best_ask_tick is not None and tick >= self.best_ask_tick:
            size, fills = self._consume(ASK, size, t, is_agent, tick, BID)
        elif side == ASK and self.best_bid_tick is not None and tick <= self.best_bid_tick:
            size, fills = self._consume(BID, size, t, is_agent, tick, ASK)
        if size <= 1e-12:
            return None, fills
        oid = self._next_oid()
        o = Order(oid, side, tick, size, is_agent=is_agent)
        lv = self.levels[tick]
        if lv is None:
            lv = PriceLevel(); self.levels[tick] = lv
        lv.append(o)
        self.registry[oid] = o
        if side == BID and (self.best_bid_tick is None or tick > self.best_bid_tick):
            self.best_bid_tick = tick
        if side == ASK and (self.best_ask_tick is None or tick < self.best_ask_tick):
            self.best_ask_tick = tick
        return oid, fills

    def market_order(self, side, size, t, taker_is_agent=False):
        opp = ASK if side == BID else BID
        _, fills = self._consume(opp, size, t, taker_is_agent, None, side)
        return fills

    def _consume(self, maker_side, size, t, taker_is_agent, limit_tick, aggressor):
        fills = []
        remaining = size
        while remaining > 1e-12:
            best = self.best_bid_tick if maker_side == BID else self.best_ask_tick
            if best is None:
                break
            if limit_tick is not None:
                if maker_side == ASK and best > limit_tick: break
                if maker_side == BID and best < limit_tick: break
            lv = self.levels[best]
            while remaining > 1e-12 and lv.head is not None:
                maker = lv.head
                traded = min(remaining, maker.size)
                price = self.cfg.tick_to_price(best)
                fills.append(Fill(maker.oid, taker_is_agent, maker.is_agent,
                                  best, price, traded, aggressor, t))
                maker.size -= traded
                lv.total_size -= traded
                remaining -= traded
                if maker.size <= 1e-12:
                    lv.unlink(maker)
                    self.registry.pop(maker.oid, None)
            if lv.empty():
                self.levels[best] = None
                if maker_side == BID: self._refresh_best_bid()
                else: self._refresh_best_ask()
        return remaining, fills

    def cancel(self, oid):
        o = self.registry.pop(oid, None)
        if o is None:
            return False
        lv = self.levels[o.tick]
        if lv is not None:
            lv.unlink(o)
            if lv.empty():
                self.levels[o.tick] = None
                if o.side == BID and o.tick == self.best_bid_tick:
                    self._refresh_best_bid()
                if o.side == ASK and o.tick == self.best_ask_tick:
                    self._refresh_best_ask()
        return True

    def bbo(self):
        bt, at = self.best_bid_tick, self.best_ask_tick
        bp = self.cfg.tick_to_price(bt) if bt is not None else float("nan")
        bs = self.levels[bt].total_size if bt is not None else float("nan")
        ap = self.cfg.tick_to_price(at) if at is not None else float("nan")
        as_ = self.levels[at].total_size if at is not None else float("nan")
        return (bp, bs, ap, as_)

    def depth(self, side, N):
        out = []
        if side == BID:
            t = self.best_bid_tick
            while t is not None and t >= 0 and len(out) < N:
                lv = self.levels[t]
                if lv is not None and not lv.empty():
                    out.append((self.cfg.tick_to_price(t), lv.total_size))
                t -= 1
        else:
            t = self.best_ask_tick; n = self.cfg.n_ticks
            while t is not None and t <= n and len(out) < N:
                lv = self.levels[t]
                if lv is not None and not lv.empty():
                    out.append((self.cfg.tick_to_price(t), lv.total_size))
                t += 1
        return out

    def mid(self):
        bp, _, ap, _ = self.bbo()
        if bp != bp or ap != ap:
            return float("nan")
        return 0.5 * (bp + ap)

    def assert_invariants(self):
        if self.best_bid_tick is not None and self.best_ask_tick is not None:
            assert self.best_bid_tick < self.best_ask_tick, \
                f"crossed book on {self.name}"
        seen = set()
        for tick, lv in enumerate(self.levels):
            if lv is None:
                continue
            tot = 0.0; node = lv.head; prev = None
            while node is not None:
                assert node.oid in self.registry, f"unlinked-but-resting {node.oid}"
                assert self.registry[node.oid] is node
                assert node.prev is prev, "DLL prev pointer broken"
                assert node.tick == tick
                seen.add(node.oid)
                tot += node.size; prev = node; node = node.next
            assert abs(tot - lv.total_size) < 1e-9, "level size mismatch"
            assert lv.tail is prev, "DLL tail pointer broken"
        assert seen == set(self.registry.keys()), "registry/DLL set mismatch"