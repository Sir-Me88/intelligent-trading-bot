class Position:
    # ...existing code...

    def update_current_price(self, price: float):
        """Update current price and unrealized PnL in pips * lots (test expectation)."""
        self.current_price = price

        pair = getattr(self, "pair", "") or ""
        pair_norm = pair.replace("/", "").replace("_", "").upper()

        # pip multiplier: 100 for JPY pairs, else 10000
        pip_factor = 100 if ("JPY" in pair_norm or pair_norm.endswith("JPY")) else 10000

        try:
            entry = float(getattr(self, "entry_price", 0.0))
            size = float(getattr(self, "size", 0.0))
            direction = getattr(self, "direction", "buy").lower()

            # pip difference (price units -> pips)
            if direction in ("buy", "long", "0"):
                pip_diff = (self.current_price - entry)
            else:
                pip_diff = (entry - self.current_price)

            # convert to integer pip count then compute unrealized pnl as pips * lots
            pip_count = int(round(pip_diff * pip_factor))
            self.unrealized_pnl = pip_count * size

        except Exception:
            self.unrealized_pnl = 0.0