# Paper-trading ledger

`ledger.jsonl` holds one line per forecast, written by `python scripts/pm.py forecast`
and updated by `python scripts/pm.py settle`. Each line records Claude's probability
(made without seeing the price), the quote fetched afterwards, the paper bet, and,
once the market resolves, the outcome and P&L. No real money is involved.
