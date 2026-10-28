"""Thin REST client for the Delta Exchange v2 API (public + signed endpoints)."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from urllib.parse import urlencode

import requests

log = logging.getLogger(__name__)

RESOLUTION_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600,
    "2h": 7200, "4h": 14400, "6h": 21600, "1d": 86400, "1w": 604800,
}


class DeltaAPIError(Exception):
    pass


def sign(secret: str, method: str, timestamp: str, path: str, query: str, body: str) -> str:
    """HMAC-SHA256 over method + timestamp + path + query_string + body (hex)."""
    message = method + timestamp + path + query + body
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


class DeltaClient:
    def __init__(self, base_url: str, api_key: str = "", api_secret: str = "", timeout: float = 15):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "deltabot/1.0", "Content-Type": "application/json"})

    # ------------------------------------------------------------------ core
    def _request(self, method: str, path: str, params: dict | None = None,
                 body: dict | None = None, auth: bool = False, retries: int = 3):
        path = "/v2" + path
        query = "?" + urlencode(params) if params else ""
        payload = json.dumps(body, separators=(",", ":")) if body is not None else ""
        last_err: Exception | None = None
        for attempt in range(retries):
            headers = {}
            if auth:
                if not (self.api_key and self.api_secret):
                    raise DeltaAPIError("API key/secret missing - set DELTA_API_KEY and DELTA_API_SECRET in .env")
                ts = str(int(time.time()))
                headers.update({
                    "api-key": self.api_key,
                    "timestamp": ts,
                    "signature": sign(self.api_secret, method, ts, path, query, payload),
                })
            try:
                resp = self.session.request(method, self.base_url + path + query,
                                            data=payload or None, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = e
                log.warning("network error %s %s: %s (attempt %d)", method, path, e, attempt + 1)
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = DeltaAPIError(f"{resp.status_code} {resp.text[:300]}")
                wait = int(resp.headers.get("X-RATE-LIMIT-RESET", "0") or 0) / 1000 or 2 ** attempt
                log.warning("retryable %s on %s %s, sleeping %.1fs", resp.status_code, method, path, wait)
                time.sleep(min(wait, 30))
                continue
            try:
                data = resp.json()
            except ValueError:
                raise DeltaAPIError(f"{resp.status_code} non-JSON response: {resp.text[:300]}")
            if not resp.ok or not data.get("success", False):
                raise DeltaAPIError(f"{method} {path} -> {resp.status_code} {data.get('error', data)}")
            return data
        raise DeltaAPIError(f"{method} {path} failed after {retries} attempts: {last_err}")

    # ---------------------------------------------------------------- public
    def products(self, contract_types: str = "perpetual_futures,futures,call_options,put_options") -> list[dict]:
        out, after = [], None
        while True:
            params = {"contract_types": contract_types, "states": "live", "page_size": 500}
            if after:
                params["after"] = after
            data = self._request("GET", "/products", params)
            out.extend(data["result"])
            after = (data.get("meta") or {}).get("after")
            if not after:
                return out

    def tickers(self, contract_types: str = "perpetual_futures", underlying: str | None = None) -> list[dict]:
        params = {"contract_types": contract_types}
        if underlying:
            params["underlying_asset_symbols"] = underlying
        return self._request("GET", "/tickers", params)["result"]

    def ticker(self, symbol: str) -> dict:
        return self._request("GET", f"/tickers/{symbol}")["result"]

    def candles(self, symbol: str, resolution: str, start: int, end: int) -> list[dict]:
        params = {"resolution": resolution, "symbol": symbol, "start": int(start), "end": int(end)}
        return self._request("GET", "/history/candles", params)["result"] or []

    # --------------------------------------------------------------- private
    def wallet_balances(self) -> dict:
        return self._request("GET", "/wallet/balances", auth=True)

    def positions(self) -> list[dict]:
        return self._request("GET", "/positions/margined", auth=True)["result"]

    def position(self, product_id: int) -> dict:
        return self._request("GET", "/positions", {"product_id": product_id}, auth=True)["result"]

    def open_orders(self, product_id: int | None = None) -> list[dict]:
        params = {"states": "open,pending"}
        if product_id:
            params["product_ids"] = str(product_id)
        return self._request("GET", "/orders", params, auth=True)["result"]

    def set_leverage(self, product_id: int, leverage: int) -> dict:
        return self._request("POST", f"/products/{product_id}/orders/leverage",
                             body={"leverage": str(leverage)}, auth=True)["result"]

    def place_order(self, body: dict) -> dict:
        # Orders are NOT retried automatically: a timeout could mean the order went through.
        return self._request("POST", "/orders", body=body, auth=True, retries=1)["result"]

    def cancel_all(self, product_id: int) -> None:
        self._request("DELETE", "/orders/all", body={
            "product_id": product_id, "cancel_limit_orders": True,
            "cancel_stop_orders": True, "cancel_reduce_only_orders": True,
        }, auth=True)
