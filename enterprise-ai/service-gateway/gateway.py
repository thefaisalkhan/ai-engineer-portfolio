"""
API Gateway: Rate Limiting, JWT Auth, Request Routing, Structured Logging
Every microservice system needs a gateway layer. This implements the core patterns.
"""

import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional


# ─── JWT (minimal, no external deps) ─────────────────────────────────────────

import base64


def base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def base64url_decode(s: str) -> bytes:
    pad = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * pad)


def create_jwt(payload: dict, secret: str) -> str:
    header = base64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = base64url_encode(json.dumps(payload).encode())
    sig = hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
    return f"{header}.{body}.{base64url_encode(sig)}"


def verify_jwt(token: str, secret: str) -> Optional[dict]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header, body, sig = parts
        expected_sig = hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(base64url_decode(sig), expected_sig):
            return None
        payload = json.loads(base64url_decode(body))
        if "exp" in payload and payload["exp"] < time.time():
            return None
        return payload
    except Exception:
        return None


# ─── Rate Limiter (Token Bucket) ──────────────────────────────────────────────

@dataclass
class TokenBucket:
    """
    Token bucket rate limiter per user.
    Allows burst traffic while enforcing sustained rate.
    """
    capacity: float
    refill_rate: float        # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self):
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def consume(self, n: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


class RateLimiter:
    def __init__(self, requests_per_minute: int = 60, burst: int = 10):
        self.rps = requests_per_minute / 60.0
        self.burst = burst
        self._buckets: dict[str, TokenBucket] = {}

    def check(self, user_id: str) -> tuple[bool, dict]:
        if user_id not in self._buckets:
            self._buckets[user_id] = TokenBucket(capacity=self.burst, refill_rate=self.rps)
        bucket = self._buckets[user_id]
        allowed = bucket.consume()
        return allowed, {
            "tokens_remaining": round(bucket.tokens, 1),
            "refill_rate_rps": self.rps,
            "retry_after": round((1 - bucket.tokens) / self.rps, 2) if not allowed else 0,
        }


# ─── Request / Response ───────────────────────────────────────────────────────

@dataclass
class GatewayRequest:
    method: str
    path: str
    headers: dict = field(default_factory=dict)
    body: dict = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)


@dataclass
class GatewayResponse:
    status: int
    body: dict
    headers: dict = field(default_factory=dict)
    latency_ms: float = 0.0


# ─── Route Registry ───────────────────────────────────────────────────────────

@dataclass
class Route:
    method: str
    path: str
    service: str
    handler: Callable
    auth_required: bool = True
    rate_limit_tier: str = "default"


# ─── Gateway ──────────────────────────────────────────────────────────────────

class APIGateway:
    JWT_SECRET = "change-me-in-production"  # use env var in prod

    def __init__(self):
        self.routes: list[Route] = []
        self.rate_limiter = RateLimiter(requests_per_minute=60, burst=10)
        self._request_log: list[dict] = []

    def register(self, route: Route):
        self.routes.append(route)

    def _find_route(self, method: str, path: str) -> Optional[Route]:
        return next(
            (r for r in self.routes if r.method == method and r.path == path),
            None,
        )

    def _authenticate(self, headers: dict) -> tuple[bool, Optional[dict]]:
        auth = headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return False, None
        token = auth[len("Bearer "):]
        payload = verify_jwt(token, self.JWT_SECRET)
        return payload is not None, payload

    def handle(self, req: GatewayRequest) -> GatewayResponse:
        start = time.perf_counter()

        route = self._find_route(req.method, req.path)
        if route is None:
            return GatewayResponse(status=404, body={"error": "Not found"})

        # Auth
        user_id = "anonymous"
        if route.auth_required:
            ok, payload = self._authenticate(req.headers)
            if not ok:
                return GatewayResponse(status=401, body={"error": "Unauthorized"})
            user_id = payload.get("sub", "unknown")

        # Rate limit
        allowed, rl_info = self.rate_limiter.check(user_id)
        if not allowed:
            return GatewayResponse(
                status=429,
                body={"error": "Rate limit exceeded", **rl_info},
                headers={"Retry-After": str(rl_info["retry_after"])},
            )

        # Route to service
        try:
            response_body = route.handler(req.body, user_id=user_id)
            status = 200
        except ValueError as e:
            response_body = {"error": str(e)}
            status = 400
        except Exception as e:
            response_body = {"error": "Internal error"}
            status = 500

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        # Structured access log
        log_entry = {
            "request_id": req.request_id,
            "method": req.method,
            "path": req.path,
            "user_id": user_id,
            "status": status,
            "latency_ms": latency_ms,
            "service": route.service,
        }
        self._request_log.append(log_entry)
        print(json.dumps(log_entry))

        return GatewayResponse(
            status=status,
            body={"request_id": req.request_id, **response_body},
            latency_ms=latency_ms,
        )

    def create_token(self, user_id: str, role: str = "user", ttl_seconds: int = 3600) -> str:
        return create_jwt(
            {"sub": user_id, "role": role, "exp": time.time() + ttl_seconds},
            self.JWT_SECRET,
        )

    def metrics(self) -> dict:
        if not self._request_log:
            return {}
        latencies = sorted(r["latency_ms"] for r in self._request_log)
        n = len(latencies)
        errors = sum(1 for r in self._request_log if r["status"] >= 400)
        return {
            "total_requests": n,
            "error_rate": round(errors / n, 4),
            "latency_p50": latencies[n // 2],
            "latency_p95": latencies[int(n * 0.95)],
            "latency_p99": latencies[int(n * 0.99)],
        }


# ─── Demo ────────────────────────────────────────────────────────────────────

def rag_handler(body: dict, user_id: str = "anonymous") -> dict:
    if not body.get("question"):
        raise ValueError("question is required")
    return {"answer": f"[RAG service] Answering: {body['question'][:50]}...", "confidence": 0.92}

def health_handler(body: dict, user_id: str = "anonymous") -> dict:
    return {"status": "ok", "services": {"rag": "healthy", "embedding": "healthy"}}


if __name__ == "__main__":
    gateway = APIGateway()
    gateway.register(Route("POST", "/v1/ask",    "rag-service",  rag_handler,    auth_required=True))
    gateway.register(Route("GET",  "/v1/health", "gateway",      health_handler, auth_required=False))

    # Create token
    token = gateway.create_token("user-123", role="user")
    print(f"JWT: {token[:50]}...\n")

    # Unauthenticated request
    resp = gateway.handle(GatewayRequest("POST", "/v1/ask", body={"question": "What is RAG?"}))
    print(f"No auth: {resp.status} {resp.body}\n")

    # Authenticated requests
    headers = {"Authorization": f"Bearer {token}"}
    for i in range(5):
        req = GatewayRequest("POST", "/v1/ask", headers=headers, body={"question": f"Question {i}?"})
        resp = gateway.handle(req)
        print(f"Request {i}: status={resp.status}, latency={resp.latency_ms}ms")

    # Health check (no auth)
    resp = gateway.handle(GatewayRequest("GET", "/v1/health"))
    print(f"\nHealth: {resp.body}")

    print(f"\n=== Gateway Metrics ===")
    print(json.dumps(gateway.metrics(), indent=2))
