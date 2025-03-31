import asyncio
import time
import random
import litellm
from litellm.exceptions import (
    RateLimitError,
    APIConnectionError,
    APIError,
    ServiceUnavailableError,
    Timeout,
)


class QPSRateLimiter:
    def __init__(self, requests_per_second: float, burst_capacity: int = 5):
        """
        Token bucket implementation for rate limiting
        """
        self.capacity = burst_capacity
        self.tokens = burst_capacity
        self.fill_rate = requests_per_second
        self.last_fill = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a token is available"""
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_fill
            self.tokens = min(
                self.capacity, self.tokens + elapsed * self.fill_rate
            )
            self.last_fill = now

            if self.tokens < 1:
                sleep_time = (1 - self.tokens) / self.fill_rate
                await asyncio.sleep(sleep_time)
                self.tokens += sleep_time * self.fill_rate
                self.tokens = min(self.tokens, self.capacity)

            self.tokens -= 1


async def evaluate_answer_llm(
    evaluator_system: str,
    evaluator_user: str,
    evaluator_model: str,
    question: str,
    answer: str,
    ground_truth: str,
    rate_limiter: QPSRateLimiter,
    request_timeout: float = 90.0,
    max_retries: int = 10,
    na_value: str = "N/A",
) -> bool:
    attempt = 0
    last_error = None

    while attempt < max_retries:
        try:
            await rate_limiter.acquire()

            user_content = evaluator_user.format(
                question=question, answer=answer, ground_truth=ground_truth
            )

            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=evaluator_model,
                    messages=[
                        {"role": "system", "content": evaluator_system},
                        {"role": "user", "content": user_content},
                    ],
                ),
                timeout=request_timeout,
            )

            response_text = response.choices[0].message.content.strip().lower()

            # Enhanced response validation
            if "1" in response_text:
                return 1
            if "0" in response_text:
                return 0
            if any(kw in response_text for kw in ["n/a", "unknown", "invalid"]):
                return na_value

            raise ValueError(f"Invalid LLM judge response: {response_text}")

        except (
            RateLimitError,
            APIConnectionError,
            APIError,
            Timeout,
            ServiceUnavailableError,
        ) as e:
            last_error = e
            attempt += 1

            if attempt >= max_retries:
                break

            backoff = calculate_backoff(e, attempt)
            await handle_retry_delay(e, attempt, backoff)

        except Exception as e:
            last_error = e
            break

    raise ServiceUnavailableError(
        f"Max retries ({max_retries}) exceeded. Last error: {str(last_error)}"
    )


def calculate_backoff(error: Exception, attempt: int) -> float:
    """Adaptive backoff with jitter and header-based delays"""
    base_delay = 1.2
    jitter = random.uniform(0.8, 1.2)

    if isinstance(error, RateLimitError):
        return getattr(error, "retry_after", (base_delay**attempt)) * jitter

    return min((base_delay**attempt) * jitter, 30)  # Cap at 30s


async def handle_retry_delay(error: Exception, attempt: int, delay: float):
    """Centralized retry handling with logging"""
    error_name = type(error).__name__
    print(f"{error_name} (attempt {attempt}), retrying in {delay:.2f}s")
    await asyncio.sleep(delay)
