import asyncio
import time
import random
import litellm
from litellm.exceptions import ServiceUnavailableError, RateLimitError


class QPSRateLimiter:
    def __init__(self, requests_per_second: int):
        """
        Simplified rate limiter enforcing only QPS
        (QPS alone determines RPM: QPS*60 = RPM)
        """
        self.requests_per_second = requests_per_second
        self.request_timestamps = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait until we can make a new request without exceeding QPS limit"""
        while True:
            async with self.lock:
                now = time.monotonic()
                one_second_ago = now - 1

                # Maintain only timestamps from the last second
                self.request_timestamps = [
                    ts for ts in self.request_timestamps if ts > one_second_ago
                ]

                if len(self.request_timestamps) < self.requests_per_second:
                    self.request_timestamps.append(now)
                    return

                # Calculate wait time until the oldest request window expires
                oldest_ts = self.request_timestamps[0]
                sleep_time = max(0.1, 1 - (now - oldest_ts))

            # Release lock before sleeping
            await asyncio.sleep(sleep_time)


async def evaluate_answer_llm(
    evaluator_system: str,
    evaluator_user: str,
    evaluator_model: str,
    question: str,
    answer: str,
    ground_truth: str,
    rate_limiter: QPSRateLimiter,
    request_timeout: float = 30.0,
    max_retries: int = 15,
    na_value: str = "N/A",
) -> bool:
    for attempt in range(max_retries):
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
            response = response.choices[0].message.content.strip().lower()
            return 1 if "1" in response else 0 if "0" in response else na_value

        except asyncio.TimeoutError:
            if attempt >= max_retries - 1:
                raise ServiceUnavailableError("Max timeout retries exceeded")

            backoff = min(
                2**attempt + random.uniform(0, 0.5), 30
            )  # Capped backoff
            print(f"Timeout (attempt {attempt+1}), retrying in {backoff:.2f}s")
            await asyncio.sleep(backoff)

        except RateLimitError as e:
            if attempt >= max_retries - 1:
                raise ServiceUnavailableError(
                    f"Persistent rate limits: {str(e)}"
                )

            # Use retry-after header if available
            backoff = getattr(
                e, "retry_after", 2**attempt + random.uniform(0, 0.5)
            )
            print(
                f"Rate limited (attempt {attempt+1}), retrying in {backoff:.2f}s"
            )
            await asyncio.sleep(backoff)

        except ServiceUnavailableError as e:
            if attempt >= max_retries - 1:
                raise ServiceUnavailableError(f"Service unavailable: {str(e)}")

            backoff = 2**attempt + random.uniform(0, 0.5)
            print(
                f"Service error (attempt {attempt+1}), retrying in {backoff:.2f}s"
            )
            await asyncio.sleep(backoff)
