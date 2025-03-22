import asyncio
import time
import random
import litellm
from litellm.exceptions import ServiceUnavailableError, RateLimitError


class RequestRateLimiter:
    def __init__(self, requests_per_minute: int):
        """
        A rate limiter that enforces only a per-minute rate limit.
        (e.g., requests_per_minute=600)
        """
        self.requests_per_minute = requests_per_minute
        self.request_timestamps = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """
        Wait until we can make a new request without exceeding the
        requests_per_minute limit.
        """
        while True:
            async with self.lock:
                now = time.monotonic()
                # Filter out timestamps older than 60 seconds
                one_minute_ago = now - 60
                self.request_timestamps = [
                    ts for ts in self.request_timestamps if ts > one_minute_ago
                ]

                # If we're under the per-minute threshold, proceed
                if len(self.request_timestamps) < self.requests_per_minute:
                    self.request_timestamps.append(now)
                    return

                # Otherwise, figure out how long until the first request
                # in the window is more than 60 seconds old
                wait_for_minute = 60 - (now - self.request_timestamps[0])

            # Release the lock before sleeping; ensure at least 0.1s wait
            await asyncio.sleep(max(0.1, wait_for_minute))


async def evaluate_answer_llm(
    evaluator_system: str,
    evaluator_user: str,
    evaluator_model: str,
    question: str,
    answer: str,
    ground_truth: str,
    rate_limiter: RequestRateLimiter,
    request_timeout: float = 30.0,
    max_retries: int = 5,
) -> bool:
    """
    Evaluates an answer using an LLM, with:
      - A rate-limiter to avoid exceeding requests_per_minute
      - Simple retry logic on RateLimitError (429) and timeouts
    """
    for attempt in range(max_retries):
        try:
            # Acquire permission to make a request under RPM constraint
            await rate_limiter.acquire()

            # Format the user prompt
            user_content = evaluator_user.format(
                question=question, answer=answer, ground_truth=ground_truth
            )

            # Call the LLM with a timeout
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

            # Convert LLM response into a boolean (true if "1" is in the text)
            llm_response = response.choices[0].message.content.strip().lower()
            return "1" in llm_response

        except asyncio.TimeoutError:
            # Retry on timeouts with exponential backoff
            if attempt < max_retries - 1:
                backoff_time = 2**attempt + random.random()
                print(
                    f"[Retry {attempt+1}/{max_retries}] "
                    f"Timeout - waiting {backoff_time:.2f}s before retry..."
                )
                await asyncio.sleep(backoff_time)
            else:
                raise ServiceUnavailableError(
                    "Request timed out after repeated attempts."
                )

        except RateLimitError:
            # Retry on 429 errors with exponential backoff
            if attempt < max_retries - 1:
                backoff_time = 2**attempt + random.random()
                print(
                    f"[Retry {attempt+1}/{max_retries}] "
                    f"429 RateLimitError - waiting {backoff_time:.2f}s before retry..."
                )
                await asyncio.sleep(backoff_time)
            else:
                raise ServiceUnavailableError(
                    "Hit rate limit too many times; giving up."
                )

        except ServiceUnavailableError:
            # Retry on other service-unavailable errors (e.g., 503)
            if attempt < max_retries - 1:
                backoff_time = 2**attempt + random.random()
                print(
                    f"[Retry {attempt+1}/{max_retries}] "
                    f"ServiceUnavailableError - waiting {backoff_time:.2f}s before retry..."
                )
                await asyncio.sleep(backoff_time)
            else:
                raise
