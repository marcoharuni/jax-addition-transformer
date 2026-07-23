"""Exercise a running Streamlit application in real desktop and mobile browsers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

CASES = (
    ("347 + 928", "1275"),
    ("What is 499 plus 501?", "1000"),
    ("Please add 91 and 909", "1000"),
    ("999 + 999", "1998"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    if args.artifacts:
        args.artifacts.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path="/usr/bin/google-chrome",
            args=["--no-sandbox"],
        )
        desktop = browser.new_page(viewport={"width": 1440, "height": 1000})
        desktop.goto(args.url, wait_until="networkidle", timeout=120_000)
        desktop.get_by_role("heading", name="JAX Addition Transformer").wait_for()
        desktop.evaluate("window.scrollTo(0, 0)")
        if args.artifacts:
            desktop.screenshot(path=args.artifacts / "desktop-initial.png", full_page=True)
        for index, (prompt, expected) in enumerate(CASES):
            chat_input = desktop.locator('[data-testid="stChatInput"] textarea')
            chat_input.fill(prompt)
            chat_input.press("Enter")
            answer = desktop.locator(".answer-number").nth(index)
            answer.wait_for(timeout=120_000)
            if answer.inner_text().strip() != expected:
                raise AssertionError(f"{prompt!r} produced {answer.inner_text()!r}, expected {expected}")

        for prompt, expected_text in (
            ("12 * 3", "addition only"),
            ("1000 + 1", "between 0 and 999"),
        ):
            chat_input = desktop.locator('[data-testid="stChatInput"] textarea')
            chat_input.fill(prompt)
            chat_input.press("Enter")
            desktop.get_by_text(expected_text, exact=False).last.wait_for(timeout=30_000)

        latency_text = desktop.locator(".latency").all_inner_texts()
        if len(latency_text) != len(CASES):
            raise AssertionError("each valid response must display inference latency")
        if args.artifacts:
            desktop.screenshot(path=args.artifacts / "desktop-results.png", full_page=True)

        mobile = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True)
        mobile.goto(args.url, wait_until="networkidle", timeout=120_000)
        mobile.get_by_role("heading", name="JAX Addition Transformer").wait_for()
        mobile.locator('[data-testid="stChatInput"] textarea').wait_for()
        mobile.evaluate("document.activeElement.blur(); window.scrollTo(0, 0)")
        mobile.locator(".editorial-masthead").scroll_into_view_if_needed()
        if args.artifacts:
            mobile.screenshot(path=args.artifacts / "mobile.png", full_page=True)
        browser.close()

    print(
        json.dumps(
            {
                "url": args.url,
                "valid_results": dict(CASES),
                "multiplication": "rejected",
                "out_of_range": "rejected",
                "desktop": "rendered",
                "mobile": "rendered",
                "latency_visible": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
