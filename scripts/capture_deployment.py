"""Capture evidence from a real running Flask app in a headless browser."""

from playwright.sync_api import sync_playwright, expect


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        channel="chrome", headless=True, args=["--no-sandbox"]
    )
    page = browser.new_page(viewport={"width": 1280, "height": 920}, device_scale_factor=1)
    page.goto("http://127.0.0.1:5000/", wait_until="networkidle")

    page.locator("#textToAnalyze").fill("I am very angry")
    page.locator("#analyzeButton").click()
    expect(page.locator("#system_response")).to_contain_text(
        "The dominant emotion is anger.", timeout=25000
    )
    page.screenshot(path="6b_deployment_test.png", full_page=True)

    page.locator("#textToAnalyze").fill("")
    page.locator("#analyzeButton").click()
    expect(page.locator("#system_response")).to_contain_text(
        "Invalid text! Please try again!", timeout=15000
    )
    page.screenshot(path="7c_error_handling_interface.png", full_page=True)
    browser.close()
