#!/usr/bin/env python3
"""Check OpenWebUI compatibility for pipeline tool backend parity.

This script verifies that the required functions and patterns from OpenWebUI
are available and have the expected signatures. Run this in CI to detect
breaking changes in OpenWebUI that could affect the pipeline.

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
    2 - OpenWebUI is not installed or cannot be imported
"""

from __future__ import annotations

import inspect
import sys
from typing import Any, Callable


def check_function_exists(module_path: str, function_name: str) -> bool:
    """Check if a function exists in the specified module."""
    try:
        parts = module_path.rsplit(".", 1)
        if len(parts) == 2:
            module_name, _ = parts
        else:
            module_name = module_path

        module = __import__(module_path, fromlist=[function_name])
        func = getattr(module, function_name, None)
        return callable(func)
    except (ImportError, AttributeError):
        return False


def check_function_signature(
    module_path: str,
    function_name: str,
    expected_params: list[str],
) -> tuple[bool, str]:
    """Check if a function has the expected parameter names."""
    try:
        module = __import__(module_path, fromlist=[function_name])
        func = getattr(module, function_name, None)
        if not callable(func):
            return False, f"Function {function_name} not found or not callable"

        sig = inspect.signature(func)
        actual_params = list(sig.parameters.keys())

        # Check if all expected params are present (allows extra params)
        missing = [p for p in expected_params if p not in actual_params]
        if missing:
            return False, f"Missing parameters: {missing}. Found: {actual_params}"

        return True, "OK"
    except Exception as e:
        return False, str(e)


def check_all() -> tuple[int, list[str]]:
    """Run all compatibility checks."""
    errors: list[str] = []
    checks_passed = 0
    checks_total = 0

    # Check 1: get_citation_source_from_tool_result exists
    checks_total += 1
    if not check_function_exists(
        "open_webui.utils.middleware",
        "get_citation_source_from_tool_result",
    ):
        errors.append(
            "FAIL: get_citation_source_from_tool_result not found in open_webui.utils.middleware"
        )
    else:
        checks_passed += 1
        print("PASS: get_citation_source_from_tool_result exists")

    # Check 2: get_citation_source_from_tool_result signature
    checks_total += 1
    ok, msg = check_function_signature(
        "open_webui.utils.middleware",
        "get_citation_source_from_tool_result",
        ["tool_name", "tool_params", "tool_result"],
    )
    if not ok:
        errors.append(f"FAIL: get_citation_source_from_tool_result signature: {msg}")
    else:
        checks_passed += 1
        print("PASS: get_citation_source_from_tool_result has expected parameters")

    # Check 3: apply_source_context_to_messages exists
    checks_total += 1
    if not check_function_exists(
        "open_webui.utils.middleware",
        "apply_source_context_to_messages",
    ):
        errors.append(
            "FAIL: apply_source_context_to_messages not found in open_webui.utils.middleware"
        )
    else:
        checks_passed += 1
        print("PASS: apply_source_context_to_messages exists")

    # Check 4: apply_source_context_to_messages signature
    checks_total += 1
    ok, msg = check_function_signature(
        "open_webui.utils.middleware",
        "apply_source_context_to_messages",
        ["request", "messages", "sources", "user_message"],
    )
    if not ok:
        errors.append(f"FAIL: apply_source_context_to_messages signature: {msg}")
    else:
        checks_passed += 1
        print("PASS: apply_source_context_to_messages has expected parameters")

    # Check 5: process_tool_result exists (for future use)
    checks_total += 1
    if not check_function_exists(
        "open_webui.utils.middleware",
        "process_tool_result",
    ):
        errors.append(
            "FAIL: process_tool_result not found in open_webui.utils.middleware"
        )
    else:
        checks_passed += 1
        print("PASS: process_tool_result exists")

    # Check 6: get_file_url_from_base64 exists (for future use)
    checks_total += 1
    if not check_function_exists(
        "open_webui.utils.files",
        "get_file_url_from_base64",
    ):
        errors.append(
            "FAIL: get_file_url_from_base64 not found in open_webui.utils.files"
        )
    else:
        checks_passed += 1
        print("PASS: get_file_url_from_base64 exists")

    print(f"\n{checks_passed}/{checks_total} checks passed")
    return (0 if not errors else 1), errors


def main() -> int:
    """Run compatibility checks and return exit code."""
    print("OpenWebUI Compatibility Check")
    print("=" * 40)
    print()

    # First check if OpenWebUI can be imported at all
    try:
        import open_webui  # noqa: F401
    except ImportError:
        print("ERROR: OpenWebUI is not installed or cannot be imported.")
        print("This script requires OpenWebUI >= 0.7.0 to be installed.")
        return 2

    # Try to get version
    try:
        from open_webui import __version__

        print(f"OpenWebUI version: {__version__}")
    except (ImportError, AttributeError):
        print("OpenWebUI version: unknown")
    print()

    exit_code, errors = check_all()

    if errors:
        print("\nErrors found:")
        for error in errors:
            print(f"  - {error}")
        print("\nThese errors indicate breaking changes in OpenWebUI.")
        print("Please update the pipeline code to match the new API.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
