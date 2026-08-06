"""Secret valves must reach Open WebUI's functions table as ciphertext.

This existed nowhere. `conftest` installed a `pydantic_core` stub unconditionally,
overwriting the REAL `core_schema` builders, so `EncryptedStr`'s chain validator was
never constructed and every secret valve was stored as a plain `str`. On top of that,
an autouse fixture deletes `WEBUI_SECRET_KEY` before every test, which makes
`encrypt()`/`decrypt()` the identity function -- so the assertions that looked like
they covered this reduced to `x == x`.

Net effect: changing the validator to store API keys in plain text passed all 5593
tests. These tests set a real key and assert the ciphertext property directly.
"""

from __future__ import annotations

from typing import Any

import re

import pytest

from open_webui_openrouter_pipe.core.config import EncryptedStr

_CREDENTIAL_NAME_RE = re.compile(
    r"(^|_)?(APIKEY|KEY|SECRET|PASSWORD|PASSPHRASE|CREDENTIAL|TOKEN|BEARER|SALT)S?(_|$)"
)

# Empty on purpose, and checked below: both former entries were already excluded by the
# type filter, so they decided nothing while reading as a reviewed clearance.
_NOT_A_SECRET: frozenset[str] = frozenset()


def _secret_valve_names(*, honour_exemptions: bool = True) -> list[str]:
    """Every string-typed valve whose name says it holds a credential."""
    from open_webui_openrouter_pipe import Pipe

    found = []
    for name, field in Pipe.Valves.model_fields.items():
        if not _CREDENTIAL_NAME_RE.search(name):
            continue
        if honour_exemptions and name in _NOT_A_SECRET:
            continue
        annotation = str(field.annotation)
        if "str" not in annotation and "EncryptedStr" not in annotation:
            continue
        found.append(name)
    return sorted(found)


_SECRET_VALVES = tuple(_secret_valve_names())


def test_the_secret_valve_rule_still_finds_the_known_secrets():
    """Guards the derivation itself: a rule that matches nothing passes everything."""
    assert set(_SECRET_VALVES) >= {
        "API_KEY",
        "ARTIFACT_ENCRYPTION_KEY",
        "SESSION_LOG_ZIP_PASSWORD",
    }, f"the credential-name rule no longer finds the known secrets; it found {_SECRET_VALVES}"


def test_every_valve_that_names_a_credential_is_encrypted_at_rest():
    """The check a hardcoded list could not make: it covers valves not yet written."""
    from open_webui_openrouter_pipe import Pipe

    plaintext = [
        name
        for name in _SECRET_VALVES
        if "EncryptedStr" not in str(Pipe.Valves.model_fields[name].annotation)
    ]
    assert not plaintext, (
        "these valves name a credential but are declared as plain str, so their value is "
        f"persisted and shown in clear: {plaintext}. Declare them EncryptedStr."
    )


@pytest.fixture
def _real_secret_key(monkeypatch):
    """Undo the suite-wide WEBUI_SECRET_KEY deletion for these tests only."""
    monkeypatch.setenv("WEBUI_SECRET_KEY", "gauntlet-unit-test-application-secret")
    return "gauntlet-unit-test-application-secret"


def test_the_encryption_chain_validator_is_actually_installed(_real_secret_key):
    """If pydantic never builds the validator, nothing below can fail."""
    from open_webui_openrouter_pipe.pipe import Pipe

    kwargs: dict[str, Any] = {"API_KEY": "sk-canary-plaintext"}
    valves = Pipe.Valves(**kwargs)
    assert isinstance(valves.API_KEY, EncryptedStr), (
        f"API_KEY came back as {type(valves.API_KEY).__name__}, not EncryptedStr -- "
        "pydantic did not run the chain validator, so nothing enforces encryption. "
        "Check that conftest is not stubbing the real pydantic_core.core_schema."
    )


@pytest.mark.parametrize("field", _SECRET_VALVES)
def test_a_secret_valve_is_ciphertext_at_rest(field, _real_secret_key, monkeypatch):
    """The stored value must not contain the plaintext."""
    from open_webui_openrouter_pipe.pipe import Pipe

    plaintext = f"sk-canary-{field.lower()}-do-not-store-me"
    kwargs: dict[str, Any] = {field: plaintext}
    valves = Pipe.Valves(**kwargs)
    stored = str(getattr(valves, field))

    assert stored != plaintext, (
        f"{field} is stored verbatim. It is persisted into Open WebUI's functions "
        "table, so the operator's secret is readable by anyone with database or "
        "backup access."
    )
    assert plaintext not in stored, (
        f"{field}'s plaintext appears inside the stored value {stored[:40]!r}..."
    )
    from open_webui_openrouter_pipe.core.config import EncryptedStr as _Enc

    with monkeypatch.context() as other:
        other.setenv("WEBUI_SECRET_KEY", "a-different-application-secret-entirely")
        clear = getattr(_Enc._get_encryption_key, "cache_clear", None)
        if callable(clear):
            clear()
        recovered = _Enc.decrypt(stored)
    assert recovered != plaintext, (
        f"{field}'s stored value decrypts under a DIFFERENT WEBUI_SECRET_KEY, so it is "
        "not keyed to this deployment's secret at all. Reversible encoding -- base64, "
        "a reversal, an XOR -- satisfies every other assertion here; only this one "
        "distinguishes encryption from obfuscation."
    )

    assert stored.startswith("encrypted:"), (
        f"{field} is stored as {stored[:40]!r}..., which does not carry the encrypted "
        "prefix, so decrypt() will hand the ciphertext back as if it were plaintext."
    )
    assert EncryptedStr.decrypt(stored) == plaintext, (
        f"{field} does not round-trip: the operator's configured value is unrecoverable"
    )


def test_encryption_is_a_no_op_without_an_application_secret(monkeypatch):
    """Open WebUI's own contract: no secret configured means no encryption.

    Pinned so the tests above cannot be "fixed" by making encryption unconditional,
    which would make every existing plain-text valve undecryptable on upgrade.
    """
    monkeypatch.delenv("WEBUI_SECRET_KEY", raising=False)
    assert EncryptedStr.encrypt("sk-plain") == "sk-plain"
    assert EncryptedStr.decrypt("sk-plain") == "sk-plain"


def test_every_exemption_actually_exempts_something():
    """An entry that changes no outcome is permission nobody asked for.

    The sibling census in test_swallowed_failure_diagnostics checks its exemptions in
    both directions for exactly this reason. Both names here are already excluded a line
    later by the annotation filter -- neither is `str`-typed -- so the list decides
    nothing today and would go on deciding nothing while a reader treats it as a
    reviewed record of what was considered and cleared.
    """
    from open_webui_openrouter_pipe import Pipe

    inert = sorted(
        name
        for name in _NOT_A_SECRET
        if name not in _secret_valve_names(honour_exemptions=False)
    )
    assert not inert, (
        f"these exemptions change nothing: {inert}. Each is already excluded by the "
        "type filter, so the list is granting permission nobody asked for. Remove them, "
        "or if one is meant to guard a future `str`-typed valve, add it when that valve "
        "arrives.\n"
        f"declared: {sorted(_NOT_A_SECRET)}\n"
        f"names the sweep would reach without the list: {_secret_valve_names(honour_exemptions=False)}"
    )
