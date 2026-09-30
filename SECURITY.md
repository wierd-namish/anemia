# Security & Data Privacy Policy

## Patient Data Protection & Privacy

> [!CAUTION]
> **Strict Clinical Privacy Rule**: Identifiable patient images, hospital records, medical histories, and personally identifiable health information (PHI) must **never** be committed to Git or transmitted across unencrypted channels.

All testing within the software repository must strictly utilize **synthetic test fixtures** (labeled under `tests/fixtures/`) or properly de-identified, ethics-approved data partitions with verified zero leakage.

## Supported Versions

| Version | Supported | Security Updates |
| :--- | :--- | :--- |
| `1.0.x` | Yes | Active maintenance |
| `< 1.0.0` | No | Deprecated |

## Reporting a Vulnerability

We take the security and medical privacy of our systems seriously. If you discover a security vulnerability or potential privacy leak:

1. **Do not create a public GitHub issue.**
2. Send a detailed report to the security maintainers at: `security@anemia-ai.internal` (or repository owner).
3. Include:
   - Description of the vulnerability or data leakage risk.
   - Steps to reproduce.
   - Potential impact assessment.
4. The maintainers will respond within 48 hours to acknowledge receipt and coordinate a fix.

## Secrets & Credential Management

- Production deployments must inject API keys, database credentials, and certificates exclusively via environment variables or secret managers (e.g. AWS Secrets Manager, HashiCorp Vault).
- Never commit `.env` files to version control.
- All commits are verified with automated pre-commit secret scans (`scripts/run_quality_checks.py`).
