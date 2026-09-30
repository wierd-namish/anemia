# Cybersecurity & Data Privacy Specification

**Document Identifier:** ETH-SUB-TECH-008  
**Version:** 1.0  
**Date:** 2026-09-30  
**Ethics Submission Status:** Prepared for ethics submission (PENDING ETHICS APPROVAL)  

---

## 1. Technical Privacy Controls

1. **In-Memory Volatile Processing:** Fingernail images captured in the mobile browser canvas are streamed as binary payloads, decoded in volatile server RAM, and immediately discarded after inference. No raw patient images are written to persistent server storage.
2. **Zero Facial Biometrics:** The camera UI enforces a guide box constrained strictly to isolated distal digits. No facial, iris, or whole-body biometric data are ever acquired.
3. **De-Identification & Tokenization:** Clinical study records are indexed strictly by random UUID-4 identifiers (e.g. `GH-PEDI-001-A9F3`).
4. **Transport Security:** All client-server communication is encrypted using TLS 1.3 with AES-256-GCM cipher suites.
5. **Regulatory Alignment:** Designed in compliance with the **Ghana Data Protection Act, 2012 (Act 843)**.
